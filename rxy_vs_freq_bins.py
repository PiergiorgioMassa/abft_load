#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rxy_vs_freq_bins.py (v4)

Diagnostic plot: Rxy vs ancestral (Y) derived-allele frequency (pY_der), binned.

Output:
- 4 subplots (HOM_ANC, HET, HOM_DER, DER)
- x = frequency (bin midpoint)
- y = Rxy for that genotype category
- regression line + bootstrap confidence band (bootstrap over sites within bins)

Binning strategies:
1) fixed-width bins   (--binning fixed --bin-width 0.05)
2) quantile bins      (--binning quantile --n-quantiles 20 --quantile-ref-class MODERATE)

Within each bin, the script samples the SAME number of SNP from each class (without replacement),
so comparisons are frequency-matched locally (bin-by-bin).

Important subtlety (your request):
- Bin EDGES are defined once.
- If you resample individuals per site at each bootstrap replicate (equalize-across-sites / equalize-per-site individuals),
  you may NOT want to keep SNPs in the bin determined by the raw pY, because the resampled pY can shift slightly.
  With --bin-membership-each-rep, the script recomputes the *bin membership* at every replicate using the
  same per-site subsampling model, but keeps the edges fixed.

Implementation detail: per-site subsampling is done exactly (sampling individuals without replacement) via
multivariate hypergeometric draws from (c00,c01,c11) genotype counts. This is equivalent to sampling individuals,
but much faster and depends only on counts.

Rxy definition (same as your other scripts):
  Rxy = (fx*(1-fy)) / (fy*(1-fx))
where fx, fy are pooled frequencies (genotype or derived allele) with alpha=0.5 pseudocount.

"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


GENO_CATS_ORDER = ["HOM_ANC", "HET", "HOM_DER", "DER"]

# Palette coerente con i tuoi plot Rxy
CLASS_COLORS = {
    "MODERATE": "#ff7f0e",                 # arancione
    "LOW": "#ffd700",                      # giallo/oro
    "MODIFIER:intergenic_region": "#2ca02c",  # verde
    "HIGH": "#d62728",                     # rosso (non usato di default)
}

DEFAULT_CLASSES = ["MODERATE", "LOW", "MODIFIER:intergenic_region"]


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def read_popmap(path):
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if df.shape[1] < 2:
        raise ValueError("popmap deve avere almeno 2 colonne: SAMPLE\\tGROUP")
    return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))


def parse_ann_field(rec):
    try:
        ann = rec.info.get("ANN")
    except Exception:
        ann = None
    if ann is None:
        return []
    if isinstance(ann, (tuple, list)):
        return [str(x) for x in ann]
    return [str(ann)]


def pick_ann(ann_list, mode="first"):
    if not ann_list:
        return None
    if mode == "first":
        return ann_list[0]
    if mode == "canonical":
        for a in ann_list:
            if "CANONICAL" in a:
                return a
        return ann_list[0]
    if mode == "worst":
        for a in ann_list:
            parts = a.split("|")
            if len(parts) > 1 and parts[1] == "intergenic_region":
                return a
        # otherwise pick highest impact rank (best-effort)
        rank = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "MODIFIER": 0}
        best = ann_list[0]
        best_r = -1
        for a in ann_list:
            parts = a.split("|")
            imp = parts[2] if len(parts) > 2 else ""
            r = rank.get(imp, -1)
            if r > best_r:
                best = a
                best_r = r
        return best
    return ann_list[0]


def classify_from_ann(ann_entry):
    """
    Returns one of:
      HIGH, MODERATE, LOW, MODIFIER:intergenic_region
    Keeps only intergenic MODIFIER as separate class (as in your workflow).
    """
    if ann_entry is None:
        return None
    parts = ann_entry.split("|")
    if len(parts) < 3:
        return None
    effect = parts[1]
    impact = parts[2]
    if impact == "MODIFIER":
        if effect == "intergenic_region":
            return "MODIFIER:intergenic_region"
        return None
    if impact in ("HIGH", "MODERATE", "LOW"):
        return impact
    return None


def counts_from_gts(gt_tuples):
    """
    gt_tuples: list of (a1,a2) where a1,a2 in {0,1} or None.
    Returns c00,c01,c11,n_called (n individuals).
    """
    c00 = c01 = c11 = n = 0
    for gt in gt_tuples:
        if gt is None:
            continue
        a, b = gt
        if a is None or b is None:
            continue
        if a < 0 or b < 0:
            continue
        if a == 0 and b == 0:
            c00 += 1
        elif a != b:
            c01 += 1
        else:
            # 1/1
            c11 += 1
        n += 1
    return c00, c01, c11, n


def freq_geno(sum_c, sum_n, alpha=0.5):
    if sum_n <= 0:
        return float("nan")
    return (sum_c + alpha) / (sum_n + 2 * alpha)


def freq_der(sum_c01, sum_c11, sum_n, alpha=0.5):
    if sum_n <= 0:
        return float("nan")
    num = 2 * sum_c11 + sum_c01
    den = 2 * sum_n
    return (num + alpha) / (den + 2 * alpha)


def rxy_from_fx_fy(fx, fy):
    fx = min(max(fx, 1e-12), 1 - 1e-12)
    fy = min(max(fy, 1e-12), 1 - 1e-12)
    return (fx * (1 - fy)) / (fy * (1 - fx))


# ---------- bin edges ----------
def make_fixed_bins(min_f, max_f, width):
    if width <= 0:
        raise ValueError("bin-width deve essere > 0")
    edges = [min_f]
    x = min_f
    while x + width < max_f - 1e-12:
        x += width
        edges.append(x)
    edges.append(max_f)
    return np.array(edges, float)


def make_quantile_bins(pvals, n_q, min_f, max_f):
    p = np.asarray(pvals, float)
    p = p[np.isfinite(p)]
    p = p[(p >= min_f) & (p <= max_f)]
    if p.size < max(50, n_q * 5):
        log(f"[warn] pochi SNP per quantili: n={p.size}, n_quantiles={n_q}")
    qs = np.linspace(0, 1, n_q + 1)
    edges = np.quantile(p, qs)
    edges[0] = min_f
    edges[-1] = max_f
    # strictly increasing
    out = [float(edges[0])]
    for e in edges[1:]:
        if e > out[-1] + 1e-10:
            out.append(float(e))
    out = np.array(out, float)
    if out.size < 3:
        raise ValueError("Quantile edges degeneri (troppi valori identici). Prova meno quantili o usa fixed bins.")
    out[-1] = max_f
    return out


def bin_midpoints(edges):
    return (edges[:-1] + edges[1:]) / 2.0


def assign_bin_vec(p, edges):
    """Vectorized assign: returns int array, -1 for out of range/NaN."""
    p = np.asarray(p, float)
    out = np.searchsorted(edges, p, side="right") - 1
    bad = (~np.isfinite(p)) | (p < edges[0]) | (p > edges[-1])
    out = out.astype(int)
    out[bad] = -1
    max_i = len(edges) - 2
    out[out > max_i] = max_i
    return out


# ---------- exact subsampling via multivariate hypergeometric ----------
def subsample_counts_vec(c00, c01, c11, n, m, rng):
    """
    Vectorized exact sampling without replacement of m individuals from genotype counts.
    Returns sampled (c00s,c01s,c11s, ms).
    c00,c01,c11,n,m can be arrays (same shape) or scalars broadcastable.
    """
    c00 = np.asarray(c00, int)
    c01 = np.asarray(c01, int)
    c11 = np.asarray(c11, int)
    n = np.asarray(n, int)
    m = np.asarray(m, int)

    # invalid where m<=0 or m>n
    valid = (m > 0) & (n > 0) & (m <= n)
    c00s = np.zeros_like(n, dtype=int)
    c01s = np.zeros_like(n, dtype=int)
    c11s = np.zeros_like(n, dtype=int)
    ms = np.where(valid, m, 0)

    if np.any(valid):
        vg = valid
        # first draw for 0/0
        c00s[vg] = rng.hypergeometric(ngood=c00[vg], nbad=(n[vg] - c00[vg]), nsample=ms[vg])
        rem = ms[vg] - c00s[vg]
        # then draw for 0/1 from remaining pool
        # remaining pool size after removing all 0/0 individuals is n - c00
        nbad2 = n[vg] - c00[vg] - c01[vg]
        # hypergeometric requires non-negative nbad
        nbad2 = np.maximum(nbad2, 0)
        c01s[vg] = rng.hypergeometric(ngood=c01[vg], nbad=nbad2, nsample=rem)
        c11s[vg] = rem - c01s[vg]

    return c00s, c01s, c11s, ms


# ---------- regression band ----------
def fit_linpred_band(x, y_reps, xgrid, min_pts=3):
    x = np.asarray(x, float)
    xgrid = np.asarray(xgrid, float)
    n_reps = y_reps.shape[0]
    preds = np.full((n_reps, xgrid.size), np.nan, float)
    slopes = np.full(n_reps, np.nan, float)
    intercepts = np.full(n_reps, np.nan, float)

    for r in range(n_reps):
        y = y_reps[r, :]
        m = np.isfinite(y) & (y > 0) & np.isfinite(x)
        if np.sum(m) < min_pts:
            continue
        X = np.column_stack([np.ones(np.sum(m)), x[m]])
        ly = np.log(y[m])
        beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
        a, b = beta[0], beta[1]
        intercepts[r] = a
        slopes[r] = b
        preds[r, :] = np.exp(a + b * xgrid)

    pred_med = np.nanmedian(preds, axis=0)
    pred_lo = np.nanquantile(preds, 0.025, axis=0)
    pred_hi = np.nanquantile(preds, 0.975, axis=0)

    slope_med = float(np.nanmedian(slopes))
    slope_lo = float(np.nanquantile(slopes, 0.025))
    slope_hi = float(np.nanquantile(slopes, 0.975))
    return pred_med, pred_lo, pred_hi, slope_med, slope_lo, slope_hi


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--popmap", required=True)
    ap.add_argument("--group-x", required=True, help="Group label for X (modern)")
    ap.add_argument("--group-y", required=True, help="Group label for Y (ancient)")
    ap.add_argument("--annotation-mode", default="first", choices=["first", "canonical", "worst"])

    ap.add_argument("--min-present-frac", type=float, default=0.7)
    ap.add_argument("--max-callrate-diff", type=float, default=0.05)

    ap.add_argument("--equalize-per-site", default="none", choices=["none", "individuals"])
    ap.add_argument("--equalize-across-sites", action="store_true", help="Use M=min across sites and resample M per site per group per bootstrap.")
    ap.add_argument("--bootstrap", type=int, default=500, help="# bootstrap reps")

    ap.add_argument("--min-fy", type=float, default=0.01)
    ap.add_argument("--max-fy", type=float, default=0.99)

    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="Comma-separated classes to include (e.g. MODERATE,LOW,MODIFIER:intergenic_region). HIGH excluded by default.")
    ap.add_argument("--binning", choices=["fixed", "quantile"], default="quantile")
    ap.add_argument("--bin-width", type=float, default=0.05)
    ap.add_argument("--n-quantiles", type=int, default=20)
    ap.add_argument("--quantile-ref-class", default="MODERATE")
    ap.add_argument("--quantile-max-bin-width", type=float, default=None,
                    help="(Solo quantile) Esclude i bin con ampiezza (hi-lo) > questo valore. Es. 0.10. Default: disattivo.")

    ap.add_argument("--k-per-bin", type=int, default=100, help="Sites sampled per class per bin (without replacement).")
    ap.add_argument("--min-sites-per-bin", type=int, default=50, help="Used only if k-per-bin not provided.")

    ap.add_argument("--bin-membership-each-rep", action="store_true",
                    help="Recompute bin membership at each bootstrap replicate using per-site subsampling (edges fixed).")

    ap.add_argument("--dump-tsv", action="store_true")
    ap.add_argument("--out-prefix", required=True)

    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(42)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    log("[info] classi incluse: " + ", ".join(classes))

    popmap = read_popmap(args.popmap)

    vcf = pysam.VariantFile(args.vcf)

    # Identify samples in X/Y
    samples = list(vcf.header.samples)
    x_samples = [s for s in samples if popmap.get(s) == args.group_x]
    y_samples = [s for s in samples if popmap.get(s) == args.group_y]
    common = set(x_samples) & set(y_samples)
    if common:
        # should not happen typically
        pass
    if len(x_samples) == 0 or len(y_samples) == 0:
        raise ValueError("Nessun campione trovato per X o Y (controlla popmap e group labels).")

    log(f"After intersect with VCF: X={len(x_samples)}, Y={len(y_samples)}")

    # Accumulate per-class site counts
    per_class = {c: {"c00x": [], "c01x": [], "c11x": [], "nx": [],
                     "c00y": [], "c01y": [], "c11y": [], "ny": [],
                     "pY_raw": []} for c in classes}

    kept_total = 0
    n_total = 0

    # Precompute sample indices for speed
    x_idx = [samples.index(s) for s in x_samples]
    y_idx = [samples.index(s) for s in y_samples]

    for rec in vcf.fetch():
        n_total += 1
        if rec.alts is None or len(rec.alts) != 1:
            continue
        if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
            continue

        ann_list = parse_ann_field(rec)
        ann_entry = pick_ann(ann_list, mode=args.annotation_mode)
        cls = classify_from_ann(ann_entry)
        if cls is None or cls not in per_class:
            continue

        # collect gts
        x_gts = []
        y_gts = []
        for i in x_idx:
            gt = rec.samples[i].get("GT")
            x_gts.append(gt if gt is not None else None)
        for i in y_idx:
            gt = rec.samples[i].get("GT")
            y_gts.append(gt if gt is not None else None)

        c00x, c01x, c11x, nx = counts_from_gts(x_gts)
        c00y, c01y, c11y, ny = counts_from_gts(y_gts)
        if nx == 0 or ny == 0:
            continue

        # present fractions
        if nx / len(x_samples) < args.min_present_frac:
            continue
        if ny / len(y_samples) < args.min_present_frac:
            continue
        # callrate diff
        if abs((nx / len(x_samples)) - (ny / len(y_samples))) > args.max_callrate_diff:
            continue

        # pY raw
        pY = (2 * c11y + c01y) / float(2 * ny) if ny > 0 else float("nan")
        if (not np.isfinite(pY)) or pY < args.min_fy or pY > args.max_fy:
            continue

        d = per_class[cls]
        d["c00x"].append(c00x); d["c01x"].append(c01x); d["c11x"].append(c11x); d["nx"].append(nx)
        d["c00y"].append(c00y); d["c01y"].append(c01y); d["c11y"].append(c11y); d["ny"].append(ny)
        d["pY_raw"].append(pY)
        kept_total += 1

    log(f"Siti tenuti dopo filtri: {kept_total}")
    for c in classes:
        log(f" {c:<28}: {len(per_class[c]['pY_raw'])}")

    # Determine M (across sites) if requested
    m_target = None
    if args.equalize_across_sites:
        mins = []
        for c in classes:
            nx = np.array(per_class[c]["nx"], int)
            ny = np.array(per_class[c]["ny"], int)
            if nx.size == 0:
                continue
            mins.append(np.min(np.minimum(nx, ny)))
        if not mins:
            raise ValueError("Nessun sito dopo filtri per calcolare M.")
        m_target = int(min(mins))
        log(f"[equalize-across-sites] M=min={m_target}")

    # Prepare bin edges (once)
    if args.binning == "fixed":
        edges = make_fixed_bins(args.min_fy, args.max_fy, args.bin_width)
    else:
        ref = args.quantile_ref_class
        if ref not in per_class:
            raise ValueError(f"quantile-ref-class '{ref}' non è tra le classi incluse: {list(per_class.keys())}")
        edges = make_quantile_bins(per_class[ref]["pY_raw"], args.n_quantiles, args.min_fy, args.max_fy)

    mids = bin_midpoints(edges)
    n_bins = len(mids)
    log(f"[binning] {args.binning} -> n_bins={n_bins}")

    # Optional filter: exclude very wide quantile bins (too heterogeneous in frequency)
    bad_bins = set()
    if args.binning == "quantile" and args.quantile_max_bin_width is not None:
        L = float(args.quantile_max_bin_width)
        if L <= 0:
            raise ValueError("--quantile-max-bin-width deve essere > 0")
        widths = edges[1:] - edges[:-1]
        bad = np.where(widths > L)[0]
        bad_bins = set(int(x) for x in bad)
        if bad_bins:
            log(f"[binning] quantile max-bin-width={L:g}: esclusi {len(bad_bins)}/{n_bins} bin troppo larghi")
        else:
            log(f"[binning] quantile max-bin-width={L:g}: nessun bin escluso")

    # Convert lists to numpy arrays per class
    arr = {}
    for c in classes:
        d = per_class[c]
        arr[c] = {k: np.array(d[k], dtype=int if k != "pY_raw" else float) for k in d.keys()}
        # pY_raw float
        arr[c]["pY_raw"] = np.array(d["pY_raw"], dtype=float)

    # Build "static" pools using raw pY to decide usable bins (edges fixed)
    static_pools = {c: defaultdict(np.ndarray) for c in classes}
    for c in classes:
        bins_c = assign_bin_vec(arr[c]["pY_raw"], edges)
        for b in range(n_bins):
            if b in bad_bins:
                static_pools[c][b] = np.array([], dtype=int)
                continue
            idx = np.where(bins_c == b)[0]
            static_pools[c][b] = idx

    usable_bins = []
    k_bin = {}
    for b in range(n_bins):
        if b in bad_bins:
            continue
        counts = [int(static_pools[c][b].size) for c in classes]
        cap = min(counts) if counts else 0
        if args.k_per_bin is not None:
            if cap >= args.k_per_bin:
                usable_bins.append(b)
                k_bin[b] = int(args.k_per_bin)
        else:
            if cap >= args.min_sites_per_bin:
                usable_bins.append(b)
                k_bin[b] = int(cap)

    if not usable_bins:
        raise ValueError("Nessun bin utilizzabile con i criteri scelti. Riduci k-per-bin/min-sites-per-bin o cambia binning.")

    ubins = np.array(sorted(usable_bins), dtype=int)
    xvals = mids[ubins]
    n_bins_eff = n_bins - len(bad_bins)
    if bad_bins:
        log(f"[binning] usable bins: {len(usable_bins)}/{n_bins_eff} (post width-filter)")
    else:
        log(f"[binning] usable bins: {len(usable_bins)}/{n_bins}")
    log(f"[binning] K per bin: min={min(k_bin.values())} max={max(k_bin.values())}")

    if args.bin_membership_each_rep:
        log("[binning] bin membership per replica: ON (edges fissi, membership ricalcolata con subsampling)")
    else:
        log("[binning] bin membership per replica: OFF (membership da pY_raw)")

    # Storage: reps x bins
    y_reps = {c: {gc: np.full((args.bootstrap, ubins.size), np.nan, float) for gc in GENO_CATS_ORDER} for c in classes}

    # For debug: how many bins actually used each rep
    used_bins_per_rep = np.zeros(args.bootstrap, dtype=int)

    # Bootstrap loop
    for r in range(args.bootstrap):
        # For each class, compute (possibly resampled) per-site counts for this replicate
        rep_counts = {}
        rep_bins = {}
        rep_n = {}

        for c in classes:
            nx = arr[c]["nx"]; ny = arr[c]["ny"]
            c00x = arr[c]["c00x"]; c01x = arr[c]["c01x"]; c11x = arr[c]["c11x"]
            c00y = arr[c]["c00y"]; c01y = arr[c]["c01y"]; c11y = arr[c]["c11y"]

            if args.equalize_across_sites and m_target is not None:
                m = np.full_like(nx, m_target)
            elif args.equalize_per_site == "individuals":
                m = np.minimum(nx, ny)
            else:
                m = None  # no subsampling

            if (m is not None):
                # exact subsampling for both X and Y
                c00xs, c01xs, c11xs, mx = subsample_counts_vec(c00x, c01x, c11x, nx, m, rng)
                c00ys, c01ys, c11ys, my = subsample_counts_vec(c00y, c01y, c11y, ny, m, rng)
                # by construction mx==my==m' where valid; use my
                m_used = my
            else:
                c00xs, c01xs, c11xs, mx = c00x, c01x, c11x, nx
                c00ys, c01ys, c11ys, my = c00y, c01y, c11y, ny
                m_used = my

            rep_counts[c] = {
                "c00x": c00xs, "c01x": c01xs, "c11x": c11xs, "nx": mx,
                "c00y": c00ys, "c01y": c01ys, "c11y": c11ys, "ny": my,
            }
            # pY for binning from this replicate's Y counts
            den = 2.0 * m_used.astype(float)
            pY_rep = np.full_like(den, np.nan, dtype=float)
            ok = den > 0
            pY_rep[ok] = (2.0 * c11ys[ok] + c01ys[ok]) / den[ok]
            if args.bin_membership_each_rep:
                bins_c = assign_bin_vec(pY_rep, edges)
            else:
                bins_c = assign_bin_vec(arr[c]["pY_raw"], edges)
            rep_bins[c] = bins_c
            rep_n[c] = m_used  # n individuals used in Y (same m)

        used_bins = 0

        for j, b in enumerate(ubins):
            K = k_bin[int(b)]
            sampled_idx = {}
            okbin = True
            for c in classes:
                pool = np.where(rep_bins[c] == int(b))[0] if args.bin_membership_each_rep else static_pools[c][int(b)]
                if pool.size < K:
                    okbin = False
                    break
                sampled_idx[c] = rng.choice(pool, size=K, replace=False)
            if not okbin:
                continue

            used_bins += 1

            # For each class compute pooled fx,fy,Rxy from sampled indices
            for c in classes:
                idx = sampled_idx[c]
                rc = rep_counts[c]
                sum_c00x = int(rc["c00x"][idx].sum())
                sum_c01x = int(rc["c01x"][idx].sum())
                sum_c11x = int(rc["c11x"][idx].sum())
                sum_nx = int(rc["nx"][idx].sum())

                sum_c00y = int(rc["c00y"][idx].sum())
                sum_c01y = int(rc["c01y"][idx].sum())
                sum_c11y = int(rc["c11y"][idx].sum())
                sum_ny = int(rc["ny"][idx].sum())

                # HOM_ANC
                fx = freq_geno(sum_c00x, sum_nx)
                fy = freq_geno(sum_c00y, sum_ny)
                y_reps[c]["HOM_ANC"][r, j] = rxy_from_fx_fy(fx, fy)

                # HET
                fx = freq_geno(sum_c01x, sum_nx)
                fy = freq_geno(sum_c01y, sum_ny)
                y_reps[c]["HET"][r, j] = rxy_from_fx_fy(fx, fy)

                # HOM_DER
                fx = freq_geno(sum_c11x, sum_nx)
                fy = freq_geno(sum_c11y, sum_ny)
                y_reps[c]["HOM_DER"][r, j] = rxy_from_fx_fy(fx, fy)

                # DER allele
                fx = freq_der(sum_c01x, sum_c11x, sum_nx)
                fy = freq_der(sum_c01y, sum_c11y, sum_ny)
                y_reps[c]["DER"][r, j] = rxy_from_fx_fy(fx, fy)

        used_bins_per_rep[r] = used_bins

    log(f"[bootstrap] bins usati per replica: median={int(np.median(used_bins_per_rep))} min={int(used_bins_per_rep.min())} max={int(used_bins_per_rep.max())}")

    # Summaries per bin
    rows = []
    for c in classes:
        for gc in GENO_CATS_ORDER:
            mat = y_reps[c][gc]
            for j, b in enumerate(ubins):
                y = mat[:, j]
                y = y[np.isfinite(y)]
                if y.size == 0:
                    continue
                rows.append({
                    "class": c,
                    "geno_cat": gc,
                    "bin_id": int(b),
                    "bin_lo": float(edges[int(b)]),
                    "bin_hi": float(edges[int(b) + 1]),
                    "x_mid": float(xvals[j]),
                    "k_used": int(k_bin[int(b)]),
                    "n_reps_used": int(y.size),
                    "Rxy_mean": float(np.mean(y)),
                    "Rxy_median": float(np.median(y)),
                    "Rxy_ci2_5": float(np.quantile(y, 0.025)),
                    "Rxy_ci97_5": float(np.quantile(y, 0.975)),
                })

    df_bins = pd.DataFrame(rows)

    # Aliases used by plotting (bin edges)
    if not df_bins.empty:
        df_bins["lo"] = df_bins["bin_lo"]
        df_bins["hi"] = df_bins["bin_hi"]

    # Regression bands + slopes
    xgrid = np.linspace(float(np.min(xvals)), float(np.max(xvals)), 200)
    slope_rows = []
    bands = {c: {} for c in classes}

    for c in classes:
        for gc in GENO_CATS_ORDER:
            pred_med, pred_lo, pred_hi, s_med, s_lo, s_hi = fit_linpred_band(xvals, y_reps[c][gc], xgrid)
            bands[c][gc] = (pred_med, pred_lo, pred_hi)
            slope_rows.append({
                "class": c,
                "geno_cat": gc,
                "slope_mean": s_med,
                "slope_ci2_5": s_lo,
                "slope_ci97_5": s_hi,
            })

    df_slopes = pd.DataFrame(slope_rows)

    # Plot PDF
    pdf_path = f"{args.out_prefix}.rxy_vs_freq.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
        axes = axes.ravel()

        # Bin info shared across panels (only bins that have points)
        bin_info_all = (
            df_bins[["bin_id", "lo", "hi", "x_mid"]]
            .drop_duplicates()
            .sort_values(["lo", "hi", "bin_id"])
            .reset_index(drop=True)
        )
        if not bin_info_all.empty:
            bin_info_all["bin_idx"] = range(1, bin_info_all.shape[0] + 1)
        nbins = int(bin_info_all.shape[0])
        if nbins <= 8:
            label_step = 1
        elif nbins <= 14:
            label_step = 2
        elif nbins <= 20:
            label_step = 3
        else:
            label_step = 4

        for ax, gc in zip(axes, GENO_CATS_ORDER):
            # baseline
            ax.axhline(1.0, linewidth=1, linestyle="--", color="gray", alpha=0.7)
            ax.grid(True, linestyle=":", alpha=0.5)

            for c in classes:
                col = CLASS_COLORS.get(c, None)
                # points from df_bins
                sub = df_bins[(df_bins["class"] == c) & (df_bins["geno_cat"] == gc)].sort_values("x_mid")
                if sub.shape[0] > 0:
                    ax.errorbar(
                        sub["x_mid"].values,
                        sub["Rxy_mean"].values,
                        yerr=[sub["Rxy_mean"].values - sub["Rxy_ci2_5"].values,
                              sub["Rxy_ci97_5"].values - sub["Rxy_mean"].values],
                        fmt="o",
                        markersize=4,
                        linewidth=1,
                        capsize=2,
                        label=c,
                        color=col,
                    )

                pred_med, pred_lo, pred_hi = bands[c][gc]
                ax.plot(xgrid, pred_med, linewidth=2, color=col)
                ax.fill_between(xgrid, pred_lo, pred_hi, alpha=0.18, color=col)

            ax.set_title(gc)
            ax.set_ylabel("Rxy")
            # Adapt x-limits to the bins actually plotted and indicate bins without clutter
            if not bin_info_all.empty:
                xmin = float(bin_info_all["lo"].min())
                xmax = float(bin_info_all["hi"].max())
                pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.01
                ax.set_xlim(xmin - pad, xmax + pad)

                # Light alternating shading to show bin extents (clearer than many vertical lines)
                for ii, row in enumerate(bin_info_all.itertuples(index=False)):
                    if ii % 2 == 0:
                        ax.axvspan(row.lo, row.hi, color="black", alpha=0.04, linewidth=0)

                # Bin indices on a secondary x-axis (top) only for bottom row to avoid clutter
                if gc in ("HOM_DER", "DER") and nbins > 0:
                    ax_top = ax.secondary_xaxis("top")
                    ax_top.set_xticks(bin_info_all["x_mid"].values[::label_step])
                    ax_top.set_xticklabels([str(int(b)) for b in bin_info_all["bin_idx"].values[::label_step]], fontsize=8)
                    ax_top.set_xlabel("bin", fontsize=9)
            # Cap y-axis for HOM_DER subplot to avoid extreme autoscaling from rare events
            if gc == "HOM_DER":
                y0, y1 = ax.get_ylim()
                if y1 > 4.0:
                    ax.set_ylim(y0, 4.0)
        axes[2].set_xlabel("pY_der (ANC) [bin midpoint]")
        axes[3].set_xlabel("pY_der (ANC) [bin midpoint]")

        # Legend once
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)

        fig.suptitle(f"Rxy vs pY_der (binning={args.binning}, K/bin={args.k_per_bin})", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

    log(f"Wrote PDF: {pdf_path}")

    if args.dump_tsv:
        bins_path = f"{args.out_prefix}.rxy_vs_freq_bins.tsv"
        slopes_path = f"{args.out_prefix}.rxy_vs_freq_slopes.tsv"
        df_bins.to_csv(bins_path, sep="\t", index=False)
        df_slopes.to_csv(slopes_path, sep="\t", index=False)
        log(f"Wrote TSV: {bins_path}")
        log(f"Wrote slopes TSV: {slopes_path}")


if __name__ == "__main__":
    main()
