#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic (expectation-based) SFS by SnpEff class, per group (MOD/ANC).

- Proiezione deterministica a H con l'aspettativa ipergeometrica (nessun random).
- Filtri di presenza, per-gruppo MAC (haploid minor count), opzionalmente require-pass-both-groups.
- Esclusione opzionale di fixed, singleton, doubleton a livello di PMF.
- Output TSV con masse (raw) e versioni normalizzate (density/fraction), bootstrap opzionale.

Nessun plotting in questo file.

Autore: you
"""

import argparse, sys, math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pysam

# ---------------------- configuration ----------------------

CLASSES = ["HIGH", "MODERATE", "LOW", "MODIFIER:intergenic_region"]
IMPACT_RANK = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "MODIFIER": 0}

# ---------------------- CLI ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Deterministic (hypergeom expectation) SFS per group & SnpEff class (no plotting).")
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--popmap", required=True)
    ap.add_argument("--group-mod", default="MOD")
    ap.add_argument("--group-anc", default="ANC")
    ap.add_argument("--annotation-mode", choices=["first","canonical","worst"], default="first")

    ap.add_argument("--H", type=int, required=True, help="Projection haploids (per group). Use one fixed H for all.")
    ap.add_argument("--min-present-frac", type=float, default=0.7,
                    help="Min fraction called in the group (per-site) to consider the site for that group's SFS.")
    ap.add_argument("--include-fixed", action="store_true",
                    help="Include k'=0 and k'=H bins (default: exclude).")

    # Per-group MAC filter (haploid minor allele count)
    ap.add_argument("--pergroup-min-mac", type=int, default=0,
                    help="Require per-group minor allele count >= this value (haploids). "
                         "Use 3 to escludere per-group singletons e doubletons.")
    ap.add_argument("--require-pass-both-groups", action="store_true",
                    help="Keep a site only if BOTH groups pass presence, h>=H and per-group MAC. Else drop for both.")

    # Extra esclusioni a livello di PMF (spesso ridondanti se si usa per-group MAC)
    ap.add_argument("--exclude-singletons", action="store_true",
                    help="Exclude singletons: zero k'=1 e k'=H-1 (e minor=1 nel folded).")
    ap.add_argument("--exclude-doubletons", action="store_true",
                    help="Exclude doubletons: zero k'=2 e k'=H-2 (e minor=2 nel folded).")

    # Bootstrap su blocchi (finestre o cromosomi)
    ap.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap replicates (0 = skip).")
    ap.add_argument("--block-mode", choices=["chrom","window"], default="chrom")
    ap.add_argument("--block-size", type=int, default=5_000_000, help="Window size if --block-mode window.")
    ap.add_argument("--min-sites-per-block", type=int, default=10, help="Skip blocks with too few sites.")

    ap.add_argument("--out-prefix", default="stable_sfs")
    return ap.parse_args()

# ---------------------- SnpEff helpers ----------------------

def split_ann_entry(entry):
    parts = entry.split('|')
    if len(parts) < 3: parts += [""]*(3-len(parts))
    annotation = parts[1] if len(parts) > 1 else ""
    impact     = parts[2] if len(parts) > 2 else ""
    trailing   = parts[-1] if parts else ""
    return annotation, impact, trailing

def choose_ann_entry(ann_list, mode):
    if not ann_list: return None
    anns = list(ann_list)
    if mode == "first":
        return anns[0]
    elif mode == "canonical":
        for e in anns:
            ann, imp, tail = split_ann_entry(e)
            if "CANONICAL" in tail: return e
        return anns[0]
    elif mode == "worst":
        # Se presente intergenic, è la più "bassa" come priorità di filtro (qui manteniamo compat)
        for e in anns:
            ann, imp, tail = split_ann_entry(e)
            if ann == "intergenic_region": return e
        best_e, best_rank = None, -1
        for e in anns:
            ann, imp, tail = split_ann_entry(e)
            r = IMPACT_RANK.get(imp, -1)
            if r > best_rank:
                best_e, best_rank = e, r
        return best_e
    return None

def classify_from_ann(entry):
    if entry is None: return None
    ann, imp, _ = split_ann_entry(entry)
    if ann == "intergenic_region": return "MODIFIER:intergenic_region"
    if imp in ("HIGH","MODERATE","LOW"): return imp
    return None

# ---------------------- genotype utilities ----------------------

def derived_k_and_h(rec, sample_names):
    """Return (k, h, n_called_individuals) for given sample names."""
    k = 0; n_called = 0
    sdata = rec.samples
    for s in sample_names:
        sd = sdata.get(s)
        if sd is None: continue
        gt = sd.get('GT')
        if gt is None or len(gt) != 2: continue
        a, b = gt[0], gt[1]
        if a not in (0,1) or b not in (0,1): continue
        n_called += 1
        k += (a + b)  # 0/1 alleles
    h = 2 * n_called
    return k, h, n_called

# ---------------------- hypergeometric PMF ----------------------

def logC(n, r):
    # log binomial coefficient via lgamma
    if r < 0 or r > n: return -np.inf
    return math.lgamma(n+1) - math.lgamma(r+1) - math.lgamma(n-r+1)

def hypergeom_pmf_vec(k, h, H):
    """
    Return an array p[k'] for k'=0..H:
      P(K'=k' | draw H from h haploids with k derived) = C(k,k')*C(h-k,H-k') / C(h,H)
    """
    p = np.zeros(H+1, dtype=float)
    kmin = max(0, H - (h - k))
    kmax = min(H, k)
    denom = logC(h, H)
    for kp in range(kmin, kmax+1):
        val = logC(k, kp) + logC(h - k, H - kp) - denom
        p[kp] = math.exp(val)
    s = p.sum()
    if s > 0:
        p /= s
    return p

def fold_sfs(unfolded):
    """
    Fold unfolded array of len H+1 into M=floor(H/2)+1 bins (minor allele count).
    Bin j gets mass from j and H-j (except middle if H even).
    """
    H = len(unfolded) - 1
    M = H//2 + 1
    folded = np.zeros(M, dtype=float)
    for j in range(M):
        folded[j] = unfolded[j]
        mirror = H - j
        if mirror != j:
            folded[j] += unfolded[mirror]
    return folded

# ---------------------- blocks ----------------------

def block_id(rec, mode, win):
    return rec.chrom if mode=="chrom" else f"{rec.chrom}:{(rec.pos-1)//win}"

# ---------------------- helpers ----------------------

def _zero_idx(arr, idx):
    if 0 <= idx < len(arr):
        arr[idx] = 0.0

# ---------------------- main ----------------------

def main():
    args = parse_args()
    H = args.H

    # popmap
    pop = pd.read_csv(args.popmap, sep="\t", header=None, names=["sample","group"], comment="#")
    MOD_all = pop.loc[pop["group"]==args.group_mod, "sample"].tolist()
    ANC_all = pop.loc[pop["group"]==args.group_anc, "sample"].tolist()
    if not MOD_all or not ANC_all:
        sys.stderr.write("ERROR: empty MOD or ANC from popmap (check labels).\n"); sys.exit(2)

    vcf = pysam.VariantFile(args.vcf)
    vcf_samples = set(vcf.header.samples)
    MOD = [s for s in MOD_all if s in vcf_samples]
    ANC = [s for s in ANC_all if s in vcf_samples]
    if not MOD or not ANC:
        sys.stderr.write("ERROR: after VCF intersect, MOD or ANC empty.\n"); sys.exit(2)

    # accumulators: per (group, class) arrays di lunghezza H+1 (unfolded mass)
    total_sfs = {(g,c): np.zeros(H+1, dtype=float) for g in (args.group_mod, args.group_anc) for c in CLASSES}
    block_sfs = defaultdict(lambda: {(g,c): np.zeros(H+1, dtype=float) for g in (args.group_mod, args.group_anc) for c in CLASSES})
    block_counts = Counter()

    kept_sites = 0
    per_class_counts = Counter()
    dropped_by_mac = {args.group_mod: 0, args.group_anc: 0}

    for rec in vcf:
        # bi-allelic SNP only
        if rec.alts is None or len(rec.alts)!=1: continue
        if len(rec.ref)!=1 or len(rec.alts[0])!=1: continue

        ann = rec.info.get("ANN")
        if not ann: continue
        chosen = choose_ann_entry(ann, args.annotation_mode)
        cls = classify_from_ann(chosen)
        if cls not in CLASSES: continue

        # per group: compute counts and filters
        out_by_group = {}
        passes_group = {}
        for label, samples, n_tot in ((args.group_mod, MOD, len(MOD)), (args.group_anc, ANC, len(ANC))):
            k, h, n_called = derived_k_and_h(rec, samples)

            # presence filter
            if n_called < math.ceil(args.min_present_frac * n_tot):
                passes_group[label] = False
                out_by_group[label] = None
                continue

            # H threshold
            if h < H:
                passes_group[label] = False
                out_by_group[label] = None
                continue

            # per-group MAC filter
            minor_mac = min(k, h - k)
            pergroup_min_mac = args.pergroup_min_mac
            if pergroup_min_mac > 0 and minor_mac < pergroup_min_mac:
                passes_group[label] = False
                out_by_group[label] = None
                dropped_by_mac[label] += 1
                continue

            passes_group[label] = True
            out_by_group[label] = (k, h)

        # require both groups?
        if args.require_pass_both_groups:
            if not (passes_group.get(args.group_mod, False) and passes_group.get(args.group_anc, False)):
                continue

        # add deterministic expectation to SFS (+ esclusioni ai bordi)
        added_any = False
        b_id = block_id(rec, args.block_mode, args.block_size)
        for label in (args.group_mod, args.group_anc):
            vals = out_by_group.get(label)
            if vals is None:
                continue
            k, h = vals
            pmf = hypergeom_pmf_vec(k, h, H)   # length H+1

            # exclude fixed (edges) if requested
            if not args.include_fixed:
                _zero_idx(pmf, 0)
                _zero_idx(pmf, len(pmf)-1)
            # legacy exclusions at PMF level
            if args.exclude_singletons:
                _zero_idx(pmf, 1); _zero_idx(pmf, len(pmf)-2)
            if args.exclude_doubletons:
                _zero_idx(pmf, 2); _zero_idx(pmf, len(pmf)-3)

            total_sfs[(label, cls)] += pmf
            block_sfs[b_id][(label, cls)] += pmf
            added_any = True

        if added_any:
            kept_sites += 1
            per_class_counts[cls] += 1
            block_counts[b_id] += 1

    vcf.close()
    sys.stderr.write(f"Processed sites kept (at least one group passed): {kept_sites}\n")
    for c in CLASSES:
        sys.stderr.write(f"  {c:28s}: {per_class_counts[c]}\n")
    if args.pergroup_min_mac > 0:
        sys.stderr.write(f"Per-group MAC filter (>= {args.pergroup_min_mac}): "
                         f"dropped MOD={dropped_by_mac[args.group_mod]}, ANC={dropped_by_mac[args.group_anc]}\n")
    if args.require_pass_both_groups:
        sys.stderr.write("Require-pass-both-groups: enabled (sites must pass in MOD AND ANC).\n")

    # ------- SAVE TSVs (unfolded & folded) -------
    rows = []
    for g in (args.group_mod, args.group_anc):
        for c in CLASSES:
            arr = total_sfs[(g,c)].copy()
            x_unf = np.arange(H+1, dtype=float) / float(H)
            for i, val in enumerate(arr):
                rows.append({"type":"unfolded", "group":g, "class":c, "bin":i, "x":x_unf[i], "mass":val})
            # folded
            arr_f = fold_sfs(arr)
            M = arr_f.size
            x_f = np.arange(M, dtype=float) / float(H)  # minor allele count / H
            for i, val in enumerate(arr_f):
                rows.append({"type":"folded", "group":g, "class":c, "bin":i, "x":x_f[i], "mass":val})
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out_prefix}.sfs_mass.tsv", sep="\t", index=False)
    sys.stderr.write(f"Wrote: {args.out_prefix}.sfs_mass.tsv (expected mass, deterministic)\n")

    # Normalized variants (density & fraction)
    def normalize_group(df_sub, folded_flag):
        arr = df_sub["mass"].to_numpy(dtype=float)
        bins = df_sub["bin"].to_numpy(dtype=int)
        maxbin = bins.max()
        y = np.zeros(maxbin+1, dtype=float); y[bins] = arr
        domain = 0.5 if folded_flag else 1.0
        bin_w = domain / max(1, maxbin)
        s_mass = y.sum()
        density = (y / (s_mass * bin_w)) if (s_mass > 0) else y
        fraction = (y / s_mass) if (s_mass > 0) else y
        return density, fraction

    out_norm = []
    for t in ("unfolded", "folded"):
        for g in (args.group_mod, args.group_anc):
            for c in CLASSES:
                sub = df[(df["type"]==t) & (df["group"]==g) & (df["class"]==c)].copy()
                if sub.empty: continue
                dens, frac = normalize_group(sub, folded_flag=(t=="folded"))
                sub = sub.sort_values("bin").reset_index(drop=True)
                sub["density"] = dens
                sub["fraction"] = frac
                out_norm.append(sub)
    df_norm = pd.concat(out_norm, ignore_index=True)
    df_norm.to_csv(f"{args.out_prefix}.sfs_norm.tsv", sep="\t", index=False)
    sys.stderr.write(f"Wrote: {args.out_prefix}.sfs_norm.tsv (density & fraction)\n")

    # ------- Bootstrap (optional) -------
    if args.bootstrap and args.bootstrap > 0:
        blocks = [b for b,cnt in block_counts.items() if cnt >= args.min_sites_per_block]
        if not blocks:
            sys.stderr.write("No eligible blocks for bootstrap; skipping.\n")
        else:
            rng = np.random.default_rng(12345)
            boot_rows = []
            B = args.bootstrap
            for b in range(B):
                sampled = rng.choice(blocks, size=len(blocks), replace=True)
                agg = {(g,c): np.zeros(H+1, dtype=float) for g in (args.group_mod, args.group_anc) for c in CLASSES}
                for bid in sampled:
                    bd = block_sfs[bid]
                    for key in agg:
                        agg[key] += bd[key]
                # save folded/unfolded normalized densities (rispetta esclusioni già applicate)
                for g in (args.group_mod, args.group_anc):
                    for c in CLASSES:
                        # unfolded
                        y = agg[(g,c)].copy()
                        if not args.include_fixed:
                            _zero_idx(y, 0); _zero_idx(y, len(y)-1)
                        if args.exclude_singletons:
                            _zero_idx(y, 1); _zero_idx(y, len(y)-2)
                        if args.exclude_doubletons:
                            _zero_idx(y, 2); _zero_idx(y, len(y)-3)
                        s = y.sum()
                        bin_w = 1.0 / H
                        dens = (y/(s*bin_w)) if s>0 else y
                        for i,val in enumerate(dens):
                            boot_rows.append({"rep":b, "type":"unfolded", "group":g, "class":c, "bin":i, "density":val})
                        # folded
                        yf = fold_sfs(agg[(g,c)])
                        if not args.include_fixed and yf.size>0:
                            yf[0] = 0.0
                        if args.exclude_singletons and yf.size > 1:
                            yf[1] = 0.0
                        if args.exclude_doubletons and yf.size > 2:
                            yf[2] = 0.0
                        sf = yf.sum()
                        bin_wf = 0.5 / max(1, yf.size-1)
                        densf = (yf/(sf*bin_wf)) if sf>0 else yf
                        for i,val in enumerate(densf):
                            boot_rows.append({"rep":b, "type":"folded", "group":g, "class":c, "bin":i, "density":val})
            df_boot = pd.DataFrame(boot_rows)
            df_boot.to_csv(f"{args.out_prefix}.bootstrap_density.tsv", sep="\t", index=False)
            sys.stderr.write(f"Wrote: {args.out_prefix}.bootstrap_density.tsv\n")

if __name__ == "__main__":
    main()

