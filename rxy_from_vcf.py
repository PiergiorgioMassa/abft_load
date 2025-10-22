#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcolo di Rxy (X=moderni, Y=antichi) per classi SnpEff e categorie genotipiche,
con block bootstrap adattativo, scelte di annotazione (first/canonical/worst)
e PDF **a una sola pagina** con:
  - Rxy boxplot (y=Rxy, x=categoria genotipica, hue=classe SnpEff)

Uso (esempio):
  python rxy_from_vcf.py \
    --vcf modanc.filt.miss10.minDP10.mac2.notrans.polarized.annot.vcf.gz \
    --popmap popmap.tsv --group-x moderni --group-y antichi \
    --annotation-mode first \
    --min-present-frac 0.7 \
    --bootstrap 1000 --block-mode chrom \
    --out-prefix rxy_out
"""

import argparse, sys, math, random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pysam
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

# ---- classi/categorie/colore ----
SNPEFF_CLASSES_ORDER = ["HIGH", "MODERATE", "LOW", "MODIFIER:intergenic_region"]  # ordine nei plot
GENO_CATS_ORDER = ["HOM_ANC", "HET", "HOM_DER", "DER"]  # DER = DAF

# Colori: verde (MODIFIER), rosso (HIGH), arancione (MODERATE), giallo (LOW)
CLASS_COLORS = {
    "MODIFIER:intergenic_region": "#2ca02c",
    "HIGH": "#d62728",
    "MODERATE": "#ff7f0e",
    "LOW": "#ffd700",
}
CLASS_OFFSETS = {"HIGH": -0.30, "MODERATE": -0.10, "LOW": 0.10, "MODIFIER:intergenic_region": 0.30}
IMPACT_RANK = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "MODIFIER": 0}

def parse_args():
    ap = argparse.ArgumentParser(description="Rxy per classi SnpEff e categorie genotipiche con block bootstrap (PDF 1 pagina).")
    ap.add_argument("--vcf", required=True, help="VCF polarizzato e annotato con snpEff (bgzip + .tbi)")
    ap.add_argument("--popmap", required=True, help="TSV: SAMPLE\\tGROUP (due colonne).")
    ap.add_argument("--group-x", required=True, help="Nome del gruppo X (moderni) presente nella popmap.")
    ap.add_argument("--group-y", required=True, help="Nome del gruppo Y (antichi) presente nella popmap.")
    ap.add_argument("--annotation-mode", choices=["first","canonical","worst"], default="first",
                    help="first=prima ANN; canonical=entry con CANONICAL (fallback first); worst=intergenic se presente, altrimenti HIGH>MODERATE>LOW.")
    ap.add_argument("--min-present-frac", type=float, default=0.7,
                    help="Frazione minima di genotipi chiamati per gruppo per tenere un sito (default: 0.7).")

    # --- NUOVE OPZIONI ---
    ap.add_argument("--max-callrate-diff", type=float, default=None,
                    help="Differenza massima assoluta nel call-rate tra X e Y, per-sito (se supera, scarta il sito).")
    ap.add_argument("--equalize-per-site", choices=["none","individuals"], default="none",
                    help="Se 'individuals', per ciascun sito sottocampiona i chiamati in X e Y fino ad avere lo stesso n.")

    ap.add_argument("--bootstrap", type=int, default=1000, help="Numero repliche bootstrap (default: 1000).")
    ap.add_argument("--block-mode", choices=["chrom","window"], default="chrom",
                    help="Blocchi per il bootstrap: 'chrom' = per cromosoma; 'window' = finestre fisse.")
    ap.add_argument("--block-size", type=int, default=5_000_000,
                    help="Dimensione finestra se --block-mode=window (default: 5Mb).")
    ap.add_argument("--min-blocks", type=int, default=10,
                    help="Min numero di blocchi non vuoti per usare block-bootstrap (default: 10).")
    ap.add_argument("--min-sites-for-block", type=int, default=50,
                    help="Min numero di siti totali nella classe per usare block-bootstrap (default: 50).")
    ap.add_argument("--seed", type=int, default=12345, help="Seed RNG (default: 12345).")
    ap.add_argument("--out-prefix", default="rxy", help="Prefisso output (TSV e PDF).")
    ap.add_argument("--pdf", default=None, help="PDF output (default: <out-prefix>.rxy_plots.pdf)")
    return ap.parse_args()

# -------- helper: ANN selection & classification --------
def split_ann_entry(entry):
    parts = entry.split('|')
    if len(parts) < 3:
        parts += [""]*(3-len(parts))
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
            if "CANONICAL" in tail:
                return e
        return anns[0]
    elif mode == "worst":
        for e in anns:
            ann, imp, tail = split_ann_entry(e)
            if ann == "intergenic_region":
                return e
        best_e, best_rank = None, -1
        for e in anns:
            ann, imp, tail = split_ann_entry(e)
            rank = IMPACT_RANK.get(imp, -1)
            if rank > best_rank:
                best_e, best_rank = e, rank
        return best_e
    else:
        return None

def classify_from_ann(entry):
    if entry is None: return None
    ann, imp, _ = split_ann_entry(entry)
    if ann == "intergenic_region":
        return "MODIFIER:intergenic_region"
    if imp in ("HIGH","MODERATE","LOW"):
        return imp
    return None  # MODIFIER non-intergenic esclusi

# -------- helper: genotype counting --------
def count_group(rec, sample_names):
    c00 = c01 = c11 = 0
    n_called = 0
    sdata = rec.samples
    for s in sample_names:
        sd = sdata.get(s)
        if sd is None: continue
        gt = sd.get('GT')
        if gt is None or len(gt) != 2: continue
        a, b = gt[0], gt[1]
        if a is None or b is None: continue
        if a not in (0,1) or b not in (0,1): continue
        n_called += 1
        ssum = a + b
        if ssum == 0: c00 += 1
        elif ssum == 1: c01 += 1
        elif ssum == 2: c11 += 1
    return c00, c01, c11, n_called

# --- helper: elenco di campioni chiamati al sito ---
def called_sample_names(rec, sample_names):
    out = []
    sdata = rec.samples
    for s in sample_names:
        sd = sdata.get(s)
        if sd is None:
            continue
        gt = sd.get('GT')
        if gt is None or len(gt) != 2:
            continue
        a, b = gt[0], gt[1]
        if a in (0,1) and b in (0,1):
            out.append(s)
    return out

def get_block_id(rec, mode, win):
    return rec.chrom if mode == "chrom" else f"{rec.chrom}:{(rec.pos-1)//win}"

# -------- Rxy with Haldane correction --------
def freq_geno_pooled(sum_c, sum_n, alpha=0.5):
    return (sum_c + alpha) / (sum_n + 2*alpha) if (sum_n is not None) else float('nan')

def freq_der_pooled(sum_c01, sum_c11, sum_n, alpha=0.5):
    num = 2*sum_c11 + sum_c01
    den = 2*sum_n
    return (num + alpha) / (den + 2*alpha) if den is not None else float('nan')

def rxy_from_fx_fy(fx, fy):
    fx = min(max(fx, 1e-12), 1-1e-12)
    fy = min(max(fy, 1e-12), 1-1e-12)
    return (fx*(1.0 - fy)) / (fy*(1.0 - fx))

# -------- plotting (unica pagina) --------
def plot_grouped_box(ax, boot_vals_dict, per_class_sites, title, ylabel):
    base_x = np.arange(len(GENO_CATS_ORDER))
    classes_present = [c for c in SNPEFF_CLASSES_ORDER if len(per_class_sites[c]) > 0]
    plotted = False
    for cls in classes_present:
        color = CLASS_COLORS.get(cls, "#999999")
        for i, gc in enumerate(GENO_CATS_ORDER):
            data = boot_vals_dict[cls][gc]
            if not data: continue
            pos = base_x[i] + CLASS_OFFSETS[cls]
            ax.boxplot(
                data, positions=[pos], widths=0.18, patch_artist=True, manage_ticks=False,
                boxprops=dict(facecolor=color, edgecolor="black"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black")
            )
            plotted = True
    ax.set_xticks(base_x); ax.set_xticklabels(GENO_CATS_ORDER)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")
    if plotted:
        handles = [Patch(color=CLASS_COLORS[c], label=c) for c in SNPEFF_CLASSES_ORDER if c in classes_present]
        ax.legend(handles=handles, title="Classe SnpEff", loc="best", frameon=False)

# -------- main --------
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    pdf_path = args.pdf or f"{args.out_prefix}.rxy_plots.pdf"
    summary_tsv = f"{args.out_prefix}.rxy_summary.tsv"

    # --- popmap ---
    pop = pd.read_csv(args.popmap, sep="\t", header=None, names=["sample","group"], comment="#")
    if args.group_x not in set(pop["group"]) or args.group_y not in set(pop["group"]):
        sys.stderr.write(f"ERROR: gruppi {args.group_x} e/o {args.group_y} non trovati nella popmap.\n"); sys.exit(2)
    group_samples = {
        "X": pop.loc[pop["group"]==args.group_x, "sample"].tolist(),
        "Y": pop.loc[pop["group"]==args.group_y, "sample"].tolist()
    }

    # --- VCF ---
    vcf = pysam.VariantFile(args.vcf)
    vcf_samples = list(vcf.header.samples)
    for key in ("X","Y"):
        group_samples[key] = [s for s in group_samples[key] if s in vcf_samples]
    nX = len(group_samples["X"]); nY = len(group_samples["Y"])
    if nX==0 or nY==0:
        sys.stderr.write("ERROR: nessun campione X o Y presente nel VCF.\n"); sys.exit(2)
    sys.stderr.write(f"After intersect with VCF: X={nX}, Y={nY}\n")

    per_class_sites = {cls: [] for cls in SNPEFF_CLASSES_ORDER}
    kept_sites_total = 0
    class_counts = Counter()

    # iterate VCF
    for rec in vcf:
        if rec.alts is None or len(rec.alts) != 1: continue
        if len(rec.ref) != 1 or len(rec.alts[0]) != 1: continue

        ann_field = rec.info.get("ANN")
        if not ann_field: continue
        chosen = choose_ann_entry(ann_field, args.annotation_mode)
        cls = classify_from_ann(chosen)
        if cls is None: continue

        # ---- NUOVA LOGICA: missing-balance + equalizzazione ----
        # individui chiamati per sito in X e Y
        called_X = called_sample_names(rec, group_samples["X"])
        called_Y = called_sample_names(rec, group_samples["Y"])
        nx, ny = len(called_X), len(called_Y)

        # filtro presenza per gruppo
        if nx < math.ceil(args.min_present_frac * nX) or ny < math.ceil(args.min_present_frac * nY):
            continue

        # filtro opzionale: sbilanciamento di call-rate tra X e Y
        if args.max_callrate_diff is not None:
            crx = nx / float(nX)
            cry = ny / float(nY)
            if abs(crx - cry) > args.max_callrate_diff:
                continue

        # equalizzazione opzionale per-sito: sottocampiona i chiamati per avere pari n in X e Y
        if args.equalize_per_site == "individuals":
            m = min(nx, ny)
            if m == 0:
                continue
            if nx > m:
                idx = rng.choice(nx, size=m, replace=False)
                useX = [called_X[i] for i in idx]
            else:
                useX = called_X
            if ny > m:
                idy = rng.choice(ny, size=m, replace=False)
                useY = [called_Y[i] for i in idy]
            else:
                useY = called_Y
        else:
            useX = called_X
            useY = called_Y

        # conte basate sui sottoinsiemi selezionati
        c00x, c01x, c11x, nx = count_group(rec, useX)
        c00y, c01y, c11y, ny = count_group(rec, useY)

        block_id = get_block_id(rec, args.block_mode, args.block_size)
        per_class_sites[cls].append({
            "block": block_id,
            "X": {"c00": c00x, "c01": c01x, "c11": c11x, "n": nx},
            "Y": {"c00": c00y, "c01": c01y, "c11": c11y, "n": ny},
        })
        kept_sites_total += 1
        class_counts[cls] += 1

    vcf.close()
    sys.stderr.write(f"Siti tenuti dopo filtri: {kept_sites_total}\n")
    for cls in SNPEFF_CLASSES_ORDER:
        sys.stderr.write(f"  {cls:28s}: {class_counts[cls]}\n")

    # pooling + bootstrap
    def pooled_fx_fy_Rxy(site_list):
        sums = {"X": {"c00":0, "c01":0, "c11":0, "n":0},
                "Y": {"c00":0, "c01":0, "c11":0, "n":0}}
        for s in site_list:
            for grp in ("X","Y"):
                for k in ("c00","c01","c11","n"):
                    sums[grp][k] += s[grp][k]
        out = {}
        fx = freq_geno_pooled(sums["X"]["c00"], sums["X"]["n"]); fy = freq_geno_pooled(sums["Y"]["c00"], sums["Y"]["n"])
        out["HOM_ANC"] = (fx, fy, rxy_from_fx_fy(fx, fy))
        fx = freq_geno_pooled(sums["X"]["c01"], sums["X"]["n"]); fy = freq_geno_pooled(sums["Y"]["c01"], sums["Y"]["n"])
        out["HET"] = (fx, fy, rxy_from_fx_fy(fx, fy))
        fx = freq_geno_pooled(sums["X"]["c11"], sums["X"]["n"]); fy = freq_geno_pooled(sums["Y"]["c11"], sums["Y"]["n"])
        out["HOM_DER"] = (fx, fy, rxy_from_fx_fy(fx, fy))
        fx = freq_der_pooled(sums["X"]["c01"], sums["X"]["c11"], sums["X"]["n"]); fy = freq_der_pooled(sums["Y"]["c01"], sums["Y"]["c11"], sums["Y"]["n"])
        out["DER"] = (fx, fy, rxy_from_fx_fy(fx, fy))
        return out

    def bootstrap_for_sites(sites, n_reps, min_blocks, min_sites_for_block):
        n_sites = len(sites)
        boot_out = {gc: [] for gc in GENO_CATS_ORDER}
        if n_sites == 0 or n_reps == 0:
            return boot_out
        block_to_idx = defaultdict(list)
        for i, s in enumerate(sites):
            block_to_idx[s["block"]].append(i)
        blocks = list(block_to_idx.keys())
        n_blocks = len(blocks)
        use_block = (n_blocks >= min_blocks) and (n_sites >= min_sites_for_block)
        for _ in range(n_reps):
            if use_block:
                sampled_blocks = rng.choice(blocks, size=n_blocks, replace=True)
                idxs = []
                for bl in sampled_blocks:
                    idxs.extend(block_to_idx[bl])
            else:
                idxs = rng.integers(low=0, high=n_sites, size=n_sites, endpoint=False).tolist()
            sampled_sites = [sites[i] for i in idxs]
            res = pooled_fx_fy_Rxy(sampled_sites)
            for gc in GENO_CATS_ORDER:
                boot_out[gc].append(res[gc][2])
        return boot_out

    # point estimates + bootstraps
    point_estimates = []
    boot_vals_overall = {cls: {gc: [] for gc in GENO_CATS_ORDER} for cls in SNPEFF_CLASSES_ORDER}
    for cls in SNPEFF_CLASSES_ORDER:
        sites = per_class_sites[cls]; n_sites = len(sites)
        if n_sites == 0:
            pe_row = {"class": cls, "n_sites": 0}
            pe_row.update({f"{gc}_Rxy": float('nan') for gc in GENO_CATS_ORDER})
            point_estimates.append(pe_row); continue
        pe_res = pooled_fx_fy_Rxy(sites)
        pe_row = {"class": cls, "n_sites": n_sites}
        pe_row.update({f"{gc}_Rxy": pe_res[gc][2] for gc in GENO_CATS_ORDER})
        point_estimates.append(pe_row)
        boots = bootstrap_for_sites(sites, args.bootstrap, args.min_blocks, args.min_sites_for_block)
        for gc in GENO_CATS_ORDER:
            boot_vals_overall[cls][gc] = boots[gc]

    # Summary TSV
    rows = []
    for cls in SNPEFF_CLASSES_ORDER:
        n_sites = len(per_class_sites[cls])
        for gc in GENO_CATS_ORDER:
            arr = np.array(boot_vals_overall[cls][gc], dtype=float)
            if arr.size > 0:
                lo, hi = np.nanpercentile(arr, [2.5, 97.5]); med = np.nanmedian(arr); mean = np.nanmean(arr)
            else:
                lo = hi = med = mean = float('nan')
            point = [r for r in point_estimates if r["class"]==cls][0][f"{gc}_Rxy"]
            rows.append({"class": cls, "geno_cat": gc, "n_sites": n_sites,
                         "Rxy_point": point, "Rxy_boot_mean": mean, "Rxy_boot_median": med,
                         "Rxy_CI2.5": lo, "Rxy_CI97.5": hi})
    pd.DataFrame(rows).to_csv(summary_tsv, sep="\t", index=False)
    sys.stderr.write(f"Wrote summary: {summary_tsv}\n")

    # ---- PLOTTING (PDF: solo Pag.1) ----
    with PdfPages(pdf_path) as pdf:
        fig1 = plt.figure(figsize=(10,6))
        ax1 = plt.gca()
        plot_grouped_box(ax1, boot_vals_overall, per_class_sites,
                         "Rxy per categoria genotipica (boxplot su bootstrap)", "Rxy")
        fig1.tight_layout(); pdf.savefig(fig1); plt.close(fig1)

    sys.stderr.write(f"Wrote PDF: {pdf_path}\n")

if __name__ == "__main__":
    main()

