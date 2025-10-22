#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CLASSES = ["HIGH", "MODERATE", "LOW", "MODIFIER"]
PAGE_ORDER = ["DER", "HOM_ANC", "HET", "HOM_DER"]  # righe nell'ordine desiderato
GENOTYPES = ["HOM_ANC", "HET", "HOM_DER", "DER"]   # per validazione colonne

# --- bin su init_freq: 3 colonne (<0.005, 0.005–0.05, >0.05)
def bin_index(x: float) -> int:
    if x < 0.005:
        return 0
    if x < 0.05:
        return 1
    return 2

BIN_LABELS = ["ALL inits", "init<0.005", "0.005–0.05", "init>0.05"]  # prima colonna aggrega tutti i bin

def present_by_mod(mod: str) -> int:
    return 300502 if mod == "mod1" else 305020

def scale_by_mod(mod: str) -> float:
    return 1.0 if mod == "mod1" else 0.1  # cycle -> generations

def cutoff_zero_by_mod(mod: str) -> int:
    return 300490 if mod == "mod1" else 304900

def nice_gen_ticks(xmax: float):
    base = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
                     1000, 2000, 5000, 10000], dtype=float)
    ticks = base[base <= max(xmax, 1)]
    if ticks.size == 0:
        ticks = np.array([0, 1, 2, 5, 10], dtype=float)
    return ticks

def load_all_tsv(dir_path: str):
    paths = [os.path.join(dir_path, f"{c}.tsv") for c in CLASSES]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("Mancano TSV: " + ", ".join(missing))
    df = pd.concat([pd.read_csv(p, sep="\t") for p in paths], ignore_index=True)

    source_col = "src_csv" if "src_csv" in df.columns else ("src_log" if "src_log" in df.columns else None)
    if source_col is None:
        raise KeyError("Colonna sorgente mancante (attese: 'src_csv' o 'src_log').")

    df["init_freq"] = pd.to_numeric(df["init_freq"], errors="coerce")
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    for g in GENOTYPES:
        if g not in df.columns:
            raise KeyError(f"Colonna mancante nei TSV: {g}")
        df[g] = pd.to_numeric(df[g], errors="coerce")

    df = df.dropna(subset=["cycle", "init_freq"]).sort_values(["class", source_col, "cycle"])
    df["init_bin"] = df["init_freq"].apply(bin_index)
    return df, source_col

def group_should_be_dropped_for_der_zero(g: pd.DataFrame, mod: str, zero_eps: float) -> bool:
    """
    True se, prima del cutoff (dipendente da mod),
    esiste almeno un punto con DER <= zero_eps.
    """
    cutoff = cutoff_zero_by_mod(mod)
    gg = g[g["cycle"] < cutoff]
    if gg.empty or "DER" not in gg.columns:
        return False
    return (gg["DER"] <= zero_eps).any()

def rolling_median(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Smussatura robusta con mediana mobile (finestra dispari)."""
    n = y.size
    if n == 0 or window <= 1:
        return y.copy()
    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        vals = y[s:e]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = np.median(vals)
    return out

# ---------- Pagina 1: Rxy per classe (intensificata, senza CI) ----------
def plot_regression_page(df: pd.DataFrame, source_col: str, pdf: PdfPages, mod: str,
                         eps: float, drop_early_der_zero: bool, zero_eps: float,
                         n_bins: int = 30):
    P = present_by_mod(mod)
    scale = scale_by_mod(mod)

    d = df[df["cycle"] <= P].copy()
    d["gens_before_present"] = (P - d["cycle"]) * scale
    d.loc[d["gens_before_present"] < 0, "gens_before_present"] = 0.0

    fig, axes = plt.subplots(4, 4, figsize=(18, 18), sharex=False, sharey=False)
    fig.suptitle(
        "Rxy vs generazioni prima del presente (mediana per bin; smussata)\n"
        "Log su (x+1); Aree: 0–2 gen (verde), 2–12 gen (rosso).",
        y=0.995, fontsize=16
    )

    # palette richiesta
    class_colors = {
        "HIGH": "#e41a1c",       # rosso
        "MODERATE": "#ff7f00",   # arancio
        "LOW": "#ffd92f",        # giallo
        "MODIFIER": "#4daf4a",   # verde
    }

    x_all = d["gens_before_present"].values
    xmax_gen = max(float(np.nanmax(x_all)) if x_all.size else 1.0, 1.0)
    x_edges_plot = np.geomspace(1.0, xmax_gen + 1.0, num=n_bins + 1)
    x_centers_plot = np.sqrt(x_edges_plot[:-1] * x_edges_plot[1:])

    for r, genotype in enumerate(PAGE_ORDER):
        for c in range(4):  # 0=ALL, 1..3=bin
            ax = axes[r, c]

            # Aree 0–2 e 2–12 gen
            green_start, green_end = 1.0, min(3.0, x_edges_plot[-1])
            red_start, red_end = 3.0, min(13.0, x_edges_plot[-1])
            ax.axvspan(green_start, green_end, color="green", alpha=0.12, zorder=0)
            ax.axvspan(red_start, red_end, color="red", alpha=0.10, zorder=0)

            panel_recent_vals = []
            added_label = set()

            for cls in CLASSES:
                sel = d[d["class"] == cls] if c == 0 else d[(d["class"] == cls) & (d["init_bin"] == (c - 1))]
                if sel.empty:
                    continue

                groups = []
                for _, g in sel.groupby(source_col, sort=False):
                    g = g.dropna(subset=[genotype, "gens_before_present"]).sort_values("cycle")
                    if g.empty:
                        continue
                    if drop_early_der_zero and group_should_be_dropped_for_der_zero(g, mod, zero_eps):
                        continue
                    gp = g[g["cycle"] == P]
                    if gp.empty or pd.isna(gp[genotype].iloc[0]):
                        continue
                    f_x = float(gp[genotype].iloc[0])
                    if 1.0 - f_x <= eps:
                        continue
                    f_y = g[genotype].astype(float)
                    denom = f_y * (1.0 - f_x)
                    valid = (f_y >= eps) & (np.abs(denom) >= eps)
                    if not valid.any():
                        continue
                    x_plot = (g.loc[valid, "gens_before_present"] + 1.0).values
                    y = (f_x * (1.0 - f_y[valid].values)) / denom[valid].values
                    finite = np.isfinite(y) & np.isfinite(x_plot)
                    if finite.any():
                        groups.append((x_plot[finite], y[finite]))

                if len(groups) < 5:
                    continue

                # mediane per bin (media per-corsa)
                medians = []
                has_bin = []
                for b_start, b_end in zip(x_edges_plot[:-1], x_edges_plot[1:]):
                    per_run_vals = []
                    for xg, yg in groups:
                        m = (xg >= b_start) & (xg < b_end)
                        if m.any():
                            per_run_vals.append(np.nanmedian(yg[m]))
                    if len(per_run_vals) >= 5:
                        medians.append(float(np.nanmedian(np.array(per_run_vals))))
                        has_bin.append(True)
                    else:
                        medians.append(np.nan)
                        has_bin.append(False)

                medians = np.array(medians, dtype=float)
                mask = np.isfinite(medians) & np.array(has_bin, dtype=bool)
                if not mask.any():
                    continue

                x_med = x_centers_plot[mask]
                y_med = medians[mask]
                y_smooth = rolling_median(y_med, window=5)

                # alone + curva principale
                ax.plot(x_med, y_smooth, color="black", linewidth=4.0, alpha=0.25, zorder=3)
                label = cls if cls not in added_label else None
                ax.plot(x_med, y_smooth, linewidth=3.0, color=class_colors[cls], alpha=0.98,
                        label=label, zorder=4)
                added_label.add(cls)
                # traccia raw
                ax.plot(x_med, y_med, linewidth=1.0, linestyle="--", color=class_colors[cls],
                        alpha=0.35, zorder=2)

                # valori recenti per ylim (centro bin <= 12 gen => x <= 13)
                recent_mask = (x_med <= 13.0)
                vals_recent = y_med[recent_mask]
                if vals_recent.size:
                    panel_recent_vals.extend(vals_recent[np.isfinite(vals_recent)])

            ax.axhline(1.0, linestyle="--", linewidth=0.9, color="black", alpha=0.9, zorder=1)
            ax.set_xscale("log")
            ax.set_xlim(1.0, x_edges_plot[-1])

            panel_df = d if c == 0 else d[d["init_bin"] == (c - 1)]
            xmax_here = float(panel_df["gens_before_present"].max()) if not panel_df.empty else 1.0
            ticks_gen = nice_gen_ticks(xmax_here)
            ax.set_xticks(ticks_gen + 1.0)
            ax.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:.2g}" for t in ticks_gen])

            ax.set_title(f"{genotype} | {BIN_LABELS[c]}", fontsize=11)
            if r == len(PAGE_ORDER) - 1:
                ax.set_xlabel("Generazioni prima del presente (log su x+1)")
            if c == 0:
                ax.set_ylabel("Rxy")

            if len(ax.lines) > 0:
                ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")

            if len(panel_recent_vals) > 0:
                arr = np.array(panel_recent_vals, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    pad = 0.1
                    y_low = float(np.nanmin(arr)) - pad
                    y_high = float(np.nanmax(arr)) + pad
                    if y_high <= y_low:
                        y_high = y_low + 1.0
                    ax.set_ylim(y_low, y_high)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

# ---------- Pagina 2: Ratio Rxy_(HIGH/MODERATE/LOW) / Rxy_MODIFIER ----------
def plot_ratio_to_modifier_page(df: pd.DataFrame, source_col: str, pdf: PdfPages, mod: str,
                                eps: float, drop_early_der_zero: bool, zero_eps: float,
                                n_bins: int = 30):
    P = present_by_mod(mod)
    scale = scale_by_mod(mod)

    d = df[df["cycle"] <= P].copy()
    d["gens_before_present"] = (P - d["cycle"]) * scale
    d.loc[d["gens_before_present"] < 0, "gens_before_present"] = 0.0

    fig, axes = plt.subplots(4, 4, figsize=(18, 18), sharex=False, sharey=False)
    fig.suptitle(
        "Ratio Rxy (classe / MODIFIER) vs generazioni prima del presente\n"
        "Log su (x+1); Aree: 0–2 gen (verde), 2–12 gen (rosso).",
        y=0.995, fontsize=16
    )

    ratio_colors = {
        "HIGH": "#e41a1c",
        "MODERATE": "#ff7f00",
        "LOW": "#ffd92f",
    }

    x_all = d["gens_before_present"].values
    xmax_gen = max(float(np.nanmax(x_all)) if x_all.size else 1.0, 1.0)
    x_edges_plot = np.geomspace(1.0, xmax_gen + 1.0, num=n_bins + 1)
    x_centers_plot = np.sqrt(x_edges_plot[:-1] * x_edges_plot[1:])

    target_classes = ["HIGH", "MODERATE", "LOW"]
    denom_class = "MODIFIER"

    for r, genotype in enumerate(PAGE_ORDER):
        for c in range(4):
            ax = axes[r, c]

            green_start, green_end = 1.0, min(3.0, x_edges_plot[-1])
            red_start, red_end = 3.0, min(13.0, x_edges_plot[-1])
            ax.axvspan(green_start, green_end, color="green", alpha=0.12, zorder=0)
            ax.axvspan(red_start, red_end, color="red", alpha=0.10, zorder=0)

            medians_map, mask_map = {}, {}

            def class_selection(cls):
                return d[d["class"] == cls] if c == 0 else d[(d["class"] == cls) & (d["init_bin"] == (c - 1))]

            for cls in target_classes + [denom_class]:
                sel = class_selection(cls)
                if sel.empty:
                    continue
                groups = []
                for _, g in sel.groupby(source_col, sort=False):
                    g = g.dropna(subset=[genotype, "gens_before_present"]).sort_values("cycle")
                    if g.empty:
                        continue
                    if drop_early_der_zero and group_should_be_dropped_for_der_zero(g, mod, zero_eps):
                        continue
                    gp = g[g["cycle"] == P]
                    if gp.empty or pd.isna(gp[genotype].iloc[0]):
                        continue
                    f_x = float(gp[genotype].iloc[0])
                    if 1.0 - f_x <= eps:
                        continue
                    f_y = g[genotype].astype(float)
                    denom = f_y * (1.0 - f_x)
                    valid = (f_y >= eps) & (np.abs(denom) >= eps)
                    if not valid.any():
                        continue
                    x_plot = (g.loc[valid, "gens_before_present"] + 1.0).values
                    y = (f_x * (1.0 - f_y[valid].values)) / denom[valid].values
                    finite = np.isfinite(y) & np.isfinite(x_plot)
                    if finite.any():
                        groups.append((x_plot[finite], y[finite]))

                if len(groups) < 5:
                    continue

                medians, has_bin = [], []
                for b_start, b_end in zip(x_edges_plot[:-1], x_edges_plot[1:]):
                    per_run_vals = []
                    for xg, yg in groups:
                        m = (xg >= b_start) & (xg < b_end)
                        if m.any():
                            per_run_vals.append(np.nanmedian(yg[m]))
                    if len(per_run_vals) >= 5:
                        medians.append(float(np.nanmedian(np.array(per_run_vals))))
                        has_bin.append(True)
                    else:
                        medians.append(np.nan)
                        has_bin.append(False)

                medians_map[cls] = np.array(medians, dtype=float)
                mask_map[cls] = np.isfinite(medians_map[cls]) & np.array(has_bin, dtype=bool)

            # se il denominatore non è disponibile, faccio solo setup assi e continuo
            if denom_class not in medians_map or not mask_map[denom_class].any():
                ax.axhline(1.0, linestyle="--", linewidth=0.9, color="black", alpha=0.9, zorder=1)
                ax.set_xscale("log")
                ax.set_xlim(1.0, x_edges_plot[-1])
                panel_df = d if c == 0 else d[d["init_bin"] == (c - 1)]
                xmax_here = float(panel_df["gens_before_present"].max()) if not panel_df.empty else 1.0
                ticks_gen = nice_gen_ticks(xmax_here)
                ax.set_xticks(ticks_gen + 1.0)
                ax.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:.2g}" for t in ticks_gen])
                ax.set_title(f"{genotype} | {BIN_LABELS[c]}", fontsize=11)
                if r == len(PAGE_ORDER) - 1:
                    ax.set_xlabel("Generazioni prima del presente (log su x+1)")
                if c == 0:
                    ax.set_ylabel("Rxy(class) / Rxy(MODIFIER)")
                continue

            panel_recent_vals = []
            added_label = set()
            denom_med = medians_map[denom_class]
            denom_mask = mask_map[denom_class]

            for cls in target_classes:
                if cls not in medians_map:
                    continue
                num_med = medians_map[cls]
                num_mask = mask_map[cls]
                mask = num_mask & denom_mask & np.isfinite(num_med) & np.isfinite(denom_med) & (np.abs(denom_med) >= eps)
                if not mask.any():
                    continue

                x_med = x_centers_plot[mask]
                ratio = num_med[mask] / denom_med[mask]
                if ratio.size == 0 or not np.isfinite(ratio).any():
                    continue

                y_smooth = rolling_median(ratio, window=5)
                ax.plot(x_med, y_smooth, color="black", linewidth=4.0, alpha=0.25, zorder=3)
                label = f"{cls} / MODIFIER" if (f"{cls}/MODIFIER" not in added_label) else None
                ax.plot(x_med, y_smooth, linewidth=3.0, color=ratio_colors[cls], alpha=0.98,
                        label=label, zorder=4)
                added_label.add(f"{cls}/MODIFIER")
                ax.plot(x_med, ratio, linewidth=1.0, linestyle="--", color=ratio_colors[cls],
                        alpha=0.35, zorder=2)

                recent_mask = (x_med <= 13.0)
                vals_recent = ratio[recent_mask]
                if vals_recent.size:
                    panel_recent_vals.extend(vals_recent[np.isfinite(vals_recent)])

            ax.axhline(1.0, linestyle="--", linewidth=0.9, color="black", alpha=0.9, zorder=1)
            ax.set_xscale("log")
            ax.set_xlim(1.0, x_edges_plot[-1])
            panel_df = d if c == 0 else d[d["init_bin"] == (c - 1)]
            xmax_here = float(panel_df["gens_before_present"].max()) if not panel_df.empty else 1.0
            ticks_gen = nice_gen_ticks(xmax_here)
            ax.set_xticks(ticks_gen + 1.0)
            ax.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:.2g}" for t in ticks_gen])

            ax.set_title(f"{genotype} | {BIN_LABELS[c]}", fontsize=11)
            if r == len(PAGE_ORDER) - 1:
                ax.set_xlabel("Generazioni prima del presente (log su x+1)")
            if c == 0:
                ax.set_ylabel("Rxy(class) / Rxy(MODIFIER)")

            if len(ax.lines) > 0:
                ax.legend(fontsize=8, frameon=False, ncol=1, loc="upper right")

            if len(panel_recent_vals) > 0:
                arr = np.array(panel_recent_vals, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    pad = 0.1
                    y_low = float(np.nanmin(arr)) - pad
                    y_high = float(np.nanmax(arr)) + pad
                    if y_high <= y_low:
                        y_high = y_low + 1.0
                    ax.set_ylim(y_low, y_high)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(
        description="PDF: (1) Rxy per classe e (2) Ratio Rxy classe/MODIFIER, vs generazioni (log su x+1)"
    )
    ap.add_argument("dir_tsv", help="Cartella con HIGH.tsv, MODERATE.tsv, LOW.tsv, MODIFIER.tsv")
    ap.add_argument("out_pdf", help="File PDF di output")
    ap.add_argument("--mod", choices=["mod1","mod2"], required=True,
                    help="mod1: P=300502, 1 cycle=1 gen | mod2: P=305020, 1 cycle=0.1 gen")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="Tolleranza per evitare divisioni per zero (default: 1e-6)")
    ap.add_argument("--drop-early-der-zero", action="store_true",
                    help="Esclude corse in cui DER==0 prima di 300490 (mod1) o 304900 (mod2).")
    ap.add_argument("--zero-eps", type=float, default=0.0,
                    help="Tolleranza per considerare DER come zero (default: 0.0).")
    ap.add_argument("--n-bins", type=int, default=30,
                    help="Numero di bin log-spaced sull'asse X (default: 30)")
    args = ap.parse_args()

    df, source_col = load_all_tsv(args.dir_tsv)
    with PdfPages(args.out_pdf) as pdf:
        # Pagina 1: Rxy per classe
        plot_regression_page(
            df, source_col, pdf, args.mod, args.eps,
            args.drop_early_der_zero, args.zero_eps, n_bins=args.n_bins
        )
        # Pagina 2: Ratio rispetto a MODIFIER
        plot_ratio_to_modifier_page(
            df, source_col, pdf, args.mod, args.eps,
            args.drop_early_der_zero, args.zero_eps, n_bins=args.n_bins
        )

    print(f"Creato PDF con 2 pagine: {args.out_pdf}")

if __name__ == "__main__":
    main()

