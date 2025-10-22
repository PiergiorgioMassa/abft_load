#!/usr/bin/env python3
import sys, os, argparse, subprocess, random, math

def parse_args():
    ap = argparse.ArgumentParser(
        description=("VCF -> est-sfs data file con gestione missing (downsampling per sito) "
                     "e strategie outgroup. Supporta più gruppi outgroup separati.")
    )
    ap.add_argument("--vcf", required=True, help="VCF compresso bgzip + index (.tbi)")
    ap.add_argument("--ingroup", required=True, help="file con ID campioni ingroup (uno per riga)")

    # OUTGROUP RIPETIBILE: --outgroup og1.txt --outgroup og2.txt ...
    ap.add_argument("--outgroup", action="append", required=True,
                    help="file con ID campioni per un gruppo outgroup (ripetibile). Passa questa opzione una volta per ciascun gruppo outgroup.")

    # Target: numero fisso OPPURE frazione dell'ingroup
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-n", type=int,
                       help="numero di individui ingroup da tenere per sito (diploidi) -> 2*target alleli")
    group.add_argument("--target-frac", type=float,
                       help="frazione (0..1] di individui ingroup da tenere per sito")
    ap.add_argument("--target-frac-rounding", choices=["floor","ceil","round"], default="floor",
                    help="Metodo di arrotondamento per --target-frac (default: floor)")

    # Filtri per-sample
    ap.add_argument("--minDP", type=int, default=1, help="scarta sample se DP<minDP (default: 1)")
    ap.add_argument("--minGQ", type=int, default=0, help="scarta sample se GQ<minGQ (default: 0)")

    ap.add_argument("--seed", type=int, default=12345, help="seed RNG per riproducibilità")
    ap.add_argument("--out", default="estsfs.data.txt", help="output est-sfs")
    ap.add_argument("--log", default="estsfs.data.log", help="log")
    ap.add_argument("--bcftools", default="bcftools", help="path a bcftools (se non nel PATH)")

    # Strategia per collassare ciascun outgroup a UNA copia
    ap.add_argument("--outgroup-strategy", choices=["missing","random","majority"], default="missing",
                    help=("Come costruire la singola copia per ciascun gruppo outgroup: "
                          "missing=conservativo (default); random=estrai 1 allele tra quelli osservati; "
                          "majority=assegna 1 copia se una base supera la soglia"))
    ap.add_argument("--majority-threshold", type=float, default=0.7,
                    help="Soglia per --outgroup-strategy majority (default: 0.7)")

    # Filtro di missingness sugli outgroup (per gruppo): se uno fallisce, il SITO è scartato
    ap.add_argument("--outgroup-min-present-frac", type=float, default=None,
                    help=("Se impostato (0..1), richiede che per OGNI outgroup la frazione di campioni non-missing "
                          "nel sito sia >= soglia; se almeno un outgroup è sotto soglia, il sito viene scartato."))

    # Mapping siti
    ap.add_argument("--sites-out", default=None,
                    help="Scrivi TSV con CHROM POS REF ALT per ogni riga output (stesso ordine).")
    ap.add_argument("--emit-header", action="store_true",
                    help="Se presente, aggiunge una riga di header al file --sites-out.")
    return ap.parse_args()

def read_list(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def base_to_idx(b):
    return {"A":0,"C":1,"G":2,"T":3}.get(b, -1)

def parse_gt(gt):
    # accetta 0/0, 0|1, 1/1; ritorna tuple (a,b) con alleli 0 o 1; None se missing/ploidi diversi
    if gt is None or gt == "." or gt == "./." or gt == ".|.":
        return None
    sep = "/" if "/" in gt else ("|" if "|" in gt else None)
    if not sep:  # haploide o altro
        return None
    a,b = gt.split(sep)
    if a == "." or b == ".":
        return None
    try:
        ai, bi = int(a), int(b)
    except ValueError:
        return None
    if ai not in (0,1) or bi not in (0,1):
        return None
    return (ai, bi)

def add_counts_for_base(letter_idx, count, ACGT):
    if letter_idx == 0:
        ACGT[0] += count
    elif letter_idx == 1:
        ACGT[1] += count
    elif letter_idx == 2:
        ACGT[2] += count
    elif letter_idx == 3:
        ACGT[3] += count

def choose_outgroup_from_group(og_ref_hom, og_alt_hom, og_het,
                               og_ref_alleles, og_alt_alleles, og_nonmissing,
                               strategy, thr):
    """
    Ritorna: (chosen, missing_reason)
      chosen in {"REF","ALT", None}
      missing_reason in {None, "zero", "het_or_conflict", "below_thr"}
    """
    if og_nonmissing == 0:
        return (None, "zero")
    if strategy == "missing":
        # 1 copia solo se tutti i non-missing sono omozigoti e concordi
        if og_het == 0 and ((og_ref_hom > 0) ^ (og_alt_hom > 0)):  # xor
            return ("REF" if og_ref_hom > 0 else "ALT", None)
        else:
            return (None, "het_or_conflict")
    elif strategy == "random":
        tot = og_ref_alleles + og_alt_alleles
        if tot == 0:
            return (None, "zero")
        p_ref = og_ref_alleles / tot
        return ("REF" if random.random() < p_ref else "ALT", None)
    elif strategy == "majority":
        tot = og_ref_alleles + og_alt_alleles
        if tot == 0:
            return (None, "zero")
        prop_ref = og_ref_alleles / tot
        prop_alt = og_alt_alleles / tot
        if prop_ref >= thr and prop_ref > prop_alt:
            return ("REF", None)
        elif prop_alt >= thr and prop_alt > prop_ref:
            return ("ALT", None)
        else:
            return (None, "below_thr")
    else:
        return (None, "het_or_conflict")

def main():
    args = parse_args()
    random.seed(args.seed)

    # Validazioni
    if args.outgroup_min_present_frac is not None:
        if not (0.0 <= args.outgroup_min_present_frac <= 1.0):
            sys.exit("ERROR: --outgroup-min-present-frac deve essere tra 0 e 1.")

    ing = read_list(args.ingroup)
    if len(ing) == 0:
        sys.exit("ERROR: nessun campione in --ingroup.")

    outgroup_paths = args.outgroup  # lista di path, uno per gruppo
    outgroups = [read_list(p) for p in outgroup_paths]

    # Costruisci lista campioni in ordine: INGROUP, poi OG1, OG2, ...
    all_samples = ing + [s for grp in outgroups for s in grp]
    n_ing = len(ing)
    n_out = [len(g) for g in outgroups]  # dimensioni per gruppo

    # Calcola target_n effettivo (numero di INDIVIDUI ingroup)
    if getattr(args, "target_frac", None) is not None:
        if not (0.0 < args.target_frac <= 1.0):
            sys.exit("ERROR: --target-frac deve essere in (0,1].")
        if args.target_frac_rounding == "floor":
            target_n = max(1, min(n_ing, math.floor(args.target_frac * n_ing)))
        elif args.target_frac_rounding == "ceil":
            target_n = max(1, min(n_ing, math.ceil(args.target_frac * n_ing)))
        else:  # "round"
            target_n = max(1, min(n_ing, int(round(args.target_frac * n_ing))))
    else:
        if args.target_n <= 0:
            sys.exit("ERROR: --target-n deve essere > 0.")
        target_n = min(args.target_n, n_ing)  # non superare il numero totale di ingroup

    fmt = r'%CHROM\t%POS\t%REF\t%ALT[\t%GT:%DP:%GQ]\n'
    cmd_view  = [args.bcftools, "view",  "-v", "snps", "-m2", "-M2", "-Ou", args.vcf, "-S", "-"]
    cmd_query = [args.bcftools, "query", "-f", fmt]

    p_view = subprocess.Popen(cmd_view, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=False)
    p_view.stdin.write(("\n".join(all_samples) + "\n").encode())
    p_view.stdin.close()
    p_query = subprocess.Popen(cmd_query, stdin=p_view.stdout, stdout=subprocess.PIPE, text=True)
    p_view.stdout.close()

    out = open(args.out, "w")
    log = open(args.log, "w")
    sites_f = open(args.sites_out, "w") if args.sites_out else None
    if sites_f and args.emit_header:
        sites_f.write("CHROM\tPOS\tREF\tALT\n")

    kept = 0
    skipped_low_ing = 0
    skipped_nonacgt = 0
    skipped_other = 0
    skipped_outgroup_missingness = 0

    # Statistiche per gruppo outgroup
    g_used_ref = [0]*len(outgroups)
    g_used_alt = [0]*len(outgroups)
    g_missing_zero = [0]*len(outgroups)
    g_missing_het_or_conflict = [0]*len(outgroups)
    g_missing_below_thr = [0]*len(outgroups)
    g_failed_missingness_thr = [0]*len(outgroups)  # quante volte il gruppo ha fatto fallire il sito

    for line in p_query.stdout:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            skipped_other += 1
            continue
        chrom, pos, REF, ALT = parts[0], parts[1], parts[2], parts[3]
        # scarta se REF/ALT non sono basi canoniche
        ridx = base_to_idx(REF); aidx = base_to_idx(ALT)
        if ridx < 0 or aidx < 0:
            skipped_nonacgt += 1
            continue

        fields = parts[4:]  # per-sample GT:DP:GQ
        if len(fields) != len(all_samples):
            skipped_other += 1
            continue

        # --- INGROUP ---
        usable = []  # (n_ref_alleles, n_alt_alleles) per sample accettati
        for i in range(n_ing):
            tok = fields[i]
            if ":" in tok:
                gt, dp, gq = tok.split(":")
            else:
                gt, dp, gq = tok, ".", "."
            dpv = int(dp) if dp.isdigit() else -1
            gqv = int(gq) if gq.isdigit() else -1
            gt2 = parse_gt(gt)
            if gt2 is None:  # missing o ploidia non 2
                continue
            if (args.minDP and dpv >= 0 and dpv < args.minDP) or (args.minGQ and gqv >= 0 and gqv < args.minGQ):
                continue
            r = (1 if gt2[0]==0 else 0) + (1 if gt2[1]==0 else 0)
            a = 2 - r
            usable.append((r, a))

        if len(usable) < target_n:
            skipped_low_ing += 1
            continue

        # sottocampiona target_n individui e somma alleli
        pick = random.sample(usable, target_n)
        ref_alleles = sum(x[0] for x in pick)
        alt_alleles = 2*target_n - ref_alleles

        # A,C,G,T dell’ingroup
        ACGT = [0,0,0,0]
        add_counts_for_base(ridx, ref_alleles, ACGT)
        add_counts_for_base(aidx, alt_alleles, ACGT)

        # --- OUTGROUPS: prima calcola statistiche e verifica missingness threshold ---
        og_stats = []  # per gruppo: dict con conteggi e non-missing
        offset = n_ing
        for g_idx, group in enumerate(outgroups):
            size = n_out[g_idx]
            og_ref_hom = og_alt_hom = og_het = 0
            og_ref_alleles = og_alt_alleles = 0
            og_nonmissing = 0
            for j in range(offset, offset + size):
                tok = fields[j]
                if ":" in tok:
                    gt, dp, gq = tok.split(":")
                else:
                    gt, dp, gq = tok, ".", "."
                dpv = int(dp) if dp.isdigit() else -1
                gqv = int(gq) if gq.isdigit() else -1
                gt2 = parse_gt(gt)
                if gt2 is None or (args.minDP and dpv >=0 and dpv < args.minDP) or (args.minGQ and gqv >=0 and gqv < args.minGQ):
                    continue
                og_nonmissing += 1
                if gt2 == (0,0):
                    og_ref_hom += 1; og_ref_alleles += 2
                elif gt2 == (1,1):
                    og_alt_hom += 1; og_alt_alleles += 2
                else:
                    og_het += 1; og_ref_alleles += 1; og_alt_alleles += 1
            og_stats.append({
                "size": size,
                "og_ref_hom": og_ref_hom,
                "og_alt_hom": og_alt_hom,
                "og_het": og_het,
                "og_ref_alleles": og_ref_alleles,
                "og_alt_alleles": og_alt_alleles,
                "og_nonmissing": og_nonmissing
            })
            offset += size

        # Applica il filtro di missingness sugli outgroup (se richiesto)
        if args.outgroup_min_present_frac is not None:
            failing = []
            for g_idx, st in enumerate(og_stats):
                size = st["size"]
                frac_present = (st["og_nonmissing"] / size) if size > 0 else 0.0
                if frac_present < args.outgroup_min_present_frac:
                    failing.append(g_idx)
            if failing:
                skipped_outgroup_missingness += 1
                for g in failing:
                    g_failed_missingness_thr[g] += 1
                continue  # scarta il sito prima di chiamare gli outgroup

        # --- OUTGROUPS: ora chiama la singola copia per ciascun gruppo e prepara i blocchi ---
        og_blocks = []
        for g_idx, st in enumerate(og_stats):
            chosen, miss = choose_outgroup_from_group(
                st["og_ref_hom"], st["og_alt_hom"], st["og_het"],
                st["og_ref_alleles"], st["og_alt_alleles"], st["og_nonmissing"],
                args.outgroup_strategy, args.majority_threshold
            )
            og_tuple_list = [0,0,0,0]
            if chosen == "REF":
                add_counts_for_base(ridx, 1, og_tuple_list); g_used_ref[g_idx] += 1
            elif chosen == "ALT":
                add_counts_for_base(aidx, 1, og_tuple_list); g_used_alt[g_idx] += 1
            else:
                if miss == "zero": g_missing_zero[g_idx] += 1
                elif miss == "het_or_conflict": g_missing_het_or_conflict[g_idx] += 1
                elif miss == "below_thr": g_missing_below_thr[g_idx] += 1
            og_blocks.append(tuple(og_tuple_list))

        # sites mapping
        if sites_f:
            sites_f.write(f"{chrom}\t{pos}\t{REF}\t{ALT}\n")

        # scrivi riga est-sfs: ingroup + OG1 + OG2 + ...
        blocks = [",".join(map(str, ACGT))] + [",".join(map(str, t)) for t in og_blocks]
        out.write("  ".join(blocks) + "\n")
        kept += 1

    p_query.stdout.close()
    rc_q = p_query.wait()
    rc_v = p_view.wait()
    out.close()
    if sites_f:
        sites_f.close()

    # Log riepilogativo
    total_sites = kept + skipped_low_ing + skipped_nonacgt + skipped_other + skipped_outgroup_missingness
    log.write(f"Ingroup samples listed: {n_ing}\n")
    if getattr(args, "target_frac", None) is not None:
        log.write(f"Target-n effective: {target_n} (from --target-frac={args.target_frac} with {args.target_frac_rounding})\n")
    else:
        log.write(f"Target-n effective: {target_n} (from --target-n)\n")
    log.write(
        f"Sites total (post biallelic SNP filter): {total_sites}\n"
        f"Kept: {kept}\n"
        f"Skipped (low ingroup usable < {target_n}): {skipped_low_ing}\n"
        f"Skipped (non-ACGT REF/ALT): {skipped_nonacgt}\n"
        f"Skipped (other/malformed): {skipped_other}\n"
        f"Skipped (outgroup missingness threshold): {skipped_outgroup_missingness}\n\n"
    )
    for g_idx, path in enumerate(outgroup_paths):
        log.write(f"Outgroup #{g_idx+1} [{os.path.basename(path)}] (siti tenuti):\n"
                  f"  used REF copy: {g_used_ref[g_idx]}\n"
                  f"  used ALT copy: {g_used_alt[g_idx]}\n"
                  f"  missing (zero non-missing): {g_missing_zero[g_idx]}\n"
                  f"  missing (het/conflict): {g_missing_het_or_conflict[g_idx]}\n"
                  f"  missing (below majority thr): {g_missing_below_thr[g_idx]}\n"
                  f"  failed missingness thr (causa esclusione sito): {g_failed_missingness_thr[g_idx]}\n\n")
    log.close()

if __name__ == "__main__":
    main()

