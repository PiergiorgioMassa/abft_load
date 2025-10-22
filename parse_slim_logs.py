#!/usr/bin/env python3
import sys, re, os, glob, argparse

# ====== pattern filename: K{K}_seed{seed_burnin}_bn{BN}_{CLASS}_rep{rep}_id{ID}.csv
FN_RE = re.compile(
    r"K(?P<K>[^_]+)_seed(?P<seed>\d+)_bn(?P<bn>[^_]+)_(?P<class>HIGH|MODERATE|LOW|MODIFIER)_rep(?P<rep>\d+)_id(?P<id>\d+)\.csv$"
)

# ====== pattern header righe "##"
ID_RE    = re.compile(r'^##ID=(?P<id>\d+)$')
S_RE     = re.compile(r'^##s=(?P<s>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$')
FREQ_RE  = re.compile(r'^##freq=(?P<freq>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$')
CLASS_RE = re.compile(r'^##(HIGH|MODERATE|LOW|MODIFIER)$')
START_RE = re.compile(r'^##START$')

CSV_HEADER = "CYCLE,POPSIZE,HOM_REF,HET,HOM_DER,DER"
CLASSES = ["HIGH", "MODERATE", "LOW", "MODIFIER"]

def parse_one_csv(path, required_last_cycle):
    """
    Ritorna (meta, rows, is_complete)
    - meta: dict con class,K,BN,seed_burnin,rep,mut_id,s,init_freq,src_csv
    - rows: lista di dict con cycle,popsize,HOM_ANC,HET,HOM_DER,DER
    - is_complete: True se esiste una riga con cycle == required_last_cycle
    """
    m = FN_RE.search(os.path.basename(path))
    if not m:
        return None, [], False
    meta = {
        "class": m.group("class"),
        "K": m.group("K"),
        "BN": m.group("bn"),
        "seed_burnin": int(m.group("seed")),
        "rep": int(m.group("rep")),
        "mut_id": m.group("id"),
        "s": None,
        "init_freq": None,
        "src_csv": os.path.basename(path),
    }

    rows = []
    in_table = False
    has_required = False

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            # metadati prima della tabella
            if not in_table:
                if START_RE.match(line) or CLASS_RE.match(line):
                    continue
                mm = S_RE.match(line)
                if mm:
                    meta["s"] = mm.group("s"); continue
                mm = ID_RE.match(line)
                if mm:
                    meta["mut_id"] = meta["mut_id"] or mm.group("id"); continue
                mm = FREQ_RE.match(line)
                if mm:
                    meta["init_freq"] = mm.group("freq"); continue
                if line == CSV_HEADER:
                    in_table = True
                    continue
                # qualunque altra riga prima della tabella: ignora
                continue

            # righe tabella CSV semplici
            parts = line.split(",")
            if len(parts) != 6 or not parts[0].isdigit():
                # fine tabella o riga non valida
                in_table = False
                continue

            cycle   = int(parts[0])
            popsize = int(parts[1])
            hom_ref = parts[2]
            het     = parts[3]
            hom_der = parts[4]
            der     = parts[5]

            if cycle == required_last_cycle:
                has_required = True

            rows.append({
                "cycle": cycle,
                "popsize": popsize,
                "HOM_ANC": hom_ref,  # HOM_REF rinominato
                "HET": het,
                "HOM_DER": hom_der,
                "DER": der,
            })

    return meta, rows, has_required

def ensure_tsvs(out_dir):
    handles = {}
    for c in CLASSES:
        p = os.path.join(out_dir, f"{c}.tsv")
        new = not os.path.exists(p)
        fh = open(p, "a", encoding="utf-8")
        if new:
            fh.write("\t".join([
                "class","K","BN","seed_burnin","rep","mut_id","s","init_freq",
                "cycle","popsize","HOM_ANC","HET","HOM_DER","DER","src_csv"
            ]) + "\n")
        handles[c] = fh
    return handles

def main():
    ap = argparse.ArgumentParser(description="Parsa i CSV di SLiM e salva 1 TSV per classe (solo file completi).")
    ap.add_argument("out_dir", help="Cartella con i CSV (es. mod1_output/)")
    ap.add_argument("--mod", choices=["mod1","mod2"], required=True,
                    help="mod1 richiede last cycle=300510; mod2 richiede last cycle=305100")
    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        print(f"Directory non trovata: {args.out_dir}", file=sys.stderr)
        sys.exit(1)

    required_last_cycle = 300510 if args.mod == "mod1" else 305100

    handles = ensure_tsvs(args.out_dir)

    files = sorted(glob.glob(os.path.join(args.out_dir, "*.csv")))
    n_files = 0
    n_rows = 0
    n_skipped_incomplete = 0
    n_skipped_nodata = 0

    for fp in files:
        meta, rows, is_complete = parse_one_csv(fp, required_last_cycle)
        if not meta:
            continue
        if not rows:
            n_skipped_nodata += 1
            continue
        if not is_complete:
            n_skipped_incomplete += 1
            continue

        n_files += 1
        fh = handles[meta["class"]]
        for r in rows:
            fh.write("\t".join([
                meta["class"], str(meta["K"]), str(meta["BN"]), str(meta["seed_burnin"]), str(meta["rep"]),
                str(meta["mut_id"] or ""), str(meta["s"] or ""), str(meta["init_freq"] or ""),
                str(r["cycle"]), str(r["popsize"]), str(r["HOM_ANC"]), str(r["HET"]),
                str(r["HOM_DER"]), str(r["DER"]), meta["src_csv"]
            ]) + "\n")
            n_rows += 1

    for fh in handles.values():
        fh.close()

    print(f"Fatto. File completi parsati: {n_files}, righe totali: {n_rows}")
    print(f"Saltati (incompleti): {n_skipped_incomplete} — richiesto last_cycle={required_last_cycle}")
    print(f"Saltati (senza dati): {n_skipped_nodata}")
    print(f"TSV creati in: {args.out_dir}/ (HIGH.tsv, MODERATE.tsv, LOW.tsv, MODIFIER.tsv)")

if __name__ == "__main__":
    main()

