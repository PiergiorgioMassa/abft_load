#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   ./polarize_and_flip_vcf.sh INPUT.vcf.gz ancprob_with_coords.tsv INGROUP.txt|'-' THRESHOLD OUTPUT_PREFIX
#
# Se passi '-' al posto di INGROUP.txt, usa TUTTI i campioni del VCF come ingroup.
# Requisiti: python3 (pysam), bgzip, tabix

if [[ $# -lt 5 ]]; then
  echo "Uso: $0 INPUT.vcf.gz ancprob_with_coords.tsv INGROUP.txt|'-' THRESHOLD OUTPUT_PREFIX" >&2
  exit 1
fi

VCF="$1"
ANCTSV="$2"      # CHROM POS REF ALT P_major
INGARG="$3"      # path o '-' per tutti i campioni
THR="$4"
OUTPFX="$5"

[[ -f "$VCF" ]]    || { echo "VCF non trovato: $VCF" >&2; exit 1; }
[[ -f "$ANCTSV" ]] || { echo "TSV non trovato: $ANCTSV" >&2; exit 1; }
if [[ "$INGARG" != "-" && ! -f "$INGARG" ]]; then
  echo "Lista INGROUP non trovata: $INGARG (oppure usa '-')" >&2; exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/polarize.py" << 'PYCODE'
import sys, argparse, pysam

def load_anc(path):
    anc = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = ln.split('\t')
            if parts[0].upper() == "CHROM":  # salta header
                continue
            if len(parts) < 5: continue
            chrom, pos, ref, alt, p = parts[0], int(parts[1]), parts[2], parts[3], float(parts[4])
            anc[(chrom, pos, ref, alt)] = p
    return anc

def load_ingroup(path_or_dash, samples_in_vcf):
    if path_or_dash == "-":
        return samples_in_vcf[:]  # tutti i campioni
    sset = set()
    with open(path_or_dash) as f:
        for l in f:
            l=l.strip()
            if l and not l.startswith("#"):
                sset.add(l)
    return [s for s in samples_in_vcf if s in sset]

def ensure_info(h, key, number, typ, desc):
    if key not in h.info:
        h.info.add(key, number, typ, desc)

def flip_gt_tuple(gt):
    if gt is None: return None
    new=[]
    for a in gt:
        if a is None: new.append(None)
        elif a==0: new.append(1)
        elif a==1: new.append(0)
        else: new.append(a)
    return tuple(new)

def compute_ac_an(rec):
    AN=0; AC=0
    for s in rec.samples:
        samp = rec.samples[s]
        gt = samp.get('GT')
        if gt is None or len(gt)!=2 or gt[0] is None or gt[1] is None:
            continue
        if gt[0] not in (0,1) or gt[1] not in (0,1):
            continue
        AN += 2
        AC += (gt[0]==1) + (gt[1]==1)
    return AC, AN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--anc", required=True)
    ap.add_argument("--ingroup", required=True, help="path oppure '-' per tutti i campioni")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--outvcf", required=True)
    args = ap.parse_args()

    invcf = pysam.VariantFile(args.vcf)

    # Header input/output: aggiungi AA/DER/APR e (se mancano) AC/AN/AF
    hdr = invcf.header.copy()
    for H in (invcf.header, hdr):
        ensure_info(H, 'AA',  1, 'String', 'Ancestral allele inferred (est-sfs)')
        ensure_info(H, 'DER', 1, 'String', 'Derived allele inferred (est-sfs)')
        ensure_info(H, 'APR', 1, 'Float',  'P_major_ancestral from est-sfs')
        ensure_info(H, 'AN',  1, 'Integer','Total number of alleles in called genotypes')
        ensure_info(H, 'AC',  'A','Integer','Allele count in genotypes, for each ALT allele')
        ensure_info(H, 'AF',  'A','Float', 'Allele frequency, for each ALT allele')

    outvcf = pysam.VariantFile(args.outvcf, "w", header=hdr)

    samples_in_vcf = list(invcf.header.samples)
    ingroup = load_ingroup(args.ingroup, samples_in_vcf)
    if not ingroup:
        sys.stderr.write("ERRORE: nessun campione ingroup trovato nel VCF\n")
        sys.exit(2)

    ancmap  = load_anc(args.anc)

    kept = flipped = removed_amb = removed_nomatch = 0

    for rec in invcf:
        # solo SNP biallelici
        if len(rec.alts or []) != 1: continue
        if len(rec.ref)!=1 or len(rec.alts[0])!=1: continue

        key = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        p = ancmap.get(key)
        if p is None:
            removed_nomatch += 1
            continue

        # Conta AC/AN solo nell’ingroup per decidere il "maggioritario"
        ANi = 0; ACi = 0
        for s in ingroup:
            samp = rec.samples[s]
            gt = samp.get('GT')
            if gt is None or len(gt)!=2 or gt[0] is None or gt[1] is None: continue
            if gt[0] not in (0,1) or gt[1] not in (0,1): continue
            ANi += 2
            ACi += (gt[0]==1) + (gt[1]==1)

        if ANi == 0:
            removed_amb += 1
            continue

        if ACi*2 > ANi:
            major, minor = rec.alts[0], rec.ref
            aa_is_alt_major = True
        elif ACi*2 < ANi:
            major, minor = rec.ref, rec.alts[0]
            aa_is_alt_major = False
        else:
            removed_amb += 1
            continue

        # decide AA/DER
        if p >= args.thr:
            aa, der = major, minor
        elif p <= (1.0 - args.thr):
            aa, der = minor, major
        else:
            removed_amb += 1
            continue

        rec.info['AA']  = aa
        rec.info['DER'] = der
        rec.info['APR'] = round(p, 6)

        # flip se AA == ALT
        if aa == rec.alts[0]:
            old_ref, old_alt = rec.ref, rec.alts[0]
            rec.ref  = old_alt
            rec.alts = (old_ref,)
            for s in samples_in_vcf:
                samp = rec.samples[s]
                gt = samp.get('GT')
                if gt is not None:
                    samp['GT'] = flip_gt_tuple(gt)
                if 'AD' in samp and samp['AD'] is not None and len(samp['AD']) >= 2:
                    ad = list(samp['AD']); ad[:2] = [ad[1], ad[0]]; samp['AD'] = tuple(ad)
                if 'PL' in samp and samp['PL'] is not None and len(samp['PL']) == 3:
                    pl = list(samp['PL']); samp['PL'] = (pl[2], pl[1], pl[0])
            flipped += 1

        # (ri)calcola AC/AN/AF sui GT attuali (tutti i campioni)
        AC, AN = compute_ac_an(rec)
        if AN == 0:
            removed_amb += 1
            continue
        rec.info['AN'] = AN
        rec.info['AC'] = (AC,)   # Number=A, biallelico -> un solo valore
        rec.info['AF'] = (AC/AN,)

        outvcf.write(rec)
        kept += 1

    outvcf.close(); invcf.close()
    sys.stderr.write(f"Kept: {kept}, Flipped: {flipped}, Removed ambiguous/no-AN: {removed_amb}, Not in ancprob: {removed_nomatch}\n")

if __name__ == "__main__":
    main()
PYCODE

# esegui python (scrive VCF non compresso)
python3 "$TMP/polarize.py" --vcf "$VCF" --anc "$ANCTSV" --ingroup "$INGARG" --thr "$THR" --outvcf "$TMP/out.vcf" 1>&2

# comprime e indicizza (nessun plugin necessario)
bgzip -f -c "$TMP/out.vcf" > "${OUTPFX}.polarized.vcf.gz"
tabix -f -p vcf "${OUTPFX}.polarized.vcf.gz"

echo "OK:"
echo "  -> ${OUTPFX}.polarized.vcf.gz"
echo "  -> ${OUTPFX}.polarized.vcf.gz.tbi"
