"""Dock the in-domain novel molecules into COX-1 and COX-2 with smina.

Pipeline (all tools live in the mldd26 conda env):
  1. download the receptor PDBs from RCSB (cached in ./docking)
  2. split each into a clean rigid receptor (chain A protein + heme) and the
     co-crystallised inhibitor (used purely to auto-place the search box)
  3. convert the receptor to PDBQT with OpenBabel
  4. build 3D ligands with RDKit (ibuprofen control + the in-domain set)
  5. run smina with --autobox_ligand around the native inhibitor
  6. collect the best (most negative) affinity per ligand/receptor

Run with the mldd26 interpreter from the project root:
    .../envs/mldd26/python.exe dock_cox.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import subprocess
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ENV_BIN = Path(r"C:/Users/adria/anaconda3/envs/mldd26/Library/bin")
SMINA = ENV_BIN / "smina.exe"
OBABEL = ENV_BIN / "obabel.exe"

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "docking"
WORK.mkdir(exist_ok=True)

ADMET_CSV = ROOT / "novel_molecules_admet.csv"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"

# Receptor definitions: PDB id + the resname of the bound inhibitor (chain A)
# used to centre the docking box.
RECEPTORS = {
    "COX1": {"pdb": "1EQG", "lig_resn": "IBP"},  # ovine COX-1 + ibuprofen
    "COX2": {"pdb": "1CX2", "lig_resn": "S58"},  # murine COX-2 + SC-558
}

EXHAUSTIVENESS = 8
NUM_MODES = 9
SEED = 42


def download_pdb(pdb_id: str) -> Path:
    dest = WORK / f"{pdb_id}.pdb"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[recv] downloading {pdb_id} from RCSB ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def clean_pdb(pdb_path: Path, lig_resn: str) -> tuple[Path, Path]:
    """Split a PDB into (rigid receptor, reference ligand) for chain A.

    Receptor = chain-A protein atoms + heme (HEM); waters, sugars, detergents
    and the inhibitor are dropped. The inhibitor is written separately to seed
    the smina search box.
    """
    rec_path = WORK / f"{pdb_path.stem}_rec.pdb"
    lig_path = WORK / f"{pdb_path.stem}_ref.pdb"
    rec_lines, lig_lines = [], []
    for line in pdb_path.read_text().splitlines():
        rec = line[:6]
        if rec not in ("ATOM  ", "HETATM"):
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):           # keep one conformer
            continue
        chain = line[21]
        if chain != "A":                        # one monomer is enough
            continue
        resn = line[17:20].strip()
        if rec == "ATOM  ":
            rec_lines.append(line)              # protein backbone/side chains
        elif resn == "HEM":
            rec_lines.append(line)              # keep the catalytic heme
        elif resn == lig_resn:
            lig_lines.append(line)              # native inhibitor -> box seed
    if not lig_lines:
        raise RuntimeError(f"ligand {lig_resn} not found in {pdb_path.name}")
    rec_path.write_text("\n".join(rec_lines) + "\nEND\n")
    lig_path.write_text("\n".join(lig_lines) + "\nEND\n")
    return rec_path, lig_path


def prep_receptor(rec_pdb: Path) -> Path:
    """OpenBabel: add hydrogens at pH 7.4 and write a rigid PDBQT receptor."""
    out = rec_pdb.with_suffix(".pdbqt")
    subprocess.run(
        [str(OBABEL), str(rec_pdb), "-O", str(out), "-xr", "-p", "7.4"],
        check=True, capture_output=True, text=True,
    )
    return out


def prep_ligand(smiles: str, name: str) -> Path | None:
    """RDKit: embed a 3D conformer (ETKDG) and MMFF/UFF-minimise -> SDF."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[lig] {name}: invalid SMILES, skipping")
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    if AllChem.EmbedMolecule(mol, params) != 0:
        print(f"[lig] {name}: 3D embedding failed, skipping")
        return None
    if AllChem.MMFFOptimizeMolecule(mol) != 0:
        AllChem.UFFOptimizeMolecule(mol)        # fallback for missing MMFF types
    mol.SetProp("_Name", name)
    out = WORK / f"lig_{name}.sdf"
    Chem.SDWriter(str(out)).write(mol)
    return out


def dock(receptor_pdbqt: Path, ref_ligand: Path, ligand_sdf: Path,
         tag: str) -> float | None:
    """Run smina; return the best (most negative) affinity in kcal/mol."""
    out_sdf = WORK / f"out_{tag}.sdf"
    proc = subprocess.run(
        [str(SMINA), "-r", str(receptor_pdbqt), "-l", str(ligand_sdf),
         "--autobox_ligand", str(ref_ligand), "--autobox_add", "4",
         "-o", str(out_sdf), "--exhaustiveness", str(EXHAUSTIVENESS),
         "--num_modes", str(NUM_MODES), "--seed", str(SEED)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"[dock] {tag}: smina failed -> {proc.stderr.strip()[:200]}")
        return None
    # Best affinity = the 'minimizedAffinity' of the top pose in the output SDF.
    best = None
    for m in Chem.SDMolSupplier(str(out_sdf), sanitize=False):
        if m is not None and m.HasProp("minimizedAffinity"):
            aff = float(m.GetProp("minimizedAffinity"))
            best = aff if best is None else min(best, aff)
    return best


def main() -> None:
    # ---- Prepare receptors ------------------------------------------------
    prepared = {}
    for name, spec in RECEPTORS.items():
        pdb = download_pdb(spec["pdb"])
        rec_pdb, ref_lig = clean_pdb(pdb, spec["lig_resn"])
        rec_pdbqt = prep_receptor(rec_pdb)
        prepared[name] = (rec_pdbqt, ref_lig)
        print(f"[recv] {name} ({spec['pdb']}) ready: {rec_pdbqt.name}")

    # ---- Collect ligands: ibuprofen control + in-domain molecules ---------
    df = pd.read_csv(ADMET_CSV)
    in_dom = df[df["in_domain"] == True].reset_index(drop=True)  # noqa: E712
    ligands = [("ibuprofen_control", IBUPROFEN)]
    for i, row in in_dom.iterrows():
        ligands.append((f"mol{i:02d}", row["smiles"]))
    print(f"[lig] {len(ligands)} ligands "
          f"(1 control + {len(in_dom)} in-domain)")

    # ---- Dock every ligand into every receptor ----------------------------
    rows = []
    for name, smiles in ligands:
        sdf = prep_ligand(smiles, name)
        if sdf is None:
            continue
        result = {"name": name, "smiles": smiles}
        for rec_name, (rec_pdbqt, ref_lig) in prepared.items():
            aff = dock(rec_pdbqt, ref_lig, sdf, tag=f"{name}_{rec_name}")
            result[f"{rec_name}_affinity"] = aff
            print(f"[dock] {name:18s} {rec_name}: "
                  f"{aff if aff is not None else 'FAIL'} kcal/mol")
        rows.append(result)

    out = pd.DataFrame(rows)
    out_csv = ROOT / "docking_results.csv"
    out.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv}")
    cols = ["name", "COX1_affinity", "COX2_affinity"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
