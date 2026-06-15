"""Dock the top shortlist candidates into EGFR kinase domain with smina.

Reuses the helper functions from dock_cox.py (download / clean / prep / dock).
Receptor: 1M17 = EGFR kinase domain co-crystallised with erlotinib (resname
AQ4), a 4-anilinoquinazoline ATP-competitive inhibitor -- the native ligand
seeds the docking box in the ATP pocket and erlotinib is redocked as control.

Run with the mldd26 interpreter from the project root:
    .../envs/mldd26/python.exe dock_egfr.py
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

import dock_cox as dc  # helper functions + tool paths

# Erlotinib (Tarceva): redock control for the EGFR ATP pocket.
ERLOTINIB = "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
RECEPTOR = {"pdb": "1M17", "lig_resn": "AQ4"}
SHORTLIST = dc.ROOT / "agent_candidates_shortlist.csv"
TOP_N = 3


def main() -> None:
    # ---- Prepare the EGFR receptor ----------------------------------------
    pdb = dc.download_pdb(RECEPTOR["pdb"])
    rec_pdb, ref_lig = dc.clean_pdb(pdb, RECEPTOR["lig_resn"])
    rec_pdbqt = dc.prep_receptor(rec_pdb)
    print(f"[recv] EGFR ({RECEPTOR['pdb']}) ready: {rec_pdbqt.name}")

    # ---- Ligands: erlotinib control + top-N shortlist candidates ----------
    df = pd.read_csv(SHORTLIST)
    top = df.head(TOP_N)
    ligands = [("erlotinib_control", ERLOTINIB)]
    for i, row in top.iterrows():
        ligands.append((f"cand{i + 1}", row["smiles"]))
    print(f"[lig] {len(ligands)} ligands (1 control + {len(top)} candidates)")

    rows = []
    for name, smiles in ligands:
        sdf = dc.prep_ligand(smiles, name)
        if sdf is None:
            continue
        aff = dc.dock(rec_pdbqt, ref_lig, sdf, tag=f"{name}_EGFR")
        print(f"[dock] {name:18s} EGFR: "
              f"{aff if aff is not None else 'FAIL'} kcal/mol")
        rows.append({"name": name, "smiles": smiles, "EGFR_affinity": aff})

    out = pd.DataFrame(rows)
    out_csv = dc.ROOT / "docking_results_egfr.csv"
    out.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv}")
    print(out[["name", "EGFR_affinity", "smiles"]].to_string(index=False))


if __name__ == "__main__":
    main()
