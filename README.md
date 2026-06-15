# Autonomous End-to-End Agentic Workflow with MTL for Drug Discovery

## Abstract
The traditional drug discovery process is extremely costly, with the vast
majority of novel compounds failing during clinical trials due to unforeseen
toxicity or poor pharmacokinetic profiles (ADMET). While generative Large
Language Models (LLMs) show promise in *de novo* drug design, they frequently
hallucinate - generating structures that are syntactically invalid, chemically
unstable, or impossible to synthesize.

This project is an autonomous pipeline that combines the reasoning of a local
LLM agent with deterministic cheminformatics validation, a predictive
Multi-Task Learning (MTL) network for ADMET, an explicit **applicability-domain**
check, and 3D molecular docking - minimizing failure risk at the earliest
stages of molecular design.

---

## Pipeline (4 stages)

```
STAGE 1  GENERATION ──────────────────────────────────────────────
  Path A  base qwen3.5:4b, prompted toward ibuprofen / NSAID space
  Path B  qwen2.5:3b fine-tuned toward EGFR actives
          │ 
          │  
          ▼  candidate SMILES
STAGE 2  VALIDATION (in-loop, deterministic) ─────────────────────
  check_molecule()  =  RDKit syntax + canonicalization
                     + physicochemical props (MW, LogP, HBD, HBA, TPSA, QED…)
                     + Lipinski / Veber drug-likeness gate
                     + PubChem novelty (InChIKey exact + 90/70% analogs)
          │
          ▼  valid, drug-like, novel SMILES  →  CSV
STAGE 3  ADMET FILTRATION (MTL network) ──────────────────────────
  9 tasks (caco2, bbb, cyp3a4, half_life, herg, hum_mic_cl, ppb, logd, water_sol)
  + Monte-Carlo Dropout  → epistemic uncertainty (mean ± std per task)
  + Tanimoto applicability-domain check → flag out-of-domain (unreliable) preds
          │
          ▼  safe, in-domain lead compounds  →  shortlist
STAGE 4  3D DOCKING (smina / AutoDock Vina) ──────────────────────
  COX-2 (1CX2) vs COX-1 (1EQG)  - selectivity for the NSAID lead (mol12)
  EGFR  (1M17)                   - kinase ATP pocket for the EGFR leads (cand1…)
          │
          ▼  final leads + binding poses (PyMOL renders)
```
---

## Repository layout

```
autodrug-agent/                       # ← git repository root (this README lives here)
├── LICENSE                           # MIT
├── README.md                         # this file
│
├── agent_base_model/                 # Stage 1 · Path A - ReAct generation agent
│   ├── main.py                       #   qwen3.5:4b (Ollama) + check_molecule tool
│   └── novel_molecules.csv           #   sample of validated agent output
│
├── agent_fine_tuned/                 # Stage 1 · Path B - fine-tuned generator + agent
│   ├── stage1_fine_tuning.ipynb      #   QLoRA SMILES-grammar fine-tune of Qwen2.5-3B (~25k ChEMBL)
│   ├── stage2_fine_tuning.ipynb      #   Stage-2 EGFR specialization → qwen25_3b_smiles_egfr_lora
│   └── orchestrator-and-fine-tuned-generator.ipynb   # ReAct agent: qwen2.5:7b (Ollama) drives the fine-tuned generator
│
├── mtl_nn/                           # Stage 3 · ADMET Multi-Task Learning network
│   ├── config.py                     #   task registry + hyperparameters (single source of truth)
│   ├── main.py                       #   TRAINING entry point (download → train → eval → save)
│   ├── predict_admet.py              #   INFERENCE CLI: MC-Dropout + Tanimoto domain check
│   ├── model/
│   │   └── network.py                #   shared trunk (2048→1024→512→256) + 9 task heads; predict_mc_dropout
│   ├── data/
│   │   ├── featurizer.py             #   Morgan / ECFP4 fingerprints + Tanimoto similarity
│   │   ├── tdc_loader.py             #   TDC + PharmaBench download / outer-merge on SMILES
│   │   └── dataset.py                #   masked multi-task Dataset + DataLoaders
│   ├── training/
│   │   ├── loss.py                   #   masked multi-task loss
│   │   └── trainer.py                #   training loop
│   ├── notebookbe21d85c75.ipynb      #   exploratory training / evaluation notebook
│   └── requirements.txt
│
├── docking/                          # Stage 4 · 3D docking (smina / AutoDock Vina)
│   ├── dock_cox.py                   #   COX-1 (1EQG) vs COX-2 (1CX2) selectivity
│   ├── dock_egfr.py                  #   EGFR kinase ATP pocket (1M17); reuses dock_cox helpers
│   ├── render_mol12.pml              #   PyMOL render script - NSAID lead
│   ├── render_cand1.pml              #   PyMOL render script - EGFR lead
│   ├── mol12_COX2_pocket.png         #   rendered COX-2 binding pose
│   └── cand1_EGFR_pocket.png         #   rendered EGFR binding pose
│
├── results/                          # generated outputs + analysis
│   ├── novel_molecules_admet.csv     #   Path A candidates + ADMET preds + domain flags
│   ├── agent_candidates.csv          #   Path B (EGFR) candidates, validated
│   ├── agent_candidates_admet.csv    #   Path B candidates + ADMET preds + domain flags
│   ├── agent_candidates_shortlist.csv#   ranked, in-domain EGFR leads → docking input
│   ├── analysis_visualization.ipynb  #   plots: ADMET distributions, uncertainty, domain, safety
│   └── *.png                         #   exported figures (agent_* and novel_* sets)
│
└── data/                             # cached TDC datasets (*.tab) - auto-downloaded, not committed
```

> **Note - trained weights are not committed.** Training (`mtl_nn/main.py`)
> writes `mtl_nn/model_weights.pt`; inference (`mtl_nn/predict_admet.py`) loads
> `admet_mtl_weights.pt` by default. Train the model to produce weights, or pass
> your own file via `--weights`.

---

## Stage details

### Stage 1 - Generation
- **Path A** (`agent_base_model/main.py`): a local **qwen3.5:4b** runs in a
  **ReAct** loop (LangGraph `create_react_agent`) prompted to design novel
  ibuprofen-like NSAIDs. A single combined `check_molecule` tool enforces
  RDKit validation + PubChem novelty in code (small models reliably skip the
  second step of a two-tool workflow). Output → `agent_base_model/novel_molecules.csv`.
- **Path B** (`agent_fine_tuned/`): a **fine-tuned**, EGFR-targeted generator
  driven by its own agent - the trainable counterpart to Path A. Three Kaggle
  notebooks (developed on 2× T4):
  - `stage1_fine_tuning.ipynb` - **QLoRA** (4-bit NF4 + LoRA `r=32`, TRL
    `SFTTrainer`) fine-tune of **Qwen2.5-3B** on ~25k broad ChEMBL SMILES to
    learn valid SMILES grammar → `qwen25_3b_smiles_lora` adapter.
  - `stage2_fine_tuning.ipynb` - **Stage 2** continues from that adapter and
    specialises on **EGFR (CHEMBL203)** actives (IC50/Ki/Kd < 1000 nM, ~5.8k
    cleaned SMILES) → `qwen25_3b_smiles_egfr_lora` (~70 % valid / ~60 %
    EGFR-like scaffold).
  - `orchestrator-and-fine-tuned-generator.ipynb` - a **ReAct** agent (LangGraph)
    where a **qwen2.5:7b** orchestrator (Ollama, reliable native tool-calling)
    drives the fine-tuned generator through three combined tools:
    `generate_candidates` (generate + RDKit-validate), `analyze_and_rank`
    (drug-likeness + Lipinski + QED ranking, auto-slims too-heavy hits via BRICS,
    saves to CSV) and `edit_molecule`. A final selection step ranks by
    QED × Tanimoto similarity to known EGFR inhibitors (gefitinib / erlotinib /
    osimertinib) and flags PAINS / BRENK structural alerts.

  Output → `results/agent_candidates.csv`, which feeds Stages 3-4.

### Stage 2 - Validation (`check_molecule`)
Every SMILES passes through, in order: RDKit parse/canonicalize → descriptor
computation → Lipinski (≤1 violation) / Veber gate → PubChem novelty
(InChIKey exact match + 2D similarity analogs at 90 % / 70 %). Verdict:
`PASS` / `FAIL` / `ERROR`. Results are de-duplicated and appended to CSV.

### Stage 3 - ADMET (MTL network)
- **Data:** TDC (Caco2, BBB, CYP3A4, Half-Life, hERG) + PharmaBench
  (hum_mic_cl, ppb, logd, water_sol), outer-merged on SMILES with a per-task
  label mask (`mtl_nn/data/tdc_loader.py`). TDC sets are cached locally under
  `data/*.tab`; PharmaBench sets are pulled from GitHub at runtime.
- **Model:** hard-parameter-sharing trunk `2048 → 1024 → 512 → 256` (LayerNorm,
  dropout 0.1) + one head per task; `half_life` is trained on `log1p` and
  inverted with `expm1` (`mtl_nn/model/network.py`). Tasks, types and weights
  are declared once in `mtl_nn/config.py`.
- **Uncertainty:** `predict_mc_dropout` keeps dropout active for N stochastic
  passes (default **10** in `predict_admet.py`, 50 in the research run in
  `mtl_nn/main.py`) → `mean ± std`.
- **Applicability domain:** max **Tanimoto** similarity (ECFP4, 2048 bits) to the
  training set; below the threshold (default **0.4**, `--tanimoto-threshold`) a
  molecule is flagged **out-of-domain** and its ADMET prediction treated as
  unreliable.

### Stage 4 - Docking (`docking/dock_cox.py`, `docking/dock_egfr.py`)
smina with `--autobox_ligand` seeded from each receptor's co-crystal ligand;
OpenBabel prepares the receptor PDBQT, RDKit embeds 3D ligands (ETKDGv3 +
MMFF/UFF). Receptors: **COX-2 `1CX2`**, **COX-1 `1EQG`** (selectivity),
**EGFR `1M17`** (erlotinib ATP pocket). Poses are rendered with the PyMOL
scripts in `docking/render_*.pml` (see `mol12_COX2_pocket.png`,
`cand1_EGFR_pocket.png`).

Example results: the NSAID lead **mol12** reproduced the canonical coxib
selectivity (COX-2 −8.80 vs COX-1 −6.72 kcal/mol; H-bonds to Arg513, His90),
beating the ibuprofen control (−7.74). The EGFR lead **cand1** scored −8.65 vs
the erlotinib control −7.12.

> The docking scripts read their input CSV from the `docking/` directory and use
> `smina`/`obabel` paths hard-coded to the author's conda env (`ENV_BIN` at the
> top of `dock_cox.py`). Edit `ENV_BIN` to point at your own tools, and place the
> relevant input CSV (`novel_molecules_admet.csv` / `agent_candidates_shortlist.csv`)
> next to the scripts before running.

---

## Tech stack
- **Generative AI & Agents:** LangChain / LangGraph, Ollama (local Qwen models)
- **LLM fine-tuning:** Hugging Face Transformers, PEFT (LoRA), TRL (SFT),
  bitsandbytes (4-bit), ChEMBL web client
- **Cheminformatics:** RDKit (descriptors, QED, BRICS, PAINS/BRENK alerts), PubChem PUG REST API
- **Machine Learning:** PyTorch (MTL network + MC-Dropout), TDC + PharmaBench data
- **Docking & visualization:** smina / AutoDock Vina, OpenBabel, PyMOL

## Quick start
Run all commands from the repository root (`autodrug-agent/`).

```bash
# 1. Install (PyTDC is needed for training + the applicability-domain check)
pip install -r mtl_nn/requirements.txt
pip install langchain langchain-ollama langgraph requests   # Stage 1 agent

# 2. Stage 1 · Path A - generate candidates (Ollama must be running)
ollama pull qwen3.5:4b
python agent_base_model/main.py            # → agent_base_model/novel_molecules.csv

# 2b. Stage 1 · Path B - fine-tuned generator + agent (GPU; developed on Kaggle 2× T4)
#     deps: pip install -r agent_fine_tuned/requirements.txt
#     stage1 → stage2 notebooks build the LoRA adapter (qwen25_3b_smiles_egfr_lora),
#     then orchestrator-and-fine-tuned-generator.ipynb runs the discovery agent
#     → results/agent_candidates.csv

# 3. Stage 3 - train the MTL network (writes mtl_nn/model_weights.pt)
python -m mtl_nn.main

# 4. Stage 3 - ADMET inference + applicability-domain check
#    (loads admet_mtl_weights.pt by default; override with --weights)
python -m mtl_nn.predict_admet --input candidates.csv --output preds.csv --domain-check

# 5. Stage 4 - docking (needs smina + OpenBabel; edit ENV_BIN in dock_cox.py)
cd docking
python dock_cox.py
python dock_egfr.py
```
