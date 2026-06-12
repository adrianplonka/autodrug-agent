"""
Zero-Cost Drug Discovery – Phase 1: ReAct AI Agent
====================================================
An autonomous agent that generates novel molecules in SMILES format,
validates their syntax with RDKit, and checks novelty via the PubChem API.
Architecture:
    LLM (qwen3.5:4b via Ollama)
        ↓
    ReAct loop (LangGraph create_react_agent)
        └── Tool: check_molecule → RDKit validation + PubChem novelty check
            (a single combined tool – small models reliably skip the second
             step of a two-tool workflow, so the pipeline is enforced in code)
Tech stack:
    • langchain / langchain-ollama / langgraph  agent framework
    • rdkit                                     cheminformatics
    • requests                                  HTTP client
Usage:
    Ensure Ollama is running locally with the qwen3.5:4b model pulled:
        $ ollama pull qwen3.5:4b
    Then run:
        $ python main.py
"""
from __future__ import annotations
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any
import requests
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
# ---------------------------------------------------------------------------
# RDKit logging: set to WARNING so parse errors are visible in real time.
# (Use RDLogger.CRITICAL to suppress everything, or RDLogger.ERROR for
#  only critical issues.)
# ---------------------------------------------------------------------------
RDLogger.logger().setLevel(RDLogger.WARNING)
# ---------------------------------------------------------------------------
# Python logging: real-time output with timestamps.
# ---------------------------------------------------------------------------
# Force UTF-8 so the box-drawing/check-mark characters in the logs don't
# crash when stdout is redirected (Windows falls back to a legacy codepage).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
# Suppress noisy httpx logs (Ollama HTTP calls: "HTTP Request: POST ...")
logging.getLogger("httpx").setLevel(logging.WARNING)
# ========================== CONSTANTS ======================================
PUBCHEM_PUG_BASE: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_TIMEOUT_SECONDS: int = 30  # generous timeout for slow connections
# PubChem allows at most ~5 requests/second – keep a safety margin
PUBCHEM_MIN_REQUEST_INTERVAL: float = 0.25
# Drug-likeness gate: a molecule failing Lipinski's Rule of Five by 2+
# criteria is rejected before the (slow) PubChem novelty check.
MAX_LIPINSKI_VIOLATIONS: int = 1
# CSV file for persisting discovered novel molecules (append mode)
CSV_PATH: Path = Path(__file__).parent / "novel_molecules.csv"
CSV_HEADERS: list[str] = [
    "timestamp", "smiles", "scaffold", "exact_match",
    "analogs_90", "analogs_70", "scaffold_known",
    "mw", "logp", "hbd", "hba", "tpsa", "rot_bonds", "qed",
    "lipinski_violations", "verdict",
]
# ===========================================================================
#  CSV persistence – append novel molecules to a running log
# ===========================================================================
def _load_discovered() -> set[str]:
    """Load canonical SMILES already saved in the CSV (for deduplication)."""
    if not CSV_PATH.exists():
        return set()
    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as fh:
            return {row["smiles"] for row in csv.DictReader(fh) if row.get("smiles")}
    except OSError as exc:
        logger.warning("[CSV] Could not read existing file: %s", exc)
        return set()
# Canonical SMILES of every molecule saved so far (survives restarts via CSV)
discovered_smiles: set[str] = _load_discovered()
def save_to_csv(
    smiles: str,
    report: dict[str, Any],
    verdict: str,
) -> None:
    """Append a row to the CSV results file (creates it with headers if new)."""
    def cell(value: Any) -> Any:
        return "" if value is None else value
    file_exists: bool = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    try:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if not file_exists:
                writer.writerow(CSV_HEADERS)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                smiles,
                cell(report.get("scaffold_smiles")),
                cell(report.get("compound_known")),
                cell(report.get("close_analogs_90")),
                cell(report.get("broad_analogs_70")),
                cell(report.get("scaffold_known")),
                cell(report.get("mw")),
                cell(report.get("logp")),
                cell(report.get("hbd")),
                cell(report.get("hba")),
                cell(report.get("tpsa")),
                cell(report.get("rot_bonds")),
                cell(report.get("qed")),
                cell(report.get("lipinski_violations")),
                verdict,
            ])
        discovered_smiles.add(smiles)
        logger.info("[CSV] Saved to %s", CSV_PATH)
    except OSError as exc:
        logger.error("[CSV] Failed to write: %s", exc)
# ===========================================================================
#  RDKit physicochemical properties & drug-likeness
# ===========================================================================
def compute_properties(mol: Chem.rdchem.Mol) -> dict[str, Any]:
    """Compute key physicochemical descriptors and drug-likeness flags.

    Returns the classic Lipinski Rule-of-Five inputs (MW, LogP, H-bond
    donors/acceptors), the Veber oral-bioavailability descriptors (TPSA,
    rotatable bonds), and the QED score (0–1, higher = more drug-like).
    """
    mw: float = Descriptors.MolWt(mol)
    logp: float = Crippen.MolLogP(mol)
    hbd: int = Lipinski.NumHDonors(mol)
    hba: int = Lipinski.NumHAcceptors(mol)
    tpsa: float = Descriptors.TPSA(mol)
    rot_bonds: int = Descriptors.NumRotatableBonds(mol)
    try:
        qed: float = QED.qed(mol)
    except Exception:  # QED can fail on exotic structures
        qed = float("nan")
    # Lipinski Rule of Five: drug-like when at most one criterion is broken
    lipinski_flags = {
        "MW>500": mw > 500,
        "LogP>5": logp > 5,
        "HBD>5": hbd > 5,
        "HBA>10": hba > 10,
    }
    violations = [name for name, broken in lipinski_flags.items() if broken]
    # Veber rules for oral bioavailability
    veber_ok = rot_bonds <= 10 and tpsa <= 140
    return {
        "mw": round(mw, 1),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 1),
        "rot_bonds": rot_bonds,
        "qed": round(qed, 3) if qed == qed else "",  # NaN check
        "lipinski_violations": len(violations),
        "lipinski_failed": violations,
        "veber_ok": veber_ok,
    }
# ===========================================================================
#  PubChem helpers – rate-limited HTTP, identity and similarity lookups
# ===========================================================================
_last_pubchem_request: float = 0.0
def _pubchem_request(
    url: str,
    *,
    data: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> requests.Response | None:
    """Single rate-limited request to PubChem.

    SMILES is sent as POST form data (never in the URL path) because SMILES
    may contain '/' and '\\', which break PUG REST path routing even when
    percent-encoded.  Transient failures (network error, HTTP 503) are
    retried once.  Returns None when the request ultimately failed, so
    callers can distinguish "no results" from "check failed".
    """
    global _last_pubchem_request
    for attempt in range(2):
        wait = PUBCHEM_MIN_REQUEST_INTERVAL - (time.monotonic() - _last_pubchem_request)
        if wait > 0:
            time.sleep(wait)
        try:
            if data is not None:
                resp = requests.post(url, data=data, params=params,
                                     timeout=PUBCHEM_TIMEOUT_SECONDS)
            else:
                resp = requests.get(url, params=params,
                                    timeout=PUBCHEM_TIMEOUT_SECONDS)
            if resp.status_code == 503 and attempt == 0:
                logger.warning("[PubChem] HTTP 503 (server busy) – retrying in 2s")
                time.sleep(2)
                continue
            return resp
        except requests.exceptions.RequestException as exc:
            if attempt == 0:
                logger.warning("[PubChem] Request failed (%s) – retrying in 2s", exc)
                time.sleep(2)
                continue
            logger.warning("[PubChem] Request failed after retry: %s", exc)
            return None
        finally:
            _last_pubchem_request = time.monotonic()
    return None
def _pubchem_identity(smiles: str) -> list[int] | None:
    """Exact-match lookup, verified via InChIKey.

    PubChem's SMILES search can return CIDs for normalised/tautomeric forms
    that don't truly match our molecule, so candidate CIDs are verified by
    comparing InChIKeys.  InChIKey is stereochemistry-aware, unlike PubChem's
    connectivity-only canonical SMILES (the old CanonicalSMILES property,
    now renamed ConnectivitySMILES), so it avoids both false positives and
    false negatives for stereo compounds.

    Returns:
        Verified CIDs (empty list = compound genuinely not found), or None
        when the check could not be completed (network error / throttling).
    """
    logger.info("[PubChem] Identity lookup: %s", smiles)
    t0 = time.time()
    resp = _pubchem_request(
        f"{PUBCHEM_PUG_BASE}/compound/smiles/cids/JSON",
        data={"smiles": smiles},
    )
    if resp is None:
        return None
    logger.info("[PubChem] Identity → HTTP %d (%.1fs)", resp.status_code, time.time() - t0)
    if resp.status_code == 404:
        return []  # PUGREST.NotFound – genuinely no matching compound
    if not resp.ok:
        # 503 ServerBusy etc. – the check did NOT succeed; never treat as novelty
        logger.warning("[PubChem] Identity HTTP %d – check inconclusive", resp.status_code)
        return None
    # PubChem returns CID 0 as a sentinel meaning "not found"
    cids = [c for c in resp.json().get("IdentifierList", {}).get("CID", []) if c != 0]
    if not cids:
        return []
    cids = cids[:5]  # verify at most 5 candidates
    logger.info("[PubChem] Identity: candidate CIDs = %s", cids)
    # --- Verify candidates by InChIKey (one batched property request) -----
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return cids
    our_key: str = Chem.MolToInchiKey(mol)
    prop_resp = _pubchem_request(
        f"{PUBCHEM_PUG_BASE}/compound/cid/"
        f"{','.join(str(c) for c in cids)}/property/InChIKey/JSON"
    )
    if prop_resp is None or not prop_resp.ok:
        # Cannot verify – conservatively keep all candidates as matches
        logger.warning("[PubChem] InChIKey verification unavailable – "
                       "keeping %d candidate CIDs as matches", len(cids))
        return cids
    props = prop_resp.json().get("PropertyTable", {}).get("Properties", [])
    verified = [p["CID"] for p in props
                if "CID" in p and p.get("InChIKey") == our_key]
    logger.info("[PubChem] Verified exact matches: %d / %d", len(verified), len(cids))
    return verified
def _pubchem_similarity(smiles: str, threshold: int = 90) -> int | None:
    """2-D fingerprint similarity search at the given Tanimoto threshold.

    Returns the number of analogs found, or None when the check failed.
    """
    logger.info("[PubChem] Similarity search (threshold=%d%%): %s", threshold, smiles)
    t0 = time.time()
    resp = _pubchem_request(
        f"{PUBCHEM_PUG_BASE}/compound/fastsimilarity_2d/smiles/cids/JSON",
        data={"smiles": smiles},
        params={"Threshold": threshold},
    )
    if resp is None:
        return None
    logger.info("[PubChem] Similarity(%d%%) → HTTP %d (%.1fs)",
                threshold, resp.status_code, time.time() - t0)
    if resp.status_code == 404:
        return 0  # no analogs at this threshold
    if not resp.ok:
        logger.warning("[PubChem] Similarity(%d%%) HTTP %d – check inconclusive",
                       threshold, resp.status_code)
        return None
    count = len(resp.json().get("IdentifierList", {}).get("CID", []))
    logger.info("[PubChem] Similarity(%d%%) analogs found: %d", threshold, count)
    return count
def _novelty_report(smiles: str) -> dict[str, Any]:
    """Build a multi-level novelty report for a given canonical SMILES.
    Checks:
        1. Exact identity match in PubChem
        2. Close analogs (≥ 90 % Tanimoto similarity)
        3. Broad analogs (≥ 70 % Tanimoto similarity)
        4. Whether the Murcko scaffold itself is a known compound
    Each field is None when that check could not be completed (network
    error or PubChem throttling) – callers must NOT treat None as novelty.
    """
    logger.info("[Novelty] Building full report for: %s", smiles)
    t0_total = time.time()
    mol: Chem.rdchem.Mol | None = Chem.MolFromSmiles(smiles)
    scaffold_smi: str = ""
    if mol is not None:
        scaffold_smi = Chem.MolToSmiles(
            MurckoScaffold.GetScaffoldForMol(mol)
        )
        logger.info("[Novelty] Murcko scaffold: %s", scaffold_smi)
    else:
        logger.warning("[Novelty] Could not compute scaffold (MolFromSmiles returned None)")
    identity = _pubchem_identity(smiles)
    report: dict[str, Any] = {
        "compound_known": None if identity is None else bool(identity),
        "close_analogs_90": None,
        "broad_analogs_70": None,
        "scaffold_known": None,
        "scaffold_smiles": scaffold_smi,
    }
    if report["compound_known"]:
        # Already known – the verdict is FAIL regardless, so skip the
        # remaining lookups to conserve the PubChem request budget.
        logger.info("[Novelty] Exact match found – skipping similarity checks")
    else:
        report["close_analogs_90"] = _pubchem_similarity(smiles, 90)
        report["broad_analogs_70"] = _pubchem_similarity(smiles, 70)
        if scaffold_smi:
            scaffold_identity = _pubchem_identity(scaffold_smi)
            report["scaffold_known"] = (
                None if scaffold_identity is None else bool(scaffold_identity)
            )
        else:
            report["scaffold_known"] = False
    elapsed_total = time.time() - t0_total
    logger.info("[Novelty] Report complete in %.1fs: %s", elapsed_total, report)
    return report
# ===========================================================================
#  The single agent tool – RDKit validation + PubChem novelty in ONE step
# ===========================================================================
@tool
def check_molecule(smiles: str) -> str:
    """Validate a SMILES string and run the full PubChem novelty assessment.

    This ONE tool performs the complete evaluation pipeline:
      1. RDKit syntax/valence validation
      2. Exact identity search    – is the compound already registered?
      3. Close-analog search      – compounds at >= 90% Tanimoto similarity
      4. Broad-analog search      – compounds at >= 70% similarity
      5. Scaffold lookup          – is the Murcko scaffold itself known?
    Args:
        smiles: A SMILES string representing the designed molecule.
    Returns:
        A syntax error message (fix the SMILES and call again), or a
        structured novelty report with a PASS / FAIL / ERROR verdict
        and guidance for the next step.
    """
    logger.info("[Tool:check_molecule] Called with: '%s'", smiles)
    if not smiles or not smiles.strip():
        logger.warning("[Tool:check_molecule] Empty SMILES received")
        return (
            "ERROR: Empty SMILES string provided. "
            "Please generate a valid, non-empty SMILES structure."
        )
    smiles = smiles.strip()
    # ---- Step 1: RDKit syntax validation ----------------------------------
    logger.info("[RDKit] Parsing SMILES: %s", smiles)
    mol: Chem.rdchem.Mol | None = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("[RDKit] ✘ INVALID SMILES: '%s'", smiles)
        return (
            f"ERROR: RDKit could not parse the SMILES '{smiles}'. "
            "The chemical structure is invalid (bad valence, unmatched "
            "parentheses, or illegal atom symbols). "
            "Please generate a new, syntactically correct SMILES."
        )
    # Canonical form: consistent deduplication and PubChem comparison
    smiles = Chem.MolToSmiles(mol)
    logger.info("[RDKit] ✔ Valid → canonical: %s", smiles)
    # ---- Local deduplication: don't re-discover the same molecule --------
    if smiles in discovered_smiles:
        logger.info("[Novelty] %s already discovered in an earlier round", smiles)
        return (
            f"VERDICT: FAIL — The molecule {smiles} was ALREADY DISCOVERED in "
            "a previous round of this session and is saved in the results file. "
            "ACTION REQUIRED: Generate a substantially DIFFERENT structure."
        )
    # ---- Step 2: RDKit physicochemical properties & drug-likeness ---------
    props: dict[str, Any] = compute_properties(mol)
    logger.info(
        "[Props] MW=%s LogP=%s HBD=%s HBA=%s TPSA=%s RotB=%s QED=%s "
        "| Lipinski violations=%d %s | Veber=%s",
        props["mw"], props["logp"], props["hbd"], props["hba"],
        props["tpsa"], props["rot_bonds"], props["qed"],
        props["lipinski_violations"],
        props["lipinski_failed"] or "(none)",
        "OK" if props["veber_ok"] else "FAIL",
    )
    # Drug-likeness gate: reject grossly non-drug-like molecules BEFORE the
    # slow PubChem lookups (saves API budget – an undruggable "novel"
    # molecule is worthless anyway).
    if props["lipinski_violations"] > MAX_LIPINSKI_VIOLATIONS:
        logger.info("[Props] ✘ Fails Lipinski Ro5 (%d violations) – skipping novelty",
                    props["lipinski_violations"])
        return (
            f"Syntax check: OK. Canonical SMILES: {smiles}\n"
            f"PROPERTIES: MW={props['mw']}, LogP={props['logp']}, "
            f"HBD={props['hbd']}, HBA={props['hba']}, TPSA={props['tpsa']}, "
            f"RotBonds={props['rot_bonds']}, QED={props['qed']}\n"
            f"VERDICT: FAIL — Not drug-like: fails Lipinski's Rule of Five "
            f"on {props['lipinski_violations']} criteria "
            f"({', '.join(props['lipinski_failed'])}). "
            f"Novelty was NOT checked.\n"
            "ACTION REQUIRED: Redesign so that at most one Lipinski criterion "
            "is broken (MW<=500, LogP<=5, H-bond donors<=5, acceptors<=10)."
        )
    report: dict[str, Any] = _novelty_report(smiles)
    report.update(props)  # fold descriptors into the report for CSV + output
    # ---- Determine verdict first -----------------------------------------
    # A None field means that check failed (network/throttling) – novelty
    # cannot be confirmed, so the molecule must never get a PASS by default.
    core_checks_failed = (
        report["compound_known"] is None or report["close_analogs_90"] is None
    )
    if report["compound_known"]:
        verdict = "FAIL"
        reason = "Exact compound already exists in PubChem."
    elif core_checks_failed:
        verdict = "ERROR"
        reason = (
            "PubChem could not be reached (network error or rate limiting). "
            "Novelty is UNKNOWN."
        )
    elif report["close_analogs_90"] > 50:
        verdict = "FAIL"
        reason = f"Too many close analogs ({report['close_analogs_90']} at >=90% similarity)."
    elif report["scaffold_known"] is False:
        verdict = "PASS"
        reason = (
            "Molecule is NOVEL and even its Murcko scaffold is unknown "
            "to PubChem (high novelty)."
        )
    elif report["broad_analogs_70"] == 0:
        verdict = "PASS"
        reason = (
            "Molecule is NOVEL with no broad analogs at >=70% similarity "
            "(high novelty)."
        )
    else:
        verdict = "PASS"
        reason = "Molecule is NOVEL — not in PubChem, acceptable analog profile."
    def fmt(value: Any, yes: str = "YES", no: str = "NO") -> str:
        if value is None:
            # None means either "skipped" (exact match already decided the
            # verdict) or "check failed" (network error)
            return "- (skipped)" if report["compound_known"] else "? (check failed)"
        if isinstance(value, bool):
            return yes if value else no
        return str(value)
    # ---- Visual log block (appears in real-time console) ------------------
    logger.info("")
    logger.info("╔══════════════════════════  NOVELTY REPORT  ══════════════════════════╗")
    logger.info("║  SMILES        : %s", smiles)
    if report["scaffold_smiles"]:
        logger.info("║  Scaffold      : %s", report["scaffold_smiles"])
    logger.info("║─────────────────────────────────────────────────────────────────────║")
    logger.info("║  Exact match   : %s", fmt(report["compound_known"], "YES ✘ (compound exists)", "NO  ✔ (not in database)"))
    logger.info("║  Analogs ≥90%%  : %s compounds", fmt(report["close_analogs_90"]))
    logger.info("║  Analogs ≥70%%  : %s compounds", fmt(report["broad_analogs_70"]))
    logger.info("║  Scaffold known: %s", fmt(report["scaffold_known"]))
    logger.info("║─────────────────────────────────────────────────────────────────────║")
    logger.info("║  MW / LogP     : %s / %s", report["mw"], report["logp"])
    logger.info("║  HBD / HBA     : %s / %s", report["hbd"], report["hba"])
    logger.info("║  TPSA / RotB   : %s / %s", report["tpsa"], report["rot_bonds"])
    logger.info("║  QED           : %s   (Lipinski viol.: %d, Veber: %s)",
                report["qed"], report["lipinski_violations"],
                "OK" if report["veber_ok"] else "FAIL")
    logger.info("║─────────────────────────────────────────────────────────────────────║")
    logger.info("║  VERDICT       : %s — %s", verdict, reason)
    logger.info("╚═════════════════════════════════════════════════════════════════════╝")
    logger.info("")
    # ---- Save only PASS results to CSV ------------------------------------
    if verdict == "PASS":
        save_to_csv(smiles, report, verdict)
    # ---- Build text result for the agent ---------------------------------
    lines: list[str] = [
        f"Syntax check: OK. Canonical SMILES: {smiles}",
        f"═══ NOVELTY REPORT for: {smiles} ═══",
        f"  Exact match in PubChem : {fmt(report['compound_known'], 'YES (KNOWN)', 'NO (not found)')}",
        f"  Close analogs (>=90%)  : {fmt(report['close_analogs_90'])} compounds",
        f"  Broad analogs (>=70%)  : {fmt(report['broad_analogs_70'])} compounds",
        f"  Scaffold known         : {fmt(report['scaffold_known'])}",
    ]
    if report["scaffold_smiles"]:
        lines.append(f"  Murcko scaffold        : {report['scaffold_smiles']}")
    lines.append(
        f"  Properties (RDKit)     : MW={report['mw']}, LogP={report['logp']}, "
        f"HBD={report['hbd']}, HBA={report['hba']}, TPSA={report['tpsa']}, "
        f"RotBonds={report['rot_bonds']}, QED={report['qed']}"
    )
    lines.append(
        f"  Drug-likeness          : Lipinski violations={report['lipinski_violations']}, "
        f"Veber={'OK' if report['veber_ok'] else 'FAIL'}"
    )
    lines.append("")
    if verdict == "ERROR":
        lines.append(f"VERDICT: ERROR — {reason}")
        lines.append(
            "ACTION REQUIRED: Do NOT treat this molecule as novel and do NOT "
            "give a final answer. Wait a moment, then call "
            "check_molecule again with the SAME SMILES."
        )
    elif verdict == "FAIL":
        lines.append(f"VERDICT: FAIL — {reason}")
        lines.append(
            "ACTION REQUIRED: Generate a substantially DIFFERENT structure. "
            "Apply scaffold hopping, fused rings, or unusual substituents."
        )
    else:
        lines.append(f"VERDICT: PASS — {reason}")
        lines.append(
            "The molecule has been saved. You can present this as the final answer."
        )
    return "\n".join(lines)
# ===========================================================================
#  LLM initialisation
# ===========================================================================
def build_llm() -> ChatOllama:
    """Instantiate the local Ollama-backed chat model.
    Uses qwen3.5:4b – a compact model that supports tool/function calling.
    Temperature is set to 0.7 to allow creative molecular generation while
    keeping outputs reasonably deterministic.
    """
    llm = ChatOllama(
        model="qwen3.5:4b",
        temperature=0.7,
    )
    return llm
# ===========================================================================
#  Agent setup (system prompt + LangGraph ReAct agent)
# ===========================================================================
# System prompt: tells the LLM *who it is* and *how to behave*
SYSTEM_PROMPT: str = """\
You are an expert medicinal chemist AI specializing in de-novo drug design.
Your task is to create a NOVEL structural modification of a given base molecule.
You will receive the base molecule as a SMILES string.
CRITICAL CONTEXT: PubChem contains over 115 million compounds. Simple, single-
substitution derivatives of well-known drugs (like adding one fluorine, swapping
-CH3 for -Cl, or replacing -COOH with -CONH2) almost certainly ALREADY EXIST.
To create something truly novel, you MUST apply MULTIPLE simultaneous changes.

=== MANDATORY WORKFLOW (follow EVERY time, NO exceptions) ===

Step 1: GENERATE a heavily modified SMILES.
   Combine AT LEAST 3 of these strategies simultaneously:
   - bioisosteric replacement (e.g. -COOH to -SO2NH2, -COOH to tetrazole,
     phenyl to thienyl or pyridyl)
   - ring expansion / contraction (e.g. 6-membered to 7-membered ring)
   - scaffold hopping (replace the phenyl core with a bicyclic or
     heterocyclic system like indole, benzimidazole, or quinoline)
   - addition of unusual substituents (e.g. trifluoromethyl, cyclopropyl,
     morpholine, piperazine, oxetane)
   - insert a linker (e.g. -CH2NH-, -OCH2-, -NHCO-) between fragments
   - add a fused ring system or spiro junction
   DO NOT make simple changes like:
   - Adding a single halogen to the existing ring
   - Just replacing COOH with another simple acid
   - Only adding one methyl or ethyl group
   KEEP THE MOLECULE DRUG-LIKE while doing this. Aim to respect Lipinski's
   Rule of Five: molecular weight <= 500, LogP <= 5, H-bond donors <= 5,
   H-bond acceptors <= 10. Avoid bloating the structure into something huge
   or extremely greasy.

Step 2: Call the `check_molecule` tool with your SMILES. It validates the
   syntax, computes drug-likeness properties (Lipinski Ro5, Veber, QED),
   AND checks novelty against PubChem — all in ONE step.
   - If it returns a syntax ERROR, fix the SMILES and call it again.
   - If VERDICT: FAIL because the molecule is NOT drug-like, redesign it to
     respect Lipinski's Rule of Five, then call check_molecule again.
   - If VERDICT: FAIL (already known / too many analogs), go back to Step 1
     with a MORE RADICAL modification.
   - If VERDICT: ERROR (PubChem unreachable), call check_molecule again
     with the SAME SMILES. Never treat an ERROR as novelty.
   - If VERDICT: PASS, proceed to Step 3.

Step 3: Return the final answer ONLY after receiving VERDICT: PASS.
   Include in your answer:
   - The canonical SMILES of the novel molecule
   - The full novelty report
   - A brief description of ALL structural changes made

=== ABSOLUTE RULES ===
- EVERY molecule you design MUST be evaluated with check_molecule.
- NEVER return a final answer without VERDICT: PASS from check_molecule.
- NEVER fabricate tool results. Always call the real tool.
- If a previous attempt got VERDICT: FAIL, make substantially different changes.
"""
def build_agent() -> Any:
    """Construct the ReAct agent with its tools and system prompt.
    Uses LangGraph's `create_react_agent` – the modern replacement for
    the legacy AgentExecutor pattern.  The system prompt is passed via
    the `prompt` parameter, and iteration limits are enforced at invoke
    time through `recursion_limit` in the config.
    Returns:
        A compiled LangGraph agent (CompiledStateGraph) ready to invoke.
    """
    # A single combined tool – two-step workflows are unreliable with
    # small models, which tend to loop on the first tool and never call
    # the second one
    tools: list = [check_molecule]
    # Initialise the LLM
    llm: ChatOllama = build_llm()
    # Create the ReAct agent with system prompt
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )
    return agent
# ===========================================================================
#  Main execution
# ===========================================================================
if __name__ == "__main__":
    # Base molecule: Ibuprofen
    IBUPROFEN_SMILES: str = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    task: str = (
        f"The base molecule is Ibuprofen with SMILES: {IBUPROFEN_SMILES}\n"
        "Please generate a NOVEL structural modification of this molecule. "
        "WARNING: Simple derivatives (adding one F, one Cl, swapping COOH for CONH2, "
        "etc.) already exist in PubChem's 115M compound database. You MUST combine "
        "at least 3 different structural changes simultaneously to create something "
        "truly new. The new molecule must pass both the RDKit syntax check and the "
        "PubChem novelty check. Describe your design rationale."
    )
    logger.info("=" * 70)
    logger.info("Zero-Cost Drug Discovery – Phase 1: ReAct Agent")
    logger.info("Base molecule: Ibuprofen (%s)", IBUPROFEN_SMILES)
    logger.info("CSV output  : %s", CSV_PATH)
    logger.info("Mode        : Continuous (Ctrl+C to stop)")
    logger.info("=" * 70)
    # Build the agent once (reused across rounds)
    agent = build_agent()
    # ---- Continuous discovery loop -----------------------------------------
    round_num: int = 0
    total_checks: int = 0
    error_streak: int = 0
    try:
        while True:
            round_num += 1
            logger.info("")
            logger.info("=" * 70)
            logger.info("  ROUND %d", round_num)
            logger.info("=" * 70)
            t0_round = time.time()
            try:
                # Each round is a fresh invocation with clean message history
                result: dict[str, Any] = agent.invoke(
                    {"messages": [("human", task)]},
                    config={"recursion_limit": 100},
                )
            except Exception as exc:
                # Exponential backoff: 5s, 10s, 20s, ... capped at 60s, so a
                # persistent failure (e.g. Ollama down) doesn't spin the loop.
                error_streak += 1
                delay = min(5 * 2 ** (error_streak - 1), 60)
                logger.error("Agent error in round %d: %s", round_num, exc)
                logger.info("Retrying in %d seconds...", delay)
                time.sleep(delay)
                continue
            error_streak = 0
            elapsed_round = time.time() - t0_round
            # ---- Print round result ----------------------------------------
            messages = result.get("messages", [])
            if messages:
                final_message = messages[-1]
                content = (
                    final_message.content
                    if hasattr(final_message, "content")
                    else str(final_message)
                )
                logger.info("[Round %d] Agent answer (%.0fs):\n%s",
                            round_num, elapsed_round, content)
            # ---- Count tool calls this round -------------------------------
            round_checks: int = 0
            for msg in messages:
                if (hasattr(msg, "type") and msg.type == "tool"
                        and getattr(msg, "name", "") == "check_molecule"):
                    round_checks += 1
            total_checks += round_checks
            logger.info(
                "[Round %d] Molecule checks: %d | Time: %.0fs",
                round_num, round_checks, elapsed_round,
            )
            # Show CSV status (set is kept in sync with the CSV file)
            logger.info("[CSV] Total novel molecules saved: %d", len(discovered_smiles))
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  STOPPED by user (Ctrl+C)")
        logger.info("=" * 70)
    # ---- Final summary -----------------------------------------------------
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"  Rounds completed : {round_num}")
    print(f"  Total molecule checks: {total_checks}")
    print(f"  CSV file         : {CSV_PATH}")
    print(f"  Novel molecules in CSV: {len(discovered_smiles)}")
    print("=" * 70)
    print()