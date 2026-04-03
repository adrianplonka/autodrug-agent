# Autonomous End-to-End Agentic Workflow with MTL for Drug Discovery

> **Zero-Cost In Silico Drug Design Pipeline**

## Abstract
The traditional drug discovery process is extremely costly, with the vast majority of novel compounds failing during clinical trials due to unforeseen toxicity or poor pharmacokinetic profiles (ADMET). While generative Large Language Models (LLMs) show promise in *de novo* drug design, they frequently suffer from hallucinations, generating structures that are syntactically invalid, chemically unstable, or impossible to synthesize.

This project introduces an autonomous pipeline designed to run locally. By combining the reasoning and planning capabilities of AI Agent with deterministic cheminformatics validation and a predictive Multi-Task Learning (MTL) network, this system minimizes failure risks at the earliest stages of molecular design.

## System Architecture

The pipeline operates autonomously, progressing through three main stages:

1. **Agent-Generator (LLM + Tool Calling):** A local LLM operates within a ReAct architecture. It iteratively designs novel molecules (SMILES) and autonomously uses **RDKit** to validate chemical syntax and the **PubChem API** to verify global structural novelty.
2. **MTL Filtration (ADMET):** Validated compounds are passed to a Deep Neural Network utilizing Multi-Task Learning. Trained on the **Therapeutics Data Commons (TDC)**, this network predicts multiple ADMET properties effectively filtering out high-risk candidates
3. **3D Validation (Molecular Docking):** The surviving safe lead compounds are subjected to molecular docking using **AutoDock Vina**, providing *in silico* proof of their binding affinity to the target protein.

## Workflow Overview

```
Target Protein & User Prompt
 │
 ├──► LLM Agent (Generator) ──┬──► RDKit (Syntax Check)
 │                            └──► PubChem (Novelty Check)
 │
 └──► Valid & Novel SMILES
       │
       └──► MTL Network (ADMET) ──┬──► Absorption & Distribution
                                  ├──► Metabolism & Excretion
                                  ├──► Toxicity
                                  │
                                  └──► Safe Lead Compounds
                                        │
                                        │
                                        └──► AutoDock Vina (Molecular Docking)
                                              │
                                              └──────────── ALL ──► Final Leads
```
## Tech Stack
This project is built on following technologies:
 - **Generative AI & Agents:** LangChain, Ollama (Local LLMs, e.g., Qwen)
 - **Cheminformatics:** RDKit, PubChem API
 - **Machine Learning:** PyTorch (MTL Network architecture), TDC (Dataset)
 - **Molecular Docking:** AutoDock Vina
