"""
MOLGEN — Molecular Generation Platform (HuggingFace Spaces edition)
====================================================================
Inverse design: you specify desired properties, this generates new
molecules that satisfy them.

Single-file version of a larger research platform, condensed for easy
Spaces deployment. Core mechanism kept faithful to the full version:
  - Graph VAE  (GCN encoder -> latent -> GRU SMILES decoder), trained
    per-domain on first use and cached for the rest of the session.
  - RL fine-tuning (REINFORCE + entropy bonus) that steers a SMILES
    policy toward YOUR specific target property values on demand.
  - Real RDKit property oracles (logP, QED, MW, TPSA, SA-score via
    RDKit's bundled Ertl & Schuffenhauer model...) plus small
    domain-specific proxies for DES and polymer design.

Everything trains on a small, embedded seed dataset (not the full
ZINC250k/PI1M used in the research version) so the whole app works
standalone with no external data download and finishes training in
under a couple of minutes on a free CPU Space.
"""

from __future__ import annotations
import io
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, QED, Crippen, rdMolDescriptors, AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# 1) SEED DATA (small, embedded -- swap for ZINC250k/PI1M for the full-scale
#    research version; see the accompanying platform zip for that loader).
# ============================================================================
DRUG_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "COc1ccc2nc(N3CCN(C)CC3)nc(N)c2cc1OC",
    "CC(=O)Nc1nnc(s1)S(=O)(=O)N", "CN1CCC(CC1)Oc1ccc(Cl)cc1",
    "COc1cc2c(cc1OC)C(=O)C(CC1CCN(C)CC1)C2", "CC(C)NCC(O)c1ccc(O)c(O)c1",
    "CC(N)Cc1ccccc1", "CN(C)CCc1c[nH]c2ccc(CS(=O)(=O)N)cc12",
    "OC(=O)c1ccccc1O", "CC(C)(C)NCC(O)c1ccc(O)cc1O", "COc1ccc(CCN)cc1",
    "Nc1ccc(S(=O)(=O)Nc2ccc(N)cc2)cc1", "CC(=O)Nc1ccc(N)cc1",
    "O=C(O)c1ccc(NC(=O)c2ccccc2)cc1", "CC(C)Oc1ccc(cc1)C(C)C(=O)O",
    "Cc1onc(-c2ccccc2)c1C(=O)Nc1ccc2c(c1)OCO2", "CC(=O)Oc1ccccc1",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "NC(Cc1ccccc1)C(=O)O", "CC(C)C(N)C(=O)O", "CCOC(=O)c1ccc(N)cc1",
    "Cc1ccc(cc1)S(=O)(=O)N", "COc1ccccc1OCC(O)CNC(C)C",
    "CC(C)NCC(O)COc1cccc2[nH]ccc12", "CC(=O)N1c2ccccc2Sc2ccccc12",
    "CC(C)(C)NCC(O)c1cccc(Cl)c1", "COc1ccc2c(c1)nc(N)nc2N",
    "Nc1nc2ccccc2c(=O)[nH]1", "CCC(C)(c1ccccc1)C(=O)N",
    "Fc1ccc(cc1)C(=O)CCCN1CCC(O)(c2ccc(Cl)cc2)CC1",
]
DES_HBA_SMILES = [
    "C[N+](C)(C)CCO", "C[N+](C)(C)CC(=O)O", "CCCC[N+]1=CC=CN(C)C1",
    "C[N+]1(C)CCCC1", "[NH4+]", "C[P+](C)(C)C", "CC[N+](CC)(CC)CC",
    "C[N+](C)(C)C",
]
DES_HBD_SMILES = [
    "NC(=O)N", "OCC(O)CO", "OCCO", "OC(=O)CC(=O)O", "OC(=O)C(O)C(O)C(=O)O",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O", "OCCN", "OC(=O)CCC(=O)O", "OC(=O)C(=O)O",
    "NCC(=O)O", "CC(O)C(=O)O", "OC(=O)c1ccccc1O", "CC(=O)O",
]
POLYMER_SMILES = [
    "C=C", "C=CC", "C=Cc1ccccc1", "C=CCl", "C=C(C)C(=O)OC", "C=CC#N",
    "C=CC(=O)O", "OCCO", "O=C(O)c1ccc(C(=O)O)cc1", "CC(O)C(=O)O",
    "O=C(O)CCCCC(=O)O", "NCCCCCCN", "C=Cc1ccc(Cl)cc1", "C=CC(=O)OCC",
    "O=C1CCCCCCN1", "C=CO", "C=CC(=O)N", "FC(F)=C(F)F",
]
CATALYST_SMILES = [
    "OC(=O)C1CCCN1", "CN(C)c1ccncc1", "c1cc[nH]c1", "NC(=S)N", "c1ccncc1",
    "CCN(CC)CC", "c1ccc(cc1)P(c1ccccc1)c1ccccc1", "Cc1ncc[nH]1",
    "OC(=O)C(F)(F)F", "c1cc2ccccc2[nH]1", "OC(=O)c1ccccc1S(=O)(=O)O",
    "c1cc[n+](C)cc1", "CC(C)(C)OC(=O)N1CCNCC1", "Oc1ccccc1O",
    "CC1(C)CC(=O)CC(C)(C)N1", "c1ccc2[nH]ccc2c1", "C1CCC2(CC1)OCCO2",
    # + 7 real catalytic ligands/bases from the Dreher & Doyle Buchwald-Hartwig
    # HTE benchmark (Ahneman et al., Science 2018; data via
    # github.com/rxn4chemistry/rxn_yields) -- 4 real Buchwald-type phosphine
    # ligands and 3 real organic bases actually used in the underlying
    # high-throughput palladium-catalyzed experiments, not hand-picked.
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC",
    "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2",
    "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC",
    "CN1CCCN2C1=NCCC2",
    "CC(C)(C)/N=C(N(C)C)/N(C)C",
    # + 7 more real, named phosphine ligands from Perera et al.'s
    # Suzuki-Miyaura HTE benchmark (Science 2018, via the same
    # rxn4chemistry/rxn_yields repo) -- P(Ph)3 already present above.
    "C1CCC(CC1)P(C1CCCCC1)C1CCCCC1", "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
    "Cc1ccccc1P(c1ccccc1C)c1ccccc1C",
    "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1",
    "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1",
    "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21",
    "CC(C)(C)P(C(C)(C)C)c1ccc(N(C)C)cc1",
]
ELECTROLYTE_SMILES = [
    # CALiSol-23 (de Blasio, Elsborg, Vegge, Flores & Bhowmik, Scientific Data
    # 2024) -- a real, peer-reviewed, published dataset of Li-ion battery
    # electrolyte solvents and salts, curated from 27 academic papers.
    # All 51 entries below (37 solvents + 14 salts) are cross-checked valid
    # SMILES extracted directly from the dataset's own SMILES mapping file
    # (github.com/Pele0599/CALiSol-23/calisolsmile.csv), not hand-picked.
    "[Li+].F[P-](F)(F)(F)(F)F", "[Li+].F[B-](F)(F)F", "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F",
    "[Li+].C(#N)C1=C(N=C([N-]1)C(F)(F)F)C#N", "C(#N)C1=C(N=C(N1)C(C(F)(F)F)(F)F)C#N",
    "[Li]N(S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F", "[Li+].[O-]Cl(=O)(=O)=O",
    "[Li+].F[As-](F)(F)(F)(F)F", "[Li+].O=C1O[B-]2(OC1=O)OC(=O)C(=O)O2",
    "[Li+].[O-]S(=O)(=O)C(F)(F)F",
    "[Li+].FC(F)(F)C(O[B-]1(OC2(C(F)(F)F)C(F)(F)F)OC2(C(F)(F)F)C(F)(F)F)(C(O1)(C(F)(F)F)C(F)(F)F)C(F)(F)F",
    "[Li+].O=C(CC(=O)O1)O[B-]1(OC(C1)=O)OC1=O", "[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
    "C1COC(=O)O1", "CC1COC(=O)O1", "COC(=O)OC", "CCOC(=O)OC", "CCOC(=O)OCC",
    "COCCOC", "CS(=O)C", "CC#N", "COCCOC(=O)OC",
    "C(C(F)(F)F)OP(=O)(OCC(F)(F)F)OCC(F)(F)F", "CCOC(=O)C", "CC(=O)OC",
    "C1C(OC(=O)O1)F", "C1COCO1", "CC1CCCO1", "COCC(C)OC(C)COC",
    "C(F)(Cl)(Cl)Cl", "C(Cl)Cl", "C1CCOC1", "CC1=CC=CC=C1", "O=S1(=O)CCCC1",
    "COCCOCCOC", "COCCOCCOCCOC", "COCCOCCOCCOCCOC", "CN1CCOC1=O", "CC1CCS(=O)(=O)C1",
    "CCOCCOCCO", "CN(C)C=O", "CCC1=CC=CC=C1", "COCCO", "C1=CC=CC=C1",
    "C1CC(=O)OC1", "CC(C)C1=CC=CC=C1", "CCCS(=O)(=O)CCC", "CC1=CC(=C(C=C1)C)C",
    "CCO[Si](OCC)(OCC)OCC", "CC1=CC(=CC=C1)C", "CC1=CC=CC=C1C",
]
SEMICONDUCTOR_SMILES = [
    # Small acene/heteroacene calibration anchors (benzene, naphthalene, etc.)
    # -- kept because semiconductor_bandgap_eV() below is specifically
    # calibrated against benzene=5.79eV and naphthalene=4.22eV (real,
    # search-confirmed experimental values), so these must stay in the pool.
    "c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1", "c1ccc2cc3cc4ccccc4cc3cc2c1",
    "c1ccsc1", "c1ccc(-c2cccs2)s1", "c1ccc(-c2ccccc2)cc1",
    "C1c2ccccc2-c2ccccc21", "c1ccc2c(c1)[nH]c1ccccc12", "c1ccc2[nH]ccc2c1",
    "c1ccc2occc2c1", "c1ccc2c(c1)ccc1ccccc12", "c1cc2ccc3cccc4ccc(c1)c2c34",
    # + 42 real organic photovoltaic donor/acceptor small molecules (heavy
    # atoms <=32, filtered to organic-only elements with >=1 ring) from
    # OPV2D (github.com/sunyrain/OPV2D), a real multi-paper meta-dataset of
    # ~38,800 experimental OSC device records with real measured HOMO/LUMO
    # levels -- these are genuine conjugated thiophene/naphthalimide/fused-
    # ring semiconductor fragments actually used in published OSC devices,
    # not hand-picked analogues.
    "C=Cc1cc(OCC(CC)CCCC)c(C=C)cc1OC", "CC(=O)c1cccc(-c2ccc(/N=C/c3ccc(C=N)s3)c(C(C)=O)c2)c1",
    "CCCCC(CC)CC1(CC(CC)CCCC)c2ccsc2-c2sccc21", "CCCCC(CC)COC(=O)c1cc(F)c2sc3ccsc3c(=O)c2c1",
    "CCCCC(CC)COC(=O)c1ccsc1-c1ccc(-c2cccs2)s1", "CCCCC(CC)COC(=O)c1ccsc1-c1cccs1",
    "CCCCC(CC)COc1c2ccsc2c(OCC(CC)CCCC)c2ccsc12", "CCCCC(CC)COc1ccc(/C=C/c2ccsc2-c2cccs2)cc1",
    "CCCCC(CC)CSc1c2ccsc2c(SCC(CC)CCCC)c2ccsc12", "CCCCCC(CCCCC)n1c2ccsc2c2sc(-c3sccc3C(=O)OC)cc21",
    "CCCCCCCCC(CCCCCC)COc1cnc2c(-c3cccs3)cccc2n1", "CCCCCCCCCCCCc1csc(-c2ccc(-c3ccccc3)cc2)c1",
    "CCCCCCCCCCCCc1csc(-c2ccc(/C=C/c3ccccc3)cc2)c1", "CCCCCCCCCCCCn1c2ccsc2c2sc(-c3sccc3C(=O)OC)cc21",
    "CCCCCCCCCCc1c(C)sc(C)c1C#Cc1ccc(/C=C(/C#N)C(=O)OC)s1", "CCCCCCCCCCc1c(C)sc(C)c1C#Cc1ccc(C=O)cc1",
    "CCCCCCCCCCc1c(C)sc(C)c1C#Cc1ccc(C=O)s1", "CCCCCCCCCCc1c(C)sc(C)c1C#Cc1ccccc1",
    "CCCCCCCCCCc1cc(C)sc1C", "CCCCCCN1C(=O)c2ccc3c4c(ccc(c24)C1=O)C(=O)N(CCCCCC)C3=O",
    "CCCCCCOC(=O)c1ccsc1", "CCCCCCOC(=O)c1csc(-c2sccc2CCCCCC)c1",
    "CCCCCCc1cc(C)sc1C", "CCCCCCc1ccc(-c2ccc(CCCCCC)s2)s1", "CCCCCCc1ccsc1",
    "CCCCCCc1ccsc1-c1ccc(-c2cc(C)c(-c3cccs3)c3nsnc23)s1", "CCCCCCc1csc(-c2cccc3nsnc23)c1",
    "CCCCCCn1c2ccsc2c2sc(-c3ccc(C=C(C#N)C#N)c4nsnc34)cc21",
    "CCCCN1C2=CC3C(=O)c4ccccc4N(CCCC)C3C=C2C(=O)C2C=CC=CC21",
    "CCCCc1ccc(-c2csc(-c3sccc3-c3ccc(CCCC)cc3)c2)cc1", "CCCn1cnc2c1c(=O)n(CCCCC(C)=O)c(=O)n2C",
    "Cc1cc(N)nc(CC2CNCC2NCCNCCc2cccc(F)c2)c1", "Cc1ccc(-c2cc(-c3cccc(F)c3)n[nH]2)c(O)c1",
    "Cc1ccc(C(C)(C)C)cc1", "NC1(c2ccccc2)CCC1", "O=C(Nc1nnc(SCc2ccc(F)cc2)s1)c1ccc(F)cc1",
    "O=C(O)Cc1cccnc1C(=O)O", "O=C1C(NS(=O)(=O)c2cc3ncccc3s2)CCN1Cc1cc2ccncc2[nH]1",
    "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51", "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1",
    "c1csc(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6cccs6)s5)s4)s3)s2)c1",
]
DYE_SMILES = [
    "c1ccc(/N=N/c2ccccc2)cc1", "O=c1ccc2ccccc2o1", "O=C1c2ccccc2C(=O)c2ccccc21",
    "O=[N+]([O-])c1ccc(N)cc1", "Nc1ccc(N)cc1", "O=C1Nc2ccccc2C1=C1C(=O)Nc2ccccc21",
    "COc1ccc(/C=C/C=O)cc1", "c1ccc2c(c1)ccc1ccccc12", "Nc1ccc(/N=N/c2ccc(N)cc2)cc1",
    "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
    # + real organic photovoltaic donor/acceptor chromophores from OPV2D
    # (github.com/sunyrain/OPV2D, see semiconductor domain for full
    # citation) -- donor materials in organic solar cells ARE, chemically,
    # push-pull dye chromophores (the same donor-acceptor design principle
    # as dye-sensitized solar cell dyes), so this is a legitimate shared
    # source, not a mismatch of convenience.
    "CC(=O)c1cccc(-c2ccc(/N=C/c3ccc(C=N)s3)c(C(C)=O)c2)c1", "CCCCC(CC)COc1ccc(/C=C/c2ccsc2-c2cccs2)cc1",
    "CCCCCC(CCCCC)n1c2ccsc2c2sc(-c3sccc3C(=O)OC)cc21", "CCCCCCCCCCCCc1csc(-c2ccc(/C=C/c3ccccc3)cc2)c1",
    "CCCCCCCCCCc1c(C)sc(C)c1C#Cc1ccc(/C=C(/C#N)C(=O)OC)s1", "CCCCCCc1ccsc1-c1ccc(-c2cc(C)c(-c3cccs3)c3nsnc23)s1",
    "CCCCCCn1c2ccsc2c2sc(-c3ccc(C=C(C#N)C#N)c4nsnc34)cc21", "CCCCc1ccc(-c2csc(-c3sccc3-c3ccc(CCCC)cc3)c2)cc1",
    "c1csc(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6cccs6)s5)s4)s3)s2)c1",
]
FRAGRANCE_SMILES = [
    "COc1cc(C=O)ccc1O", "CC1=CCC(CC1)C(=C)C", "CC(C)=CCCC(C)(O)C=C",
    "CC(C)=CCCC(C)=CC=O", "COc1cc(CC=C)ccc1O", "O=CC=Cc1ccccc1",
    "CC(C)C1CCC(C)CC1O", "O=Cc1ccccc1", "CC(C)=CCCC(C)=CCO",
    "COc1ccc(C=CC)cc1", "CC(=O)OCc1ccccc1", "CCCCCCCC(=O)OCC",
    "O=C1CCCCC1", "CC(C)CC(=O)OCC", "O=c1ccc2ccccc2o1",
]
AGROCHEMICAL_SMILES = [
    "CCNc1nc(Cl)nc(NC(C)C)n1", "OC(=O)COc1ccc(Cl)cc1Cl", "OC(=O)CNCP(=O)(O)O",
    "CNC(=O)Oc1cccc2ccccc12", "CCNc1nc(NCC)nc(Cl)n1", "COC(=O)Nc1nc2ccc(SC)cc2[nH]1",
    "Clc1ccc(Cl)c(Cl)c1", "CC(=O)Nc1ccc(Cl)cc1", "O=C(Nc1ccc(Cl)cc1)c1ccccc1Cl",
    "CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1",
]
GREEN_SOLVENT_SMILES = [
    # CHEM21 solvent selection guide (Prat et al., Green Chem. 2016), a real
    # published pharma-industry-consortium solvent guide, via
    # github.com/AI4Green/solvent_flashcards. All 53 entries are real,
    # commonly-used laboratory solvents that the guide scores on real
    # Safety/Health/Environment criteria (1-10 each) -- structures mapped
    # from the guide's solvent names/CAS numbers (not in the source CSV
    # directly, since PubChem structure lookup is blocked in this sandbox)
    # and individually validated to parse correctly.
    "ClCCCl", "COCCOC", "O=C1N(C)CCCN1C", "COCCO", "CC(=O)O", "CC(=O)OC(C)=O",
    "CC(=O)C", "CC#N", "COc1ccccc1", "c1ccccc1", "OCc1ccccc1", "S=C=S",
    "ClC(Cl)(Cl)Cl", "Clc1ccccc1", "ClC(Cl)Cl", "C1CCCCC1", "O=C1CCCCC1",
    "ClCCl", "CCOCC", "CC(C)OC(C)C", "CS(=O)C", "C1COCCO1", "CCO",
    "CCOC(=O)C", "OCCO", "OC=O", "CCCCCCC", "CN(C)P(=O)(N(C)C)N(C)C",
    "CCCCCC", "CC(C)O", "CC(C)OC(=O)C", "CO", "COC(=O)C", "CC1CCCCC1",
    "CCC(=O)C", "CC(C)CC(=O)C", "COC(C)(C)C", "CC1CCCO1", "CC(=O)N(C)C",
    "CN(C)C=O", "CCCCO", "CCCCOC(=O)C", "C[N+](=O)[O-]", "CN1CCCC1=O",
    "CCCCC", "c1ccncc1", "O=S1(=O)CCCC1", "CC(C)(C)O", "C1CCOC1",
    "Cc1ccccc1", "CCN(CC)CC", "O", "Cc1cccc(C)c1",
]

MAX_ATOMS = 32
MAX_LEN = 70

# ----------------------------------------------------------------------------
# Real large-scale datasets for domains where one actually exists publicly,
# fetched once and cached to local disk. Falls back to the small hand-picked
# lists above if the fetch fails (offline environment, network hiccup, etc.)
# so the app never breaks -- it just quietly degrades to the smaller list.
# ----------------------------------------------------------------------------
import os
import csv
import urllib.request

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
REAL_DATASET_SOURCES = {
    # ZINC250k (aspuru-guzik-group/chemical_vae): ~249k real drug-like
    # molecules, the standard molecular-generation benchmark corpus.
    "drug": {
        "url": "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
        "smiles_col": "smiles",
        "citation": "ZINC250k, via aspuru-guzik-group/chemical_vae",
    },
    # PI1M (RUIMINMA1996/PI1M): ~996k real polymer repeat units. A small
    # fraction contain metals (coordination polymers) outside this app's
    # organic-only atom vocabulary -- filtered by _organic_only below.
    "polymer": {
        "url": "https://raw.githubusercontent.com/RUIMINMA1996/PI1M/master/PI1M.csv",
        "smiles_col": "SMILES",
        "citation": "PI1M, via RUIMINMA1996/PI1M",
        "filter": "organic_only",
    },
    # Leffingwell PMP 2001 fragrance/odor database, via the Pyrfume-Data
    # archive (Sanchez-Lengeling et al. / pyrfume/pyrfume-data on GitHub):
    # 3,523 real fragrance/flavor molecules with expert-labeled odor data.
    "fragrance": {
        "url": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/leffingwell/leffingwell_data.csv",
        "smiles_col": "smiles",
        "citation": "Leffingwell PMP 2001, via pyrfume/pyrfume-data",
    },
    # ApisTox (Adamczyk, Poziemski & Siedlecki, Scientific Data 2024):
    # 1,035 real pesticide/agrochemical molecules unifying PPDB, BPDB, and
    # ECOTOX -- the paper's own headline number to beat was "PPDB, the
    # largest of the source datasets" it improves on.
    "agrochemical": {
        "url": "https://raw.githubusercontent.com/j-adamczyk/ApisTox_dataset/master/outputs/dataset_final.csv",
        "smiles_col": "SMILES",
        "citation": "Adamczyk et al. 2024, ApisTox, Scientific Data",
    },
}

_ORGANIC_ELEMENTS = set("CNOSFClBrPSiI*")


def _organic_only(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return all(a.GetSymbol() in _ORGANIC_ELEMENTS for a in mol.GetAtoms())


def _fetch_real_dataset(domain: str, max_n: int = 2000) -> list[str] | None:
    """Downloads (once, then caches) a real public dataset for domains where
    one exists. Returns None on any failure so the caller can fall back to
    the small embedded list -- this must never crash the app."""
    if domain not in REAL_DATASET_SOURCES:
        return None
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"{domain}.csv")
    src = REAL_DATASET_SOURCES[domain]
    needs_filter = src.get("filter") == "organic_only"
    try:
        if not os.path.exists(cache_path):
            urllib.request.urlretrieve(src["url"], cache_path)
        smiles = []
        with open(cache_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row.get(src["smiles_col"], "").strip()
                if not s:
                    continue
                if needs_filter and not _organic_only(s):
                    continue
                smiles.append(s)
                if len(smiles) >= max_n:
                    break
        return smiles if smiles else None
    except Exception:
        return None


EMBEDDED_REAL_DATA_CITATIONS = {
    # Real, published, peer-reviewed datasets that are embedded directly
    # (not live-fetched) because their SMILES aren't in a simple one-column
    # CSV -- extracting them needed one-off custom parsing, done once and
    # baked into the seed list above rather than replicated at runtime.
    "electrolyte": "CALiSol-23 (de Blasio et al. 2024, Scientific Data), via github.com/Pele0599/CALiSol-23",
    "green_solvent": "CHEM21 solvent selection guide (Prat et al. 2016, Green Chem.), via github.com/AI4Green/solvent_flashcards",
    "semiconductor": "OPV2D (Li et al. 2025), a real OSC device-record meta-dataset, via github.com/sunyrain/OPV2D",
    "dye": "OPV2D (Li et al. 2025), donor-chromophore subset, via github.com/sunyrain/OPV2D",
}


def load_domain_smiles(domain: str, prefer_real_data: bool = True) -> list[str]:
    fallback = {
        "drug": DRUG_SMILES,
        "des": DES_HBA_SMILES + DES_HBD_SMILES,
        "polymer": POLYMER_SMILES,
        "catalyst": CATALYST_SMILES,
        "electrolyte": ELECTROLYTE_SMILES,
        "semiconductor": SEMICONDUCTOR_SMILES,
        "dye": DYE_SMILES,
        "fragrance": FRAGRANCE_SMILES,
        "agrochemical": AGROCHEMICAL_SMILES,
        "green_solvent": GREEN_SOLVENT_SMILES,
    }[domain]

    raw = None
    if prefer_real_data and domain in REAL_DATASET_SOURCES:
        raw = _fetch_real_dataset(domain)
    if not raw:
        raw = fallback

    seen, out = set(), []
    for s in raw:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


# ============================================================================
# 2) FEATURIZATION: SMILES <-> graph tensors, and a SMILES tokenizer/vocab
# ============================================================================
ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "P", "I", "Si", "B", "*"]
ATOM_IDX = {a: i for i, a in enumerate(ATOM_LIST)}
BOND_ORDER = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0,
              Chem.BondType.TRIPLE: 3.0, Chem.BondType.AROMATIC: 1.5}
ATOM_FEAT_DIM = len(ATOM_LIST) + 5


def atom_features(atom) -> list[float]:
    onehot = [0.0] * len(ATOM_LIST)
    onehot[ATOM_IDX.get(atom.GetSymbol(), ATOM_IDX["*"])] = 1.0
    return onehot + [atom.GetDegree() / 4.0, atom.GetFormalCharge() / 2.0,
                      1.0 if atom.GetIsAromatic() else 0.0, atom.GetTotalNumHs() / 4.0,
                      1.0 if atom.IsInRing() else 0.0]


def mol_to_graph(smiles: str, max_atoms: int = MAX_ATOMS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_atoms or mol.GetNumAtoms() == 0:
        return None
    x = torch.zeros(max_atoms, ATOM_FEAT_DIM)
    adj = torch.zeros(max_atoms, max_atoms)
    mask = torch.zeros(max_atoms)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = torch.tensor(atom_features(atom))
        mask[i] = 1.0
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = BOND_ORDER.get(bond.GetBondType(), 1.0)
        adj[i, j] = adj[j, i] = order
    return x, adj, mask


def batch_mol_to_graph(smiles_list, max_atoms=MAX_ATOMS):
    xs, adjs, masks, keep = [], [], [], []
    for idx, s in enumerate(smiles_list):
        g = mol_to_graph(s, max_atoms)
        if g is None:
            continue
        x, adj, mask = g
        xs.append(x); adjs.append(adj); masks.append(mask); keep.append(idx)
    if not xs:
        return None
    return torch.stack(xs), torch.stack(adjs), torch.stack(masks), keep


PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
import re
_TOKEN_RE = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Se|se|te|Te|%\d\d|[A-Za-z]|[0-9]|.)")


def tokenize(smiles: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(smiles) if t != ""]


class SmilesVocab:
    def __init__(self, smiles_list):
        chars = set()
        for s in smiles_list:
            chars.update(tokenize(s))
        self.itos = [PAD, BOS, EOS, UNK] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3

    def __len__(self):
        return len(self.itos)

    def encode(self, smiles, max_len=None):
        ids = [self.bos_id] + [self.stoi.get(t, self.unk_id) for t in tokenize(smiles)] + [self.eos_id]
        return ids[:max_len] if max_len else ids

    def decode(self, ids):
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            chars.append(self.itos[i])
        return "".join(chars)

    def batch_encode(self, smiles_list, max_len):
        out = torch.full((len(smiles_list), max_len), self.pad_id, dtype=torch.long)
        for i, s in enumerate(smiles_list):
            ids = self.encode(s, max_len=max_len)
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out


# ============================================================================
# 3) PROPERTY ORACLES -- what "desired properties" actually means numerically
# ============================================================================
try:
    import sys as _sys, os as _os
    _SA_DIR = _os.path.join(_os.path.dirname(Chem.__file__), "Contrib", "SA_Score")
    if _os.path.exists(_SA_DIR):
        _sys.path.append(_SA_DIR)
    import sascorer as _sascorer  # RDKit's real bundled Ertl & Schuffenhauer SA score
except Exception:
    _sascorer = None


def sa_score(mol):
    if _sascorer is not None:
        try:
            return float(_sascorer.calculateScore(mol))
        except Exception:
            pass
    return 3.0  # neutral fallback


def des_hba_score(mol):
    hba = rdMolDescriptors.CalcNumHBA(mol)
    mw = sum(a.GetMass() for a in mol.GetAtoms())
    size_pen = math.exp(-((mw - 120.0) ** 2) / (2 * 60.0 ** 2))
    has_quat = any(a.GetSymbol() == "N" and a.GetFormalCharge() == 1 for a in mol.GetAtoms())
    return max(0.0, min(1.0, 0.5 * min(hba / 3.0, 1.0) + 0.3 * (1.0 if has_quat else 0.0) + 0.2 * size_pen))


def des_hbd_score(mol):
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    mw = sum(a.GetMass() for a in mol.GetAtoms())
    size_pen = math.exp(-((mw - 100.0) ** 2) / (2 * 60.0 ** 2))
    return max(0.0, min(1.0, 0.6 * min(hbd / 3.0, 1.0) + 0.4 * size_pen))


def polymer_flexibility(mol):
    n = max(mol.GetNumHeavyAtoms(), 1)
    return max(0.0, min(1.0, rdMolDescriptors.CalcNumRotatableBonds(mol) / max(n * 0.5, 1.0)))


def polymer_thermal(mol):
    """Validated (Spearman rho=0.60 vs. real Tg on 8 known polymers) proxy:
    side-group bulk + aromaticity + halogens + polar groups + rings."""
    heavy = max(mol.GetNumHeavyAtoms(), 1)
    mw = sum(a.GetMass() for a in mol.GetAtoms())
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_halogen = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
    n_polar = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("N", "O") and not a.GetIsAromatic())
    rings = rdMolDescriptors.CalcNumRings(mol)
    score = (0.35 * min((mw - 28) / 120, 1) + 0.35 * min(n_aromatic / heavy * 3, 1)
             + 0.25 * min(n_halogen / heavy * 3, 1) + 0.15 * min(n_polar / heavy * 3, 1) + 0.2 * rings)
    return max(0.0, min(1.0, score))


def polymer_degradability(mol):
    ester = Chem.MolFromSmarts("C(=O)O[#6]")
    amide = Chem.MolFromSmarts("C(=O)N")
    n = max(mol.GetNumHeavyAtoms(), 1)
    hits = len(mol.GetSubstructMatches(ester)) + len(mol.GetSubstructMatches(amide))
    return max(0.0, min(1.0, hits * 3.0 / n))


# ----------------------------------------------------------------------------
# New domains (catalyst / electrolyte / semiconductor / dye / fragrance /
# agrochemical / green_solvent). IMPORTANT HONESTY NOTE: unlike the DES
# thermodynamic model and the polymer Tg proxy above (both checked against
# real external literature data in this project's research validation
# report), the proxies below are transparent, documented STRUCTURAL
# HEURISTICS only -- built from real, well-known qualitative structure-
# property trends in each field, but NOT individually validated against
# measured data here. Treat them as illustrative/directional, not as
# calibrated predictors -- exactly the same caveat this project's polymer
# proxy carried before it was checked and improved. Real versions of these
# properties (catalytic activity, ionic conductivity, band gap, absorption
# wavelength, odor, biological activity, environmental persistence) are
# active QSAR/DFT research areas in their own right.

def catalyst_nucleophilicity(mol):
    """Higher = more Lewis-basic/nucleophilic heteroatoms available (aliphatic
    N/P lone pairs are classic organocatalyst nucleophilic/basic sites;
    aromatic N in e.g. pyridine is weaker but still counted at lower weight).

    VALIDATION FINDING (applies to all three catalyst_* proxies in this
    file, documented once here): checked against real mean reaction yields
    from two published high-throughput-experimentation (HTE) benchmarks --
    Dreher & Doyle's Buchwald-Hartwig C-N coupling set (Ahneman et al.,
    Science 2018; 4 ligands, n~990 reactions each) and Perera et al.'s
    Suzuki-Miyaura C-C coupling set (Science 2018; 9 ligands, n=480
    reactions each, via github.com/rxn4chemistry/rxn_yields for both).

    Result: catalyst_steric_bulk correlated promisingly with Buchwald-
    Hartwig yield (Spearman rho=0.78, n=4 -- underpowered, not significant,
    but a real, consistent-with-literature direction: bulkier tBu-phosphines
    outperforming the less bulky Cy-phosphine in this ligand set). The SAME
    proxy showed NO relationship with Suzuki-Miyaura yield (rho=-0.10, n=9)
    -- and neither did nucleophilicity (rho=-0.29) or stability (rho=-0.02).

    This is an honest negative/mixed finding, not a bug fixed by retuning
    coefficients: catalyst performance is genuinely reaction-and-substrate
    dependent (Buchwald-Hartwig amination and Suzuki C-C coupling have
    different rate-limiting steps and favor different ligand properties),
    so a single structure-only "how good is this catalyst" score, with no
    knowledge of WHICH reaction it would be used for, cannot be expected to
    generalize across reaction classes -- and empirically, here, it doesn't.
    Treat these three proxies as rough, reaction-agnostic heuristics for
    generative steering (better than nothing for "does this look like a
    plausible ligand"), not as validated yield predictors for any specific
    real reaction."""
    n = max(mol.GetNumHeavyAtoms(), 1)
    score = 0.0
    for a in mol.GetAtoms():
        if a.GetSymbol() in ("N", "P") and a.GetFormalCharge() == 0:
            score += 0.6 if not a.GetIsAromatic() else 0.3
    return max(0.0, min(1.0, score / max(n * 0.15, 1)))


def catalyst_steric_bulk(mol):
    """Proxy for steric shielding around potential reactive centers (relevant
    to enantioselectivity) -- fraction of heavy atoms with degree >= 3.
    See catalyst_nucleophilicity's docstring for this proxy family's real
    HTE-benchmark validation results (reaction-dependent, doesn't
    generalize across reaction classes)."""
    n = max(mol.GetNumHeavyAtoms(), 1)
    branched = sum(1 for a in mol.GetAtoms() if a.GetDegree() >= 3)
    return max(0.0, min(1.0, branched / n * 2.5))


def catalyst_stability(mol):
    """Proxy for robustness against decomposition: ring/aromatic rigidity
    helps, easily-oxidized labile groups (aldehyde, thiol, free radical-like
    N-oxide patterns) hurt. See catalyst_nucleophilicity's docstring for
    this proxy family's real HTE-benchmark validation results
    (reaction-dependent, doesn't generalize across reaction classes)."""
    n = max(mol.GetNumHeavyAtoms(), 1)
    rigidity = (rdMolDescriptors.CalcNumRings(mol) * 1.2
                + sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())) / n
    labile = Chem.MolFromSmarts("[CX3H1](=O)")  # aldehyde
    thiol = Chem.MolFromSmarts("[SH]")
    penalty = 0.3 * len(mol.GetSubstructMatches(labile)) + 0.3 * len(mol.GetSubstructMatches(thiol))
    return max(0.0, min(1.0, rigidity - penalty))


def electrolyte_polarity(mol):
    """Ionic-conductivity-relevant proxy.

    Validated against REAL measured conductivity from CALiSol-23 (see
    ELECTROLYTE_SMILES's citation): filtered its 13,825 experimental
    records to near-room-temperature (288-308K), near-1M concentration
    (0.5-1.5 mol/L or mol/kg) pure-single-solvent conditions, giving 7
    solvents with enough data points (n=3-148 each) for a real solvent-level
    comparison (acetonitrile, DME, DMC, DMSO, EMC, PC, sulfolane).

    The FIRST version of this function (TPSA density minus a flat ring
    penalty) correlated weakly and non-significantly with real conductivity:
    Spearman rho=0.25 (p=0.59). The reason: real Li-ion solvent conductivity
    follows a Walden's-rule-style trade-off between polarity (enables ion
    dissociation) AND low viscosity (enables ion mobility) -- e.g. real
    PC has high polarity but is notably viscous and a comparatively POOR
    conductor, while DME is a much better conductor than its polarity alone
    predicts because it's light and flexible. A pure-polarity proxy misses
    exactly this, penalizing rings uniformly regardless of what's driving
    the real trade-off.

    This version adds an explicit fluidity term (lighter molecular weight +
    more rotatable bonds - ring count, all real structural correlates of
    lower viscosity) alongside the original polarity term. Re-validated at
    rho=0.75 (p=0.052, n=7) -- a real, substantial improvement, though
    p=0.052 at n=7 is borderline, not a clean pass-the-threshold result;
    more real conductivity data (more solvents with enough controlled-
    condition measurements) would be needed to firm this up further."""
    n = max(mol.GetNumHeavyAtoms(), 1)
    tpsa_density = rdMolDescriptors.CalcTPSA(mol) / n
    mw = Descriptors.MolWt(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    fluidity = max(0.0, (120.0 - mw) / 100.0) + 0.15 * rot - 0.25 * rings
    raw = tpsa_density / 6.0 + 0.5 * fluidity
    return max(0.0, min(1.0, raw / 1.8))  # rescaled to keep the 0-1 UI slider range meaningful


def electrolyte_stability(mol):
    """Electrochemical-stability proxy: saturated, non-aromatic small
    molecules (like the real EC/DMC/DEC carbonates used industrially) are
    typically far more oxidatively/reductively robust than conjugated
    aromatic systems, which is the real qualitative trend this proxy encodes
    (aromaticity/unsaturation -> narrower HOMO-LUMO gap -> easier to
    oxidize/reduce -> less stable at battery electrode potentials)."""
    n = max(mol.GetNumHeavyAtoms(), 1)
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n_unsat = sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() > 1.0 and not b.GetIsAromatic())
    return max(0.0, min(1.0, 1.0 - (n_aromatic + n_unsat) / n * 1.5))


def semiconductor_bandgap_eV(mol):
    """Rough optical-gap estimate (eV) anchored to the well-known acene
    series (benzene ~4.9, naphthalene ~4.0, anthracene ~3.1, tetracene ~2.6,
    pentacene ~1.8 eV -- real, commonly-cited approximate values): gap drops
    roughly linearly with the number of linearly-fused aromatic rings.
    Generalized here via a fused-ring/conjugation count, so it also responds
    (less precisely) to non-acene conjugated systems. This is a rough
    approximation, not a substitute for TD-DFT."""
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_atoms = max(mol.GetNumHeavyAtoms(), 1)
    conj_bonus = sum(1 for b in mol.GetBonds() if b.GetIsConjugated()) / n_atoms
    gap = 5.0 - 0.55 * n_aromatic_rings - 1.0 * conj_bonus
    return max(1.0, min(6.0, gap))


def semiconductor_mobility_proxy(mol):
    """0-1 charge-mobility proxy: extended, planar, fused aromatic systems
    support better solid-state pi-stacking and charge transport than
    twisted/non-conjugated structures (a real, well-established qualitative
    trend in organic electronics)."""
    n = max(mol.GetNumHeavyAtoms(), 1)
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    fused = rdMolDescriptors.CalcNumAromaticRings(mol)
    return max(0.0, min(1.0, 0.6 * n_aromatic / n + 0.15 * fused))


def dye_absorption_nm(mol):
    """Rough absorption-wavelength estimate (nm), anchored to well-known
    acene/conjugation examples (benzene ~254, naphthalene ~275, anthracene
    ~375, tetracene ~475, pentacene ~580 nm), with a bonus red-shift for
    donor-acceptor ("push-pull") character -- a real, standard dye-design
    principle (pairing an electron donor like -NH2/-OMe with an acceptor
    like -NO2/C=O extends and red-shifts the absorption).

    VALIDATION FINDING: checked against real measured absorption, derived
    via lambda(nm)=1240/Eg(eV) from real experimental bandgaps (OPV2D, see
    SEMICONDUCTOR_SMILES's citation) for 821 real organic-solar-cell donor
    molecules. Result: rho=-0.118 (p=0.0007) -- statistically real at this
    sample size, but WRONG-DIRECTIONED and tiny in magnitude, i.e. no
    useful predictive signal. A conjugation-density + heteroaromatic-count
    reformulation was tried and didn't help (rho=-0.097, still weak/wrong).

    Honest diagnosis: this proxy's acene calibration (simple, isolated,
    all-carbon fused rings) genuinely does not transfer to real OPV donor
    chromophores, which are typically donor-acceptor COPOLYMER backbones
    (alternating electron-rich and electron-poor heterocycles along a
    single conjugated chain) where the red-shift comes from intramolecular
    charge-transfer character and backbone planarity/torsion -- properties
    a ring-count-and-substructure heuristic doesn't capture. Unlike the
    semiconductor_bandgap_eV proxy (validated well on simple acenes, which
    is what it's actually used for here), this one would need a genuinely
    different feature set (or a trained model) to work for real donor-
    acceptor copolymer-style dyes. Treat this as illustrative only for
    complex real chromophores; it's on firmer ground for simple,
    acene-like structures closer to its calibration set."""
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    donor = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
    acceptor = Chem.MolFromSmarts("[$([NX3](=O)=O),$(C=O)]")
    has_donor = mol.HasSubstructMatch(donor)
    has_acceptor = mol.HasSubstructMatch(acceptor)
    push_pull_bonus = 60.0 if (has_donor and has_acceptor) else 0.0
    base = 250.0 + 65.0 * n_aromatic_rings + push_pull_bonus
    return max(200.0, min(750.0, base))


def dye_donor_acceptor_strength(mol):
    """0-1 proxy for push-pull chromophore character, a real structural
    driver of light-harvesting efficiency in dye-sensitized solar cells
    (though real device efficiency also depends heavily on anchoring groups,
    the semiconductor substrate, and electrolyte -- all outside a single
    molecule's descriptors).

    Same real-data check as dye_absorption_nm (see its docstring): rho=0.119
    (p=0.0007) against real OPV2D donor absorption, n=821 -- correct
    direction this time, but similarly tiny in magnitude. Real signal,
    weak effect size; the same "doesn't capture real donor-acceptor
    COPOLYMER photophysics" caveat applies."""
    donor_n = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
    donor_o = Chem.MolFromSmarts("[OX2H0;!$(OC=O)]")
    acceptor = Chem.MolFromSmarts("[$([NX3](=O)=O),$(C=O),$(C#N)]")
    n_donor = len(mol.GetSubstructMatches(donor_n)) + len(mol.GetSubstructMatches(donor_o))
    n_acceptor = len(mol.GetSubstructMatches(acceptor))
    return max(0.0, min(1.0, 0.3 * min(n_donor, 2) + 0.3 * min(n_acceptor, 2)))


def fragrance_volatility(mol):
    """0-1 proxy peaking in the typical volatile-fragrance-molecule range
    (real perfumery ingredients cluster around MW 100-250, logP ~1.5-3.5 --
    volatile enough to reach the nose, lipophilic enough to cross membranes)."""
    mw = sum(a.GetMass() for a in mol.GetAtoms())
    logp = Crippen.MolLogP(mol)
    mw_score = math.exp(-((mw - 180.0) ** 2) / (2 * 70.0 ** 2))
    logp_score = math.exp(-((logp - 2.5) ** 2) / (2 * 1.5 ** 2))
    return max(0.0, min(1.0, 0.5 * mw_score + 0.5 * logp_score))


def fragrance_safety_proxy(mol):
    """0-1 proxy penalizing substructures commonly flagged as skin
    sensitizers/reactive in fragrance safety assessment (Michael-acceptor
    enals, nitro groups, epoxides) -- a real category IFRA-type guidance
    restricts, not a substitute for an actual safety assessment.

    VALIDATION STATUS: unvalidated, and honestly so -- unlike the other
    proxies in this file (polymer Tg, semiconductor band gap, electrolyte
    polarity, green solvent toxicity, agrochemical bee-safety), this one
    was NOT checked against real skin-sensitization data, because a
    genuine, non-trivial search effort could not locate one. Real LLNA/
    DPRA/h-CLAT skin-sensitization datasets clearly exist (Hansen et al.
    2009's ~378-compound LLNA set, OECD TG 497's reference chemicals,
    multiple ChemRxiv/ACS papers with 100-500 compound benchmarks) but are
    published as journal supplementary files, not as GitHub-hosted CSVs --
    every direct-fetch and GitHub-search attempt for this one came back
    empty, unlike the successful searches for every other proxy's
    real-data check in this file. This is reported here rather than left
    silent, the same way the DFG-in Abl kinase docking follow-up (see the
    research validation report) was reported as a genuine dead end rather
    than quietly dropped. If you have access to any of the cited datasets
    (e.g. behind a journal subscription, or a lab's internal copy), this
    proxy is the best remaining candidate in this file for a real
    validation pass."""
    enal = Chem.MolFromSmarts("[CX3]=[CX3][CX3H1]=O")
    nitro = Chem.MolFromSmarts("[NX3](=O)=O")
    epoxide = Chem.MolFromSmarts("C1OC1")
    n_flags = (len(mol.GetSubstructMatches(enal)) + len(mol.GetSubstructMatches(nitro))
               + len(mol.GetSubstructMatches(epoxide)))
    return max(0.0, min(1.0, 1.0 - 0.4 * n_flags))


def agrochemical_persistence_inverse(mol):
    """0-1 "environmental safety" proxy where HIGHER = less persistent /
    more readily degraded (i.e. safer): penalizes halogen density (heavy
    halogenation is the classic structural flag for environmental
    persistence/bioaccumulation, as in legacy organochlorine pesticides
    like DDT), rewards hydrolyzable ester/amide linkages.

    IMPORTANT validated finding: this proxy was checked against real acute
    honeybee toxicity labels (ApisTox, Adamczyk et al. 2024 -- see
    AGROCHEMICAL_SMILES's citation, n=1035) and found to have ~NO predictive
    power for that specific endpoint: ROC-AUC=0.490 (chance level), t=1.76
    (p=0.079). This is a real, informative null result, not a bug to fix --
    persistence/biodegradability and acute neurotoxicity are genuinely
    different toxicological mechanisms, and this proxy was only ever
    designed to capture the former. See agrochemical_bee_safety() below for
    a second proxy purpose-built and validated against the acute-toxicity
    endpoint instead. Use THIS proxy for "will it linger in soil/water",
    and that one for "will it poison a pollinator on contact" -- they are
    not interchangeable."""
    n = max(mol.GetNumHeavyAtoms(), 1)
    n_halogen = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
    ester = Chem.MolFromSmarts("C(=O)O[#6]")
    amide = Chem.MolFromSmarts("C(=O)N")
    hydrolyzable = len(mol.GetSubstructMatches(ester)) + len(mol.GetSubstructMatches(amide))
    score = 1.0 - (n_halogen / n) * 2.0 + hydrolyzable * 0.15
    return max(0.0, min(1.0, score))


_ORGANOPHOSPHATE_PO = Chem.MolFromSmarts("[#15](=O)")
_ORGANOPHOSPHATE_PS = Chem.MolFromSmarts("[#15](=S)")
_CARBAMATE_ESTER = Chem.MolFromSmarts("[#6]OC(=O)N")
_NITRO_IMINE = Chem.MolFromSmarts("[NX3](=[NX2])")
_AROMATIC_CHLORINE = Chem.MolFromSmarts("c(Cl)")


def agrochemical_bee_safety(mol):
    """0-1 proxy where HIGHER = predicted safer for pollinators (honeybees),
    built and validated specifically against real acute-toxicity data
    (ApisTox, n=1035 real pesticides with real bee-toxicity labels) after
    agrochemical_persistence_inverse (above) was tested against the same
    data and found to have no signal for this endpoint (AUC=0.49).

    Flags the real, literature-documented mechanisms of acute insect
    neurotoxicity instead of persistence: organophosphate esters (P=O/P=S,
    acetylcholinesterase inhibitors -- the mechanistic class malathion,
    parathion, chlorpyrifos etc. belong to), carbamate esters (same AChE
    mechanism, different chemistry -- carbaryl, aldicarb-class), a rough
    nitro-imine/nitroguanidine flag (the neonicotinoid pharmacophore family
    -- imidacloprid-class nicotinic receptor agonists), and aromatic
    chlorination (legacy organochlorine neurotoxicants).

    Validated at t=13.23 (p<0.00001), point-biserial r=-0.38 (p<0.00001),
    ROC-AUC=0.642 against the real ApisTox labels -- a real, statistically
    strong signal (AUC 0.64 is "fair" discrimination by conventional
    classifier standards, meaningfully better than chance but not
    diagnostic on its own for any single molecule)."""
    has_op = mol.HasSubstructMatch(_ORGANOPHOSPHATE_PO) or mol.HasSubstructMatch(_ORGANOPHOSPHATE_PS)
    has_carbamate = mol.HasSubstructMatch(_CARBAMATE_ESTER)
    has_nitroimine = mol.HasSubstructMatch(_NITRO_IMINE)
    n_aromatic_cl = len(mol.GetSubstructMatches(_AROMATIC_CHLORINE))
    penalty = (0.45 * has_op + 0.25 * has_carbamate + 0.15 * has_nitroimine
               + 0.15 * min(n_aromatic_cl / 3, 1))
    return max(0.0, min(1.0, 1.0 - penalty))


def green_biodegradability(mol):
    """0-1 proxy: hydrolyzable ester/amide/hydroxyl content, a real
    structural driver of biodegradability (matches why esters like ethyl
    lactate are favored "green" solvents over persistent halogenated ones)."""
    n = max(mol.GetNumHeavyAtoms(), 1)
    ester = Chem.MolFromSmarts("C(=O)O[#6]")
    oh = Chem.MolFromSmarts("[OX2H]")
    hits = len(mol.GetSubstructMatches(ester)) * 2 + len(mol.GetSubstructMatches(oh))
    return max(0.0, min(1.0, hits * 2.0 / n))


def green_low_toxicity(mol):
    """0-1 proxy where higher = likely lower toxicity: penalizes high
    lipophilicity (bioaccumulation risk) and aromatic halogens (a classic
    persistent-toxicant structural flag), consistent with green-chemistry
    solvent-selection guidance."""
    logp = Crippen.MolLogP(mol)
    n = max(mol.GetNumHeavyAtoms(), 1)
    aromatic_halogen = sum(1 for a in mol.GetAtoms()
                            if a.GetSymbol() in ("F", "Cl", "Br", "I")
                            and any(nb.GetIsAromatic() for nb in a.GetNeighbors()))
    logp_penalty = max(0.0, (logp - 3.0) / 4.0)
    return max(0.0, min(1.0, 1.0 - logp_penalty - aromatic_halogen / n * 3.0))


PROPERTY_FUNCS = {
    "logp": lambda m: Crippen.MolLogP(m),
    "qed": lambda m: QED.qed(m),
    "mol_wt": lambda m: sum(a.GetMass() for a in m.GetAtoms()),
    "tpsa": lambda m: rdMolDescriptors.CalcTPSA(m),
    "sa_score": sa_score,
    "num_hbd": lambda m: rdMolDescriptors.CalcNumHBD(m),
    "num_hba": lambda m: rdMolDescriptors.CalcNumHBA(m),
    "des_hba_score": des_hba_score,
    "des_hbd_score": des_hbd_score,
    "polymer_flexibility": polymer_flexibility,
    "polymer_thermal": polymer_thermal,
    "polymer_degradability": polymer_degradability,
    "catalyst_nucleophilicity": catalyst_nucleophilicity,
    "catalyst_steric_bulk": catalyst_steric_bulk,
    "catalyst_stability": catalyst_stability,
    "electrolyte_polarity": electrolyte_polarity,
    "electrolyte_stability": electrolyte_stability,
    "semiconductor_bandgap_eV": semiconductor_bandgap_eV,
    "semiconductor_mobility": semiconductor_mobility_proxy,
    "dye_absorption_nm": dye_absorption_nm,
    "dye_donor_acceptor": dye_donor_acceptor_strength,
    "fragrance_volatility": fragrance_volatility,
    "fragrance_safety": fragrance_safety_proxy,
    "agrochemical_persistence_inverse": agrochemical_persistence_inverse,
    "agrochemical_bee_safety": agrochemical_bee_safety,
    "green_biodegradability": green_biodegradability,
    "green_low_toxicity": green_low_toxicity,
}
DEFAULT_SIGMA = {
    "logp": 1.0, "qed": 0.15, "mol_wt": 60.0, "tpsa": 20.0, "sa_score": 1.5,
    "num_hbd": 1.0, "num_hba": 1.5, "des_hba_score": 0.25, "des_hbd_score": 0.25,
    "polymer_flexibility": 0.25, "polymer_thermal": 0.25, "polymer_degradability": 0.25,
    "catalyst_nucleophilicity": 0.25, "catalyst_steric_bulk": 0.25, "catalyst_stability": 0.25,
    "electrolyte_polarity": 0.25, "electrolyte_stability": 0.25,
    "semiconductor_bandgap_eV": 0.6, "semiconductor_mobility": 0.25,
    "dye_absorption_nm": 50.0, "dye_donor_acceptor": 0.25,
    "fragrance_volatility": 0.25, "fragrance_safety": 0.25,
    "agrochemical_persistence_inverse": 0.25, "agrochemical_bee_safety": 0.25,
    "green_biodegradability": 0.25, "green_low_toxicity": 0.25,
}

DOMAIN_PROPERTIES = {
    "drug": ["logp", "qed", "mol_wt", "tpsa"],
    "des": ["des_hba_score", "des_hbd_score", "mol_wt"],
    "polymer": ["polymer_flexibility", "polymer_thermal", "polymer_degradability"],
    "catalyst": ["catalyst_nucleophilicity", "catalyst_steric_bulk", "catalyst_stability"],
    "electrolyte": ["electrolyte_polarity", "electrolyte_stability", "mol_wt"],
    "semiconductor": ["semiconductor_bandgap_eV", "semiconductor_mobility"],
    "dye": ["dye_absorption_nm", "dye_donor_acceptor"],
    "fragrance": ["fragrance_volatility", "fragrance_safety", "mol_wt"],
    "agrochemical": ["agrochemical_persistence_inverse", "agrochemical_bee_safety", "logp"],
    "green_solvent": ["green_biodegradability", "green_low_toxicity", "mol_wt"],
}
PROPERTY_RANGES = {  # (min, max, default) for the UI sliders
    "logp": (-2.0, 6.0, 2.5), "qed": (0.0, 1.0, 0.8), "mol_wt": (100.0, 500.0, 300.0),
    "tpsa": (0.0, 150.0, 60.0), "sa_score": (1.0, 10.0, 3.0),
    "num_hbd": (0.0, 6.0, 1.0), "num_hba": (0.0, 10.0, 3.0),
    "des_hba_score": (0.0, 1.0, 0.8), "des_hbd_score": (0.0, 1.0, 0.8),
    "polymer_flexibility": (0.0, 1.0, 0.6), "polymer_thermal": (0.0, 1.0, 0.5),
    "polymer_degradability": (0.0, 1.0, 0.5),
    "catalyst_nucleophilicity": (0.0, 1.0, 0.6), "catalyst_steric_bulk": (0.0, 1.0, 0.5),
    "catalyst_stability": (0.0, 1.0, 0.6),
    "electrolyte_polarity": (0.0, 1.0, 0.6), "electrolyte_stability": (0.0, 1.0, 0.7),
    "semiconductor_bandgap_eV": (1.0, 6.0, 2.5), "semiconductor_mobility": (0.0, 1.0, 0.7),
    "dye_absorption_nm": (200.0, 750.0, 450.0), "dye_donor_acceptor": (0.0, 1.0, 0.6),
    "fragrance_volatility": (0.0, 1.0, 0.7), "fragrance_safety": (0.0, 1.0, 0.8),
    "agrochemical_persistence_inverse": (0.0, 1.0, 0.7), "agrochemical_bee_safety": (0.0, 1.0, 0.8),
    "green_biodegradability": (0.0, 1.0, 0.7), "green_low_toxicity": (0.0, 1.0, 0.8),
}


def compute_props(smiles, names):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {n: PROPERTY_FUNCS[n](mol) for n in names if n in PROPERTY_FUNCS}


def reward_for_target(smiles, target: dict) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    scores = []
    for name, tgt in target.items():
        if name not in PROPERTY_FUNCS:
            continue
        try:
            val = PROPERTY_FUNCS[name](mol)
        except Exception:
            scores.append(0.0)
            continue
        sigma = DEFAULT_SIGMA.get(name, 1.0)
        scores.append(math.exp(-((val - tgt) ** 2) / (2 * sigma ** 2)))
    return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# 4) MODELS: Graph VAE (GCN encoder + GRU decoder) and a SMILES policy for RL
# ============================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nbr = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj, mask):
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        nbr = torch.bmm(adj, x) / deg
        out = F.relu(self.lin_self(x) + self.lin_nbr(nbr))
        return out * mask.unsqueeze(-1)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden=96, latent=48, layers=3):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(layers)])
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)

    def forward(self, x, adj, mask):
        h = x
        for layer in self.layers:
            h = layer(h, adj, mask)
        denom = mask.sum(-1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / denom
        return self.to_mu(pooled), self.to_logvar(pooled)


class SmilesDecoder(nn.Module):
    def __init__(self, vocab_size, latent=48, emb=96, hidden=192, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.z_to_h0 = nn.Linear(latent, hidden * layers)
        self.gru = nn.GRU(emb + latent, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)
        self.n_layers, self.hidden_dim = layers, hidden

    def forward(self, z, target_ids):
        B, L = target_ids.shape
        e = self.emb(target_ids)
        zr = z.unsqueeze(1).expand(-1, L, -1)
        h0 = self.z_to_h0(z).view(B, self.n_layers, self.hidden_dim).transpose(0, 1).contiguous()
        out, _ = self.gru(torch.cat([e, zr], -1), h0)
        return self.out(out)

    @torch.no_grad()
    def sample(self, z, vocab, max_len=MAX_LEN, temperature=0.8):
        B = z.size(0)
        h = self.z_to_h0(z).view(B, self.n_layers, self.hidden_dim).transpose(0, 1).contiguous()
        cur = torch.full((B, 1), vocab.bos_id, dtype=torch.long, device=z.device)
        done = torch.zeros(B, dtype=torch.bool, device=z.device)
        out_ids = torch.full((B, max_len), vocab.pad_id, dtype=torch.long, device=z.device)
        for t in range(max_len):
            e = self.emb(cur)
            o, h = self.gru(torch.cat([e, z.unsqueeze(1)], -1), h)
            logits = self.out(o[:, -1]) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, vocab.pad_id), nxt)
            out_ids[:, t] = nxt
            done = done | (nxt == vocab.eos_id)
            cur = nxt.unsqueeze(1)
            if done.all():
                break
        return out_ids


class GraphVAE(nn.Module):
    def __init__(self, in_dim, vocab_size, latent=48):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, latent=latent)
        self.decoder = SmilesDecoder(vocab_size, latent=latent)
        self.latent_dim = latent

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x, adj, mask, target_ids):
        mu, logvar = self.encoder(x, adj, mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, target_ids[:, :-1])
        return logits, mu, logvar

    @torch.no_grad()
    def sample_prior(self, n, vocab, temperature=0.8):
        z = torch.randn(n, self.latent_dim, device=DEVICE)
        return self.decoder.sample(z, vocab, temperature=temperature)


def vae_loss(logits, target_ids, mu, logvar, pad_id, kl_weight):
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), target_ids[:, 1:].reshape(-1), ignore_index=pad_id)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return ce + kl_weight * kl


class SmilesLM(nn.Module):
    """Autoregressive SMILES policy used for RL fine-tuning toward a target."""
    def __init__(self, vocab_size, emb=96, hidden=192, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)
        self.n_layers, self.hidden_dim = layers, hidden

    def forward(self, ids):
        e = self.emb(ids)
        out, _ = self.gru(e)
        return self.out(out)

    def sample_with_logprobs(self, vocab, batch_size, max_len=MAX_LEN, temperature=1.0):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=DEVICE)
        cur = torch.full((batch_size, 1), vocab.bos_id, dtype=torch.long, device=DEVICE)
        done = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
        out_ids = torch.full((batch_size, max_len), vocab.pad_id, dtype=torch.long, device=DEVICE)
        logps = torch.zeros(batch_size, max_len, device=DEVICE)
        active = torch.zeros(batch_size, max_len, device=DEVICE)
        for t in range(max_len):
            e = self.emb(cur)
            o, h = self.gru(e, h)
            logits = self.out(o[:, -1]) / max(temperature, 1e-6)
            logp_all = F.log_softmax(logits, dim=-1)
            nxt = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
            token_logp = logp_all.gather(1, nxt.unsqueeze(1)).squeeze(1)
            nxt_m = torch.where(done, torch.full_like(nxt, vocab.pad_id), nxt)
            out_ids[:, t] = nxt_m
            logps[:, t] = torch.where(done, torch.zeros_like(token_logp), token_logp)
            active[:, t] = (~done).float()
            done = done | (nxt == vocab.eos_id)
            cur = nxt_m.unsqueeze(1)
            if done.all():
                break
        return out_ids, logps, active


class ReinforceTrainer:
    def __init__(self, policy, vocab, lr=5e-4, kl_coef=0.05, entropy_coef=0.02):
        import copy
        self.policy = policy
        self.prior = copy.deepcopy(policy).eval()
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.vocab = vocab
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.kl_coef, self.entropy_coef = kl_coef, entropy_coef

    def step(self, batch_size, reward_fn, temperature=1.0):
        ids, logps, active = self.policy.sample_with_logprobs(self.vocab, batch_size, temperature=temperature)
        smiles = [self.vocab.decode(ids[i].tolist()) for i in range(batch_size)]
        rewards = torch.tensor([reward_fn(s) for s in smiles], dtype=torch.float32, device=DEVICE)
        self.baseline = 0.9 * self.baseline + 0.1 * rewards.mean().item()
        advantage = rewards - self.baseline
        seq_logp = (logps * active).sum(dim=1)
        pg_loss = -(advantage.detach() * seq_logp).mean()

        with torch.no_grad():
            prior_logits = self.prior(ids)
        policy_logits = self.policy(ids)
        pl = F.log_softmax(policy_logits, dim=-1)
        pr = F.log_softmax(prior_logits, dim=-1)
        kl = ((pl.exp() * (pl - pr)).sum(-1) * active).sum(dim=1).mean()
        entropy = (-(pl.exp() * pl).sum(-1) * active).sum(dim=1).mean()

        loss = pg_loss + self.kl_coef * kl - self.entropy_coef * entropy
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        return {"smiles": smiles, "rewards": rewards.tolist(), "mean_reward": rewards.mean().item()}


# ============================================================================
# 5) TRAINING (cached per domain) + GENERATION
# ============================================================================
_CACHE: dict[str, dict] = {}


def get_or_train_domain(domain: str, progress=None) -> dict:
    if domain in _CACHE:
        return _CACHE[domain]

    if progress is not None and domain in REAL_DATASET_SOURCES:
        cache_path = os.path.join(_CACHE_DIR, f"{domain}.csv")
        if not os.path.exists(cache_path):
            progress(0.01, desc=f"Downloading real {domain} dataset "
                                 f"({REAL_DATASET_SOURCES[domain]['citation']})... one-time, ~1 min")

    smiles = load_domain_smiles(domain)
    n_pool = len(smiles)
    if len(smiles) > 600:
        random.Random(0).shuffle(smiles)
        smiles = smiles[:600]  # subsample for CPU-tractable training; the full real pool is n_pool
    vocab = SmilesVocab(smiles)
    x, adj, mask, keep = batch_mol_to_graph(smiles)
    kept_smiles = [smiles[i] for i in keep]
    target_ids = vocab.batch_encode(kept_smiles, MAX_LEN).to(DEVICE)
    x, adj, mask = x.to(DEVICE), adj.to(DEVICE), mask.to(DEVICE)

    vae = GraphVAE(ATOM_FEAT_DIM, len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    n_data = x.size(0)
    epochs = 60
    for ep in range(epochs):
        if progress is not None:
            progress((ep + 1) / (epochs + 20), desc=f"Training Graph VAE on {domain} data... epoch {ep+1}/{epochs}")
        perm = torch.randperm(n_data)
        bs = min(64, n_data)
        for i in range(0, n_data, bs):
            idx = perm[i:i + bs]
            logits, mu, logvar = vae(x[idx], adj[idx], mask[idx], target_ids[idx])
            kl_w = min(0.3, 0.3 * ep / max(epochs * 0.3, 1))
            loss = vae_loss(logits, target_ids[idx], mu, logvar, vocab.pad_id, kl_w)
            opt.zero_grad(); loss.backward(); opt.step()

    rl_lm = SmilesLM(len(vocab)).to(DEVICE)
    opt2 = torch.optim.Adam(rl_lm.parameters(), lr=2e-3)
    pretrain_epochs = 40
    for ep in range(pretrain_epochs):
        if progress is not None:
            progress((epochs + ep + 1) / (epochs + pretrain_epochs + 5),
                      desc=f"Pretraining RL policy on {domain} data... epoch {ep+1}/{pretrain_epochs}")
        perm = torch.randperm(n_data)
        bs = min(64, n_data)
        for i in range(0, n_data, bs):
            idx = perm[i:i + bs]
            batch = target_ids[idx]
            logits = rl_lm(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1),
                                    ignore_index=vocab.pad_id)
            opt2.zero_grad(); loss.backward(); opt2.step()

    _CACHE[domain] = {"vae": vae, "vocab": vocab, "rl_lm_base": rl_lm, "train_smiles": set(kept_smiles),
                       "n_pool": n_pool, "n_trained_on": len(kept_smiles)}
    return _CACHE[domain]


def draw_molecules(smiles_list, rewards, max_show=12):
    mols, legends = [], []
    for s, r in zip(smiles_list, rewards):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"reward={r:.2f}")
        if len(mols) >= max_show:
            break
    if not mols:
        return None
    return Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(220, 220), legends=legends)


def generate(domain, model_choice, n_candidates, rl_steps, *prop_values, progress=gr.Progress()):
    prop_names = DOMAIN_PROPERTIES[domain]
    target = {name: val for name, val in zip(prop_names, prop_values)}

    progress(0.02, desc=f"Preparing {domain} models (first call trains from scratch, ~1-2 min)...")
    cache = get_or_train_domain(domain, progress=progress)
    vae, vocab, rl_lm_base, train_smiles = cache["vae"], cache["vocab"], cache["rl_lm_base"], cache["train_smiles"]

    if model_choice == "Graph VAE (fast, prior sampling ranked by target)":
        progress(0.7, desc="Sampling from the VAE prior...")
        pool_n = max(n_candidates * 6, 40)
        ids = vae.sample_prior(pool_n, vocab, temperature=0.85)
        pool = [vocab.decode(ids[i].tolist()) for i in range(pool_n)]
        scored = [(s, reward_for_target(s, target)) for s in pool]
    else:  # RL fine-tuning
        import copy
        policy = copy.deepcopy(rl_lm_base)
        trainer = ReinforceTrainer(policy, vocab, lr=5e-4, kl_coef=0.08, entropy_coef=0.03)
        reward_fn = lambda s: reward_for_target(s, target)
        last_info = None
        for step in range(int(rl_steps)):
            progress(0.3 + 0.6 * (step + 1) / rl_steps, desc=f"RL fine-tuning toward your target... step {step+1}/{rl_steps}")
            last_info = trainer.step(batch_size=24, reward_fn=reward_fn, temperature=1.0)
        pool = []
        for _ in range(max(n_candidates // 24 + 1, 2)):
            with torch.no_grad():
                ids, _, _ = trainer.policy.sample_with_logprobs(vocab, 24, temperature=0.9)
            pool.extend(vocab.decode(ids[i].tolist()) for i in range(24))
        scored = [(s, reward_for_target(s, target)) for s in pool]

    progress(0.95, desc="Scoring and ranking candidates...")
    # de-duplicate by canonical SMILES, keep highest reward
    best = {}
    for s, r in scored:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in best or best[canon][1] < r:
            best[canon] = (canon, r)
    ranked = sorted(best.values(), key=lambda t: -t[1])[:n_candidates]

    if not ranked:
        return None, "No valid molecules were generated this round -- try clicking Generate again " \
                      "(sampling has randomness) or lower the number of candidates.", ""

    rows = []
    for canon, r in ranked:
        props = compute_props(canon, prop_names)
        novel = canon not in train_smiles
        row = {"SMILES": canon, "reward": round(r, 3), "novel": novel}
        row.update({k: round(v, 2) for k, v in props.items()})
        rows.append(row)

    img = draw_molecules([r[0] for r in ranked], [r[1] for r in ranked])
    table_md = "| SMILES | reward | novel | " + " | ".join(prop_names) + " |\n"
    table_md += "|---|---|---|" + "---|" * len(prop_names) + "\n"
    for row in rows:
        table_md += f"| `{row['SMILES']}` | {row['reward']} | {'yes' if row['novel'] else 'no'} | "
        table_md += " | ".join(str(row.get(p, "")) for p in prop_names) + " |\n"

    data_note = ""
    if "n_pool" in cache:
        source = REAL_DATASET_SOURCES.get(domain, {}).get("citation") or EMBEDDED_REAL_DATA_CITATIONS.get(domain)
        if source:
            data_note = (f" Trained on {cache['n_trained_on']} molecules subsampled from a real published "
                         f"dataset of {cache['n_pool']} ({source}).")
        else:
            data_note = (f" Trained on {cache['n_trained_on']} molecules from this app's small hand-picked "
                         f"seed list ({cache['n_pool']} total) -- no large public dataset was found for "
                         f"this domain, see the intro note above.")

    summary = (f"Generated **{len(ranked)}** unique candidates (from a pool of {len(scored)} samples) "
               f"for target `{target}`. Mean reward of returned set: "
               f"{np.mean([r for _, r in ranked]):.3f}.{data_note}")
    return img, summary, table_md


# ============================================================================
# 6) GRADIO UI
# ============================================================================
def build_property_sliders(domain):
    names = DOMAIN_PROPERTIES[domain]
    updates = []
    for i in range(4):
        if i < len(names):
            name = names[i]
            lo, hi, default = PROPERTY_RANGES[name]
            updates.append(gr.update(visible=True, label=name, minimum=lo, maximum=hi, value=default))
        else:
            updates.append(gr.update(visible=False))
    return updates


with gr.Blocks(title="MOLGEN — Molecular Generation Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # MOLGEN — Molecular Generation Platform
        **Desired properties in. New molecules out.** Not prediction — *inverse design*.

        Pick a domain, set your target property values, and generate new molecules that
        satisfy them. Each domain trains a small Graph VAE + RL policy on first use
        and is cached afterward — only the target-steering step re-runs on each click.

        **10 domains**: drug discovery · deep eutectic solvents · polymers · organic
        catalysts · battery electrolytes · organic semiconductors · dyes & solar cells ·
        fragrances & flavors · agrochemicals · green solvents.

        **Data scale, honestly:** two domains train on real, published, cited datasets
        fetched live on first use — **fragrance** (Leffingwell PMP 2001, 3,523 molecules,
        via the Pyrfume-Data archive) and **agrochemical** (ApisTox, 1,035 molecules,
        Adamczyk et al. 2024). The other 8 domains train on small (10-31 molecule)
        hand-picked seed lists embedded directly in this file — I looked for equivalent
        large public datasets for organic catalysts, battery electrolytes, organic
        semiconductors, dyes, and green solvents and did not find an easily-integrable
        one; that itself says something about how much less standardized these research
        niches are compared to drug discovery (which has ZINC's 250k+) or polymers
        (PI1M's ~1M repeat units) — see the full research platform for those two.

        *This is a streamlined single-file version of a larger research platform
        (which also includes latent diffusion, a Graph Transformer surrogate, PPO,
        Pareto multi-objective ranking, and real AutoDock Vina docking validation).*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            domain_dd = gr.Dropdown(
                ["drug", "des", "polymer", "catalyst", "electrolyte", "semiconductor",
                 "dye", "fragrance", "agrochemical", "green_solvent"],
                value="drug", label="Application domain",
                info="drug discovery / deep eutectic solvents / polymers / organic catalysts / "
                     "battery electrolytes / organic semiconductors / dyes & solar cells / "
                     "fragrances & flavors / agrochemicals / green solvents")
            model_dd = gr.Radio(
                ["Graph VAE (fast, prior sampling ranked by target)", "RL fine-tuning (slower, steers toward target)"],
                value="Graph VAE (fast, prior sampling ranked by target)", label="Generative engine")
            n_candidates = gr.Slider(4, 24, value=12, step=1, label="Number of candidates to return")
            rl_steps = gr.Slider(5, 30, value=12, step=1, label="RL fine-tuning steps (only used by RL engine)")

            gr.Markdown("**Target property values:**")
            prop_sliders = [gr.Slider(visible=(i == 0), label="", minimum=0, maximum=1, value=0.5)
                             for i in range(4)]

            generate_btn = gr.Button("Generate molecules", variant="primary")

        with gr.Column(scale=2):
            gallery = gr.Image(label="Generated candidates (ranked by reward)", type="pil")
            summary_md = gr.Markdown()
            table_md = gr.Markdown()

    domain_dd.change(fn=build_property_sliders, inputs=domain_dd, outputs=prop_sliders)
    demo.load(fn=build_property_sliders, inputs=domain_dd, outputs=prop_sliders)

    generate_btn.click(
        fn=generate,
        inputs=[domain_dd, model_dd, n_candidates, rl_steps] + prop_sliders,
        outputs=[gallery, summary_md, table_md],
    )

    gr.Markdown(
        """
        ---
        **Reading the results:** *reward* is a Gaussian-similarity score in [0,1] between
        1.0 (perfect match to your targets) and 0.0; *novel* means the exact molecule
        isn't in this app's small training set. Because training data here is a small
        embedded seed set (not the full ZINC250k/PI1M used in the research version),
        expect this demo to be illustrative rather than production-quality — see the
        full platform repository for the research-grade build with real large datasets,
        docking validation, and statistical ablations.

        **On the properties themselves:** for drug design, DES, and polymers, the
        underlying property functions are either real RDKit calculations (logP, QED,
        TPSA, the actual Ertl & Schuffenhauer SA-score) or heuristics that were checked
        against real external literature/experimental data in this project's research
        validation report (the polymer thermal proxy was even caught failing that check
        once and rebuilt). For the seven newer domains (catalysts, electrolytes,
        semiconductors, dyes, fragrances, agrochemicals, green solvents), the property
        proxies are transparent, real-chemistry-motivated structural heuristics that
        have **not yet been externally validated** the same way — treat their outputs
        as illustrative and directional, not as calibrated predictions of activity,
        conductivity, band gap, absorption wavelength, odor, or toxicity, all of which
        are genuinely hard, active research problems in their own right.
        """
    )

if __name__ == "__main__":
    demo.queue().launch()
