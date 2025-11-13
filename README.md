Autoreferential Protection – Replication Package
Version: v1.0.0
Author: Alberto Alejandro Duarte
Contact: albertoduarte@paradoxsystems.xyz

SUMMARY
This repository contains the complete replication package for the paper:
“The Autoreferential Protection Principle: Viability, Shielding, and the Reduction Paradox in Reflexive Agent–Environment Systems”.

It includes:

EV-02C and EV-04 simulation code

Configuration files for all experiments

Analysis scripts for mortality, stability, shielding (AIM-3), temporal coupling, phase-space structure, and Granger causality (AIM-4)

Processed datasets used in all figures and tables

Ablation results for environmental-governor experiments

Aggregated metrics required for all statistical analyses

Raw per-tick logs are archived offline due to size, but the full pipeline is reproducible using the files and scripts included here.

REPOSITORY STRUCTURE
/
├── LICENSE
├── README.md
├── simulation/
│   └── simulation_app_base_v2_FINAL.py
├── config/
│   ├── config_summary.yaml
│   └── config_v5_warmstart.yaml
├── scripts/
│   ├── analyze_aim12_from_ticks.py
│   └── analyze_perturbation.py
└── data/
    ├── AIM3_mb_shielding.csv
    ├── mortality_data_validated.csv
    ├── stability_data_REAL.csv
    ├── all_metrics_alivewin.csv
    ├── granger_causality.zip
    ├── granger_causality_ablation.zip
    ├── phase_space_analysis.zip
    └── temporal_analysis.zip


LICENSE
CC BY-NC 4.0 (non-commercial use only)

REPRODUCIBILITY STATEMENT
All results, figures, and tables in the manuscript can be reproduced using the simulation engine, configuration files, and analysis scripts in this repository. No proprietary software is required beyond standard Python scientific libraries.
Each AIM (AIM-1 to AIM-4) maps directly to specific datasets and scripts provided here.

AIM MODULES

AIM-1: Mortality Partitioning
Data: mortality_data_validated.csv
Script: analyze_aim12_from_ticks.py

AIM-2: Stability Analysis
Data: stability_data_REAL.csv
Script: analyze_aim12_from_ticks.py

AIM-3: Markov-Blanket Shielding
Data: AIM3_mb_shielding.csv
Script: analyze_mb_shielding.py (to be added)

AIM-4: Granger Causality
Data: granger_causality.zip
granger_causality_ablation.zip
Script: analyze_granger.py (to be added)

AIM-Aux: Temporal Coupling and Phase-Space Structure
Data: temporal_analysis.zip
phase_space_analysis.zip

HOW TO RUN THE SIMULATOR

Example command:
python simulation/simulation_app_base_v2_FINAL.py --regime RE --condition APP --T 4000

CONFIGURATION FILES
All parameter sets used in the experiments are provided in the config folder.

CITATION
Please cite this replication package as:

Duarte, A. A. (2025).
PPA/APP Replication Package (Version v1.0.0). Zenodo. DOI: XXXXX
(Replace XXXXX with the actual Zenodo DOI once generated.)

LICENSE
Creative Commons Attribution–NonCommercial 4.0.
Academic and research use allowed.
Commercial use prohibited without written permission.

CONTACT
albertoduarte@paradoxsystems.xyz
