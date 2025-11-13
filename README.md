PPA/APP Replication Package — Systemic Reduction Analysis
DOI: https://doi.org/10.5281/zenodo.17604852

This repository contains the complete replication package for the paper:

"The Autoreferential Protection Principle: Viability, Shielding, and the Reduction Paradox in Reflexive Agent–Environment Systems."
Duarte, A. A. (2025). Zenodo. https://doi.org/10.5281/zenodo.17604852

It includes simulation code, configuration files, analysis scripts, stability metrics, perturbation results, and aggregated datasets used to generate all figures and tables in the manuscript.
Raw tick-level logs are archived offline due to size constraints, but all results in the paper are reproducible with the content of this repository.

CONTENTS

Simulation Code

ev04/: EV-04 architecture (APP-Fixed and VOFF)

ev02c/: EV-02C baseline configuration

run_ev04_cli.py: Command-line interface for full experimental batches

simulation_app_base_v2_FINAL.py: Core implementation of APP

Configuration Files

YAML configurations for AIM-1 to AIM-4

Warm-start and fixed-policy configurations

Environmental parameter sets for RE, RH, and RC regimes

Analysis Scripts

analyze_aim12_from_ticks.py: Mortality and viability structure

analyze_perturbation.py: Stability and ablation analysis

granger/: Granger causality pipeline

Additional scripts for temporal coupling, lag, coherence, and shielding

Aggregated Datasets

Mortality partitioning

Stability metrics

Markov-blanket shielding (AIM-3)

Perturbation and recovery results

Temporal phase-space reconstruction

Granger causality matrices

(Per-tick logs stored offline due to size limits.)

REPRODUCIBILITY

All plots, tables, and metrics from the paper can be reproduced using:

python run_ev04_cli.py --regimes RE RH RC --conditions APP VOFF --seeds 0 1 2 3 4 5 6 7 8 9

Analysis scripts in the analysis/ directory reproduce all reported figures and tablas.

CITATION

If you use this package, please cite:

Duarte, A. A. (2025). PPA/APP Replication Package (Version v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.17604852

BibTeX:

@dataset{Duarte2025PPAAPP,
author = {Duarte, Alberto Alejandro},
title = {PPA/APP Replication Package (Version v1.0.1)},
year = {2025},
howpublished = {Zenodo},
doi = {10.5281/zenodo.17604852},
url = {https://doi.org/10.5281/zenodo.17604852}

}

LICENSE

Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0).
Non-commercial academic use is allowed.
Commercial use requires prior written permission.

CONTACT

albertoduarte@paradoxsystems.xyz
