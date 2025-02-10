# DEGU: Distilling Ensembles for Genomic Uncertainty-aware Models

## Overview
DEGU (**Distilling Ensembles for Genomic Uncertainty-aware models**) is a deep learning framework that combines ensemble learning and knowledge distillation to improve the robustness, interpretability, and uncertainty quantification of genomic deep learning models. This repository contains the official implementation of DEGU, as described in our **ICLR MLGenX 2025** paper.

## Paper
**Title:** [Uncertainty-Aware Genomic Deep Learning with Knowledge Distillation](https://www.biorxiv.org/content/10.1101/2024.11.13.623485v1)

**Authors:** Jessica Zhou, Kaeli Rizzo, Ziqi (Amber) Tang, Peter K Koo

## Features
- **Knowledge Distillation**: Transfers knowledge from an ensemble of models (teachers) to a single model (student) to reduce computational overhead.
- **Uncertainty Quantification**:
  - **Epistemic uncertainty** (model-based) estimated from ensemble prediction variability.
  - **Aleatoric uncertainty** (data-based) modeled using experimental replicates.
- **Improved Attribution Analysis**: More stable and biologically meaningful feature attributions.
- **Generalization under Covariate Shift**: Enhanced robustness when applied to out-of-distribution genomic sequences.
- **Computational Efficiency**: Retains ensemble performance in a single distilled model, reducing inference costs.

## Datasets
We used publicly available genomic datasets, including:
- **STARR-seq (fly enhancer activity)**
- **lentiMPRA (human cis-regulatory sequence activity)**
- **Base-resolution ATAC-seq profiles**

For details on dataset preprocessing, refer to `data/README.md`.

## Documentation
For details on how knowledge distillation was applied to each dataset, refer to the corresponding markdown file in `docs/`. 

## Citation
If you use this repository, please cite our paper:
```
@article {Zhou2024.11.13.623485,
	author = {Zhou, Jessica and Rizzo, Kaeli and Tang, Ziqi and Koo, Peter K},
	title = {Uncertainty-aware genomic deep learning with knowledge distillation},
	elocation-id = {2024.11.13.623485},
	year = {2024},
	doi = {10.1101/2024.11.13.623485},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Deep neural networks (DNNs) have advanced predictive modeling for regulatory genomics, but challenges remain in ensuring the reliability of their predictions and understanding the key factors behind their decision making. Here we introduce DEGU (Distilling Ensembles for Genomic Uncertainty-aware models), a method that integrates ensemble learning and knowledge distillation to improve the robustness and explainability of DNN predictions. DEGU distills the predictions of an ensemble of DNNs into a single model, capturing both the average of the ensemble{\textquoteright}s predictions and the variability across them, with the latter representing epistemic (or model-based) uncertainty. DEGU also includes an optional auxiliary task to estimate aleatoric, or data-based, uncertainty by modeling variability across experimental replicates. By applying DEGU across various functional genomic prediction tasks, we demonstrate that DEGU-trained models inherit the performance benefits of ensembles in a single model, with improved generalization to out-of-distribution sequences and more consistent explanations of cis-regulatory mechanisms through attribution analysis. Moreover, DEGU-trained models provide calibrated uncertainty estimates, with conformal prediction offering coverage guarantees under minimal assumptions. Overall, DEGU paves the way for robust and trustworthy applications of deep learning in genomics research.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/11/15/2024.11.13.623485},
	eprint = {https://www.biorxiv.org/content/early/2024/11/15/2024.11.13.623485.full.pdf},
	journal = {bioRxiv}
}
```

## Acknowledgments
This project builds upon recent advances in ensemble learning, knowledge distillation, and uncertainty quantification for genomic deep learning.
