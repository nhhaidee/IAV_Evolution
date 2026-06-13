# Explainable and Calibrated AI for Decoding Host-Adaptive Changes in Influenza A Virus


## Introduction

### Background

Influenza A virus (IAV) is a major public health burden, causing seasonal epidemics and occasional pandemics. Its transmission from avian species to mammals and subsequent spread requires adaptive changes in the viral genome. Understanding these molecular adaptations is essential for pandemic preparedness, and machine learning offers a powerful approach to uncover the evolution and biology of IAV.

### Results

This study established a well-calibrated WaveSeekerNet model that accurately predicted the host source across all 8 IAV segments (macro F1-score: 0.9728), significantly improving the reliability of predicted probabilities with calibration errors approaching zero. Model interpretation revealed that avian-adapted IAVs consistently activated G/C content, whereas mammalian-adapted IAVs generally activated A/T content. This distinction was confirmed by codon-level analysis, in which G/C-rich codons were rewarded for the avian hosts and A/T-rich codons for the mammalian hosts. In the feature space learned by WaveSeekerNet, we defined host-adaptive distance to quantify species barriers and proposed it as a risk-assessment metric. We hypothesized the Mammalian Adaptation Zone (MAZ), a zone where the virus is expected to adjust its host-adaptive distance to reach, thereby helping it establish persistent mammalian lineages. The analysis also revealed the Hard Distance of avian-origin viruses (e.g., H5Nx, H9N2), indicating they have not yet established persistent mammalian lineages. Finally, analysis of human H7N9 (2013, China) and non-human mammalian H5Nx (North America) viruses showed that WaveSeekerNet accurately identified key mammalian-adaptive mutations, including PB2-E627K and PB2-D701N.

### Conclusions

WaveSeekerNet elucidated IAV host-adaptation mechanisms in silico, providing insights into the underlying mechanisms of host adaptation and informing improved surveillance and intervention strategies.

#### The preprint is available at https://doi.org/10.64898/2026.05.23.726879
## Requirements
1. Pytorch >= 2.4.1
2. [Pytorch Wavelet package] 1.3.0
3. [Pytorch Optimizer] 3.1.1
4. Other requirements: Python 3.12+, pytorch 2.4.1, pytorch-optimizer 3.1.1, pytorch-wavelets 1.3.0, scikit-learn 1.5.1, seaborn 0.13.2, pyfastx 2.1.0, pandas 2.2.2, numpy 1.26.4, shap 0.48.0.

## Data and Source Code

1. Data is available on [Google Drive].
2. IAV whole genome sequences can be downloaded from EpiFLu GISAID database (https://www.gisaid.org/).
3. The WaveSeekerNet model architecture was published with GigaScience (https://doi.org/10.1093/gigascience/giaf089) and code is available at https://github.com/nhhaidee/WaveSeekerNet
4. Code analyses can be found in the `analyses` directory, which includes:
    - `analyses/calibration`: Code for model calibration and evaluation of calibration performance (performed using Jupyter Notebook).
    - `analyses/compositional_nucleotide_shap`: Code for compositional nucleotide SHAP analysis (performed on high-performance computing cluster).
    - `analyses/compute_shap_gradient_explainer`: Code for computing SHAP GradientExplainer (performed on high-performance computing cluster).
    - `analyses/host_adaptive_distance`: Code for computing host-adaptive distance and visualizing the results (performed using Jupyter Notebook).
    - `analyses/model_training`: Code for model training (performed on high-performance computing cluster).
    - `analyses/PB2_E627K_D701N_mutations`: Code for analyzing the PB2-E627K and PB2-D701N mutations in human H7N9 and non-human mammalian H5Nx viruses (performed using Jupyter Notebook).
    - `analyses/rscu_codon_shap`: Code for computing RSCU and codon-SHAP values (performed on high-performance computing cluster).
    - `analyses/trajectory`: Code for visualizing the trajectory of codon-SHAP and Spearman correlation analysis (performed using Jupyter Notebook).
5. Analysis results (Figures, Tables, etc.) for the manuscript are available in the `results` directory.

## Contributors and Maintainers

* [Hai Nguyen](https://github.com/nhhaidee) ([CFIA-NCFAD](https://github.com/CFIA-NCFAD), Department of Computer Science, University of Manitoba) - designed the models, wrote the code/manuscript, prepared data, trained models, performed experiments and completed the data analysis.
* [Josip Rudar](https://github.com/jrudar) ([CFIA-NCFAD](https://github.com/CFIA-NCFAD), Department of Integrative Biology & Centre for Biodiversity Genomics, University of Guelph) - designed models, wrote the code, reviewed/edited the manuscript, provided guidance on the project, and provided feedback on the experiments.

[Pytorch Wavelet package]: https://github.com/fbcotter/pytorch_wavelets
[Pytorch Optimizer]:https://github.com/kozistr/pytorch_optimizer
[Google Drive]: https://drive.google.com/drive/folders/13Lr_bbepWhigeiPHK4fRLnRNCmR26ZlU