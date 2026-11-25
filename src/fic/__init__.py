"""FIC-CFX: Fashion Image Classification with Counterfactual Explainable AI.

KF7029 MSc project. Package layout:

    fic.config            load the YAML config
    fic.data              dataset loading (small + big dataset)
    fic.models            self-built CNN, ResNet-18, classical ML
    fic.training          training loops for PyTorch and scikit-learn models
    fic.xai               local and global explanations (Grad-CAM, saliency, importance)
    fic.counterfactuals   Wachter (Method A), prototype-guided (Method B), black-box
    fic.evaluation        accuracy and counterfactual quality metrics
    fic.stats             statistical tests (McNemar, Wilcoxon, Holm)
    fic.visualization     figures for the report
"""

__version__ = "0.1.0"
