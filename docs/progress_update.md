# Progress update for supervisor meeting (19 Nov 2026)

## 1. What I have done
* Set up the full project in Python (PyTorch + scikit-learn), following your plan:
  Fashion Image Classification, small and big dataset, self-built CNN, ResNet-18,
  classical ML (Logistic Regression, Random Forest), local and global XAI, and
  counterfactual explanations.
* All 8 steps of the pipeline run end to end (data, training, evaluation, XAI,
  counterfactuals, statistics), with unit tests.

## 2. First results (Fashion-MNIST, 1 seed, short training)

These are early numbers from a short run (CNN trained for only 6 epochs, 100 explained images).
The final experiment will use 5 seeds and 1,000 images.

**Test accuracy (10,000 test images)**

| Model | Accuracy | 95% CI |
|---|---|---|
| Logistic Regression | 84.3% | 83.6 to 85.0 |
| Random Forest | 87.5% | 86.8 to 88.1 |
| Self-built CNN (6 epochs) | 93.0% | 92.5 to 93.5 |
| ResNet-18 | not yet run (needs a GPU) | |

All pairs of models differ significantly (McNemar, Holm-corrected, p < 0.001).

**Counterfactual quality (100 images, medians)**

| Model | Method | Validity | L1 change | Pixels changed | Plausibility (AE error) |
|---|---|---|---|---|---|
| CNN | Wachter (A) | 100% | 19.3 | 543 | 0.009 |
| CNN | Prototype (B) | 100% | 45.1 | 221 | 0.022 |
| Logistic Regression | Wachter (A) | 100% | 12.0 | 518 | 0.008 |
| Logistic Regression | Prototype (B) | 100% | 45.5 | 250 | 0.022 |
| Random Forest | Black-box prototype | 100% | 48.8 | 326 | 0.007 |

**First observations**
* Wachter changes almost every pixel by a tiny amount. The changes look like noise
  (similar to adversarial examples), which matches Pawelczyk et al. (2022).
* The prototype method changes fewer than half as many pixels, and the changes sit
  on the clothing (for example, it cuts the top of an ankle boot to make it a sneaker).
* But prototype changes are larger, and the autoencoder plausibility score prefers
  the tiny Wachter changes. This supports Delaney et al. (2023): the "smallest change"
  and a "meaningful change" are not the same thing.

## 3. Questions for you
1. Should the big dataset be a fashion dataset (for example Fashion Product Images)
   instead of CIFAR-10?
2. Should Random Forest and Logistic Regression be explained with counterfactuals too,
   or only used as accuracy baselines?
3. Do I need to update the approval form for the classical ML models?
4. Is the autoencoder error a good enough plausibility measure, or should I add another
   one (for example IM1)?

## 4. Next steps
* Train ResNet-18 on a GPU (Google Colab).
* Confirm and add the big dataset.
* Full run: 5 seeds, 1,000 images per dataset.
* Restructure the literature review: ML for FIC, then XAI, then counterfactuals.
