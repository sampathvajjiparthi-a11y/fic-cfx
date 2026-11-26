# FIC-CFX: Counterfactual XAI for Fashion Image Classification

KF7029 MSc Project, Northumbria University
Student: Sai Sampath Srinivas Vajjiparthi (w25034093) · Supervisor: Hua Mao

**Question:** What would need to change in a fashion image for a model to predict a
different class, and how does the answer depend on the explanation method and the
type of model?

The project compares:
* **Models:** Logistic Regression, Random Forest, a self-built CNN, and a pretrained ResNet-18
* **Datasets:** a small one (Fashion-MNIST, grey 28 x 28) and a big one (real colour product photos, e.g. Kaggle Fashion Product Images)
* **XAI:** local (Grad-CAM, saliency) and global (weight maps, importance, class-average Grad-CAM) explanations
* **Counterfactuals for ALL four models:** gradient-based (Wachter et al., 2018), prototype-guided (Van Looveren and Klaise, 2021), and a model-agnostic black-box method that runs on every model (the only one possible for Random Forest)

See [`docs/architecture.md`](docs/architecture.md) for the full design.

---

## 1. Setup (VS Code)

1. Install **Python 3.11** and **VS Code**.
2. Open this folder in VS Code (`File > Open Folder`). Install the recommended extensions when asked.
3. Open a terminal in VS Code (`` Ctrl+` ``) and create a virtual environment:

   **Mac / Linux**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

   **Windows**
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

4. Select the interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv`.
5. Check everything works:
   ```bash
   pytest -q
   python scripts/run_pipeline.py --dataset small --quick
   ```
   The quick run uses tiny subsets and 1 epoch, so it finishes in a few minutes.
   It downloads Fashion-MNIST (about 30 MB) and the ResNet-18 weights (about 45 MB) the first time.

> **GPU:** a GPU is used automatically when available (NVIDIA CUDA or Apple Silicon `mps`).
> ResNet-18 at 224 × 224 is slow on CPU. If your laptop has no GPU, run step 3b in Google Colab.

## 2. Run the experiment

You can run each step from the VS Code **Run and Debug** panel (configurations 0 to 8),
or from the terminal:

| Step | Command | What it does |
|---|---|---|
| 0 | `python scripts/00_prepare_fashion_products.py --source <folder>` | prepare the big dataset (once) |
| 1 | `python scripts/01_check_data.py --dataset small` | load data, save sample images |
| 2 | `python scripts/02_train_classical.py --dataset small` | train Logistic Regression and Random Forest |
| 3a | `python scripts/03_train_deep.py --dataset small --model cnn` | train the self-built CNN |
| 3b | `python scripts/03_train_deep.py --dataset small --model resnet18` | fine-tune ResNet-18 |
| 4 | `python scripts/04_train_autoencoder.py --dataset small` | autoencoder for plausibility and prototypes |
| 5 | `python scripts/05_evaluate_models.py --dataset small` | **RQ1**: test accuracy + McNemar tests |
| 6 | `python scripts/06_explain_xai.py --dataset small` | local and global XAI maps |
| 7 | `python scripts/07_generate_counterfactuals.py --dataset small --model cnn --method wachter` | counterfactuals for one model and method (`wachter`, `prototype` or `blackbox`) |
| 8 | `python scripts/08_analyse_counterfactuals.py --dataset small` | **RQ2, RQ3**: statistics |

Or run everything at once:
```bash
python scripts/run_pipeline.py --dataset small
python scripts/run_pipeline.py --dataset big
```

Useful options for every script:
* `--seeds 0` run one seed instead of all five
* `--quick` tiny smoke test
* `--config configs/first_results.yaml` short run for a first look at the results

## 3. The big dataset (step 0)

Your supervisor asked for two datasets. The second one is a real-photo fashion dataset.

1. Download **Fashion Product Images (Small)** from Kaggle and unzip it anywhere.
2. Prepare it (keeps the 10 most common classes, max 2,000 images each, resized to 64 x 64):
   ```bash
   python scripts/00_prepare_fashion_products.py --source ~/Downloads/fashion-product-images-small
   ```
   Pick your own classes with `--classes Tshirts Shirts Watches ...` and change the cap with `--max-per-class`.
3. Check it: `python scripts/01_check_data.py --dataset big`
4. Run everything: `python scripts/run_pipeline.py --dataset big`

`configs/default.yaml` already points `datasets.big` at `data/raw/fashion_products`.
To go back to CIFAR-10, set `datasets.big.name: cifar10`.

## 3b. ResNet-18 on a free GPU (Google Colab)

Open `notebooks/02_colab_gpu_run.ipynb` in Google Colab, choose **Runtime > Change runtime type > T4 GPU**,
and run the cells. It clones this repository, trains ResNet-18 and saves the results to your Google Drive.

## 4. Where the results go

| Folder | Content |
|---|---|
| `outputs/metrics/<dataset>/` | `accuracy_summary.csv`, `h1_mcnemar.csv`, `cf_summary.csv`, `cf_statistics.csv` |
| `outputs/figures/<dataset>/` | sample images, training curves, confusion matrices, XAI maps, counterfactual grids |
| `outputs/counterfactuals/<dataset>/` | one CSV per model and method (one row per image) |
| `outputs/models/<dataset>/` | trained models |

## 5. References

* He, K. et al. (2016) Deep residual learning for image recognition. *CVPR*.
* Selvaraju, R.R. et al. (2017) Grad-CAM. *ICCV*.
* Van Looveren, A. and Klaise, J. (2021) Interpretable counterfactual explanations guided by prototypes. *ECML PKDD*.
* Wachter, S., Mittelstadt, B. and Russell, C. (2018) Counterfactual explanations without opening the black box. *Harvard JOLT*.
* Xiao, H., Rasul, K. and Vollgraf, R. (2017) Fashion-MNIST. *arXiv:1708.07747*.
