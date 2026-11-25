"""Fast unit tests (no dataset download). Run: pytest -q"""

import numpy as np
import torch

from fic.counterfactuals import (ConvAutoencoder, PrototypeBank, prototype_counterfactual,
                                 second_best_target, wachter_counterfactual)
from fic.evaluation import counterfactual_metrics, wilson_ci
from fic.models import LogisticRegressionTorch, SimpleCNN, SklearnModel
from fic.stats import holm_correction, mcnemar_test, rank_biserial, wilcoxon_test
from fic.xai import grad_cam, gradient_saliency


def test_cnn_shapes():
    for c, s in [(1, 28), (3, 32), (3, 64)]:
        out = SimpleCNN(c, 10)(torch.rand(2, c, s, s))
        assert out.shape == (2, 10)


def test_autoencoder_shapes():
    for c, s in [(1, 28), (3, 32)]:
        ae = ConvAutoencoder(c, s, 16)
        x = torch.rand(3, c, s, s)
        assert ae(x).shape == x.shape
        assert ae.encode(x).shape == (3, 16)


def test_logistic_regression_torch_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.random((200, 1, 28, 28)).astype("float32")
    y = rng.integers(0, 3, 200)
    sk = SklearnModel("logistic_regression", {"max_iter": 200}).fit(X, y)
    t = LogisticRegressionTorch(sk)
    p_torch = torch.softmax(t(torch.tensor(X)), 1).detach().numpy()
    assert np.allclose(p_torch, sk.predict_proba(X), atol=1e-4)


def test_counterfactual_methods_run_and_stay_in_range():
    torch.manual_seed(0)
    model = SimpleCNN(1, 10).eval()
    ae = ConvAutoencoder(1, 28, 8).eval()
    x = torch.rand(4, 1, 28, 28)
    target = second_best_target(torch.softmax(model(x), 1))
    cf = wachter_counterfactual(model, x, target, steps=30)
    assert cf.x_cf.min() >= 0 and cf.x_cf.max() <= 1
    bank = PrototypeBank(ae.encode(torch.rand(50, 1, 28, 28)).detach(), torch.arange(50) % 10)
    cf2 = prototype_counterfactual(model, ae, bank, x, target, steps=30, k_neighbours=3)
    df = counterfactual_metrics(cf2, ae)
    assert len(df) == 4 and {"valid", "l1", "sparsity", "plausibility_ae"} <= set(df.columns)


def test_resnet_wrapper_keeps_original_resolution():
    from fic.models import ResNet18Classifier
    net = ResNet18Classifier(1, 10, pretrained=False, input_size=64).eval()
    x = torch.rand(2, 1, 28, 28)
    assert net(x).shape == (2, 10)
    assert grad_cam(net, x).shape == (2, 28, 28)   # map comes back at 28 x 28


def test_blackbox_counterfactual_on_random_forest():
    from fic.counterfactuals import blackbox_counterfactual
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.random((300, 1, 28, 28)), dtype=torch.float32)
    X[150:] *= 0.3                                    # two easy-to-separate groups
    y = np.array([0] * 150 + [1] * 150)
    rf = SklearnModel("random_forest", {"n_estimators": 20}).fit(X, y)
    ae = ConvAutoencoder(1, 28, 8).eval()
    with torch.no_grad():
        bank = PrototypeBank(ae.encode(X), torch.tensor(rf.predict(X)), X)
    x = X[:3]
    cf = blackbox_counterfactual(rf, ae, bank, x, torch.ones(3, dtype=torch.long), max_iter=5)
    assert cf.valid.all()


def test_xai_maps():
    model = SimpleCNN(1, 10).eval()
    x = torch.rand(2, 1, 28, 28)
    assert grad_cam(model, x).shape == (2, 28, 28)
    assert gradient_saliency(model, x).shape == (2, 28, 28)


def test_stats():
    a = np.array([1] * 80 + [0] * 20, bool)
    b = np.array([1] * 60 + [0] * 40, bool)
    assert mcnemar_test(a, b)["p"] < 0.05
    x = np.arange(100, dtype=float)
    assert rank_biserial(x + 1, x) == 1.0
    assert wilcoxon_test(x + np.random.default_rng(0).random(100), x)["p"] < 0.05
    lo, hi = wilson_ci(90, 100)
    assert lo < 0.9 < hi
    import pandas as pd
    res = holm_correction(pd.DataFrame({"p": [0.01, 0.04, 0.5]}))
    assert list(res.significant) == [True, False, False]
