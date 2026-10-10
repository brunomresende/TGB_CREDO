"""Orquestracao do benchmark; os metodos permanecem em modulos independentes."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .credal.gamma import AdaptiveGamma, FixedGamma
from .data.preprocessing import load_processed_dataset
from .data.splitting import split_train_calibration_test
from .evaluation.aggregation import aggregate_metrics
from .evaluation.metrics import evaluate_intervals
from .evaluation.outliers import detect_outlier_groups
from .methods.cqr import CQRRegressor
from .methods.credo import CredalRegressor
from .methods.epicscore import EPICScoreRegressor
from .methods.uacqr import UACQRRegressor
from .models.catboost_quantile import CatBoostQuantileRegressor
from .models.quantile_neural_network import QuantileNeuralNetwork
from .utils.random import seed_everything


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _qnn_from_config(config: dict, input_size: int, alpha: float, seed: int, n: int, dataset: str):
    values = config["qnn"]
    if dataset == "WEC":
        batch_size = values.get("batch_size_wec", values.get("batch_size", 250))
    elif n >= 10_000:
        batch_size = values.get("batch_size_large", values.get("batch_size", 120))
    else:
        batch_size = values.get("batch_size_small", values.get("batch_size", 32))
    return QuantileNeuralNetwork(
        input_size=input_size,
        alpha=alpha,
        hidden_layers=tuple(values["hidden_layers"]),
        dropout=values["dropout"],
        epochs=values["epochs"],
        patience=values["patience"],
        learning_rate=values["learning_rate"],
        weight_decay=values["weight_decay"],
        scheduler_step=values.get("scheduler_step", 10),
        scheduler_gamma=values.get("scheduler_gamma", 0.99),
        batch_size=batch_size,
        random_state=seed,
    )


def _evaluate(method: str, intervals, split, groups, alpha: float, **extra):
    metrics = evaluate_intervals(
        split.y_test,
        intervals,
        alpha=alpha,
        outlier_indices=groups.outliers,
        inlier_indices=groups.ilr_inliers,
    )
    return {"method": method, **metrics, **extra}


def run_single_repetition(
    X,
    y,
    *,
    dataset: str,
    repeat: int,
    seed: int,
    config: dict,
    include_catboost: bool = True,
) -> list[dict]:
    alpha = float(config["alpha"])
    split_cfg = config["split"]
    split = split_train_calibration_test(
        X,
        y,
        train_fraction=split_cfg["train"],
        calibration_fraction=split_cfg["calibration"],
        test_fraction=split_cfg["test"],
        random_state=seed,
    )
    outlier_cfg = config["outliers"]
    groups = detect_outlier_groups(split.X_test, random_state=seed, **outlier_cfg)
    common = {"dataset": dataset, "repeat": repeat, "seed": seed}
    rows: list[dict] = []

    # Uma unica QNN compartilhada: comparacao controlada.
    qnn = _qnn_from_config(config, split.X_train.shape[1], alpha, seed, len(X), dataset)
    start = time.perf_counter()
    qnn.fit(split.X_train, split.y_train)
    qnn_fit_time = time.perf_counter() - start

    qnn_methods = {
        "CQR-QNN": CQRRegressor(qnn, alpha, relative=False, base_model_is_fitted=True),
        "CQR-r-QNN": CQRRegressor(qnn, alpha, relative=True, base_model_is_fitted=True),
    }
    for name, method in qnn_methods.items():
        method.calibrate(split.X_calibration, split.y_calibration)
        start = time.perf_counter()
        intervals = method.predict(split.X_test)
        predict_time = time.perf_counter() - start
        rows.append(
            {**common, **_evaluate(name, intervals, split, groups, alpha), "backend": "QNN",
             "shared_fit_seconds": qnn_fit_time, "predict_seconds": predict_time}
        )

    gamma_cfg = config["gamma"]
    credo_methods = {
        "CREDO-QNN": CredalRegressor(
            qnn,
            FixedGamma(gamma_cfg["fixed"]),
            alpha,
            config["qnn"]["n_mc_samples"],
            True,
            seed,
        ),
        "CREDO-QNN-Adaptive": CredalRegressor(
            qnn,
            AdaptiveGamma(
                gamma_min=gamma_cfg["min"],
                gamma_max=gamma_cfg["max"],
                tau=gamma_cfg["tau"],
                epsilon=gamma_cfg.get("epsilon", 1e-6),
                neighbor_constant=gamma_cfg.get("neighbor_constant", 6.672),
            ),
            alpha,
            config["qnn"]["n_mc_samples"],
            True,
            seed,
        ),
    }
    for name, method in credo_methods.items():
        method.fit(split.X_train, split.y_train)
        method.calibrate(split.X_calibration, split.y_calibration)
        start = time.perf_counter()
        intervals, decomposition = method.predict(split.X_test, return_decomposition=True)
        predict_time = time.perf_counter() - start
        out_epi = float(np.mean(decomposition["epistemic_fraction"][groups.outliers]))
        in_epi = float(np.mean(decomposition["epistemic_fraction"][groups.decomposition_inliers]))
        rows.append(
            {
                **common,
                **_evaluate(name, intervals, split, groups, alpha),
                "backend": "QNN",
                "shared_fit_seconds": qnn_fit_time,
                "predict_seconds": predict_time,
                "epistemic_fraction_outliers": out_epi,
                "epistemic_fraction_inliers": in_epi,
            }
        )

    if include_catboost:
        cat_cfg = config.get("catboost", {})
        catboost = CatBoostQuantileRegressor(alpha=alpha, random_state=seed, **cat_cfg)
        start = time.perf_counter()
        catboost.fit(split.X_train, split.y_train)
        cat_fit_time = time.perf_counter() - start

        for name, relative in (("CQR-CatBoost", False), ("CQR-r-CatBoost", True)):
            method = CQRRegressor(catboost, alpha, relative, True)
            method.calibrate(split.X_calibration, split.y_calibration)
            intervals = method.predict(split.X_test)
            rows.append(
                {**common, **_evaluate(name, intervals, split, groups, alpha),
                 "backend": "CatBoost", "shared_fit_seconds": cat_fit_time}
            )

        uacqr = UACQRRegressor(
            catboost,
            alpha=alpha,
            ensemble_size=config["uacqr"]["ensemble_size"],
            base_model_is_fitted=True,
            random_state=seed,
        ).fit(split.X_train, split.y_train)
        uacqr.calibrate(split.X_calibration, split.y_calibration)
        for name, intervals in uacqr.predict(split.X_test).items():
            rows.append(
                {**common, **_evaluate(f"{name}-CatBoost", intervals, split, groups, alpha),
                 "backend": "CatBoost", "shared_fit_seconds": cat_fit_time}
            )

        if "epicscore" in config:
            epic_cfg = dict(config["epicscore"])
            epic_cfg["hidden_layers"] = tuple(epic_cfg["hidden_layers"])
            if dataset == "WEC":
                epic_batch = epic_cfg.pop("batch_size_wec", epic_cfg.get("batch_size", 250))
            elif len(X) >= 10_000:
                epic_batch = epic_cfg.pop("batch_size_large", epic_cfg.get("batch_size", 130))
            else:
                epic_batch = epic_cfg.pop("batch_size_small", epic_cfg.get("batch_size", 40))
            epic_cfg.pop("batch_size_large", None)
            epic_cfg.pop("batch_size_wec", None)
            epic_cfg["batch_size"] = epic_batch
            epic = EPICScoreRegressor(
                catboost,
                alpha=alpha,
                base_model_is_fitted=True,
                random_state=seed,
                **epic_cfg,
            ).fit(split.X_train, split.y_train)
            epic.calibrate(split.X_calibration, split.y_calibration)
            rows.append(
                {**common, **_evaluate("EPICScore-CatBoost-MDN", epic.predict(split.X_test),
                                      split, groups, alpha),
                 "backend": "CatBoost+MDN", "shared_fit_seconds": cat_fit_time}
            )
    return rows


def run_benchmark(
    config_path: str | Path,
    *,
    data_dir: str | Path = "data/processed",
    output_dir: str | Path = "results",
    include_catboost: bool = True,
) -> Path:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "raw" / f"{config['experiment_name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    rng = np.random.default_rng(config["seed"])
    seeds = rng.integers(0, np.iinfo(np.int32).max, config["n_repetitions"]).tolist()
    (run_dir / "seeds.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    all_rows = []
    for dataset in config["datasets"]:
        X, y = load_processed_dataset(data_dir / f"{dataset}.csv")
        for repeat, seed in enumerate(seeds):
            rows = run_single_repetition(
                X,
                y,
                dataset=dataset,
                repeat=repeat,
                seed=int(seed),
                config=config,
                include_catboost=include_catboost,
            )
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(run_dir / "metrics.csv", index=False)
    summary = aggregate_metrics(pd.DataFrame(all_rows))
    summary.to_csv(run_dir / "summary.csv", index=False)
    return run_dir
