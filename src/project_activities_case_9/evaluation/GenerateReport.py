from __future__ import annotations

import csv
import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.multiclass import unique_labels


# Поддерживаемые названия метрик и их внутренние обозначения.

_CLASSIFICATION_METRICS = {
    "accuracy": "accuracy", "accuracy_score": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "balanced_accuracy_score": "balanced_accuracy",
    "precision": "precision", "precision_score": "precision",
    "recall": "recall", "recall_score": "recall",
    "f1": "f1", "f1_score": "f1",
    "jaccard": "jaccard", "jaccard_score": "jaccard",
    "matthews_corrcoef": "matthews_corrcoef",
    "cohen_kappa": "cohen_kappa", "cohen_kappa_score": "cohen_kappa",
}

_REGRESSION_METRICS = {
    "mae": "mae", "mean_absolute_error": "mae",
    "mse": "mse", "mean_squared_error": "mse",
    "rmse": "rmse", "root_mean_squared_error": "rmse",
    "r2": "r2", "r2_score": "r2",
    "mape": "mape", "mean_absolute_percentage_error": "mape",
    "median_absolute_error": "median_absolute_error",
    "explained_variance": "explained_variance", "max_error": "max_error",
}

_METRIC_TITLES = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision", "recall": "Recall", "f1": "F1-score",
    "jaccard": "Jaccard score", "matthews_corrcoef": "Коэффициент Мэтьюса",
    "cohen_kappa": "Каппа Коэна", "mae": "MAE", "mse": "MSE",
    "rmse": "RMSE", "r2": "R²", "mape": "MAPE",
    "median_absolute_error": "Медианная абсолютная ошибка",
    "explained_variance": "Объяснённая дисперсия", "max_error": "Максимальная ошибка",
}

_DEFAULT_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1"],
    "regression": ["mae", "rmse", "r2"],
}

# Подготовка конфигурации и входных данных

def _plain_config(config: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
    except (ImportError, TypeError, ValueError):
        pass
    return asdict(config) if is_dataclass(config) and not isinstance(config, type) else config

def _cfg_get(config: Any, *paths: str, default: Any = None) -> Any:
    for path in paths:
        value = config
        for part in path.split("."):
            if isinstance(value, Mapping):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                break
        if value is not None:
            return value
    return default

def _normalise_task(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("type", value.get("name", value.get("_target_")))
    text = str(value).strip().lower()
    if any(word in text for word in ("classif", "classifier", "классиф")):
        return "classification"
    if any(word in text for word in ("regress", "regressor", "регресс")):
        return "regression"
    return None

def _metric_specs(raw_metrics: Any, task: str) -> list[dict[str, Any]]:
    if raw_metrics is None:
        raw_metrics = _DEFAULT_METRICS[task]
    elif isinstance(raw_metrics, str):
        raw_metrics = [raw_metrics]

    if isinstance(raw_metrics, Mapping):
        if any(key in raw_metrics for key in ("name", "metric", "type")):
            raw_metrics = [raw_metrics]
        else:
            raw_metrics = [
                {"name": name, **(dict(options) if isinstance(options, Mapping) else {})}
                for name, options in raw_metrics.items()
            ]

    if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
        raise TypeError("Список метрик должен быть строкой, списком или словарём")

    result = []
    for item in raw_metrics:
        if isinstance(item, str):
            result.append({"name": item})
        elif isinstance(item, Mapping):
            name = item.get("name", item.get("metric", item.get("type")))
            if not name:
                raise ValueError(f"В настройке метрики отсутствует название: {item!r}")
            result.append({**dict(item), "name": str(name)})
        else:
            raise TypeError(f"Неизвестный формат настройки метрики: {item!r}")
    if not result:
        raise ValueError("В конфигурации не задано ни одной метрики")
    return result

def _detect_task(y_true: np.ndarray, model: Any, configured: Any,
                 metric_specs: list[dict[str, Any]]) -> str:
    task = _normalise_task(configured)
    if task:
        return task

    if model is not None:
        try:
            if is_classifier(model) or getattr(model, "_estimator_type", None) == "classifier":
                return "classification"
            if is_regressor(model) or getattr(model, "_estimator_type", None) == "regressor":
                return "regression"
        except (TypeError, AttributeError):
            pass

    names = {str(item["name"]).strip().lower() for item in metric_specs}
    has_classification = bool(names & set(_CLASSIFICATION_METRICS))
    has_regression = bool(names & set(_REGRESSION_METRICS))
    if has_classification != has_regression:
        return "classification" if has_classification else "regression"

    if y_true.dtype.kind in "OUSb":
        return "classification"
    if y_true.dtype.kind in "iu" and np.unique(y_true).size <= max(20, int(math.sqrt(y_true.size))):
        return "classification"
    return "regression"

def _contains_missing(values: np.ndarray) -> bool:
    try:
        return bool(np.any(np.asarray(values, dtype=float) != np.asarray(values, dtype=float)))
    except (TypeError, ValueError):
        for value in values.tolist():
            if value is None:
                return True
            try:
                if bool(np.isnan(value)):
                    return True
            except (TypeError, ValueError):
                continue
    return False

def _safe_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None

def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)

def _json_safe(value: Any) -> Any:
    if value is None or value is np.ma.masked:
        return None
    if isinstance(value, np.ma.MaskedArray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)

# Расчёт метрик и построение графиков

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str,
                       specs: list[dict[str, Any]], average: str
                       ) -> tuple[dict[str, float | None], dict | None, np.ndarray | None]:
    values: dict[str, float | None] = {}
    class_report = None
    confusion = None

    if task == "classification":
        labels = unique_labels(y_true, y_pred)
        confusion = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels)
        class_report = sk_metrics.classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        functions = {
            "accuracy": sk_metrics.accuracy_score,
            "balanced_accuracy": sk_metrics.balanced_accuracy_score,
            "precision": sk_metrics.precision_score,
            "recall": sk_metrics.recall_score,
            "f1": sk_metrics.f1_score,
            "jaccard": sk_metrics.jaccard_score,
            "matthews_corrcoef": sk_metrics.matthews_corrcoef,
            "cohen_kappa": sk_metrics.cohen_kappa_score,
        }
        for spec in specs:
            raw_name = str(spec["name"]).strip().lower()
            name = _CLASSIFICATION_METRICS.get(raw_name)
            if name is None:
                available = ", ".join(sorted(set(_CLASSIFICATION_METRICS.values())))
                raise ValueError(f"Метрика {raw_name!r} не поддерживается. Доступно: {available}")
            kwargs = {}
            if name in {"precision", "recall", "f1", "jaccard"}:
                kwargs = {"average": spec.get("average", average), "zero_division": 0}
            values[name] = _safe_float(functions[name](y_true, y_pred, **kwargs))
    else:
        true_values = np.asarray(y_true, dtype=float)
        predicted_values = np.asarray(y_pred, dtype=float)
        functions = {
            "mae": lambda: sk_metrics.mean_absolute_error(true_values, predicted_values),
            "mse": lambda: sk_metrics.mean_squared_error(true_values, predicted_values),
            "rmse": lambda: np.sqrt(sk_metrics.mean_squared_error(true_values, predicted_values)),
            "r2": lambda: sk_metrics.r2_score(true_values, predicted_values),
            "mape": lambda: sk_metrics.mean_absolute_percentage_error(true_values, predicted_values),
            "median_absolute_error": lambda: sk_metrics.median_absolute_error(true_values, predicted_values),
            "explained_variance": lambda: sk_metrics.explained_variance_score(true_values, predicted_values),
            "max_error": lambda: sk_metrics.max_error(true_values, predicted_values),
        }
        for spec in specs:
            raw_name = str(spec["name"]).strip().lower()
            name = _REGRESSION_METRICS.get(raw_name)
            if name is None:
                available = ", ".join(sorted(set(_REGRESSION_METRICS.values())))
                raise ValueError(f"Метрика {raw_name!r} не поддерживается. Доступно: {available}")
            values[name] = _safe_float(functions[name]())

    return values, class_report, confusion

def _get_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _save_plots(output_dir: Path, task: str, y_true: np.ndarray, y_pred: np.ndarray,
                metric_values: dict[str, float | None], confusion: np.ndarray | None) -> list[str]:
    plt = _get_matplotlib()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Общий график рассчитанных метрик.

    valid = [(name, value) for name, value in metric_values.items() if value is not None]
    if valid:
        labels = [_METRIC_TITLES.get(name, name) for name, _ in valid]
        scores = [score for _, score in valid]
        fig, ax = plt.subplots(figsize=(max(7, len(valid) * 1.25), 4.5))
        bars = ax.bar(labels, scores)
        ax.set(title="Метрики качества модели", ylabel="Значение метрики")
        ax.tick_params(axis="x", labelrotation=25)
        for bar, score in zip(bars, scores):
            ax.annotate(f"{score:.4g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append("plots/metrics.png")

    if task == "classification" and confusion is not None:
        labels = [str(label) for label in unique_labels(y_true, y_pred)]
        size = min(max(6, len(labels) * 0.75), 16)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        image = ax.imshow(confusion, interpolation="nearest")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels,
               xlabel="Предсказанный класс", ylabel="Истинный класс",
               title="Матрица ошибок")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        threshold = confusion.max() / 2 if confusion.size else 0
        for row in range(confusion.shape[0]):
            for column in range(confusion.shape[1]):
                color = "white" if confusion[row, column] > threshold else "black"
                ax.text(column, row, str(confusion[row, column]), ha="center", va="center", color=color)
        fig.tight_layout()
        fig.savefig(plot_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append("plots/confusion_matrix.png")

    if task == "regression":
        true_values = np.asarray(y_true, dtype=float)
        predicted_values = np.asarray(y_pred, dtype=float)
        low = float(min(true_values.min(), predicted_values.min()))
        high = float(max(true_values.max(), predicted_values.max()))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(true_values, predicted_values, alpha=0.65)
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set(xlabel="Истинные значения", ylabel="Предсказанные значения",
               title="Истинные и предсказанные значения")
        fig.tight_layout()
        fig.savefig(plot_dir / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append("plots/actual_vs_predicted.png")

        residuals = true_values - predicted_values
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(residuals, bins="auto")
        ax.set(xlabel="Остаток (истинное - предсказанное)", ylabel="Частота",
               title="Распределение остатков")
        fig.tight_layout()
        fig.savefig(plot_dir / "residuals.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append("plots/residuals.png")

    return paths

# Обработка результатов GridSearchCV

def _grid_get(grid: Any, name: str, default: Any = None) -> Any:
    return grid.get(name, default) if isinstance(grid, Mapping) else getattr(grid, name, default)

def _grid_rows(cv_results: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if cv_results is None:
        return [], []
    if isinstance(cv_results, Sequence) and not isinstance(cv_results, (str, bytes, Mapping)):
        rows = [_json_safe(dict(row)) for row in cv_results if isinstance(row, Mapping)]
        return rows, list(dict.fromkeys(key for row in rows for key in row))
    if not isinstance(cv_results, Mapping):
        return [], []

    columns = list(cv_results)
    lengths = [len(value) for value in cv_results.values() if hasattr(value, "__len__")]
    if not lengths:
        return [], [str(column) for column in columns]
    count = max(lengths)
    rows = []
    for index in range(count):
        row = {}
        for name, values in cv_results.items():
            try:
                item = values[index]
            except (IndexError, KeyError, TypeError):
                item = values
            row[str(name)] = _json_safe(item)
        rows.append(row)
    return rows, [str(column) for column in columns]

def _extract_grid_info(grid: Any) -> dict[str, Any] | None:
    if grid is None:
        return None

    cv_results = _grid_get(grid, "cv_results_", _grid_get(grid, "cv_results"))
    if cv_results is None and isinstance(grid, Mapping) and "params" in grid:
        if any(str(key).startswith("mean_test_") for key in grid):
            cv_results = grid
    rows, columns = _grid_rows(cv_results)

    best_params = _grid_get(grid, "best_params_", _grid_get(grid, "best_params"))
    best_score = _grid_get(grid, "best_score_", _grid_get(grid, "best_score"))
    best_index = _grid_get(grid, "best_index_", _grid_get(grid, "best_index"))
    refit = _grid_get(grid, "refit")
    scoring = _grid_get(grid, "scoring")
    score_key = None

    score_columns = [key for key in columns if key.startswith("mean_test_")]
    if isinstance(refit, str) and f"mean_test_{refit}" in columns:
        score_key = f"mean_test_{refit}"
    elif "mean_test_score" in columns:
        score_key = "mean_test_score"
    elif score_columns:
        score_key = score_columns[0]

    # Если объект не предоставил индекс лучшей комбинации, определяем его по рангу.

    if best_index is None and rows and score_key:
        rank_key = score_key.replace("mean_test_", "rank_test_", 1)
        try:
            if rank_key in columns:
                best_index = min(range(len(rows)), key=lambda i: float(rows[i].get(rank_key, math.inf)))
            else:
                best_index = max(range(len(rows)), key=lambda i: float(rows[i].get(score_key, -math.inf)))
        except (TypeError, ValueError):
            best_index = None

    if best_index is not None:
        try:
            best_row = rows[int(best_index)]
            if best_score is None and score_key:
                best_score = best_row.get(score_key)
            if best_params is None:
                best_params = best_row.get("params")
        except (IndexError, TypeError, ValueError):
            pass

    if callable(refit):
        refit = getattr(refit, "__name__", repr(refit))

    best_estimator = _grid_get(grid, "best_estimator_", _grid_get(grid, "best_estimator"))
    info = {
        "available": True,
        "search_class": type(grid).__name__,
        "best_params": _json_safe(best_params),
        "best_score": _json_safe(best_score),
        "best_index": _json_safe(best_index),
        "scoring": _json_safe(scoring),
        "refit": _json_safe(refit),
        "cv": _json_safe(_grid_get(grid, "cv")),
        "n_splits": _json_safe(_grid_get(grid, "n_splits_", _grid_get(grid, "n_splits"))),
        "best_estimator_class": type(best_estimator).__name__ if best_estimator is not None else None,
        "candidates_count": len(rows),
        "score_key": score_key,
        "cv_results": rows,
    }
    if best_params is None and best_score is None and not rows:
        info["warning"] = "Результаты GridSearch не найдены. Возможно, объект ещё не обучен."
    return info


def _ranked_grid_rows(info: dict[str, Any], limit: int | None = None) -> list[tuple[int, dict[str, Any]]]:
    rows = info.get("cv_results", [])
    score_key = info.get("score_key")
    if not rows or not score_key:
        return []

    rank_key = score_key.replace("mean_test_", "rank_test_", 1)

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[float, int]:
        index, row = item
        if rank_key in row and row[rank_key] is not None:
            try:
                return float(row[rank_key]), index
            except (TypeError, ValueError):
                pass
        try:
            return -float(row[score_key]), index
        except (KeyError, TypeError, ValueError):
            return float("inf"), index

    ordered = sorted(enumerate(rows), key=sort_key)
    if limit is not None:
        ordered = ordered[:limit]
    return [(index + 1, row) for index, row in ordered]

def _save_grid_artifacts(output_dir: Path, info: dict[str, Any] | None) -> str | None:
    if info is None:
        return None

    rows = info.get("cv_results", [])
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        if "params" not in columns:
            columns.insert(0, "params")
        with (output_dir / "grid_search.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                })

    score_key = info.get("score_key")
    ranked = _ranked_grid_rows(info, limit=20)
    scored = []
    for rank, row in ranked:
        try:
            score = float(row[score_key])
            if math.isfinite(score):
                scored.append((rank, score))
        except (KeyError, TypeError, ValueError):
            continue

    plot_path = None
    if scored:
        plt = _get_matplotlib()
        labels = [f"#{rank}" for rank, _ in scored]
        scores = [score for _, score in scored]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.48), 4.8))
        bars = ax.bar(labels, scores)
        ax.set(title="Результаты GridSearch", xlabel="Комбинация (по рангу)", ylabel=score_key)
        ax.tick_params(axis="x", labelrotation=45)
        for bar, score in zip(bars, scores):
            ax.annotate(f"{score:.4g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        fig.tight_layout()
        path = output_dir / "plots" / "grid_search_scores.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plot_path = "plots/grid_search_scores.png"

    info["grid_plot"] = plot_path
    with (output_dir / "grid_search.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2, default=_json_default)
    return plot_path

# Формирование отчётов

def _grid_summary(info: dict[str, Any] | None) -> list[str]:
    if not info:
        return []
    lines = ["", "## Результаты GridSearch", ""]
    if info.get("warning"):
        lines.append(f"> {info['warning']}")
    if info.get("best_params") is not None:
        lines.extend(["**Лучшие параметры:**", "", "```json",
                      json.dumps(info["best_params"], ensure_ascii=False, indent=2), "```", ""])
    for title, key in (
        ("Лучшая оценка", "best_score"),
        ("Количество проверенных комбинаций", "candidates_count"),
        ("Количество фолдов", "n_splits"),
        ("Метрика GridSearch (scoring)", "scoring"),
    ):
        if info.get(key) is not None:
            lines.append(f"- **{title}:** {info[key]}")
    if isinstance(info.get("refit"), str):
        lines.append(f"- **Метрика выбора лучшей модели (refit):** {info['refit']}")
    elif info.get("refit") is False:
        lines.append("- **Переобучение лучшей модели после поиска:** отключено")

    rows = _ranked_grid_rows(info, limit=10)
    if rows:
        score_key = info["score_key"]
        std_key = score_key.replace("mean_test_", "std_test_", 1)
        lines.extend(["", "### Лучшие комбинации параметров", "",
                      "| Ранг | Средняя оценка | Стандартное отклонение | Параметры |",
                      "|---:|---:|---:|---|"])
        rank_key = score_key.replace("mean_test_", "rank_test_", 1)
        for candidate, row in rows:
            rank = row.get(rank_key, candidate)
            params = json.dumps(row.get("params", {}), ensure_ascii=False).replace("|", "\\|")
            lines.append(f"| {rank} | {row.get(score_key, '—')} "
                         f"| {row.get(std_key, '—')} | `{params}` |")
    if info.get("grid_plot"):
        lines.extend(["", "### Сравнение комбинаций параметров", "",
                      f"![Результаты GridSearch]({info['grid_plot']})", ""])
    return lines

def _build_markdown_report(data: dict[str, Any]) -> str:
    task_title = "Классификация" if data["task_type"] == "classification" else "Регрессия"
    lines = [
        "# Отчёт о качестве модели", "",
        f"- **Модель:** {data['model_name']}",
        f"- **Тип задачи:** {task_title}",
        f"- **Количество объектов:** {data['sample_count']}",
        f"- **Дата формирования (UTC):** {data['generated_at']}",
        "", "## Метрики качества", "", "| Метрика | Значение |", "|---|---:|",
    ]
    for name, value in data["metrics"].items():
        displayed = "не определено" if value is None else f"{value:.6g}"
        lines.append(f"| {_METRIC_TITLES.get(name, name)} | {displayed} |")

    class_report = data.get("classification_report")
    if class_report:
        lines.extend(["", "## Метрики по классам", "",
                      "| Класс | Precision | Recall | F1-score | Support |",
                      "|---|---:|---:|---:|---:|"])
        for label, values in class_report.items():
            if isinstance(values, Mapping) and "precision" in values:
                lines.append(f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} "
                             f"| {values['f1-score']:.4f} | {values['support']} |")

    if data["plots"]:
        lines.extend(["", "## Графики", ""])
        for path in data["plots"]:
            caption = Path(path).stem.replace("_", " ").title()
            lines.extend([f"### {caption}", "", f"![{caption}]({path})", ""])
    lines.extend(_grid_summary(data.get("grid_search")))
    return "\n".join(lines).rstrip() + "\n"

def _grid_html(info: dict[str, Any] | None) -> str:
    if not info:
        return ""
    parts = ["<h2>Результаты GridSearch</h2>"]
    if info.get("warning"):
        parts.append(f"<p>{html.escape(str(info['warning']))}</p>")
    if info.get("best_params") is not None:
        params = json.dumps(info["best_params"], ensure_ascii=False, indent=2)
        parts.append(f"<h3>Лучшие параметры</h3><pre>{html.escape(params)}</pre>")

    labels = {
        "best_score": "Лучшая оценка", "candidates_count": "Проверено комбинаций",
        "n_splits": "Количество фолдов", "scoring": "Метрика GridSearch (scoring)",
    }
    for key, title in labels.items():
        if info.get(key) is not None:
            parts.append(f"<p><b>{title}:</b> {html.escape(str(info[key]))}</p>")
    if isinstance(info.get("refit"), str):
        parts.append(f"<p><b>Метрика выбора лучшей модели:</b> {html.escape(info['refit'])}</p>")
    elif info.get("refit") is False:
        parts.append("<p><b>Переобучение лучшей модели после поиска:</b> отключено</p>")

    rows = _ranked_grid_rows(info, limit=10)
    if rows:
        score_key = info["score_key"]
        std_key = score_key.replace("mean_test_", "std_test_", 1)
        table_rows = []
        rank_key = score_key.replace("mean_test_", "rank_test_", 1)
        for candidate, row in rows:
            params = json.dumps(row.get("params", {}), ensure_ascii=False)
            values = (row.get(rank_key, candidate), row.get(score_key, "—"),
                      row.get(std_key, "—"), html.escape(params))
            table_rows.append(f"<tr><td>{html.escape(str(values[0]))}</td>"
                              f"<td>{html.escape(str(values[1]))}</td>"
                              f"<td>{html.escape(str(values[2]))}</td>"
                              f"<td><code>{values[3]}</code></td></tr>")
        parts.append("<h3>Лучшие комбинации параметров</h3><table><thead><tr>"
                     "<th>Ранг</th><th>Средняя оценка</th><th>Стандартное отклонение</th>"
                     f"<th>Параметры</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>")
    if info.get("grid_plot"):
        path = html.escape(str(info["grid_plot"]))
        parts.append(f'<figure><img src="{path}" alt="Результаты GridSearch">'
                     "<figcaption>Сравнение результатов GridSearch</figcaption></figure>")
    return "\n".join(parts)

def _build_html_report(data: dict[str, Any]) -> str:
    metrics_rows = "".join(
        f"<tr><td>{html.escape(_METRIC_TITLES.get(name, name))}</td>"
        f"<td>{'не определено' if value is None else f'{value:.6g}'}</td></tr>"
        for name, value in data["metrics"].items()
    )
    plots = "".join(
        f'<figure><img src="{html.escape(path)}" alt="{html.escape(Path(path).stem)}">'
        f"<figcaption>{html.escape(Path(path).stem.replace('_', ' ').title())}</figcaption></figure>"
        for path in data["plots"]
    )
    class_rows = []
    for label, values in (data.get("classification_report") or {}).items():
        if isinstance(values, Mapping) and "precision" in values:
            class_rows.append(
                f"<tr><td>{html.escape(str(label))}</td>"
                f"<td>{values['precision']:.4f}</td><td>{values['recall']:.4f}</td>"
                f"<td>{values['f1-score']:.4f}</td><td>{values['support']}</td></tr>"
            )
    classes = ""
    if class_rows:
        classes = ("<h2>Метрики по классам</h2><table><thead><tr>"
                   "<th>Класс</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th>"
                   f"</tr></thead><tbody>{''.join(class_rows)}</tbody></table>")

    task_title = "Классификация" if data["task_type"] == "classification" else "Регрессия"
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отчёт о качестве модели</title>
<style>
body {{ max-width: 1050px; margin: 36px auto; padding: 0 22px; font: 16px/1.55 Arial, sans-serif; color: #222; }}
h1, h2 {{ color: #183153; }}
.meta {{ color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; }}
th, td {{ border: 1px solid #d8dee8; padding: 10px 12px; text-align: left; }}
th {{ background: #f1f4f8; }}
figure {{ margin: 26px 0 34px; }}
img {{ max-width: 100%; height: auto; border: 1px solid #e1e5eb; }}
figcaption {{ color: #666; margin-top: 6px; }}
</style>
</head>
<body>
<h1>Отчёт о качестве модели</h1>
<div class="meta"><p><b>Модель:</b> {html.escape(data['model_name'])}<br>
<b>Тип задачи:</b> {task_title}<br><b>Количество объектов:</b> {data['sample_count']}<br>
<b>Дата формирования (UTC):</b> {html.escape(data['generated_at'])}</p></div>
<h2>Метрики качества</h2>
<table><thead><tr><th>Метрика</th><th>Значение</th></tr></thead><tbody>{metrics_rows}</tbody></table>
{classes}
<h2>Графики</h2>
{plots}
{_grid_html(data.get('grid_search'))}
</body>
</html>
"""

# Основная функция генерации отчёта

def generate_report(y_true, y_pred, cfg, model, output_dir, grid=None):
    """
    Аргументы:
        y_true: реальные значения целевой переменной.
        y_pred: предсказанные значения той же длины.
        cfg: настройки модели и отчёта.
        model: обученная модель.
        output_dir: папка для сохранения результатов. Если передан путь
            файла модели с расширением .joblib, .pkl или .pickle, отчёт
            сохраняется в соседнюю папку report.
        grid: необязательный обученный GridSearchCV или словарь его результатов.

    Файлы результата:
        metrics.json — общие сведения и рассчитанные метрики;
        report.html и report.md — отчёты для пользователя;
        plots/ — графики метрик, ошибок и GridSearch;
        grid_search.json и grid_search.csv — подробные результаты подбора параметров.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true и y_pred должны быть одномерными массивами")
    if y_true.size == 0:
        raise ValueError("Нельзя сформировать отчёт по пустому набору данных")
    if y_true.size != y_pred.size:
        raise ValueError(f"Размеры y_true и y_pred не совпадают: {y_true.size} != {y_pred.size}")
    if _contains_missing(y_true) or _contains_missing(y_pred):
        raise ValueError("y_true и y_pred не должны содержать пропущенные значения")

    config = _plain_config(cfg)
    raw_metrics = _cfg_get(config, "evaluation.metrics", "metrics")
    initial_specs = _metric_specs(raw_metrics, "classification") if raw_metrics is not None else []
    task = _detect_task(
        y_true, model,
        _cfg_get(config, "evaluation.task_type", "task_type", "task", "problem_type"),
        initial_specs,
    )
    specs = _metric_specs(raw_metrics, task)

    if task == "regression":
        try:
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Для регрессии y_true и y_pred должны содержать числа") from error
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            raise ValueError("Для регрессии значения должны быть конечными числами")

    average = str(_cfg_get(config, "evaluation.average", "average", default="weighted"))
    model_name = _cfg_get(config, "name", "model.name", "model_name", "model._target_")
    model_name = str(model_name or (type(model).__name__ if model is not None else "Не указана"))

    # train.py передаёт output_dir равным пути сохранения модели. Отчёт размещаем рядом.

    requested_path = Path(output_dir).expanduser()
    if requested_path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        target_dir = requested_path.parent / "report"
    else:
        target_dir = requested_path
    target_dir.mkdir(parents=True, exist_ok=True)

    metric_values, class_report, confusion = _calculate_metrics(y_true, y_pred, task, specs, average)
    plot_paths = _save_plots(target_dir, task, y_true, y_pred, metric_values, confusion)
    grid_info = _extract_grid_info(grid)
    grid_plot = _save_grid_artifacts(target_dir, grid_info)
    if grid_plot:
        plot_paths.append(grid_plot)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    grid_summary = ({key: value for key, value in grid_info.items() if key != "cv_results"}
                    if grid_info is not None else None)
    result = {
        "generated_at": generated_at,
        "model": {"name": model_name, "class": type(model).__name__ if model is not None else None},
        "task_type": task,
        "sample_count": int(y_true.size),
        "metrics": metric_values,
        "classification_report": class_report,
        "confusion_matrix": confusion.tolist() if confusion is not None else None,
        "plots": plot_paths,
        "grid_search": grid_summary,
    }
    with (target_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, default=_json_default)

    # Сводные отчёты используют одну и ту же подготовленную структуру данных.
    
    report_data = {
        "model_name": model_name,
        "task_type": task,
        "sample_count": int(y_true.size),
        "generated_at": generated_at,
        "metrics": metric_values,
        "classification_report": class_report,
        "plots": [path for path in plot_paths if path != grid_plot],
        "grid_search": grid_info,
    }
    (target_dir / "report.md").write_text(_build_markdown_report(report_data), encoding="utf-8")
    (target_dir / "report.html").write_text(_build_html_report(report_data), encoding="utf-8")
