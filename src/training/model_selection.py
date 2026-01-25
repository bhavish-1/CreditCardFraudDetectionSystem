from typing import Dict


def select_best_model(
    models: Dict,
    primary_metric: str = "pr_auc"
):
    """
    Select the best model based on a primary metric.
    """

    if not models:
        raise ValueError("No models provided for selection")

    best_model_name = None
    best_score = -1

    for model_name, model_info in models.items():
        metrics = model_info.get("metrics", {})

        if primary_metric not in metrics:
            continue

        score = metrics[primary_metric]

        if score > best_score:
            best_score = score
            best_model_name = model_name

    if best_model_name is None:
        raise ValueError(
            f"No model contains the metric '{primary_metric}'"
        )

    best_model_info = models[best_model_name]

    return (
        best_model_name,
        best_model_info["model"],
        best_model_info["metrics"]
    )
