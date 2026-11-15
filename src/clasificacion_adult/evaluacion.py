"""Métricas y gráficos de evaluación.

Las funciones de métricas solo calculan; las de gráficos solo dibujan y devuelven la
figura (no escriben a disco: de eso se encarga ``experimentos``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier

from . import config as cfg
from .modelos import NOMBRES_MODELOS

METRICAS_COMPARACION = [
    "accuracy", "precision", "recall", "f1", "macro_f1", "balanced_accuracy", "roc_auc",
]  # fmt: skip


def color_modelo(nombre: str) -> str:
    """Color fijo por modelo, igual en todas las figuras.

    Acepta etiquetas que empiezan por el nombre del modelo, p. ej. "Perceptrón (semilla 5)".
    """
    for indice, base in enumerate(NOMBRES_MODELOS):
        if nombre.startswith(base):
            return cfg.PALETA_MODELOS[indice]
    return cfg.PALETA_MODELOS[-1]


# --- Métricas --------------------------------------------------------------
def puntajes(estimador: Any, X: Any) -> np.ndarray:
    """Puntaje continuo de la clase positiva.

    Usa ``predict_proba`` si existe (MLP, línea base) y si no ``decision_function``
    (perceptrón: distancia con signo al hiperplano). En un ``Pipeline``, ``hasattr``
    ya refleja si el último paso tiene el método (``available_if`` de scikit-learn).
    """
    if hasattr(estimador, "predict_proba"):
        return estimador.predict_proba(X)[:, 1]
    return estimador.decision_function(X)


def calcular_metricas(
    y_true: ArrayLike, y_pred: ArrayLike, y_score: ArrayLike | None = None
) -> dict[str, float]:
    """Métricas de la clase positiva (>50K = 1), macro-F1, balanced accuracy y ROC-AUC.

    ``roc_auc`` es NaN si no se pasan puntajes (no se inventa un valor).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else float("nan"),
    }


def conteos_confusion(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, int]:
    """Verdaderos negativos, falsos positivos, falsos negativos y verdaderos positivos."""
    vn, fp, fn, vp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"vn": int(vn), "fp": int(fp), "fn": int(fn), "vp": int(vp)}


def entropia_a_priori(y: ArrayLike) -> float:
    """Log-loss del mejor modelo constante (predice siempre la proporción de positivos)."""
    p = float(np.mean(np.asarray(y, dtype=float)))
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


# --- Gráficos --------------------------------------------------------------
def grafico_comparacion_metricas(metricas: pd.DataFrame, nombre_base: str) -> Figure:
    """Barras agrupadas de cada métrica por modelo; la línea gris es el accuracy de la base."""
    fig, ax = plt.subplots(figsize=(12, 5))
    metricas[METRICAS_COMPARACION].T.plot.bar(ax=ax, width=0.8)
    ax.set(title="Comparación de métricas en test", ylabel="Valor", ylim=(0, 1))
    accuracy_base = float(metricas["accuracy"].to_numpy()[metricas.index.get_loc(nombre_base)])
    ax.axhline(accuracy_base, ls="--", color="gray", lw=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def grafico_matrices_confusion(y_true: ArrayLike, predicciones: Mapping[str, ArrayLike]) -> Figure:
    """Matrices de confusión absolutas (fila 1) y normalizadas por fila (fila 2)."""
    n = len(predicciones)
    fig, ejes = plt.subplots(2, n, figsize=(4.6 * n, 8.5))
    ejes = np.atleast_2d(ejes).reshape(2, n)
    for j, (nombre, y_pred) in enumerate(predicciones.items()):
        for i, normalizar in enumerate([None, "true"]):
            cm = confusion_matrix(y_true, y_pred, normalize=normalizar)
            ConfusionMatrixDisplay(cm, display_labels=cfg.NOMBRES_CLASES).plot(
                ax=ejes[i, j],
                cmap="Blues",
                colorbar=False,
                values_format="d" if normalizar is None else ".2f",
            )
            tipo = "absoluta" if normalizar is None else "normalizada"
            ejes[i, j].set_title(f"{nombre}\n{tipo}")
            ejes[i, j].set(xlabel="Clase predicha", ylabel="Clase real")
            ejes[i, j].grid(False)
    fig.tight_layout()
    return fig


def grafico_roc(y_true: ArrayLike, scores: Mapping[str, ArrayLike]) -> Figure:
    """Curvas ROC superpuestas con su AUC."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for nombre, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        auc = roc_auc_score(y_true, s)
        ax.plot(fpr, tpr, color=color_modelo(nombre), lw=2, label=f"{nombre} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.500)")
    ax.set(
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos (recall)",
        title="Curvas ROC en el conjunto de prueba",
    )
    ax.legend(loc="lower right")
    return fig


def grafico_curvas_perdida(mlps: Mapping[str, MLPClassifier]) -> Figure:
    """Pérdida (log-loss) de entrenamiento por época y accuracy de validación interna."""
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
    for nombre, mlp in mlps.items():
        color = color_modelo(nombre)
        ejes[0].plot(mlp.loss_curve_, color=color, lw=2, label=nombre)
        if getattr(mlp, "validation_scores_", None) is not None:
            ejes[1].plot(mlp.validation_scores_, color=color, lw=2, label=nombre)
            ejes[1].axvline(int(np.argmax(mlp.validation_scores_)), color=color, ls=":", lw=1)
    ejes[0].set(title="Pérdida de entrenamiento (log-loss)", xlabel="Época", ylabel="Pérdida")
    ejes[1].set(
        title="Accuracy en validación (early stopping, 10 % de train)",
        xlabel="Época",
        ylabel="Accuracy",
    )
    for ax in ejes:
        ax.legend()
    fig.tight_layout()
    return fig


def grafico_proporciones(resumen: pd.DataFrame) -> Figure:
    """F1 de la clase >50K (media ± desviación) frente a la proporción de entrenamiento."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for modelo, grupo in resumen.groupby(level="modelo", sort=False):
        props = grupo.index.get_level_values("proporcion_train") * 100
        ax.errorbar(
            props,
            grupo["f1_mean"],
            yerr=grupo["f1_std"],
            marker="o",
            capsize=4,
            lw=2,
            color=color_modelo(str(modelo)),
            label=modelo,
        )
    ax.set(
        xlabel="Proporción de entrenamiento (%)",
        ylabel="F1 de la clase >50K",
        title="F1 frente a la proporción de entrenamiento (media ± desv., 5 semillas)",
        xticks=[p * 100 for p in cfg.PROPORCIONES_TRAIN],
    )
    ax.legend()
    return fig


def grafico_colapso(tabla: pd.DataFrame) -> Figure:
    """F1 de cada semilla por modelo; en rojo las semillas que colapsan.

    Recibe la tabla de ``experimentos.experimento_colapso`` (columnas ``modelo``,
    ``semilla``, ``f1`` y ``colapso``). Cada panel indica "k/n colapsos".
    """
    modelos = list(dict.fromkeys(tabla["modelo"]))
    fig, ejes = plt.subplots(1, len(modelos), figsize=(6.5 * len(modelos), 4.5), sharey=True)
    for ax, modelo in zip(np.atleast_1d(ejes), modelos, strict=True):
        datos = tabla[tabla["modelo"] == modelo]
        colapso = datos["colapso"].to_numpy(dtype=bool)
        ax.scatter(
            datos["semilla"][~colapso],
            datos["f1"][~colapso],
            s=22,
            color=color_modelo(modelo),
            label="Entrena",
        )
        ax.scatter(
            datos["semilla"][colapso],
            datos["f1"][colapso],
            s=40,
            color=cfg.COLOR_RESALTE,
            label="Colapso",
        )
        ax.axhline(cfg.UMBRAL_F1_COLAPSO, ls="--", color="gray", lw=1)
        ax.text(
            0.02,
            0.5,
            f"{int(colapso.sum())}/{len(datos)} colapsos",
            transform=ax.transAxes,
            fontsize=12,
            color=cfg.COLOR_RESALTE,
        )
        if colapso.any():
            semillas = ", ".join(str(int(s)) for s in datos["semilla"][colapso])
            ax.text(
                0.02,
                0.42,
                f"semillas: {semillas}",
                transform=ax.transAxes,
                fontsize=10,
                color=cfg.COLOR_RESALTE,
            )
        ax.set(title=modelo, xlabel="random_state del modelo", ylim=(-0.03, 0.8))
        ax.legend(loc="center right")
    np.atleast_1d(ejes)[0].set_ylabel("F1 de la clase >50K (test)")
    fig.suptitle(f"Colapso por inicialización (F1 < {cfg.UMBRAL_F1_COLAPSO:.2f})", fontsize=13)
    fig.tight_layout()
    return fig
