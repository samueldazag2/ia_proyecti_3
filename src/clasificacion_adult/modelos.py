"""Fábrica de las tres arquitecturas feed-forward (todas las neuronas con bias) e inspección
de los modelos entrenados (formas de pesos y neuronas ReLU muertas)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

from . import config as cfg

NOMBRE_PERCEPTRON = "Perceptrón"
NOMBRE_MLP_1 = "MLP 1 capa oculta"
NOMBRE_MLP_2 = "MLP 2 capas ocultas (2, 2)"
NOMBRES_MODELOS = (NOMBRE_PERCEPTRON, NOMBRE_MLP_1, NOMBRE_MLP_2)
# Identificadores cortos para tablas y figuras
ID_MODELO = {NOMBRE_PERCEPTRON: "perceptron", NOMBRE_MLP_1: "mlp_1capa", NOMBRE_MLP_2: "mlp_2x2"}


def crear_perceptron(semilla: int = cfg.SEMILLA) -> Perceptron:
    """Perceptrón simple de Rosenblatt: una neurona con bias (fit_intercept=True)."""
    return Perceptron(**cfg.HIPER_PERCEPTRON, random_state=semilla)


def crear_mlp_una_capa(n_entradas: int, semilla: int = cfg.SEMILLA) -> MLPClassifier:
    """MLP con una capa oculta de tantas neuronas como entradas tenga la red."""
    return MLPClassifier(hidden_layer_sizes=(n_entradas,), **cfg.HIPER_MLP, random_state=semilla)


def crear_mlp_dos_capas(semilla: int = cfg.SEMILLA) -> MLPClassifier:
    """MLP con dos capas ocultas de 2 neuronas cada una."""
    return MLPClassifier(hidden_layer_sizes=cfg.CAPAS_MLP_2, **cfg.HIPER_MLP, random_state=semilla)


def crear_modelos(
    n_entradas: int, semilla: int = cfg.SEMILLA
) -> dict[str, Perceptron | MLPClassifier]:
    """Devuelve las tres arquitecturas sin entrenar, indexadas por nombre."""
    return {
        NOMBRE_PERCEPTRON: crear_perceptron(semilla),
        NOMBRE_MLP_1: crear_mlp_una_capa(n_entradas, semilla),
        NOMBRE_MLP_2: crear_mlp_dos_capas(semilla),
    }


def describir_pesos(modelo: Perceptron | MLPClassifier) -> pd.DataFrame:
    """Formas de las matrices de pesos y de los vectores de bias de un modelo entrenado."""
    if isinstance(modelo, MLPClassifier):
        filas = [
            {
                "capa": f"{i} -> {i + 1}",
                "pesos (coefs_)": tuple(W.shape),
                "bias (intercepts_)": tuple(b.shape),
            }
            for i, (W, b) in enumerate(zip(modelo.coefs_, modelo.intercepts_, strict=True))
        ]
    else:
        filas = [
            {
                "capa": "entrada -> salida",
                "pesos (coef_)": tuple(modelo.coef_.shape),
                "bias (intercept_)": tuple(modelo.intercept_.shape),
            }
        ]
    return pd.DataFrame(filas).set_index("capa")


def fraccion_activacion(mlp: MLPClassifier, X_transformado: np.ndarray) -> list[np.ndarray]:
    """Para cada capa oculta, fracción de ejemplos en que cada neurona se activa (Z > 0).

    Z = a·W + b es la preactivación; con ReLU la neurona solo propaga señal (y gradiente)
    cuando Z > 0. Devuelve una lista con un vector por capa oculta (una entrada por neurona).
    """
    a = X_transformado
    fracciones = []
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1], strict=True):
        z = a @ W + b
        fracciones.append((z > 0).mean(axis=0))
        a = np.maximum(0, z)
    return fracciones


def neuronas_inactivas(
    mlp: MLPClassifier,
    X_transformado: np.ndarray,
    umbral: float = cfg.UMBRAL_ACTIVACION,
) -> list[tuple[int, int]]:
    """Por capa oculta: (neuronas inactivas, neuronas que nunca se activan).

    Inactiva = activa (Z > 0) en menos de ``umbral`` de los ejemplos de ``X_transformado``.
    """
    return [
        (int((f < umbral).sum()), int((f == 0).sum()))
        for f in fraccion_activacion(mlp, X_transformado)
    ]


def neuronas_muertas(mlp: MLPClassifier, X_transformado: np.ndarray) -> list[int]:
    """Número de neuronas ReLU muertas por capa oculta.

    Una neurona está "muerta" si su salida es 0 para todos los ejemplos de
    ``X_transformado`` (preactivación nunca positiva): su gradiente es 0 y ya no
    puede volver a aprender.
    """
    return [nunca for _, nunca in neuronas_inactivas(mlp, X_transformado)]
