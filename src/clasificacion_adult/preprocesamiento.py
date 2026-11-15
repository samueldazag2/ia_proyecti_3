"""Preprocesamiento: ColumnTransformer (sin fuga de datos) y detección de atípicos (IQR).

Esquema del ColumnTransformer (todo se ajusta SOLO con el conjunto de entrenamiento):

* ``num``    age, education_num, hours_per_week      -> StandardScaler
* ``log``    capital_gain, capital_loss               -> log1p -> StandardScaler
* ``cat``    workclass, marital_status, occupation,
             relationship, race, sex                  -> imputar "Desconocido" -> One-Hot
* ``pais``   native_country                           -> imputar "Desconocido"
                                                         -> agrupar < 1 % en "Otro" -> One-Hot
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from . import config as cfg


class AgrupadorCategoriasRaras(BaseEstimator, TransformerMixin):
    """Reemplaza por ``valor`` las categorías con frecuencia relativa < ``umbral``.

    Las frecuencias se aprenden en ``fit`` (solo con train); en ``transform`` cualquier
    categoría no frecuente o no vista en entrenamiento pasa a ``valor``.
    """

    def __init__(
        self, umbral: float = cfg.UMBRAL_FRECUENCIA_PAIS, valor: str = cfg.CATEGORIA_RARA
    ) -> None:
        self.umbral = umbral
        self.valor = valor

    def fit(self, X: Any, y: Any = None) -> AgrupadorCategoriasRaras:
        """Aprende, por columna, las categorías con frecuencia >= ``umbral``."""
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = pd.DataFrame(X).astype(object)
        self.n_features_in_ = X.shape[1]
        self.categorias_frecuentes_ = [
            set(X[c].value_counts(normalize=True).loc[lambda s: s >= self.umbral].index)
            for c in X.columns
        ]
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Sustituye las categorías no frecuentes (o no vistas) por ``valor``."""
        check_is_fitted(self, "categorias_frecuentes_")
        X = pd.DataFrame(X).astype(object).copy()
        for i, c in enumerate(X.columns):
            X[c] = X[c].where(X[c].isin(self.categorias_frecuentes_[i]), self.valor)
        return X.to_numpy()

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Mismos nombres de entrada (transformador uno a uno)."""
        check_is_fitted(self, "categorias_frecuentes_")
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def _imputador() -> SimpleImputer:
    return SimpleImputer(strategy="constant", fill_value=cfg.VALOR_NULO)


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def construir_preprocesador() -> ColumnTransformer:
    """Crea el ColumnTransformer sin ajustar."""
    numericas_lineales = [c for c in cfg.NUMERICAS if c not in cfg.NUMERICAS_LOG]
    categoricas_sin_pais = [c for c in cfg.CATEGORICAS if c != cfg.COLUMNA_PAIS]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas_lineales),
            (
                "log",
                Pipeline(
                    [
                        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                cfg.NUMERICAS_LOG,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputar", _imputador()),
                        ("onehot", _one_hot()),
                    ]
                ),
                categoricas_sin_pais,
            ),
            (
                "pais",
                Pipeline(
                    [
                        ("imputar", _imputador()),
                        ("agrupar", AgrupadorCategoriasRaras()),
                        ("onehot", _one_hot()),
                    ]
                ),
                [cfg.COLUMNA_PAIS],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def construir_pipeline(modelo: Any) -> Pipeline:
    """Pipeline completo preprocesador + clasificador."""
    return Pipeline([("preprocesador", construir_preprocesador()), ("modelo", modelo)])


def numero_entradas(X_train: pd.DataFrame) -> int:
    """Número de entradas de la red tras el preprocesamiento (ajustado con train)."""
    return construir_preprocesador().fit(X_train).transform(X_train.head(1)).shape[1]


def detectar_atipicos_iqr(
    datos: pd.DataFrame,
    columnas: Sequence[str] = tuple(cfg.NUMERICAS),
    objetivo: str = cfg.OBJETIVO,
    factor: float = cfg.FACTOR_IQR,
) -> pd.DataFrame:
    """Cuenta atípicos por la regla de Tukey (fuera de [Q1 - k·IQR, Q3 + k·IQR], k = 1.5).


    Solo se reportan: no se eliminan. La última columna muestra qué fracción de los
    atípicos pertenece a la clase >50K (frente al ~24 % global).
    """
    positivo = datos[objetivo] == cfg.CLASE_POSITIVA
    filas = []
    for col in columnas:
        q1, q3 = datos[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        inf, sup = q1 - factor * iqr, q3 + factor * iqr
        m = (datos[col] < inf) | (datos[col] > sup)
        filas.append(
            {
                "variable": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "limite_inferior": inf,
                "limite_superior": sup,
                "n_atipicos": int(m.sum()),
                "pct_atipicos": round(m.mean() * 100, 2),
                "pct_>50K_en_atipicos": round(positivo[m].mean() * 100, 2) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(filas).set_index("variable")


def grafico_log1p(datos: pd.DataFrame) -> Figure:
    """Distribución de capital_gain / capital_loss antes y después de log1p."""
    fig, ejes = plt.subplots(2, 2, figsize=(12, 7))
    for fila, col in enumerate(cfg.NUMERICAS_LOG):
        ejes[fila, 0].hist(datos[col], bins=50, color=cfg.PALETA_MODELOS[0])
        ejes[fila, 0].set(title=f"{col} original", yscale="log", ylabel="Frecuencia (log)")
        ejes[fila, 1].hist(np.log1p(datos[col]), bins=50, color=cfg.COLOR_SECUNDARIO)
        ejes[fila, 1].set(title=f"log1p({col})", yscale="log")
    fig.suptitle("Efecto de log1p en las variables de capital", fontsize=14)
    fig.tight_layout()
    return fig
