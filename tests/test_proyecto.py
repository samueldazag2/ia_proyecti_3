"""Pruebas del pipeline de datos y de las arquitecturas."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

from clasificacion_adult import config as cfg
from clasificacion_adult.datos import cargar_datos, limpiar_datos, separar_x_y
from clasificacion_adult.experimentos import dividir
from clasificacion_adult.modelos import crear_modelos
from clasificacion_adult.preprocesamiento import (
    AgrupadorCategoriasRaras,
    construir_preprocesador,
    numero_entradas,
)


@pytest.fixture(scope="module")
def crudos():
    return cargar_datos()


@pytest.fixture(scope="module")
def particion(crudos):
    X, y = separar_x_y(limpiar_datos(crudos))
    return dividir(X, y)


@pytest.fixture(scope="module")
def modelos_entrenados(particion):
    """Entrena los tres modelos con una muestra pequeña (rápido) para revisar sus pesos."""
    X_train, _, y_train, _ = particion
    prep = construir_preprocesador().fit(X_train)
    n = len(prep.get_feature_names_out())
    muestra = X_train.sample(3000, random_state=0)
    Xt, yt = prep.transform(muestra), y_train.loc[muestra.index]
    modelos = crear_modelos(n)
    for m in modelos.values():
        if isinstance(m, MLPClassifier):
            m.set_params(max_iter=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(Xt, yt)
    return n, modelos


# --- Carga -----------------------------------------------------------------
def test_carga_48842_filas(crudos):
    assert crudos.shape == (48842, 15)
    assert list(crudos.columns) == cfg.COLUMNAS


def test_etiquetas_unificadas(crudos):
    assert set(crudos[cfg.OBJETIVO].unique()) == {"<=50K", ">50K"}


def test_limpieza_quita_52_duplicados_y_columnas(crudos):
    limpios = limpiar_datos(crudos)
    assert len(crudos) - len(limpios) == 52
    assert not set(cfg.COLUMNAS_ELIMINADAS) & set(limpios.columns)


# --- Preprocesamiento --------------------------------------------------------
def test_sin_nulos_tras_preprocesamiento(particion):
    X_train, X_test, _, _ = particion
    assert X_train.isna().any().any()  # los datos crudos sí tienen nulos
    prep = construir_preprocesador().fit(X_train)
    for X in (X_train, X_test):
        Xt = prep.transform(X)
        assert not np.isnan(Xt).any()


def test_53_entradas(particion):
    assert numero_entradas(particion[0]) == 53


def test_sin_fuga_escalador_ajustado_solo_con_train(particion):
    X_train, X_test, _, _ = particion
    prep = construir_preprocesador().fit(X_train)
    escalador = prep.named_transformers_["num"]
    columnas = ["age", "education_num", "hours_per_week"]
    np.testing.assert_allclose(escalador.mean_, X_train[columnas].mean().to_numpy())
    assert not np.allclose(escalador.mean_, X_test[columnas].mean().to_numpy())
    # Las numéricas de train quedan con media 0 exacta; las de test no.
    Xt_train, Xt_test = prep.transform(X_train), prep.transform(X_test)
    np.testing.assert_allclose(Xt_train[:, :3].mean(axis=0), 0, atol=1e-10)
    assert np.abs(Xt_test[:, :3].mean(axis=0)).max() > 1e-6


def test_agrupador_paises_raros():
    X = pd.DataFrame({"pais": ["A"] * 98 + ["B", "C"]})
    ag = AgrupadorCategoriasRaras(umbral=0.05).fit(X)
    out = ag.transform(pd.DataFrame({"pais": ["A", "B", "Z"]}))
    assert out.ravel().tolist() == ["A", cfg.CATEGORIA_RARA, cfg.CATEGORIA_RARA]


# --- Modelos ---------------------------------------------------------------
def test_todas_las_capas_con_bias(modelos_entrenados):
    _, modelos = modelos_entrenados
    for modelo in modelos.values():
        if isinstance(modelo, Perceptron):
            assert modelo.fit_intercept
            assert modelo.intercept_.shape == (1,)
        else:
            assert len(modelo.intercepts_) == len(modelo.coefs_)
            for W, b in zip(modelo.coefs_, modelo.intercepts_, strict=True):
                assert b.shape == (W.shape[1],)


def test_capa_oculta_igual_a_entradas(modelos_entrenados):
    n, modelos = modelos_entrenados
    mlp1 = next(
        m
        for m in modelos.values()
        if isinstance(m, MLPClassifier) and len(m.hidden_layer_sizes) == 1
    )
    assert mlp1.hidden_layer_sizes == (n,)
    assert mlp1.coefs_[0].shape == (n, n)


def test_mlp_dos_capas_2_2(modelos_entrenados):
    n, modelos = modelos_entrenados
    mlp2 = next(
        m
        for m in modelos.values()
        if isinstance(m, MLPClassifier) and len(m.hidden_layer_sizes) == 2
    )
    assert [W.shape for W in mlp2.coefs_] == [(n, 2), (2, 2), (2, 1)]
