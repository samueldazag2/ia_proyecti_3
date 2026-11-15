"""Pruebas unitarias rápidas con datos sintéticos: métricas, puntajes, EDA, atípicos,
agrupador sin fuga, neuronas muertas, reproducibilidad y guardado de resultados."""

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

from clasificacion_adult import config as cfg
from clasificacion_adult import eda, evaluacion, utilidades
from clasificacion_adult.experimentos import (
    ajustar_sin_avisos,
    dividir,
    experimento_proporciones,
    resumir_proporciones,
)
from clasificacion_adult.modelos import NOMBRE_MLP_1, crear_perceptron, neuronas_muertas
from clasificacion_adult.preprocesamiento import (
    AgrupadorCategoriasRaras,
    construir_pipeline,
    construir_preprocesador,
    detectar_atipicos_iqr,
)


def _datos_sinteticos(n: int = 400, semilla: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """X con las columnas del modelo (tras limpiar) y una y que depende de age/education."""
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "workclass": rng.choice(["Private", "Self-emp", None], n),
            "education_num": rng.integers(1, 17, n),
            "marital_status": rng.choice(["Married", "Never-married"], n),
            "occupation": rng.choice(["Sales", "Tech", None], n),
            "relationship": rng.choice(["Husband", "Wife", "Own-child"], n),
            "race": rng.choice(["White", "Black"], n),
            "sex": rng.choice(["Male", "Female"], n),
            "capital_gain": rng.choice([0, 0, 0, 5000], n),
            "capital_loss": rng.choice([0, 0, 0, 1500], n),
            "hours_per_week": rng.integers(10, 70, n),
            "native_country": rng.choice(["United-States"] * 20 + ["Mexico", "Peru"], n),
        }
    )
    y = pd.Series(((X["age"] > 40) & (X["education_num"] > 9)).astype(int), name="y")
    return X, y


# --- Métricas y puntajes ---------------------------------------------------
def test_metricas_usan_clase_positiva_1():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]  # TP=1, FP=1, FN=1
    m = evaluacion.calcular_metricas(y_true, y_pred, y_score=[0.1, 0.2, 0.7, 0.9, 0.4])
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["roc_auc"] == pytest.approx(5 / 6)


def test_roc_auc_es_nan_sin_puntajes():
    assert math.isnan(evaluacion.calcular_metricas([0, 1], [0, 1])["roc_auc"])


def test_linea_base_tiene_auc_05_y_f1_0():
    X, y = _datos_sinteticos()
    base = DummyClassifier(strategy="most_frequent").fit(X, y)
    m = evaluacion.calcular_metricas(y, base.predict(X), evaluacion.puntajes(base, X))
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["f1"] == 0


def test_puntajes_perceptron_usa_decision_function():
    X, y = _datos_sinteticos()
    pipe = ajustar_sin_avisos(construir_pipeline(crear_perceptron()), X, y)
    assert not hasattr(pipe, "predict_proba")  # available_if del Pipeline
    np.testing.assert_allclose(evaluacion.puntajes(pipe, X), pipe.decision_function(X))


def test_entropia_a_priori():
    assert evaluacion.entropia_a_priori([0, 1]) == pytest.approx(math.log(2))


# --- EDA y atípicos --------------------------------------------------------
def test_v_cramer_extremos():
    x = pd.Series(["a", "b"] * 50)
    assert eda.v_cramer(x, x) == pytest.approx(1.0)
    independiente = pd.Series(["u", "u", "v", "v"] * 25)
    assert eda.v_cramer(x, independiente) == pytest.approx(0.0)


def test_distribucion_clases_suma_100():
    datos = pd.DataFrame({cfg.OBJETIVO: ["<=50K"] * 3 + [">50K"]})
    tabla = eda.distribucion_clases(datos)
    assert tabla.loc[">50K", "porcentaje"] == 25.0
    assert tabla["porcentaje"].sum() == pytest.approx(100)


def test_graficos_devuelven_figura_sin_escribir(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DIR_FIGURAS", tmp_path / "figuras")
    datos = pd.DataFrame({cfg.OBJETIVO: ["<=50K"] * 3 + [">50K"]})
    fig = eda.grafico_distribucion_clases(eda.distribucion_clases(datos))
    assert isinstance(fig, Figure)
    assert not (tmp_path / "figuras").exists()


def test_atipicos_iqr():
    datos = pd.DataFrame({"v": [*range(1, 11), 1000], cfg.OBJETIVO: ["<=50K"] * 10 + [">50K"]})
    tabla = detectar_atipicos_iqr(datos, columnas=["v"])
    assert tabla.loc["v", "n_atipicos"] == 1
    assert tabla.loc["v", "pct_>50K_en_atipicos"] == 100.0


# --- Preprocesamiento sin fuga ---------------------------------------------
def test_agrupador_aprende_frecuencias_solo_de_train():
    X, y = _datos_sinteticos(600)
    X_train, X_test, _, _ = dividir(X, y)
    X_train, X_test = X_train.copy(), X_test.copy()
    # En train: Peru < 1 % (1 fila); en test: Peru es el 100 %.
    X_train["native_country"] = ["Peru"] + ["United-States"] * (len(X_train) - 1)
    X_test["native_country"] = "Peru"
    prep = construir_preprocesador().fit(X_train)
    nombres = prep.get_feature_names_out()
    assert "pais__native_country_Peru" not in nombres
    assert "pais__native_country_Otro" in nombres
    agrupador = prep.named_transformers_["pais"].named_steps["agrupar"]
    assert agrupador.transform(X_test[["native_country"]]).ravel().tolist() == (
        [cfg.CATEGORIA_RARA] * len(X_test)
    )


def test_agrupador_nombres_de_salida():
    ag = AgrupadorCategoriasRaras().fit(pd.DataFrame({"pais": ["A", "B"]}))
    assert ag.get_feature_names_out().tolist() == ["pais"]


# --- Modelos ---------------------------------------------------------------
def test_neuronas_muertas():
    mlp = MLPClassifier(hidden_layer_sizes=(2,))
    mlp.coefs_ = [np.array([[1.0, -1.0]]), np.array([[1.0], [1.0]])]
    mlp.intercepts_ = [np.array([0.0, -5.0]), np.array([0.0])]
    X = np.array([[1.0], [2.0], [3.0]])  # la 2.ª neurona tiene preactivación siempre < 0
    assert neuronas_muertas(mlp, X) == [1]


def test_entrenamiento_reproducible():
    X, y = _datos_sinteticos()
    a = ajustar_sin_avisos(construir_pipeline(crear_perceptron(7)), X, y)
    b = ajustar_sin_avisos(construir_pipeline(crear_perceptron(7)), X, y)
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))


# --- Experimentos y guardado -----------------------------------------------
def test_experimento_proporciones_humo():
    X, y = _datos_sinteticos(500)
    detalle, resumen = experimento_proporciones(
        X, y, proporciones=[0.6], semillas=[0, 1], guardar=False, verbose=False
    )
    assert len(detalle) == 1 * 2 * 3
    assert set(detalle["modelo"]) >= {NOMBRE_MLP_1}
    assert {"f1_mean", "f1_std", "f1"} <= set(resumen.columns)


def test_resumir_proporciones_formato():
    detalle = pd.DataFrame(
        {
            "modelo": ["m", "m"],
            "proporcion_train": [0.5, 0.5],
            "accuracy": [0.8, 0.6],
            "precision": [0.5, 0.5],
            "recall": [0.5, 0.5],
            "f1": [0.4, 0.6],
        }
    )
    resumen = resumir_proporciones(detalle)
    assert resumen.loc[("m", 0.5), "f1_mean"] == pytest.approx(0.5)
    assert resumen.loc[("m", 0.5), "f1"] == "0.5000 ± 0.1414"


def test_guardar_tabla_y_figura(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DIR_TABLAS", tmp_path / "tablas")
    monkeypatch.setattr(cfg, "DIR_FIGURAS", tmp_path / "figuras")
    utilidades.guardar_tabla(pd.DataFrame({"a": [1]}), "t")
    fig = Figure()
    fig.add_subplot()
    utilidades.guardar_figura(fig, "f")
    assert (tmp_path / "tablas" / "t.csv").exists()
    assert (tmp_path / "figuras" / "f.png").exists()


# --- Humo de orquestadores (datos sintéticos, salidas en tmp_path) ----------
@pytest.fixture
def dirs_temporales(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DIR_TABLAS", tmp_path / "tablas")
    monkeypatch.setattr(cfg, "DIR_FIGURAS", tmp_path / "figuras")
    return tmp_path


def test_ejecutar_eda_guarda_tablas_y_figuras(dirs_temporales):
    X, y = _datos_sinteticos(300)
    crudos = X.assign(
        fnlwgt=np.arange(len(X)) * 1000,
        education=X["education_num"].astype(str),
        income=np.where(y == 1, cfg.CLASE_POSITIVA, cfg.CLASE_NEGATIVA),
    )
    tablas = eda.ejecutar_eda(crudos)
    assert set(tablas) == {"tipos", "descriptivas", "clases", "nulos", "correlacion", "v_cramer"}
    assert len(list((dirs_temporales / "tablas").glob("*.csv"))) == 6
    assert len(list((dirs_temporales / "figuras").glob("*.png"))) == 7


def test_entrenar_y_diagnostico_guardan_resultados(dirs_temporales):
    from clasificacion_adult.experimentos import (
        NOMBRE_BASE,
        diagnostico_semillas_mlp2,
        entrenar_y_evaluar,
    )

    X, y = _datos_sinteticos(400)
    res = entrenar_y_evaluar(X, y)
    assert NOMBRE_BASE in res.metricas.index and len(res.metricas) == 4
    assert res.metricas.loc[NOMBRE_BASE, "roc_auc"] == pytest.approx(0.5)
    diag = diagnostico_semillas_mlp2(res, semillas=[0])
    assert list(diag.columns[:2]) == ["muertas_capa1", "muertas_capa2"]
    # Las matrices, la ROC y la pérdida las genera experimento_semillas (representativos)
    figuras = {p.stem for p in (dirs_temporales / "figuras").glob("*.png")}
    assert figuras == {"comparacion_metricas"}
    tablas = {p.stem for p in (dirs_temporales / "tablas").glob("*.csv")}
    assert tablas == {"metricas_modelos", "diagnostico_mlp_2_2_semillas"}


# --- Experimento de semillas y colapso ---------------------------------------
@pytest.mark.parametrize(
    ("f1s", "esperada"),
    [
        # 10 corridas: índice (10 - 1) // 2 = 4 -> 5.ª más baja (F1 0.5 -> semilla 4)
        ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 4),
        # orden desordenado: F1 ordenados [0.1, 0.3, 0.6, 0.9] -> índice 1 -> semilla 2
        ([0.6, 0.9, 0.3, 0.1], 2),
        # empate en el F1: se desempata por semilla
        ([0.5, 0.5, 0.5], 1),
    ],
)
def test_semilla_representativa_es_mediana_inferior(f1s, esperada):
    from clasificacion_adult.experimentos import semilla_representativa

    corridas = pd.DataFrame({"semilla": range(len(f1s)), "f1": f1s})
    assert semilla_representativa(corridas) == esperada


def test_neuronas_inactivas_umbral_y_nunca():
    from clasificacion_adult.modelos import fraccion_activacion, neuronas_inactivas

    mlp = MLPClassifier(hidden_layer_sizes=(3,))
    # neurona 0: siempre activa; 1: activa solo en x = 100 (1 de 200 = 0,5 %); 2: nunca
    mlp.coefs_ = [np.array([[1.0, 1.0, -1.0]]), np.ones((3, 1))]
    mlp.intercepts_ = [np.array([1.0, -99.5, 0.0]), np.array([0.0])]
    X = np.append(np.linspace(0, 1, 199), 100.0).reshape(-1, 1)
    np.testing.assert_allclose(fraccion_activacion(mlp, X)[0], [1.0, 1 / 200, 0.0])
    assert neuronas_inactivas(mlp, X, umbral=0.01) == [(2, 1)]
    assert neuronas_muertas(mlp, X) == [1]


def test_experimento_semillas_guarda_y_resume(dirs_temporales):
    from clasificacion_adult.experimentos import METRICAS_SEMILLAS, experimento_semillas

    X, y = _datos_sinteticos(400)
    res = experimento_semillas(X, y, semillas=[0, 1, 2], verbose=False)
    assert len(res.por_semilla) == 3 * 3
    assert {"vn", "fp", "fn", "vp", "f1_macro"} <= set(res.por_semilla.columns)
    assert "macro_f1" not in res.por_semilla.columns
    for met in METRICAS_SEMILLAS:
        assert {f"{met}_mean", f"{met}_std"} <= set(res.resumen.columns)
    # desviación muestral (n - 1)
    f1_mlp1 = res.por_semilla.loc[res.por_semilla["modelo"] == NOMBRE_MLP_1, "f1"]
    assert res.resumen.loc[NOMBRE_MLP_1, "f1_std"] == pytest.approx(f1_mlp1.std(ddof=1))
    # la corrida representativa es la de la mediana inferior y sus conteos suman el test
    rep = res.representativos[NOMBRE_MLP_1]
    assert rep.semilla == res.resumen.loc[NOMBRE_MLP_1, "semilla_representativa"]
    fila = res.por_semilla.query("modelo == @NOMBRE_MLP_1 and semilla == @rep.semilla").iloc[0]
    assert fila[["vn", "fp", "fn", "vp"]].sum() == len(res.y_test)
    tablas = {p.stem for p in (dirs_temporales / "tablas").glob("*.csv")}
    assert tablas == {"metricas_por_semilla", "metricas_10_semillas"}
    figuras = {p.stem for p in (dirs_temporales / "figuras").glob("*.png")}
    assert figuras == {"matrices_confusion", "curvas_roc", "curvas_perdida_mlp"}


def test_experimento_colapso_guarda_tabla_y_figura(dirs_temporales):
    from clasificacion_adult.experimentos import experimento_colapso, resumir_colapso

    X, y = _datos_sinteticos(400)
    tabla = experimento_colapso(X, y, semillas=[0, 1], verbose=False)
    assert tabla["id"].tolist() == ["mlp_1capa"] * 2 + ["mlp_2x2"] * 2
    assert tabla["colapso"].tolist() == (tabla["f1"] < cfg.UMBRAL_F1_COLAPSO).tolist()
    mlp1 = tabla[tabla["id"] == "mlp_1capa"]
    assert mlp1["inactivas_capa2"].isna().all()  # el MLP de 1 capa no tiene 2.ª capa
    assert (tabla["nunca_activas_capa1"] <= tabla["inactivas_capa1"]).all()
    resumen = resumir_colapso(tabla)
    assert resumen["semillas"].tolist() == [2, 2]
    assert (dirs_temporales / "tablas" / "colapso_inicializacion.csv").exists()
    assert (dirs_temporales / "figuras" / "colapso_inicializacion.png").exists()


def test_grafico_colapso_indica_k_de_n():
    tabla = pd.DataFrame(
        {
            "modelo": [NOMBRE_MLP_1] * 2 + ["MLP 2 capas ocultas (2, 2)"] * 2,
            "semilla": [0, 1, 0, 1],
            "f1": [0.6, 0.65, 0.0, 0.62],
            "colapso": [False, False, True, False],
        }
    )
    fig = evaluacion.grafico_colapso(tabla)
    textos = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "0/2 colapsos" in textos and "1/2 colapsos" in textos


def test_descarga_fallida_no_deja_archivo(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request

    from clasificacion_adult.datos import descargar_datos

    def falla(*_args, **_kwargs):
        raise urllib.error.URLError("sin red")

    monkeypatch.setattr(urllib.request, "urlopen", falla)
    with pytest.raises(RuntimeError, match="No se pudo descargar"):
        descargar_datos(tmp_path)
    assert list(tmp_path.iterdir()) == []
