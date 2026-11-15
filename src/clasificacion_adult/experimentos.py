"""Experimentos: entrenamiento principal (80/20), diagnóstico del MLP (2, 2), experimento de
semillas (10 inicializaciones y modelo representativo), colapso por inicialización
(50 semillas) y experimento de proporciones de entrenamiento.

Cada experimento devuelve sus resultados; si ``guardar=True`` escribe además sus tablas y
figuras en ``resultados/`` (el guardado está separado en funciones ``guardar_*``).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from . import config as cfg
from .evaluacion import (
    calcular_metricas,
    conteos_confusion,
    grafico_colapso,
    grafico_comparacion_metricas,
    grafico_curvas_perdida,
    grafico_matrices_confusion,
    grafico_proporciones,
    grafico_roc,
    puntajes,
)
from .modelos import (
    ID_MODELO,
    NOMBRE_MLP_1,
    NOMBRE_MLP_2,
    crear_mlp_dos_capas,
    crear_modelos,
    neuronas_inactivas,
    neuronas_muertas,
)
from .preprocesamiento import construir_pipeline, numero_entradas
from .utilidades import guardar_figura, guardar_tabla

NOMBRE_BASE = "Línea base (siempre <=50K)"
METRICAS_PROPORCIONES = ["accuracy", "precision", "recall", "f1"]


def dividir(
    X: pd.DataFrame,
    y: pd.Series,
    proporcion_train: float = 1 - cfg.PROPORCION_TEST,
    semilla: int = cfg.SEMILLA,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición estratificada; devuelve (X_train, X_test, y_train, y_test)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=proporcion_train, stratify=y, random_state=semilla
    )
    return X_train, X_test, y_train, y_test


@dataclass
class ResultadoEntrenamiento:
    """Todo lo necesario para analizar el experimento principal."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    n_entradas: int
    pipelines: dict[str, Pipeline] = field(default_factory=dict)
    predicciones: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    tiempos: dict[str, float] = field(default_factory=dict)
    metricas: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def redes(self) -> list[str]:
        """Nombres de los modelos entrenados, sin la línea base."""
        return [k for k in self.pipelines if k != NOMBRE_BASE]

    def mlps(self) -> dict[str, MLPClassifier]:
        """Los MLP entrenados (los que tienen curva de pérdida)."""
        return {
            k: p.named_steps["modelo"]
            for k, p in self.pipelines.items()
            if isinstance(p.named_steps["modelo"], MLPClassifier)
        }


def ajustar_sin_avisos(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Ajusta el pipeline silenciando ConvergenceWarning (el early stopping es intencional)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        return pipeline.fit(X, y)


# --- Experimento principal -------------------------------------------------
def entrenar_y_evaluar(
    X: pd.DataFrame, y: pd.Series, guardar: bool = True
) -> ResultadoEntrenamiento:
    """Entrena los 3 modelos + línea base con la partición 80/20 y calcula las métricas."""
    X_train, X_test, y_train, y_test = dividir(X, y)
    n_entradas = numero_entradas(X_train)
    res = ResultadoEntrenamiento(X_train, X_test, y_train, y_test, n_entradas)

    modelos: dict[str, Any] = {NOMBRE_BASE: DummyClassifier(strategy="most_frequent")}
    modelos.update(crear_modelos(n_entradas))

    filas = []
    for nombre, modelo in modelos.items():
        inicio = time.perf_counter()
        pipeline = ajustar_sin_avisos(construir_pipeline(modelo), X_train, y_train)
        res.tiempos[nombre] = time.perf_counter() - inicio
        res.pipelines[nombre] = pipeline
        res.predicciones[nombre] = pipeline.predict(X_test)
        res.scores[nombre] = puntajes(pipeline, X_test)
        metricas = calcular_metricas(y_test, res.predicciones[nombre], res.scores[nombre])
        filas.append(
            {
                "modelo": nombre,
                **metricas,
                "epocas": getattr(modelo, "n_iter_", None),
                "tiempo_s": round(res.tiempos[nombre], 2),
            }
        )
    res.metricas = pd.DataFrame(filas).set_index("modelo")
    if guardar:
        guardar_resultados_principales(res)
    return res


def guardar_resultados_principales(res: ResultadoEntrenamiento) -> None:
    """Escribe la tabla de métricas y la figura comparativa del experimento principal.

    Las matrices de confusión, la ROC y las curvas de pérdida oficiales se generan con los
    modelos representativos de :func:`experimento_semillas`.
    """
    guardar_tabla(res.metricas.round(4), "metricas_modelos")
    _guardar_figuras(
        {"comparacion_metricas": grafico_comparacion_metricas(res.metricas, NOMBRE_BASE)}
    )


def _guardar_figuras(figuras: dict[str, Figure]) -> None:
    for nombre, fig in figuras.items():
        guardar_figura(fig, nombre)
        plt.close(fig)


# --- Diagnóstico del MLP (2, 2) --------------------------------------------
def diagnostico_semillas_mlp2(
    res: ResultadoEntrenamiento,
    semillas: Sequence[int] = cfg.SEMILLAS_DIAGNOSTICO,
    guardar: bool = True,
) -> pd.DataFrame:
    """Reentrena el MLP (2, 2) con varias semillas sobre la partición principal.

    Para cada semilla reporta las neuronas ReLU muertas por capa, las épocas, la pérdida
    final de entrenamiento y el F1 / ROC-AUC en test.
    """
    filas = []
    for semilla in semillas:
        pipeline = ajustar_sin_avisos(
            construir_pipeline(crear_mlp_dos_capas(semilla)), res.X_train, res.y_train
        )
        mlp = pipeline.named_steps["modelo"]
        x_transformado = pipeline.named_steps["preprocesador"].transform(res.X_train)
        muertas = neuronas_muertas(mlp, x_transformado)
        m = calcular_metricas(
            res.y_test, pipeline.predict(res.X_test), puntajes(pipeline, res.X_test)
        )
        filas.append(
            {
                "semilla": semilla,
                "muertas_capa1": muertas[0],
                "muertas_capa2": muertas[1],
                "epocas": mlp.n_iter_,
                "perdida_final": round(mlp.loss_curve_[-1], 4),
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
            }
        )
    tabla = pd.DataFrame(filas).set_index("semilla").round(4)
    if guardar:
        guardar_tabla(tabla, "diagnostico_mlp_2_2_semillas")
    return tabla


# --- Experimento de proporciones -------------------------------------------
def experimento_proporciones(
    X: pd.DataFrame,
    y: pd.Series,
    proporciones: Sequence[float] = tuple(cfg.PROPORCIONES_TRAIN),
    semillas: Sequence[int] = tuple(cfg.SEMILLAS_EXPERIMENTO),
    guardar: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena los 3 modelos para cada proporción de train y cada semilla.

    La semilla controla tanto la partición estratificada como la inicialización de la
    red, así la desviación estándar recoge ambas fuentes de variabilidad. El conjunto
    de prueba es siempre el complemento (p. ej. 30 % de test cuando train = 70 %).

    Devuelve (detalle por corrida, resumen con media ± desviación).
    """
    filas = []
    for p in proporciones:
        for s in semillas:
            X_tr, X_te, y_tr, y_te = dividir(X, y, p, s)
            n_entradas = numero_entradas(X_tr)
            for nombre, modelo in crear_modelos(n_entradas, semilla=s).items():
                pipeline = ajustar_sin_avisos(construir_pipeline(modelo), X_tr, y_tr)
                m = calcular_metricas(y_te, pipeline.predict(X_te))
                filas.append(
                    {
                        "proporcion_train": p,
                        "semilla": s,
                        "modelo": nombre,
                        **{k: m[k] for k in METRICAS_PROPORCIONES},
                    }
                )
            if verbose:
                print(f"  train = {p:.0%} | semilla {s} listo")

    detalle = pd.DataFrame(filas)
    resumen = resumir_proporciones(detalle)
    if guardar:
        guardar_tabla(resumen.round(4), "proporciones")
        guardar_tabla(detalle.round(4), "proporciones_detalle", indice=False)
        fig = grafico_proporciones(resumen)
        guardar_figura(fig, "proporciones_f1")
        plt.close(fig)
    return detalle, resumen


def resumir_proporciones(detalle: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación por modelo y proporción, más columnas legibles "media ± desv."."""
    resumen = detalle.groupby(["modelo", "proporcion_train"])[METRICAS_PROPORCIONES].agg(
        ["mean", "std"]
    )
    # Columnas MultiIndex (métrica, estadístico) -> "métrica_estadístico", p. ej. "f1_mean"
    resumen.columns = pd.Index(["_".join(map(str, col)) for col in resumen.columns])
    for met in METRICAS_PROPORCIONES:
        resumen[met] = (
            resumen[f"{met}_mean"].map("{:.4f}".format)
            + " ± "
            + resumen[f"{met}_std"].map("{:.4f}".format)
        )
    return resumen


# --- Corrida individual ------------------------------------------------------
@dataclass
class Corrida:
    """Un modelo entrenado con una semilla sobre la partición principal, y su evaluación."""

    nombre: str
    semilla: int
    pipeline: Pipeline
    prediccion: np.ndarray
    puntaje: np.ndarray
    metricas: dict[str, float]
    epocas: int
    tiempo_s: float

    @property
    def etiqueta(self) -> str:
        """Nombre para figuras, p. ej. "Perceptrón (semilla 5)"."""
        return f"{self.nombre} (semilla {self.semilla})"


def entrenar_corrida(
    modelo: Any,
    nombre: str,
    semilla: int,
    particion: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
) -> Corrida:
    """Ajusta ``construir_pipeline(modelo)`` con train y lo evalúa en test."""
    X_train, X_test, y_train, y_test = particion
    inicio = time.perf_counter()
    pipeline = ajustar_sin_avisos(construir_pipeline(modelo), X_train, y_train)
    tiempo = time.perf_counter() - inicio
    prediccion = pipeline.predict(X_test)
    puntaje = puntajes(pipeline, X_test)
    return Corrida(
        nombre=nombre,
        semilla=semilla,
        pipeline=pipeline,
        prediccion=prediccion,
        puntaje=puntaje,
        metricas=calcular_metricas(y_test, prediccion, puntaje),
        epocas=int(pipeline.named_steps["modelo"].n_iter_),
        tiempo_s=tiempo,
    )


# --- Experimento de semillas (10 inicializaciones) ---------------------------
METRICAS_SEMILLAS = [
    "accuracy", "precision", "recall", "f1", "f1_macro", "balanced_accuracy", "roc_auc",
    "epocas", "tiempo_s",
]  # fmt: skip


@dataclass
class ResultadoSemillas:
    """Métricas por semilla, su resumen y la corrida representativa de cada modelo."""

    y_test: pd.Series
    por_semilla: pd.DataFrame
    resumen: pd.DataFrame
    representativos: dict[str, Corrida]


def semilla_representativa(corridas: pd.DataFrame) -> int:
    """Semilla de la mediana INFERIOR del F1.

    Ordena las corridas de menor a mayor F1 (empates por semilla) y toma la de índice
    ``(n - 1) // 2``; con 10 corridas es la 5.ª más baja.
    """
    ordenadas = corridas.sort_values(["f1", "semilla"], kind="mergesort").reset_index(drop=True)
    return int(ordenadas.loc[(len(ordenadas) - 1) // 2, "semilla"])


def experimento_semillas(
    X: pd.DataFrame,
    y: pd.Series,
    semillas: Sequence[int] = tuple(cfg.SEMILLAS_MODELO),
    guardar: bool = True,
    verbose: bool = True,
) -> ResultadoSemillas:
    """Partición 80/20 fija (semilla 42); cada modelo se entrena con cada ``random_state``.

    Aísla la variabilidad debida a la inicialización. Con ``guardar=True`` escribe
    ``metricas_por_semilla.csv``, ``metricas_10_semillas.csv`` y las matrices de confusión,
    la ROC y las curvas de pérdida de los modelos representativos (mediana inferior del F1).
    """
    particion = dividir(X, y)
    n_entradas = numero_entradas(particion[0])
    corridas: dict[tuple[str, int], Corrida] = {}
    filas = []
    for s in semillas:
        for nombre, modelo in crear_modelos(n_entradas, semilla=s).items():
            c = entrenar_corrida(modelo, nombre, s, particion)
            corridas[(nombre, s)] = c
            metricas = {("f1_macro" if k == "macro_f1" else k): v for k, v in c.metricas.items()}
            filas.append(
                {
                    "modelo": nombre,
                    "id": ID_MODELO[nombre],
                    "semilla": s,
                    **metricas,
                    "epocas": c.epocas,
                    "tiempo_s": round(c.tiempo_s, 2),
                    **conteos_confusion(particion[3], c.prediccion),
                }
            )
        if verbose:
            print(f"  semilla {s} lista")

    por_semilla = pd.DataFrame(filas)
    representativos = {
        nombre: corridas[(nombre, semilla_representativa(grupo))]
        for nombre, grupo in por_semilla.groupby("modelo", sort=False)
    }
    res = ResultadoSemillas(
        y_test=particion[3],
        por_semilla=por_semilla,
        resumen=resumir_semillas(por_semilla, representativos),
        representativos=representativos,
    )
    if guardar:
        guardar_resultados_semillas(res)
    return res


def resumir_semillas(
    por_semilla: pd.DataFrame, representativos: dict[str, Corrida]
) -> pd.DataFrame:
    """Media y desviación muestral (n − 1) por modelo, más la semilla representativa."""
    resumen = por_semilla.groupby("modelo", sort=False)[METRICAS_SEMILLAS].agg(["mean", "std"])
    resumen.columns = pd.Index(["_".join(map(str, col)) for col in resumen.columns])
    resumen.insert(0, "id", [ID_MODELO[str(m)] for m in resumen.index])
    resumen.insert(1, "n_semillas", por_semilla.groupby("modelo", sort=False).size())
    resumen["semilla_representativa"] = [representativos[str(m)].semilla for m in resumen.index]
    resumen["f1_representativo"] = [representativos[str(m)].metricas["f1"] for m in resumen.index]
    return resumen


def guardar_resultados_semillas(res: ResultadoSemillas) -> None:
    """Escribe las tablas del experimento de semillas y las figuras de los representativos."""
    guardar_tabla(res.por_semilla.round(4), "metricas_por_semilla", indice=False)
    guardar_tabla(res.resumen.round(4), "metricas_10_semillas")
    reps = list(res.representativos.values())
    mlps = {
        c.etiqueta: c.pipeline.named_steps["modelo"]
        for c in reps
        if isinstance(c.pipeline.named_steps["modelo"], MLPClassifier)
    }
    _guardar_figuras(
        {
            "matrices_confusion": grafico_matrices_confusion(
                res.y_test, {c.etiqueta: c.prediccion for c in reps}
            ),
            "curvas_roc": grafico_roc(res.y_test, {c.etiqueta: c.puntaje for c in reps}),
            "curvas_perdida_mlp": grafico_curvas_perdida(mlps),
        }
    )


# --- Colapso por inicialización (50 semillas) --------------------------------
def experimento_colapso(
    X: pd.DataFrame,
    y: pd.Series,
    semillas: Sequence[int] = tuple(cfg.SEMILLAS_COLAPSO),
    guardar: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Entrena los dos MLP con cada ``random_state`` sobre la partición principal.

    Una corrida colapsa si su F1 de >50K es menor que ``cfg.UMBRAL_F1_COLAPSO``. Para cada
    capa oculta cuenta, sobre los ejemplos de prueba, las neuronas inactivas (Z > 0 en menos
    de ``cfg.UMBRAL_ACTIVACION`` de los ejemplos) y las que no se activan nunca.
    """
    particion = dividir(X, y)
    n_entradas = numero_entradas(particion[0])
    filas = []
    for s in semillas:
        modelos = crear_modelos(n_entradas, semilla=s)
        for nombre in (NOMBRE_MLP_1, NOMBRE_MLP_2):
            c = entrenar_corrida(modelos[nombre], nombre, s, particion)
            mlp = c.pipeline.named_steps["modelo"]
            x_test = c.pipeline.named_steps["preprocesador"].transform(particion[1])
            fila: dict[str, Any] = {
                "modelo": nombre,
                "id": ID_MODELO[nombre],
                "semilla": s,
                "f1": c.metricas["f1"],
                "colapso": c.metricas["f1"] < cfg.UMBRAL_F1_COLAPSO,
                "epocas": c.epocas,
            }
            capas = neuronas_inactivas(mlp, x_test)
            for i, (inactivas, nunca) in enumerate(capas, start=1):
                fila[f"neuronas_capa{i}"] = mlp.hidden_layer_sizes[i - 1]
                fila[f"inactivas_capa{i}"] = inactivas
                fila[f"nunca_activas_capa{i}"] = nunca
            filas.append(fila)
        if verbose and (s + 1) % 10 == 0:
            print(f"  {s + 1}/{len(semillas)} semillas listas")

    tabla = pd.DataFrame(filas).sort_values(["id", "semilla"], kind="mergesort")
    conteos = [c for c in tabla.columns if "capa" in c]
    tabla = tabla.astype(dict.fromkeys(conteos, "Int64")).reset_index(drop=True)
    if guardar:
        guardar_tabla(tabla.round(4), "colapso_inicializacion", indice=False)
        _guardar_figuras({"colapso_inicializacion": grafico_colapso(tabla)})
    return tabla


def resumir_colapso(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por modelo: número de semillas, colapsos y lista de semillas que colapsan."""
    filas = []
    for modelo, grupo in tabla.groupby("modelo", sort=False):
        colapsadas = grupo.loc[grupo["colapso"], "semilla"].astype(int).tolist()
        filas.append(
            {
                "modelo": modelo,
                "semillas": len(grupo),
                "colapsos": len(colapsadas),
                "semillas_colapso": ", ".join(map(str, colapsadas)),
            }
        )
    return pd.DataFrame(filas).set_index("modelo")
