"""Punto de entrada por línea de comandos.

Requiere el paquete instalado (``pip install -r requirements.txt``, que incluye ``-e .``).

Uso:
    python main.py eda           # análisis exploratorio (figuras y tablas)
    python main.py entrenar      # partición 80/20, 3 modelos + línea base, diagnóstico MLP (2, 2)
    python main.py semillas      # 80/20 fija, random_state 0–9, modelos representativos
    python main.py colapso       # MLP con random_state 0–49: colapso por inicialización
    python main.py proporciones  # experimento 50–90 % × 5 semillas
    python main.py todo          # todo lo anterior
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # sin ventanas: las figuras solo se guardan

import matplotlib.pyplot as plt
import pandas as pd

from clasificacion_adult import config as cfg
from clasificacion_adult.datos import cargar_datos, limpiar_datos, obtener_datos_modelado
from clasificacion_adult.eda import configurar_estilo, ejecutar_eda
from clasificacion_adult.experimentos import (
    diagnostico_semillas_mlp2,
    entrenar_y_evaluar,
    experimento_colapso,
    experimento_proporciones,
    experimento_semillas,
    resumir_colapso,
)
from clasificacion_adult.preprocesamiento import detectar_atipicos_iqr, grafico_log1p
from clasificacion_adult.utilidades import guardar_figura, guardar_tabla

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def comando_eda() -> None:
    """EDA sobre los datos crudos + atípicos y log1p sobre los datos sin duplicados."""
    datos = cargar_datos()
    print(f"Datos unificados: {datos.shape[0]:,} filas × {datos.shape[1]} columnas")
    tablas = ejecutar_eda(datos)
    print("\nDistribución de clases:\n", tablas["clases"])
    print("\nNulos y tasa de >50K:\n", tablas["nulos"])
    print("\nV de Cramér frente a income:\n", tablas["v_cramer"].round(3))

    limpios = limpiar_datos(datos)
    print(f"\nTras quitar {len(datos) - len(limpios)} duplicados: {len(limpios):,} filas")
    atipicos = detectar_atipicos_iqr(limpios)
    guardar_tabla(atipicos, "atipicos_iqr")
    print("\nAtípicos (IQR, no se eliminan):\n", atipicos)
    fig = grafico_log1p(limpios)
    guardar_figura(fig, "prep_log1p_capital")
    plt.close(fig)
    print(f"\nFiguras en {cfg.DIR_FIGURAS}\nTablas en {cfg.DIR_TABLAS}")


def comando_entrenar() -> None:
    """Experimento principal 80/20 y diagnóstico del MLP (2, 2) con varias semillas."""
    X, y = obtener_datos_modelado()
    res = entrenar_y_evaluar(X, y)
    print(
        f"Train: {len(res.X_train):,} | Test: {len(res.X_test):,} | "
        f"Entradas de la red: {res.n_entradas}"
    )
    print("\nMétricas en test (clase positiva = >50K):\n", res.metricas.round(4))
    print("\nDiagnóstico del MLP (2, 2) por semilla:\n", diagnostico_semillas_mlp2(res))


def comando_semillas() -> None:
    """Partición 80/20 fija; cada modelo con random_state 0–9 y modelo representativo."""
    X, y = obtener_datos_modelado()
    print(f"Experimento de semillas ({len(cfg.SEMILLAS_MODELO)} semillas × 3 modelos):")
    res = experimento_semillas(X, y)
    columnas = [
        "f1_mean",
        "f1_std",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "roc_auc_mean",
        "semilla_representativa",
    ]
    print("\n", res.resumen[columnas].round(4))


def comando_colapso() -> None:
    """MLP de 1 capa y (2, 2) con random_state 0–49: cuántas inicializaciones colapsan."""
    X, y = obtener_datos_modelado()
    print(f"Colapso por inicialización ({len(cfg.SEMILLAS_COLAPSO)} semillas × 2 MLP):")
    tabla = experimento_colapso(X, y)
    print("\n", resumir_colapso(tabla))
    print("\n", tabla[tabla["colapso"]].to_string(index=False))


def comando_proporciones() -> None:
    """Experimento de proporciones de entrenamiento."""
    X, y = obtener_datos_modelado()
    print("Experimento de proporciones (5 proporciones × 5 semillas × 3 modelos):")
    _, resumen = experimento_proporciones(X, y)
    print("\n", resumen[["accuracy", "precision", "recall", "f1"]])


COMANDOS: dict[str, list[Callable[[], None]]] = {
    "eda": [comando_eda],
    "entrenar": [comando_entrenar],
    "semillas": [comando_semillas],
    "colapso": [comando_colapso],
    "proporciones": [comando_proporciones],
    "todo": [
        comando_eda,
        comando_entrenar,
        comando_semillas,
        comando_colapso,
        comando_proporciones,
    ],
}


def main(argv: Sequence[str] | None = None) -> None:
    """Interpreta los argumentos y ejecuta los pasos pedidos."""
    parser = argparse.ArgumentParser(description="Clasificación Adult con redes neuronales")
    parser.add_argument("comando", choices=list(COMANDOS))
    args = parser.parse_args(argv)
    configurar_estilo()
    for paso in COMANDOS[args.comando]:
        titulo = paso.__name__.replace("comando_", "").upper()
        print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")
        paso()


if __name__ == "__main__":
    main()
