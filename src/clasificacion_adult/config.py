"""Configuración central del proyecto: rutas, columnas, semilla, hiperparámetros y estilo.

Es la única fuente de constantes del paquete: ningún otro módulo define números o
cadenas de configuración propios.
"""

import os
from pathlib import Path
from typing import Any

# --- Rutas -----------------------------------------------------------------
# Por defecto, la raíz del repositorio (src layout + instalación editable). Se puede
# sobrescribir con la variable de entorno CLASIFICACION_ADULT_RAIZ.
RAIZ = Path(os.environ.get("CLASIFICACION_ADULT_RAIZ", Path(__file__).resolve().parents[2]))
DIR_DATOS_CRUDOS = RAIZ / "data" / "raw"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
DIR_TABLAS = DIR_RESULTADOS / "tablas"

URL_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
ARCHIVOS = ("adult.data", "adult.test", "adult.names")
TIMEOUT_DESCARGA_S = 60

# --- Columnas --------------------------------------------------------------
COLUMNAS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]  # fmt: skip
OBJETIVO = "income"
CLASE_POSITIVA = ">50K"
CLASE_NEGATIVA = "<=50K"
NOMBRES_CLASES = [CLASE_NEGATIVA, CLASE_POSITIVA]

COLUMNAS_ELIMINADAS = ["fnlwgt", "education"]
COLUMNAS_CON_NULOS = ["workclass", "occupation", "native_country"]

NUMERICAS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
NUMERICAS_LOG = ["capital_gain", "capital_loss"]  # muy sesgadas -> log1p
NUMERICAS_ORIGINALES = [*NUMERICAS, "fnlwgt"]  # para el EDA (antes de eliminar)

COLUMNA_PAIS = "native_country"
CATEGORICAS = [
    "workclass", "marital_status", "occupation", "relationship", "race", "sex", COLUMNA_PAIS,
]  # fmt: skip
CATEGORICAS_ORIGINALES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", COLUMNA_PAIS,
]  # fmt: skip

VALOR_NULO = "Desconocido"
CATEGORIA_RARA = "Otro"
UMBRAL_FRECUENCIA_PAIS = 0.01  # países con < 1 % pasan a "Otro"
FACTOR_IQR = 1.5  # regla de Tukey para atípicos

# --- Reproducibilidad y particiones ---------------------------------------
SEMILLA = 42
PROPORCION_TEST = 0.20
PROPORCIONES_TRAIN = [0.5, 0.6, 0.7, 0.8, 0.9]
SEMILLAS_EXPERIMENTO = [0, 1, 2, 3, 4]
SEMILLAS_DIAGNOSTICO = [SEMILLA, *SEMILLAS_EXPERIMENTO]  # diagnóstico del MLP (2, 2)
SEMILLAS_MODELO = list(range(10))  # experimento de semillas (random_state del modelo)
SEMILLAS_COLAPSO = list(range(50))  # experimento de colapso por inicialización

# --- Colapso por inicialización ------------------------------------------
UMBRAL_F1_COLAPSO = 0.10  # F1 de >50K por debajo de este valor = colapso
UMBRAL_ACTIVACION = 0.01  # neurona inactiva: Z > 0 en menos del 1 % de los ejemplos

# --- Hiperparámetros -------------------------------------------------------
HIPER_PERCEPTRON: dict[str, Any] = {"fit_intercept": True, "max_iter": 1000, "tol": 1e-3}

HIPER_MLP: dict[str, Any] = {
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "alpha": 1e-4,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
}
CAPAS_MLP_2 = (2, 2)

# --- Figuras ---------------------------------------------------------------
DPI = 200
ESTILO_SEABORN = "whitegrid"
COLORES_CLASES = {CLASE_NEGATIVA: "#4C72B0", CLASE_POSITIVA: "#DD8452"}
PALETA_MODELOS = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
COLOR_RESALTE = "#C44E52"
COLOR_NEUTRO = "#BBBBBB"
COLOR_BARRAS = "#9DB4D3"
COLOR_SECUNDARIO = "#55A868"
MAX_CATEGORIAS_GRAFICO = 15  # categorías más frecuentes que se muestran por variable
