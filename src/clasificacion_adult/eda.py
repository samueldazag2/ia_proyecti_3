"""Análisis exploratorio: tablas resumen y gráficas.

Las funciones de este módulo son puras: las de tablas solo calculan y las de gráficos solo
dibujan a partir de datos ya calculados. Nada se escribe a disco salvo en
:func:`ejecutar_eda`, que orquesta el EDA completo y guarda sus salidas.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import chi2_contingency

from . import config as cfg
from .utilidades import guardar_figura, guardar_tabla


def configurar_estilo() -> None:
    """Aplica el estilo de seaborn a todas las figuras siguientes."""
    sns.set_theme(style=cfg.ESTILO_SEABORN)


def _es_positivo(datos: pd.DataFrame) -> pd.Series:
    return datos[cfg.OBJETIVO] == cfg.CLASE_POSITIVA


# --- Tablas ----------------------------------------------------------------
def resumen_tipos(datos: pd.DataFrame) -> pd.DataFrame:
    """Tipo, nulos (conteo y %) y número de valores únicos por columna."""
    return pd.DataFrame(
        {
            "tipo": datos.dtypes.astype(str),
            "nulos": datos.isna().sum(),
            "pct_nulos": (datos.isna().mean() * 100).round(2),
            "unicos": datos.nunique(),
        }
    )


def estadisticas_descriptivas(datos: pd.DataFrame) -> pd.DataFrame:
    """describe() de las variables numéricas originales."""
    return datos[cfg.NUMERICAS_ORIGINALES].describe().T.round(2)


def distribucion_clases(datos: pd.DataFrame) -> pd.DataFrame:
    """Conteo y porcentaje de cada clase del objetivo."""
    conteo = datos[cfg.OBJETIVO].value_counts()
    return pd.DataFrame({"conteo": conteo, "porcentaje": (conteo / conteo.sum() * 100).round(2)})


def analisis_nulos(datos: pd.DataFrame) -> pd.DataFrame:
    """Evidencia de que los nulos no son aleatorios: tasa de >50K con y sin nulo."""
    positivo = _es_positivo(datos)
    filas = []
    for col in cfg.COLUMNAS_CON_NULOS:
        m = datos[col].isna()
        filas.append(
            {
                "columna": col,
                "nulos": int(m.sum()),
                "tasa_>50K_con_nulo_%": round(positivo[m].mean() * 100, 2),
                "tasa_>50K_sin_nulo_%": round(positivo[~m].mean() * 100, 2),
            }
        )
    return pd.DataFrame(filas).set_index("columna")


def correlacion_pearson(datos: pd.DataFrame) -> pd.DataFrame:
    """Matriz de Pearson entre las numéricas originales y el objetivo binario."""
    tabla = datos[cfg.NUMERICAS_ORIGINALES].copy()
    tabla["income_>50K"] = _es_positivo(datos).astype(int)
    return tabla.corr(method="pearson")


def v_cramer(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér entre dos variables categóricas (0 = independencia, 1 = asociación total)."""
    contingencia = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingencia, correction=False)[0]
    n = contingencia.to_numpy().sum()
    k = min(contingencia.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def v_cramer_objetivo(datos: pd.DataFrame) -> pd.Series:
    """V de Cramér de cada categórica original frente a income, de mayor a menor."""
    return pd.Series(
        {
            col: v_cramer(datos[col].fillna(cfg.VALOR_NULO), datos[cfg.OBJETIVO])
            for col in cfg.CATEGORICAS_ORIGINALES
        },
        name="V_de_Cramer",
    ).sort_values(ascending=False)


def redundancia_education(datos: pd.DataFrame) -> pd.DataFrame:
    """Comprueba la correspondencia 1:1 entre education y education_num."""
    return (
        datos.groupby("education")["education_num"]
        .agg(["nunique", "min"])
        .rename(columns={"nunique": "valores_distintos", "min": "education_num"})
        .sort_values("education_num")
    )


# --- Gráficos --------------------------------------------------------------
def grafico_distribucion_clases(tabla: pd.DataFrame) -> Figure:
    """Barras con el conteo y porcentaje de cada clase (recibe :func:`distribucion_clases`)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(tabla.index, tabla["conteo"], color=[cfg.COLORES_CLASES[c] for c in tabla.index])
    for i, (n, p) in enumerate(zip(tabla["conteo"], tabla["porcentaje"], strict=True)):
        ax.text(i, n, f"{n:,}\n({p:.1f} %)", ha="center", va="bottom")
    ax.set(
        title="Distribución de la clase objetivo",
        ylabel="Número de personas",
        ylim=(0, tabla["conteo"].max() * 1.2),
    )
    return fig


def histogramas_numericas(datos: pd.DataFrame) -> Figure:
    """Histogramas (densidad) de cada numérica separados por clase."""
    fig, ejes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(ejes.ravel(), cfg.NUMERICAS_ORIGINALES, strict=True):
        sns.histplot(
            data=datos,
            x=col,
            hue=cfg.OBJETIVO,
            bins=40,
            stat="density",
            common_norm=False,
            element="step",
            palette=cfg.COLORES_CLASES,
            ax=ax,
        )
        if col in cfg.NUMERICAS_LOG:
            ax.set_yscale("log")
            ax.set_title(f"{col} (eje y logarítmico)")
        else:
            ax.set_title(col)
    fig.suptitle("Variables numéricas por clase (densidad)", fontsize=14)
    fig.tight_layout()
    return fig


def barras_categoricas(datos: pd.DataFrame) -> Figure:
    """Frecuencia de cada categoría (barras) y tasa de >50K (puntos, eje derecho)."""
    fig, ejes = plt.subplots(4, 2, figsize=(16, 22))
    positivo = _es_positivo(datos)
    tasa_global = positivo.mean() * 100
    for ax, col in zip(ejes.ravel(), cfg.CATEGORICAS_ORIGINALES, strict=True):
        serie = datos[col].fillna("(nulo)")
        resumen = (
            pd.DataFrame({"cat": serie, "pos": positivo})
            .groupby("cat")["pos"]
            .agg(["size", "mean"])
            .sort_values("size", ascending=False)
            .head(cfg.MAX_CATEGORIAS_GRAFICO)
        )
        x = np.arange(len(resumen))
        ax.bar(x, resumen["size"], color=cfg.COLOR_BARRAS)
        ax.set_xticks(x, resumen.index, rotation=60, ha="right")
        ax.set_ylabel("Frecuencia")
        ax2 = ax.twinx()
        ax2.plot(x, resumen["mean"] * 100, "o", color=cfg.COLOR_RESALTE)
        ax2.axhline(tasa_global, ls="--", color=cfg.COLOR_RESALTE, lw=1)
        ax2.set_ylabel("% con >50K", color=cfg.COLOR_RESALTE)
        ax2.set_ylim(0, 100)
        ax2.grid(False)
        recortada = serie.nunique() > cfg.MAX_CATEGORIAS_GRAFICO
        ax.set_title(
            f"{col} ({cfg.MAX_CATEGORIAS_GRAFICO} categorías más frecuentes)" if recortada else col
        )
    fig.suptitle(
        "Variables categóricas: frecuencia (barras) y tasa de >50K "
        "(puntos; línea punteada = tasa global)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def boxplots_numericas(datos: pd.DataFrame) -> Figure:
    """Diagramas de caja por clase para la detección visual de atípicos."""
    fig, ejes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(ejes.ravel(), cfg.NUMERICAS_ORIGINALES, strict=True):
        sns.boxplot(
            data=datos,
            x=cfg.OBJETIVO,
            y=col,
            hue=cfg.OBJETIVO,
            palette=cfg.COLORES_CLASES,
            legend=False,
            ax=ax,
            flierprops={"markersize": 2, "alpha": 0.4},
        )
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Diagramas de caja de las variables numéricas (atípicos)", fontsize=14)
    fig.tight_layout()
    return fig


def grafico_nulos(datos: pd.DataFrame) -> Figure:
    """Barras horizontales con el número de nulos por columna."""
    nulos = datos.isna().sum().sort_values()
    valores = nulos.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(nulos.index, valores, color=np.where(valores > 0, cfg.COLOR_RESALTE, cfg.COLOR_NEUTRO))
    for i, v in enumerate(valores):
        if v:
            ax.text(v, i, f" {v:,} ({v / len(datos) * 100:.1f} %)", va="center")
    ax.set(
        title="Valores nulos por columna ('?' en los archivos originales)",
        xlabel="Número de nulos",
        xlim=(0, valores.max() * 1.4),
    )
    return fig


def grafico_correlacion(corr: pd.DataFrame) -> Figure:
    """Mapa de calor de una matriz de correlación (recibe :func:`correlacion_pearson`)."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlación de Pearson (numéricas + objetivo)")
    return fig


def grafico_v_cramer(valores: pd.Series) -> Figure:
    """Barras horizontales de la V de Cramér (recibe :func:`v_cramer_objetivo`)."""
    invertidos = valores.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(invertidos.index, invertidos.to_numpy(), color=cfg.COLOR_SECUNDARIO)
    for i, v in enumerate(invertidos.to_numpy()):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.set(
        title="V de Cramér de cada categórica frente a income",
        xlabel="V de Cramér",
        xlim=(0, valores.max() * 1.2),
    )
    return fig


# --- Orquestación ----------------------------------------------------------
def ejecutar_eda(datos: pd.DataFrame, guardar: bool = True) -> dict[str, pd.DataFrame | pd.Series]:
    """Ejecuta todo el EDA sobre los datos crudos unificados.

    Si ``guardar`` es True, escribe las tablas en ``resultados/tablas`` y las figuras en
    ``resultados/figuras``. Devuelve las tablas calculadas.
    """
    configurar_estilo()
    clases = distribucion_clases(datos)
    correlacion = correlacion_pearson(datos)
    v_cramer_tabla = v_cramer_objetivo(datos)
    tablas: dict[str, pd.DataFrame | pd.Series] = {
        "tipos": resumen_tipos(datos),
        "descriptivas": estadisticas_descriptivas(datos),
        "clases": clases,
        "nulos": analisis_nulos(datos),
        "correlacion": correlacion,
        "v_cramer": v_cramer_tabla,
    }
    if guardar:
        archivos = {
            "tipos": "eda_tipos_nulos",
            "descriptivas": "eda_estadisticas_descriptivas",
            "clases": "eda_distribucion_clases",
            "nulos": "eda_analisis_nulos",
        }
        for clave, nombre in archivos.items():
            guardar_tabla(pd.DataFrame(tablas[clave]), nombre)
        guardar_tabla(correlacion.round(4), "eda_correlacion_pearson")
        guardar_tabla(v_cramer_tabla.round(4).to_frame(), "eda_v_cramer")
        figuras = {
            "eda_distribucion_clases": grafico_distribucion_clases(clases),
            "eda_histogramas_numericas": histogramas_numericas(datos),
            "eda_barras_categoricas": barras_categoricas(datos),
            "eda_boxplots_numericas": boxplots_numericas(datos),
            "eda_nulos_por_columna": grafico_nulos(datos),
            "eda_correlacion_pearson": grafico_correlacion(correlacion),
            "eda_v_cramer": grafico_v_cramer(v_cramer_tabla),
        }
        for nombre, fig in figuras.items():
            guardar_figura(fig, nombre)
            plt.close(fig)
    return tablas
