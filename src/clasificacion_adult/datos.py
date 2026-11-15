"""Descarga, carga, limpieza y unificación del dataset Adult (UCI)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from . import config as cfg


def descargar_datos(directorio: Path = cfg.DIR_DATOS_CRUDOS) -> None:
    """Descarga adult.data, adult.test y adult.names si no existen localmente.

    Cada archivo se escribe primero en ``<nombre>.part`` y solo se renombra al terminar,
    así una descarga interrumpida nunca deja un archivo incompleto con el nombre final.

    Raises:
        RuntimeError: si algún archivo no se puede descargar.
    """
    directorio.mkdir(parents=True, exist_ok=True)
    for nombre in cfg.ARCHIVOS:
        destino = directorio / nombre
        if destino.exists():
            continue
        temporal = destino.with_name(nombre + ".part")
        url = cfg.URL_BASE + nombre
        print(f"Descargando {nombre} ...")
        try:
            with urllib.request.urlopen(url, timeout=cfg.TIMEOUT_DESCARGA_S) as respuesta:
                temporal.write_bytes(respuesta.read())
        except OSError as error:  # URLError, HTTPError y timeouts heredan de OSError
            temporal.unlink(missing_ok=True)
            raise RuntimeError(
                f"No se pudo descargar {url}. Descárguelo manualmente en {directorio}."
            ) from error
        temporal.replace(destino)


def _leer(ruta: Path, saltar: int = 0) -> pd.DataFrame:
    return pd.read_csv(
        ruta, header=None, names=cfg.COLUMNAS, skipinitialspace=True, na_values="?", skiprows=saltar
    )


def cargar_datos(directorio: Path = cfg.DIR_DATOS_CRUDOS) -> pd.DataFrame:
    """Carga y concatena adult.data y adult.test (48 842 filas).

    adult.test tiene una primera línea de cabecera ("|1x3 Cross validator") y sus
    etiquetas terminan en punto ("<=50K."), que se elimina para unificarlas.
    """
    descargar_datos(directorio)
    entrenamiento = _leer(directorio / "adult.data")
    prueba = _leer(directorio / "adult.test", saltar=1)
    prueba[cfg.OBJETIVO] = prueba[cfg.OBJETIVO].str.rstrip(".")
    datos = pd.concat([entrenamiento, prueba], ignore_index=True)
    datos[cfg.OBJETIVO] = datos[cfg.OBJETIVO].str.strip()
    return datos


def limpiar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    """Limpieza previa a la partición (no aprende parámetros de los datos: sin fuga).

    1. Quita filas duplicadas exactas.
    2. Elimina ``fnlwgt`` (peso muestral) y ``education`` (redundante con education_num).

    Los nulos, la agrupación de países, log1p, el escalado y el one-hot se aplican
    dentro del Pipeline, que se ajusta solo con el conjunto de entrenamiento.
    """
    return datos.drop_duplicates().drop(columns=cfg.COLUMNAS_ELIMINADAS).reset_index(drop=True)


def separar_x_y(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve X y la etiqueta binaria y (1 si el ingreso es > 50K)."""
    y = (datos[cfg.OBJETIVO] == cfg.CLASE_POSITIVA).astype(int).rename("y")
    return datos.drop(columns=[cfg.OBJETIVO]), y


def obtener_datos_modelado() -> tuple[pd.DataFrame, pd.Series]:
    """Atajo: carga, limpia y separa X / y."""
    return separar_x_y(limpiar_datos(cargar_datos()))
