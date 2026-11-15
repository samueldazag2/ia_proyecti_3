"""Funciones auxiliares para guardar figuras y tablas en resultados/."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import config as cfg


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    """Guarda la figura como PNG (dpi 200) en resultados/figuras."""
    cfg.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.DIR_FIGURAS / f"{nombre}.png", dpi=cfg.DPI, bbox_inches="tight")


def guardar_tabla(tabla: pd.DataFrame, nombre: str, indice: bool = True) -> None:
    """Guarda la tabla como CSV (UTF-8) en resultados/tablas."""
    cfg.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(cfg.DIR_TABLAS / f"{nombre}.csv", index=indice, encoding="utf-8")
