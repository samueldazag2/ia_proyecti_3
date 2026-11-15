# Proyecto Aplicativo 3 – Redes neuronales feed-forward sobre *Adult (Census Income)*

**Curso:** Introducción a la Inteligencia Artificial · **Profesor:** Palacio Niño
**Equipo:** Samuel Daza, Nicolás Parra, Thomas López, Samuel Velandia

Clasificación binaria de ingresos (`>50K` frente a `<=50K`) con el dataset
[Adult de UCI](https://archive.ics.uci.edu/dataset/2/adult), comparando un perceptrón simple,
un MLP con una capa oculta (tantas neuronas como entradas) y un MLP con dos capas ocultas de
2 neuronas.

## Estructura

```
ia_pro_tres/
├── data/raw/                 adult.data, adult.test, adult.names (se descargan si faltan)
├── src/clasificacion_adult/
│   ├── config.py             rutas, columnas, SEMILLA = 42, hiperparámetros
│   ├── datos.py              descarga, carga, limpieza y unificación (48 842 filas)
│   ├── eda.py                tablas y gráficas del análisis exploratorio (funciones puras)
│   ├── preprocesamiento.py   ColumnTransformer + detección de atípicos (IQR)
│   ├── modelos.py            fábrica de las 3 arquitecturas + inspección (pesos, activación)
│   ├── evaluacion.py         métricas y gráficos (confusión, ROC, pérdida, proporciones, colapso)
│   ├── experimentos.py       principal, diagnóstico, 10 semillas, colapso (50), proporciones
│   └── utilidades.py         guardado de figuras (200 dpi) y tablas (CSV)
├── notebooks/Proyecto_Redes_Neuronales_Adult.ipynb   informe narrado (secciones 1–6)
├── tests/                    test_proyecto.py (datos reales) y test_unidades.py (sintéticos)
├── resultados/figuras/*.png  resultados/tablas/*.csv   (los genera main.py)
├── main.py                   CLI
├── requirements.txt          dependencias + `-e .` (instala el paquete)
└── pyproject.toml            metadatos, dependencias y configuración de pytest, ruff y mypy
```

Solo los orquestadores escriben a disco (`eda.ejecutar_eda`, `experimentos.*` con
`guardar=True` y `main.py`); las funciones de cálculo y de gráficos son puras. El notebook
usa `guardar=False`, así que no modifica `resultados/`.

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt     # incluye `-e .`: el paquete queda importable
pip install -e ".[dev]"             # opcional: ruff, mypy, pytest-cov
```

Probado con Python 3.14, pandas 3.1, scikit-learn 1.9 y matplotlib 3.11.

## Uso

```bash
python main.py eda            # análisis exploratorio
python main.py entrenar       # partición 80/20, 3 modelos + línea base, diagnóstico MLP (2, 2)
python main.py semillas       # 80/20 fija, random_state 0–9, modelos representativos
python main.py colapso        # MLP con random_state 0–49: colapso por inicialización
python main.py proporciones   # train = 50–90 % × semillas 0–4
python main.py todo           # todo lo anterior (≈ 15 min)
pytest                        # 36 pruebas
pytest --cov=src              # cobertura (≈ 94 %)
ruff check . && ruff format --check . && mypy src
jupyter notebook notebooks/Proyecto_Redes_Neuronales_Adult.ipynb
```

## Decisiones de preprocesamiento

| Paso | Decisión | Justificación |
|---|---|---|
| Duplicados | Se eliminan las 52 filas duplicadas exactas | Evitan doble peso y fuga entre train y test |
| `fnlwgt` | Se elimina | Peso muestral del censo, no es atributo de la persona; correlación ≈ 0 |
| `education` | Se elimina | Redundante 1:1 con `education_num` (ordinal) |
| Nulos | Categoría `"Desconocido"` | Los nulos no son aleatorios: `workclass`/`occupation` faltan juntos y esas filas tienen ~9 % de `>50K` frente a ~24 % |
| `native_country` | Países con < 1 % → `"Otro"` | 89 % es United-States; evita columnas casi vacías |
| Atípicos | Se detectan con IQR, **no** se eliminan; `log1p` en `capital_gain`/`capital_loss` | Son valores reales y concentran ejemplos de la clase minoritaria |
| Numéricas | `StandardScaler` | Misma escala para el descenso de gradiente |
| Categóricas | `OneHotEncoder(handle_unknown='ignore')` | Variables nominales |

Todo el preprocesamiento aprendido vive en un `ColumnTransformer` dentro de un `Pipeline`,
ajustado **solo con train**. Resultado: **53 entradas**.

## Modelos

| Modelo | Arquitectura | Parámetros |
|---|---|---|
| Perceptrón | 53 → 1 | 54 |
| MLP 1 capa oculta | 53 → 53 → 1 | 2 916 |
| MLP 2 capas ocultas | 53 → 2 → 2 → 1 | 117 |

MLP: ReLU, Adam (lr = 0.001), α = 1e-4, máx. 300 épocas, *early stopping* (10 % validación,
paciencia 10). Todas las capas tienen bias.

## Resultados

Partición 80/20 estratificada (`random_state=42`), métricas en test; precisión, recall y F1
son de la clase `>50K`.

| Modelo | Accuracy | Precisión | Recall | F1 | Macro-F1 | Bal. acc. | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Línea base (siempre `<=50K`) | 0.761 | 0.000 | 0.000 | 0.000 | 0.432 | 0.500 | 0.500 |
| Perceptrón | 0.797 | 0.560 | **0.713** | 0.627 | 0.744 | 0.768 | 0.867 |
| **MLP 1 capa oculta (53)** | **0.857** | **0.728** | 0.640 | **0.681** | **0.794** | **0.782** | **0.914** |
| MLP 2 capas ocultas (2, 2) | 0.760 | 0.200 | 0.000 | 0.001 | 0.432 | 0.500 | 0.500 |

* El **MLP de una capa oculta** es el mejor modelo en casi todas las métricas.
* Con `random_state=42` el **MLP (2, 2) colapsa** a la clase mayoritaria: una neurona de la
  segunda capa queda muerta (ReLU siempre 0) y la pérdida se estanca en la entropía a priori.
  Con las semillas 0–4 la misma arquitectura sí aprende (F1 ≈ 0.66); el notebook incluye el
  diagnóstico (sección 6.4, `resultados/tablas/diagnostico_mlp_2_2_semillas.csv`).
* El accuracy engaña: la línea base ya obtiene ~76 %. Se priorizan F1, recall, balanced
  accuracy y ROC-AUC.

**Experimento de proporciones** (F1 de `>50K`, media ± desv. sobre 5 semillas):

| Train | Perceptrón | MLP 1 capa | MLP (2, 2) |
|---|---|---|---|
| 50 % | 0.577 ± 0.041 | 0.661 ± 0.009 | 0.647 ± 0.011 |
| 60 % | 0.498 ± 0.065 | 0.666 ± 0.007 | 0.651 ± 0.012 |
| 70 % | 0.536 ± 0.097 | 0.672 ± 0.008 | 0.651 ± 0.010 |
| 80 % | 0.501 ± 0.146 | 0.674 ± 0.010 | 0.649 ± 0.011 |
| 90 % | 0.591 ± 0.070 | 0.674 ± 0.008 | 0.648 ± 0.015 |

Los MLP se estabilizan a partir del 70–80 %; el perceptrón (no convergente, datos no
linealmente separables) es muy inestable entre semillas.

**Experimento de semillas** (`python main.py semillas`): partición 80/20 fija (semilla 42),
cada modelo con `random_state` 0–9; media ± desviación muestral
(`resultados/tablas/metricas_10_semillas.csv`, detalle en `metricas_por_semilla.csv`).

| Modelo | F1 | Accuracy | Bal. acc. | ROC-AUC | Representativo (mediana inferior del F1) |
|---|---|---|---|---|---|
| Perceptrón | 0.578 ± 0.058 | 0.782 | 0.737 | 0.855 | semilla 5 (F1 0.590) |
| MLP 1 capa (53) | 0.679 ± 0.006 | 0.857 | 0.780 | 0.913 | semilla 2 (F1 0.677) |
| MLP (2, 2) | 0.663 ± 0.005 | 0.848 | 0.771 | 0.900 | semilla 8 (F1 0.662) |

Las figuras `matrices_confusion.png`, `curvas_roc.png` y `curvas_perdida_mlp.png` usan esos
modelos representativos.

**Colapso por inicialización** (`python main.py colapso`): los dos MLP con `random_state`
0–49 sobre la misma partición; colapso = F1 < 0.10. El MLP 1 capa colapsa 0/50 veces y el
MLP (2, 2) 4/50 (semillas 13, 15, 16 y 42); en las cuatro, las 2 neuronas de la segunda capa
están inactivas (Z > 0 en menos del 1 % de los ejemplos de prueba), y en 13 y 16 no se activan
nunca (`colapso_inicializacion.csv` y `.png`).
