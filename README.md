# Práctica 1 - Machine Learning para Robótica e Inteligencia Artificial

## Estudiante

**Cristian Zurita**

---

## Descripción del proyecto

Este proyecto implementa un pipeline completo y reproducible de Machine Learning
utilizando el dataset **Wall-Following Robot Navigation Data**.

El dataset contiene mediciones obtenidas mediante sensores ultrasónicos de un
robot móvil que navega siguiendo paredes.

El proyecto incluye las principales etapas de un flujo de Machine Learning:

- Carga y exploración de datos.
- Análisis exploratorio de datos (EDA).
- Preprocesamiento sin fuga de información.
- Entrenamiento de modelos supervisados.
- Entrenamiento de modelos no supervisados.
- Comparación de modelos mediante métricas.
- Reducción dimensional mediante PCA y t-SNE.
- Medición de latencia de inferencia.
- Documentación de resultados.
- Versionamiento mediante Git y GitHub.

El objetivo es construir un flujo modular, reproducible y documentado que permita
analizar distintas técnicas de Machine Learning aplicadas a un problema real de
navegación robótica.

---

## Dataset

Se utilizó el dataset:

**Wall-Following Robot Navigation Data**

Características principales:

- **5456 observaciones**
- **24 variables de entrada**
- Lecturas de sensores ultrasónicos `US1` a `US24`
- Variable objetivo: `Class`
- 4 acciones posibles de navegación

Clases presentes:

- `Move-Forward`
- `Sharp-Right-Turn`
- `Slight-Right-Turn`
- `Slight-Left-Turn`

Durante el análisis exploratorio se identificó un desbalance entre las clases,
por lo que se utilizó una división estratificada para conservar sus proporciones
en los conjuntos de entrenamiento y prueba.

---

## Estructura del proyecto

```text
ml-practica1-zurita/
│
├── data/
│   ├── raw/
│   │   ├── sensor_readings_24.data
│   │   └── Wall-following.names
│   │
│   └── processed/
│       ├── dataset.parquet
│       ├── eda_summary.json
│       └── benchmark_latency.json
│
├── models/
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── kmeans.pkl
│   └── dbscan.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_supervised.ipynb
│   └── 03_unsupervised.ipynb
│
├── reports/
│   ├── clase_04_preprocesamiento.md
│   ├── clase_05_random_forest.md
│   ├── clase_07_unsupervised.md
│   └── clase_07_elbow.png
│
├── src/
│   ├── data_loader.py
│   ├── train_supervised.py
│   └── train_unsupervised.py
│
├── .gitignore
├── README.md
└── requirements.txt