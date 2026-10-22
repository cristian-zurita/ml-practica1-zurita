# Práctica 1 - Machine Learning para Robótica e Inteligencia Artificial

## Estudiante

**Cristian Zurita**

## Descripción del proyecto

Este proyecto implementa un pipeline completo y reproducible de Machine Learning utilizando el dataset **Wall-Following Robot Navigation Data**, correspondiente a un problema de navegación robótica basado en lecturas de sensores ultrasónicos.

El flujo desarrollado comprende análisis exploratorio de datos, preprocesamiento, aprendizaje supervisado, aprendizaje no supervisado, reducción dimensional, comparación de modelos y medición de latencia.

El objetivo es analizar diferentes técnicas de Machine Learning aplicadas a datos reales de navegación robótica y estudiar tanto su capacidad predictiva como su comportamiento computacional.

---

## Dataset

El dataset contiene:

- **5456 observaciones**
- **24 variables de entrada**, correspondientes a sensores ultrasónicos `US1` a `US24`
- Una variable objetivo denominada `Class`
- Cuatro acciones posibles de navegación

Las clases son:

- `Move-Forward`
- `Sharp-Right-Turn`
- `Slight-Right-Turn`
- `Slight-Left-Turn`

Durante el análisis exploratorio se identificó desbalance entre las clases. Por esta razón, para los modelos supervisados se utilizó una división estratificada del conjunto de datos.

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
│   ├── clase_07_elbow.png
│   └── clase_07_unsupervised.md
│
├── src/
│   ├── data_loader.py
│   ├── train_supervised.py
│   └── train_unsupervised.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

`data/raw/` contiene los datos originales.

`data/processed/` almacena datos procesados y resultados estructurados.

`notebooks/` contiene el análisis exploratorio, experimentación y visualizaciones.

`src/` contiene código Python modular y reutilizable.

`models/` contiene los modelos entrenados.

`reports/` contiene análisis y resultados generados durante la práctica.

---

## Instalación y configuración del entorno

Esta sección permite reconstruir el proyecto en otra computadora.

### Clonar el repositorio

```bash
git clone https://github.com/cristian-zurita/ml-practica1-zurita.git
cd ml-practica1-zurita
```

### Crear un entorno virtual

```bash
python -m venv venv
```

### Activar en Bash

```bash
source venv/bin/activate
```

### Activar en Fish

```fish
source venv/bin/activate.fish
```

### Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

---

## Ejecución

### Preprocesamiento

```bash
python src/data_loader.py
```

Este script carga el dataset, separa variables de entrada y objetivo, realiza la división de entrenamiento y prueba y aplica el escalado de los datos.

El conjunto procesado se almacena en:

```text
data/processed/dataset.parquet
```

### Modelos supervisados

```bash
python src/train_supervised.py
```

Entrena y evalúa:

- Random Forest
- Support Vector Machine (SVM)

Los modelos se almacenan en:

```text
models/random_forest.pkl
models/svm.pkl
```

### Modelos no supervisados

```bash
python src/train_unsupervised.py
```

Ejecuta:

- Método del codo
- K-Means
- Búsqueda de parámetros para DBSCAN
- DBSCAN
- Silhouette Score

Los modelos obtenidos se almacenan en:

```text
models/kmeans.pkl
models/dbscan.pkl
```

---

## Análisis exploratorio

El notebook:

```text
notebooks/01_eda.ipynb
```

incluye:

- Dimensiones y tipos de datos
- Estadísticas descriptivas
- Valores faltantes
- Duplicados
- Distribución de clases
- Histogramas
- Boxplots
- Matriz de correlación

El dataset no presentó valores faltantes ni registros duplicados.

La distribución aproximada de las clases fue:

| Clase | Proporción |
|---|---:|
| Move-Forward | 40.41 % |
| Sharp-Right-Turn | 38.43 % |
| Slight-Right-Turn | 15.14 % |
| Slight-Left-Turn | 6.01 % |

Debido a este desbalance se utilizó una división estratificada en la etapa supervisada.

---

## Preprocesamiento y prevención de Data Leakage

La división utilizada fue aproximadamente:

```text
80 % entrenamiento
20 % prueba
```

Se utilizó `RobustScaler`.

La secuencia de procesamiento fue:

```text
Dataset
   ↓
Separación X / y
   ↓
Train / Test
   ↓
RobustScaler.fit() únicamente con TRAIN
   ↓
Transformación de TRAIN
   ↓
Transformación de TEST
```

El escalador nunca se ajustó utilizando el conjunto de prueba.

De esta manera se evita **data leakage**, es decir, que información correspondiente a los datos utilizados para evaluación influya durante el entrenamiento.

---

## Resultados supervisados

Se compararon Random Forest y SVM.

| Modelo | Accuracy | F1 ponderado | F1 macro |
|---|---:|---:|---:|
| Random Forest | 0.9927 | 0.9927 | 0.9894 |
| SVM | 0.8690 | 0.8684 | 0.8692 |

### Random Forest

Random Forest obtuvo el mejor rendimiento supervisado.

Sobre **1092 observaciones de prueba**:

```text
1084 clasificaciones correctas
8 clasificaciones incorrectas
```

Accuracy:

```text
≈ 99.27 %
```

El F1 macro de aproximadamente `0.9894` muestra que el modelo presentó un rendimiento elevado incluso considerando las clases minoritarias.

### SVM

SVM obtuvo aproximadamente:

```text
Accuracy = 86.90 %
```

Por lo tanto, bajo las condiciones evaluadas, Random Forest presentó una capacidad predictiva claramente superior.

Esto no implica que Random Forest sea universalmente superior a SVM; la conclusión corresponde al dataset, preprocesamiento y configuración utilizados en este experimento.

---

## Aprendizaje no supervisado

Para el análisis no supervisado se utilizaron únicamente las 24 variables correspondientes a los sensores.

La etiqueta `Class` no participó en la formación de los clusters.

Se evaluaron:

- K-Means
- DBSCAN

---

## K-Means

Para determinar un número razonable de clusters se utilizó el método del codo evaluando diferentes valores de `k`.

Se seleccionó:

```text
k = 4
```

La selección se realizó mediante el comportamiento de la inercia y no simplemente por la existencia de cuatro clases en el dataset.

Resultado:

```text
Silhouette Score = 0.4702
```

Distribución obtenida:

| Cluster | Observaciones |
|---:|---:|
| 0 | 374 |
| 1 | 429 |
| 2 | 4370 |
| 3 | 283 |

Los identificadores de los clusters no corresponden directamente a las etiquetas originales del problema supervisado.

---

## DBSCAN

Después de evaluar diferentes configuraciones se seleccionaron:

```text
eps = 5.5
min_samples = 20
```

Resultados:

```text
6 clusters
308 observaciones consideradas ruido
5.65 % de ruido
Silhouette Score = 0.4638
```

DBSCAN identificó un cluster principal de gran tamaño y varios grupos pequeños, además de observaciones consideradas ruido.

---

## Comparación K-Means vs DBSCAN

Resultados:

```text
K-Means  → Silhouette = 0.4702
DBSCAN   → Silhouette = 0.4638
```

K-Means obtuvo un Silhouette Score ligeramente superior.

DBSCAN, sin embargo, proporciona una característica adicional: la capacidad de identificar observaciones consideradas ruido.

Por esta razón, la comparación entre ambos algoritmos no debe realizarse únicamente mediante una métrica.

---

## Reducción dimensional con PCA

El dataset contiene originalmente 24 dimensiones.

PCA permitió reducirlas a dos componentes para facilitar su visualización.

Resultados:

```text
PC1 = 30.28 %
PC2 = 16.87 %

Varianza explicada total = 47.16 %
```

Las dos primeras componentes conservan aproximadamente el **47.16 % de la variabilidad total** de las 24 variables originales.

Por esta razón, las visualizaciones PCA representan únicamente una parte de la información disponible en el espacio original.

---

## Visualización mediante t-SNE

También se utilizó t-SNE para representar los datos en dos dimensiones.

A diferencia de PCA, t-SNE está orientado principalmente a conservar relaciones locales entre observaciones.

Las visualizaciones permitieron observar con mayor detalle las estructuras locales producidas por K-Means y DBSCAN.

Las distancias globales entre grupos en una representación t-SNE no deben interpretarse directamente como distancias equivalentes en las 24 dimensiones originales.

---

## Benchmark de latencia

Se realizaron **100 predicciones individuales por modelo**.

| Modelo | Latencia promedio |
|---|---:|
| K-Means | 1.1957 ms |
| SVM | 1.3383 ms |
| Random Forest | 10.3156 ms |

K-Means presentó la menor latencia promedio, seguido por SVM.

Random Forest presentó una latencia mayor, pero también obtuvo claramente el mejor rendimiento de clasificación.

Esto demuestra que:

```text
mayor precisión ≠ menor latencia
```

Por tanto, para seleccionar un modelo destinado a una aplicación robótica se deben considerar simultáneamente precisión, tiempo de respuesta y recursos computacionales.

DBSCAN no fue incluido en el benchmark de inferencia porque la implementación utilizada en scikit-learn no dispone de un método estándar `predict()` para nuevas observaciones.

---

## Reproducibilidad

Cuando correspondía se utilizó:

```python
random_state=42
```

para mantener resultados reproducibles.

Las dependencias necesarias se encuentran en:

```text
requirements.txt
```

La separación entre datos, código fuente, notebooks, modelos y reportes facilita además la reproducción y mantenimiento del proyecto.

---

## Conclusiones

El proyecto permitió implementar un pipeline completo de Machine Learning aplicado a un problema real de navegación robótica.

Random Forest presentó el mejor rendimiento supervisado, alcanzando aproximadamente **99.27 % de Accuracy** y un F1 macro de `0.9894`.

K-Means y DBSCAN permitieron analizar estructuras presentes en las lecturas de los sensores sin utilizar las etiquetas reales durante el entrenamiento.

PCA permitió representar aproximadamente el **47.16 % de la variabilidad** del dataset en dos dimensiones, mientras que t-SNE facilitó el análisis de estructuras locales.

Finalmente, el benchmark mostró que el algoritmo con mejor rendimiento predictivo no necesariamente presenta la menor latencia.

En una aplicación robótica real, la selección final de un modelo debe considerar conjuntamente:

- calidad predictiva,
- costo computacional,
- tiempo de respuesta,
- robustez,
- y restricciones del hardware disponible.