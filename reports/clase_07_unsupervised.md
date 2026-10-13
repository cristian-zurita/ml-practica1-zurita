# Clase 7 - Modelos no supervisados

## Objetivo

Analizar la estructura interna del dataset Wall-Following Robot Navigation
mediante algoritmos de aprendizaje no supervisado.

Se utilizaron:

- K-Means.
- DBSCAN.

Durante el entrenamiento no se utilizó la variable objetivo `Class`.
Los algoritmos trabajaron únicamente con las 24 variables correspondientes
a los sensores ultrasónicos.

## Datos utilizados

Se utilizaron:

- 5456 observaciones.
- 24 variables numéricas.
- Datos previamente escalados mediante RobustScaler.

Las columnas `Class` y `split` fueron excluidas del entrenamiento de los
modelos no supervisados.

## Método del codo

Se evaluaron valores de k comprendidos entre 2 y 10.

| k | Inercia |
|---:|---:|
| 2 | 276703.5387 |
| 3 | 233209.5963 |
| 4 | 203348.2523 |
| 5 | 186287.4747 |
| 6 | 169085.0545 |
| 7 | 156094.2298 |
| 8 | 146950.3959 |
| 9 | 138836.4781 |
| 10 | 130242.0916 |

La reducción de la inercia es especialmente pronunciada hasta
aproximadamente k=4. Después de este punto la mejora comienza a disminuir,
por lo que se seleccionó:

`k = 4`

La gráfica correspondiente fue almacenada en:

`reports/clase_07_elbow.png`

## K-Means

El modelo definitivo se entrenó utilizando cuatro clusters.

### Resultados

- Número de clusters: 4.
- Silhouette Score: 0.4702.
- Tiempo de entrenamiento aproximado: 0.04 segundos.

### Distribución de observaciones

| Cluster | Observaciones | Proporción aproximada |
|---:|---:|---:|
| 0 | 374 | 6.85 % |
| 1 | 429 | 7.86 % |
| 2 | 4370 | 80.10 % |
| 3 | 283 | 5.19 % |

El Silhouette Score de 0.4702 indica una separación moderada entre los
clusters.

Sin embargo, se observa una distribución considerablemente desigual,
ya que el cluster 2 concentra aproximadamente el 80 % de las observaciones.

Esto muestra que los cuatro clusters descubiertos por K-Means no deben
interpretarse directamente como las cuatro clases reales del dataset.

## Búsqueda de parámetros para DBSCAN

DBSCAN depende principalmente de los parámetros `eps` y `min_samples`.

Se probaron diferentes combinaciones considerando:

- Número de clusters encontrados.
- Cantidad de puntos considerados ruido.
- Porcentaje de ruido.
- Silhouette Score.

No se seleccionó simplemente la configuración con mayor Silhouette Score,
ya que algunas combinaciones obtenían valores altos descartando una gran
cantidad de observaciones como ruido.

Por ejemplo, con `eps=0.5` y `min_samples=5` se obtuvo un Silhouette Score
elevado, pero aproximadamente el 83 % de las observaciones fueron
clasificadas como ruido.

Después de analizar diferentes configuraciones se seleccionó:

- `eps = 5.5`
- `min_samples = 20`

## DBSCAN definitivo

### Resultados

- Clusters encontrados: 6.
- Puntos considerados ruido: 308.
- Porcentaje de ruido: 5.65 %.
- Silhouette Score definitivo: 0.4638.
- Tiempo de entrenamiento aproximado: 0.12 segundos.

### Distribución de etiquetas

| Etiqueta | Observaciones |
|---:|---:|
| -1 (ruido) | 308 |
| 0 | 4960 |
| 1 | 27 |
| 2 | 50 |
| 3 | 47 |
| 4 | 44 |
| 5 | 20 |

Aunque DBSCAN detectó seis clusters, el cluster 0 contiene aproximadamente
el 90.9 % de las observaciones.

Por lo tanto, el número de clusters por sí solo no implica que se hayan
encontrado seis grupos equilibrados o claramente diferenciados.

## Comparación

| Característica | K-Means | DBSCAN |
|---|---:|---:|
| Clusters | 4 | 6 |
| Silhouette Score | 0.4702 | 0.4638 |
| Ruido | 0 % | 5.65 % |
| Cluster dominante | 80.10 % | 90.91 % |

K-Means obtuvo un Silhouette Score ligeramente superior y una estructura
más simple.

DBSCAN permitió identificar observaciones como ruido, pero produjo un
cluster dominante que concentró la mayor parte del dataset.

Los resultados indican que existe cierta estructura interna en los datos,
aunque los clusters obtenidos no presentan una separación completamente
definida ni reproducen necesariamente las clases conocidas del problema.

## Modelos generados

Los modelos fueron almacenados en:

- `models/kmeans.pkl`
- `models/dbscan.pkl`

## Conclusión

Los dos algoritmos lograron identificar agrupaciones dentro de las lecturas
de los sensores, aunque ambos produjeron clusters considerablemente
desbalanceados.

K-Means obtuvo un Silhouette Score de 0.4702 y DBSCAN un valor de 0.4638.

En las condiciones evaluadas, K-Means ofrece una solución ligeramente más
simple y con una separación interna marginalmente superior.

DBSCAN, por otra parte, proporciona información adicional al identificar
aproximadamente un 5.65 % de las observaciones como ruido.

Estos resultados servirán como base para la siguiente etapa, donde se
utilizarán técnicas de reducción dimensional para visualizar la estructura
de los clusters.