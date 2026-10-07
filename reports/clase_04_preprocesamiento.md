# Clase 4 - Preprocesamiento de datos

## Objetivo

Preparar el dataset Wall-Following Robot Navigation para el entrenamiento
de modelos de Machine Learning, evitando fuga de información entre los
conjuntos de entrenamiento y prueba.

## Dataset utilizado

El dataset original contiene:

- 5456 observaciones.
- 24 variables de entrada correspondientes a sensores ultrasónicos.
- 1 variable objetivo denominada `Class`.

## División de los datos

Se realizó una división estratificada utilizando una proporción de:

- 80 % para entrenamiento.
- 20 % para prueba.

Los conjuntos obtenidos fueron:

| Conjunto | Observaciones | Variables |
|---|---:|---:|
| Train | 4364 | 24 |
| Test | 1092 | 24 |

La estratificación permitió conservar aproximadamente la distribución
original de las clases en ambos conjuntos.

## Distribución de clases

### Train

| Clase | Proporción |
|---|---:|
| Move-Forward | 40.42 % |
| Sharp-Right-Turn | 38.43 % |
| Slight-Right-Turn | 15.15 % |
| Slight-Left-Turn | 6.00 % |

### Test

| Clase | Proporción |
|---|---:|
| Move-Forward | 40.38 % |
| Sharp-Right-Turn | 38.46 % |
| Slight-Right-Turn | 15.11 % |
| Slight-Left-Turn | 6.04 % |

Las proporciones obtenidas en train y test son prácticamente equivalentes,
lo que confirma que la división estratificada funcionó correctamente.

## Escalado

Se utilizó `RobustScaler` para transformar las 24 variables numéricas.

El escalador fue ajustado únicamente con `X_train` mediante:

`fit_transform(X_train)`

Posteriormente, `X_test` fue transformado utilizando:

`transform(X_test)`

De esta forma, los datos de prueba no participaron en el cálculo de los
parámetros del escalado y se evitó fuga de información o `data leakage`.

## Archivo generado

Los datos procesados fueron almacenados en:

`data/processed/dataset.parquet`

El archivo contiene las 24 variables escaladas, la variable objetivo
`Class` y una columna adicional `split` que identifica si cada observación
pertenece al conjunto de entrenamiento o prueba.

## Conclusión

El proceso de preprocesamiento produjo correctamente los conjuntos de
entrenamiento y prueba, mantuvo la distribución de las clases y aplicó
el escalado sin utilizar información del conjunto de prueba durante el
ajuste del `RobustScaler`.

Los datos resultantes quedan preparados para el entrenamiento de los
modelos supervisados de las siguientes etapas.