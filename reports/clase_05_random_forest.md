# Clase 5 - Modelo supervisado Random Forest

## Objetivo

Entrenar y evaluar un modelo supervisado de clasificación Random Forest
utilizando los datos procesados durante la etapa de preprocesamiento.

El modelo busca determinar la acción de navegación del robot a partir de
las 24 lecturas de sus sensores ultrasónicos.

## Datos utilizados

Para el entrenamiento se utilizaron:

- 4364 observaciones.
- 24 variables de entrada.
- 4 clases de navegación.

Para la evaluación se reservaron:

- 1092 observaciones.

El conjunto de prueba no participó durante el entrenamiento del modelo.

## Modelo utilizado

Se utilizó `RandomForestClassifier` de Scikit-learn.

El entrenamiento se realizó con los hiperparámetros por defecto y se fijó:

`random_state=42`

para mantener la reproducibilidad del experimento.

## Resultados generales

El modelo obtuvo las siguientes métricas:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9927 |
| F1-score ponderado | 0.9927 |
| F1-score macro | 0.9894 |

El accuracy de 0.9927 indica que aproximadamente el 99.27 % de las
observaciones del conjunto de prueba fueron clasificadas correctamente.

De las 1092 observaciones evaluadas:

- 1084 fueron clasificadas correctamente.
- 8 fueron clasificadas incorrectamente.

## Matriz de confusión

| Clase real | Move-Forward | Sharp-Right-Turn | Slight-Left-Turn | Slight-Right-Turn |
|---|---:|---:|---:|---:|
| Move-Forward | 437 | 0 | 0 | 4 |
| Sharp-Right-Turn | 0 | 419 | 0 | 1 |
| Slight-Left-Turn | 2 | 0 | 64 | 0 |
| Slight-Right-Turn | 1 | 0 | 0 | 164 |

La diagonal principal representa las clasificaciones correctas.

La suma de los valores de la diagonal es:

437 + 419 + 64 + 164 = 1084

Los valores fuera de la diagonal representan los errores de clasificación,
que suman un total de 8 observaciones.

## Análisis por clase

### Move-Forward

De 441 observaciones reales de esta clase:

- 437 fueron clasificadas correctamente.
- 4 fueron confundidas con `Slight-Right-Turn`.

El recall aproximado para esta clase es de 99.09 %.

### Sharp-Right-Turn

De 420 observaciones:

- 419 fueron clasificadas correctamente.
- 1 fue confundida con `Slight-Right-Turn`.

El recall aproximado es de 99.76 %.

### Slight-Left-Turn

De 66 observaciones:

- 64 fueron clasificadas correctamente.
- 2 fueron confundidas con `Move-Forward`.

El recall aproximado es de 96.97 %.

Esta clase es especialmente importante debido a que representa
aproximadamente el 6 % del dataset original y, a pesar de ser la clase
minoritaria, mantiene un rendimiento elevado.

### Slight-Right-Turn

De 165 observaciones:

- 164 fueron clasificadas correctamente.
- 1 fue confundida con `Move-Forward`.

El recall aproximado es de 99.39 %.

## Interpretación del F1-score

El F1-score ponderado alcanzó un valor de 0.9927.

Esta métrica considera la cantidad de observaciones pertenecientes a cada
clase, por lo que las clases con mayor representación tienen mayor peso.

El F1-score macro alcanzó 0.9894.

El F1 macro asigna la misma importancia a cada una de las cuatro clases.
El hecho de que este valor también sea elevado indica que el buen
rendimiento del modelo no se debe únicamente a las clases mayoritarias.

Esto es especialmente relevante debido al desbalance de clases detectado
durante el análisis exploratorio.

## Análisis de los errores

Los errores detectados fueron:

- 4 casos de `Move-Forward` clasificados como `Slight-Right-Turn`.
- 1 caso de `Sharp-Right-Turn` clasificado como `Slight-Right-Turn`.
- 2 casos de `Slight-Left-Turn` clasificados como `Move-Forward`.
- 1 caso de `Slight-Right-Turn` clasificado como `Move-Forward`.

En total se produjeron únicamente 8 errores sobre 1092 observaciones de
prueba.

## Modelo generado

El Random Forest entrenado fue almacenado en:

`models/random_forest.pkl`

El archivo tiene aproximadamente 2.4 MB y contiene el modelo ya entrenado.

Puede volver a cargarse posteriormente utilizando `joblib` sin necesidad
de repetir el proceso de entrenamiento.

## Conclusión

Random Forest presenta un rendimiento muy alto para la clasificación de
las acciones de navegación del robot.

El accuracy de 99.27 %, junto con un F1-score macro de 98.94 %, muestra
que el modelo mantiene un comportamiento consistente incluso frente al
desbalance existente entre las clases.

La clase minoritaria `Slight-Left-Turn` también alcanza un rendimiento
elevado, con aproximadamente 96.97 % de sus observaciones identificadas
correctamente.

Estos resultados constituyen una primera referencia de rendimiento y
servirán posteriormente para comparar Random Forest con un segundo modelo
supervisado.