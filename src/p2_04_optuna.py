"""Práctica 2 · Sesión 4: optimización de Random Forest con Optuna.

Desde la raíz del repositorio (archivo ubicado en src/):
    python src/p2_04_optuna.py

Selecciona hiperparámetros usando exclusivamente validación estratificada.
NO utiliza el conjunto de prueba para seleccionar hiperparámetros.
"""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from p2_02_costos import CLASE_POSITIVA, RANDOM_STATE, encontrar_dataset

N_TRIALS = 25  # La guía requiere al menos 20.
UMBRAL = 0.35  # Seleccionado en P2-3, NO reajustado aquí.


def preparar_datos(ruta_dataset: Path):
    """Lee sensores y genera las particiones de P2-3 (60/20/20 %)."""
    datos = pd.read_parquet(ruta_dataset)
    columnas = [f"US{i}" for i in range(1, 25)]
    faltantes = [c for c in [*columnas, "Class"] if c not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas del dataset: {faltantes}")

    x = datos[columnas].apply(pd.to_numeric, errors="raise")
    y = datos["Class"].astype(str).str.strip().eq(CLASE_POSITIVA).astype(int)
    if x.isna().any().any():
        raise ValueError("El dataset contiene valores faltantes.")
    if y.nunique() != 2:
        raise ValueError("No hay dos clases binarias para la clasificación.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.25, stratify=y_train, random_state=RANDOM_STATE
    )
    # x_test e y_test se reservan para la Sesión 5.
    return x_fit, x_val, y_fit, y_val, len(y_test)


def crear_objetivo(x_fit, x_val, y_fit, y_val):
    """Construye una función objetivo que maximiza F1 de Slight-Left-Turn."""
    def objetivo(trial: optuna.Trial) -> float:
        """Entrena un bosque y devuelve F1 positivo en validación."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 20, 30]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
        modelo = RandomForestClassifier(
            **params, random_state=RANDOM_STATE, n_jobs=-1
        )
        modelo.fit(x_fit, y_fit)
        proba = modelo.predict_proba(x_val)[:, 1]
        predicciones = (proba >= UMBRAL).astype(int)
        return float(f1_score(y_val, predicciones, zero_division=0))

    return objetivo


def main():
    """Ejecuta >=20 ensayos, conserva el estudio y exporta el mejor trial."""
    ruta_dataset = encontrar_dataset()
    x_fit, x_val, y_fit, y_val, n_test = preparar_datos(ruta_dataset)
    carpeta = ruta_dataset.parents[2] / "reports" / "optuna"
    carpeta.mkdir(parents=True, exist_ok=True)
    db = carpeta / "p2_04_estudio.db"
    json_best = carpeta / "p2_04_best.json"

    # Si ya existe un estudio, se recupera y se agregan nuevos ensayos.
    estudio = optuna.create_study(
        study_name="p2_04_rf_f1",
        direction="maximize",
        storage=f"sqlite:///{db}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    print("\nOPTUNA — PRÁCTICA 2, SESIÓN 4")
    print(f"Dataset: {ruta_dataset}")
    print(f"Particiones: entrenamiento={len(y_fit)}, validación={len(y_val)}, prueba={n_test}")
    print(f"Ensayos: {N_TRIALS}; objetivo: maximizar F1 de clase 1")
    print(f"Umbral fijo (P2-3): {UMBRAL:.2f}")
    print("El conjunto de prueba NO interviene en la optimización.\n")

    estudio.optimize(
        crear_objetivo(x_fit, x_val, y_fit, y_val), n_trials=N_TRIALS
    )
    mejor = estudio.best_trial
    resultado = {
        "sesion": "P2-4",
        "study_name": estudio.study_name,
        "objetivo": "maximizar F1 positivo en validación",
        "umbral_fijo": UMBRAL,
        "n_trials_ejecucion": N_TRIALS,
        "n_trials_totales": len(estudio.trials),
        "mejor_f1_validacion": mejor.value,
        "mejor_numero_trial": mejor.number,
        "mejores_hiperparametros": mejor.params,
        "random_state": RANDOM_STATE,
        "filas_entrenamiento": len(y_fit),
        "filas_validacion": len(y_val),
        "filas_prueba_reservadas": n_test,
        "estudio_sqlite": str(db),
    }
    json_best.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("\nRESULTADO")
    print(f"Mejor F1 en validación: {mejor.value:.4f}")
    print(f"Trials acumulados en el estudio: {len(estudio.trials)}")
    print(f"Trial ganador: {mejor.number}")
    print("Hiperparámetros:")
    print(json.dumps(mejor.params, ensure_ascii=False, indent=2))
    print(f"Estudio guardado en: {db}")
    print(f"Resumen guardado en: {json_best}")
    print("Nota: aún NO hemos comprobado la mejora en test; eso corresponde a P2-5.")


if __name__ == "__main__":
    main()
