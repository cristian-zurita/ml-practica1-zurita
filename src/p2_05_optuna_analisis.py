"""Práctica 2 · Sesión 5: análisis de Optuna y comparación en prueba.

Ubicación: src/p2_05_optuna_analisis.py
Ejecución desde la raíz del repositorio:
    python src/p2_05_optuna_analisis.py

Lee el estudio de la sesión 4, dibuja las gráficas, exporta config.json,
entrena dos Random Forest sobre el 80 % de entrenamiento y los compara en
el 20 % de prueba que NO se utilizó durante la optimización.

No reentrena ni modifica los modelos de otros proyectos ni añade ensayos
nuevos a Optuna. No guarda ningún modelo dentro de models/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from p2_02_costos import (
    CLASE_POSITIVA,
    COSTO_FN,
    COSTO_FP,
    RANDOM_STATE,
    encontrar_dataset,
    evaluar_costo,
)

UMBRAL = 0.35  # Elegido en P2-3 con validación, sin usar la prueba.
NOMBRE_ESTUDIO = "p2_04_rf_f1"


def cargar_datos(ruta_dataset: Path):
    """Obtiene el mismo conjunto de prueba estratificado que P2-1 a P2-4."""
    datos = pd.read_parquet(ruta_dataset)
    columnas = [f"US{i}" for i in range(1, 25)]
    faltantes = [c for c in (*columnas, "Class") if c not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas del dataset: {faltantes}")
    x = datos[columnas].apply(pd.to_numeric, errors="raise")
    y = datos["Class"].astype(str).str.strip().eq(CLASE_POSITIVA).astype(int)
    if x.isna().any().any() or y.nunique() != 2:
        raise ValueError("Revisa los valores faltantes y las clases del dataset.")

    # Es la misma división 80/20 de las sesiones anteriores. El subconjunto
    # de validación se usó dentro de entrenamiento durante P2-3 y P2-4.
    return train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )


def cargar_estudio(carpeta: Path):
    """Recupera exclusivamente los ensayos existentes de Optuna."""
    ruta_db = carpeta / "p2_04_estudio.db"
    ruta_resumen = carpeta / "p2_04_best.json"
    if not ruta_db.is_file() or not ruta_resumen.is_file():
        raise FileNotFoundError(
            "No se encuentran p2_04_estudio.db y p2_04_best.json en reports/optuna/. "
            "Recupera los archivos de P2-4 antes de continuar."
        )
    estudio = optuna.load_study(
        study_name=NOMBRE_ESTUDIO, storage=f"sqlite:///{ruta_db}"
    )
    resumen = json.loads(ruta_resumen.read_text(encoding="utf-8"))
    if estudio.direction != optuna.study.StudyDirection.MAXIMIZE:
        raise ValueError("El estudio de Optuna no está configurado para maximizar F1.")
    completos = [
        t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completos) < 20:
        raise ValueError("El estudio debe contener al menos 20 ensayos completos.")
    if resumen.get("mejores_hiperparametros") != estudio.best_trial.params:
        raise ValueError(
            "Los hiperparámetros del JSON no coinciden con los de la base de datos. "
            "Verifica que ambos archivos correspondan al mismo estudio P2-4."
        )
    if not np.isclose(float(resumen.get("umbral_fijo", -1)), UMBRAL):
        raise ValueError("El umbral guardado en P2-4 no coincide con el de P2-3.")
    return estudio, completos


def crear_grafica_historial(ensayos: list, destino: Path):
    """Dibuja F1 por ensayo y el mejor F1 acumulado en validación."""
    numeros = [t.number for t in ensayos]
    valores = [float(t.value) for t in ensayos]
    mejores = np.maximum.accumulate(valores)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(numeros, valores, marker="o", label="F1 de cada ensayo")
    ax.plot(numeros, mejores, linewidth=2, label="Mejor F1 acumulado")
    ax.set(
        title="Optuna: historial de optimización (validación)",
        xlabel="Número de ensayo",
        ylabel="F1 de la clase minoritaria",
    )
    ax.set_ylim(max(0, min(valores) - 0.05), min(1.01, max(valores) + 0.03))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def crear_grafica_importancias(estudio: optuna.Study, destino: Path) -> str:
    """Estima importancias con Optuna; advierte si la señal es insuficiente."""
    mensaje = ""
    try:
        importancias = optuna.importance.get_param_importances(estudio)
        if not importancias:
            raise ValueError("No se pudieron estimar las importancias.")
        nombres = list(importancias.keys())[::-1]
        valores = [float(importancias[n]) for n in nombres]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(nombres, valores)
        ax.set(
            title="Optuna: importancia estimada de hiperparámetros",
            xlabel="Importancia relativa (fANOVA)",
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(destino, dpi=160)
        plt.close(fig)
        if sum(float(t.value) == 1.0 for t in estudio.trials if t.value is not None) > 1:
            mensaje = (
                "Varios ensayos empataron con F1=1,0; interpreta las importancias "
                "con cautela porque la validación discrimina poco entre modelos."
            )
    except (ValueError, RuntimeError, ImportError) as error:
        mensaje = f"Importancias no estimables de forma fiable: {error}"
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.text(
            0.5, 0.5,
            "Importancia no estimable con los ensayos disponibles.\n"
            "Revisa los resultados numéricos y las advertencias.",
            ha="center", va="center", wrap=True,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(destino, dpi=160)
        plt.close(fig)
    return mensaje


def evaluar_modelo(nombre, modelo, x_test, y_test, umbral):
    """Calcula métricas de clase positiva y costo económico sobre test."""
    probabilidades = modelo.predict_proba(x_test)[:, 1]
    pred = (probabilidades >= umbral).astype(int)
    costos = evaluar_costo(nombre, y_test, pred)
    return {
        "Modelo": nombre,
        "Umbral": umbral,
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        **{k: costos[k] for k in ("TN", "FP", "FN", "TP", "Costo total (USD)")},
    }


def main():
    """Entrega la configuración, las figuras y la comparación reproducible."""
    ruta_dataset = encontrar_dataset()
    raiz = ruta_dataset.parents[2]
    carpeta_estudio = raiz / "reports" / "optuna"
    carpeta_figuras = raiz / "reports" / "figures"
    carpeta_figuras.mkdir(parents=True, exist_ok=True)

    estudio, completos = cargar_estudio(carpeta_estudio)
    mejor = estudio.best_trial
    params_optimos = dict(mejor.params)
    config = {
        "sesion": "P2-5",
        "modelo": "RandomForestClassifier",
        "clase_positiva": CLASE_POSITIVA,
        "umbral_decision": UMBRAL,
        "random_state": RANDOM_STATE,
        "hiperparametros": params_optimos,
        "optimizacion": {
            "estudio": estudio.study_name,
            "objetivo": "F1 clase positiva en validación",
            "mejor_trial": mejor.number,
            "mejor_f1_validacion": float(mejor.value),
            "ensayos_completos": len(completos),
        },
    }
    ruta_config = carpeta_estudio / "config.json"
    ruta_config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    ruta_historial = carpeta_figuras / "optuna_history.png"
    ruta_importancia = carpeta_figuras / "optuna_importance.png"
    crear_grafica_historial(completos, ruta_historial)
    aviso = crear_grafica_importancias(estudio, ruta_importancia)

    x_train, x_test, y_train, y_test = cargar_datos(ruta_dataset)
    original = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    optimizado = RandomForestClassifier(
        **params_optimos, random_state=RANDOM_STATE, n_jobs=-1
    )
    original.fit(x_train, y_train)
    optimizado.fit(x_train, y_train)

    filas = [
        evaluar_modelo("Original (umbral 0.50)", original, x_test, y_test, 0.50),
        evaluar_modelo("Original (umbral 0.35)", original, x_test, y_test, UMBRAL),
        evaluar_modelo("Optuna (umbral 0.35)", optimizado, x_test, y_test, UMBRAL),
    ]
    tabla = pd.DataFrame(filas).set_index("Modelo")
    ruta_csv = carpeta_estudio / "p2_05_comparacion.csv"
    tabla.to_csv(ruta_csv, encoding="utf-8")

    print("\nOPTUNA — PRÁCTICA 2, SESIÓN 5")
    print(f"Dataset: {ruta_dataset}")
    print(f"Ensayos completos: {len(completos)}")
    print(f"Mejor ensayo: {mejor.number}; F1 validación: {mejor.value:.4f}")
    print(f"Entrenamiento final: {len(y_train)} filas; prueba independiente: {len(y_test)} filas")
    print("\nCOMPARACIÓN EN PRUEBA:")
    print(tabla.round(4).to_string())
    costo_original = int(tabla.loc["Original (umbral 0.35)", "Costo total (USD)"])
    costo_optuna = int(tabla.loc["Optuna (umbral 0.35)", "Costo total (USD)"])
    ahorro = costo_original - costo_optuna
    if ahorro > 0:
        print(f"\nOptuna REDUCE el costo en prueba en ${ahorro:,} respecto al original con umbral 0.35.")
    elif ahorro == 0:
        print("\nOptuna NO cambia el costo en prueba respecto al original con umbral 0.35.")
    else:
        print(f"\nOptuna AUMENTA el costo en prueba en ${-ahorro:,} respecto al original con umbral 0.35.")
    print("Los costos son supuestos didácticos: FP=$50, FN=$500.")
    print("Se mantuvo fijo el umbral 0.35, sin seleccionarlo en test.")
    if aviso:
        print(f"\nADVERTENCIA: {aviso}")
    print("\nARCHIVOS GENERADOS:")
    for ruta in (ruta_config, ruta_historial, ruta_importancia, ruta_csv):
        print("  ", ruta.relative_to(raiz))
    print("\nNo se ha modificado ningún archivo en models/.")


if __name__ == "__main__":
    main()
