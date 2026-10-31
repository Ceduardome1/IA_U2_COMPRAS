#!/usr/bin/env python3
"""
cli_evaluar_todos.py
Tipo: LÍNEA DE COMANDOS (CLI)
Responsable: CESAR
Entregable: Apartado 5 - Evaluación Comparativa sobre Test (20%) y Referencia Base

Objetivos y Especificaciones:
1. Carga de los conjuntos de prueba (20% de evaluación final):
   - 'data/test_categorico.csv' (para Naive Bayes).
   - 'data/test_numerico.csv' (para Random Forest, MLP y Baseline).
2. Carga de Modelos y Artefactos:
   - Naive Bayes: carga 'artifacts/tablas_cpt_laplace.json' y ejecuta inferencia.
   - Random Forest: carga 'artifacts/random_forest.joblib'.
   - Perceptrón Multicapa: carga 'artifacts/scaler_mlp.joblib' y 'artifacts/mlp_model.joblib'.
   - Referencia Base (Baseline): clasificador mayoritario que siempre predice "No compra".
3. Regla de Decisión Común (Umbral 0.50):
   - Para cada modelo, si P(Compra) >= 0.50 -> Predice Compra (True / 1).
   - Si P(Compra) < 0.50 -> Predice No Compra (False / 0).
4. Métricas de Evaluación Requeridas:
   - Matriz de confusión completa (TN, FP, FN, TP).
   - Precisión (Precision) para la clase Compra: TP / (TP + FP).
   - Sensibilidad (Recall) para la clase Compra: TP / (TP + FN).
   - Puntuación F1 para la clase Compra: 2 * (Prec * Rec) / (Prec + Rec).
   - Exactitud global (Accuracy).
   - Tiempos de cómputo: inferencia sobre test.
5. Salida y Serialización:
   - Impresión en consola de tabla comparativa formateada.
   - Serialización de todas las métricas en 'artifacts/metricas_evaluacion.json'.

Uso:
    python src/comparacion/cli_evaluar_todos.py
"""

import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TEST_CAT_PATH,
    TEST_NUM_PATH,
    CPT_LAPLACE_JSON,
    RF_MODEL_JOBLIB,
    MLP_MODEL_JOBLIB,
    SCALER_MLP_JOBLIB,
    METRICAS_JSON,
    DECISION_THRESHOLD,
    TARGET_COL
)
from src.redBayesiana.cli_pipeline_red_naive_bayes import (
    cargar_tablas_cpt,
    inferir_probabilidad_naive_bayes
)


def cargar_modelos_y_datos_test():
    """Carga los conjuntos de prueba y los modelos serializados."""
    if not TEST_CAT_PATH.exists() or not TEST_NUM_PATH.exists():
        raise FileNotFoundError("No se encontraron los datasets de test. Ejecute primero src/util/01_cli_preparar_datos.py")

    df_test_cat = pd.read_csv(TEST_CAT_PATH)
    df_test_num = pd.read_csv(TEST_NUM_PATH)

    cpt_data = cargar_tablas_cpt(CPT_LAPLACE_JSON)
    rf_model = joblib.load(RF_MODEL_JOBLIB) if RF_MODEL_JOBLIB.exists() else None
    mlp_model = joblib.load(MLP_MODEL_JOBLIB) if MLP_MODEL_JOBLIB.exists() else None
    scaler_mlp = joblib.load(SCALER_MLP_JOBLIB) if SCALER_MLP_JOBLIB.exists() else None

    return df_test_cat, df_test_num, cpt_data, rf_model, mlp_model, scaler_mlp


def evaluar_modelo(nombre, y_true, y_probas, umbral=DECISION_THRESHOLD, tiempo_inferencia=0.0) -> dict:
    """Calcula matriz de confusión, precisión, sensibilidad, F1 y exactitud para clase Compra (True)."""
    y_pred = (np.array(y_probas) >= umbral).astype(int)
    y_true_int = np.array(y_true).astype(int)

    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y_true_int, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true_int, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true_int, y_pred, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y_true_int, y_pred))

    return {
        "modelo": nombre,
        "umbral": umbral,
        "tiempo_inferencia_ms": round(tiempo_inferencia * 1000, 2),
        "exactitud": round(acc, 4),
        "precision": round(prec, 4),
        "sensibilidad": round(rec, 4),
        "f1_score": round(f1, 4),
        "matriz_confusion": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        }
    }


def ejecutar_evaluacion_completa():
    """Ejecuta la evaluación simultánea de los 4 enfoques y guarda resultados."""
    df_cat, df_num, cpt_data, rf, mlp, scaler = cargar_modelos_y_datos_test()
    y_true = df_cat[TARGET_COL].astype(bool).values
    n_test = len(y_true)

    resultados = {}

    # 1. Naive Bayes
    t0 = time.time()
    probas_nb = []
    for _, fila in df_cat.iterrows():
        ev = {
            "VisitorType": fila["VisitorType"],
            "ProductRelated": fila["ProductRelated"],
            "ProductRelated_Duration": fila["ProductRelated_Duration"]
        }
        res = inferir_probabilidad_naive_bayes(cpt_data, ev)
        probas_nb.append(res["prob_true"])
    t_nb = time.time() - t0
    resultados["Naive Bayes"] = evaluar_modelo("Naive Bayes", y_true, probas_nb, tiempo_inferencia=t_nb)

    # 2. Random Forest
    if rf:
        features_num = [c for c in df_num.columns if c != TARGET_COL]
        x_num = df_num[features_num]
        t0 = time.time()
        probas_rf = rf.predict_proba(x_num)[:, 1]
        t_rf = time.time() - t0
        resultados["Random Forest"] = evaluar_modelo("Random Forest", y_true, probas_rf, tiempo_inferencia=t_rf)

    # 3. Perceptrón Multicapa MLP
    if mlp and scaler:
        features_num = [c for c in df_num.columns if c != TARGET_COL]
        x_scaled = scaler.transform(df_num[features_num])
        t0 = time.time()
        probas_mlp = mlp.predict_proba(x_scaled)[:, 1]
        t_mlp = time.time() - t0
        resultados["Perceptrón Multicapa (MLP)"] = evaluar_modelo("Perceptrón Multicapa (MLP)", y_true, probas_mlp, tiempo_inferencia=t_mlp)

    # 4. Referencia Base (Mayoritaria "No Compra")
    t0 = time.time()
    probas_base = np.zeros(n_test, dtype=float)
    t_base = time.time() - t0
    resultados["Referencia Base (Mayoritaria)"] = evaluar_modelo("Referencia Base (Mayoritaria)", y_true, probas_base, tiempo_inferencia=t_base)

    # Guardar en artifacts
    METRICAS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICAS_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    return resultados


def main():
    print("=" * 82)
    print(" APARTADO 5: EVALUACIÓN COMPARATIVA EN TEST (20% - 2,466 MUESTRAS)")
    print("=" * 82)

    res = ejecutar_evaluacion_completa()

    header = f" {'Modelo':<30} {'Exactitud':<11} {'Precisión':<11} {'Recall':<11} {'F1-Score':<11} {'Tiempo (ms)':<12}"
    print("\n" + header)
    print("-" * len(header))
    for mod, data in res.items():
        print(f" {mod:<30} {data['exactitud']:<11.4f} {data['precision']:<11.4f} {data['sensibilidad']:<11.4f} {data['f1_score']:<11.4f} {data['tiempo_inferencia_ms']:<12.1f}")

    print("\nMATRICES DE CONFUSIÓN:")
    for mod, data in res.items():
        cm = data["matriz_confusion"]
        print(f" • {mod:<30}: TN={cm['TN']:<4} | FP={cm['FP']:<4} | FN={cm['FN']:<4} | TP={cm['TP']:<4}")

    print(f"\n[OK] Métricas serializadas en: {METRICAS_JSON}")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()
