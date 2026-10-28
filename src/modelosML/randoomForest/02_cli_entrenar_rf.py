#!/usr/bin/env python3
"""
02_cli_entrenar_rf.py
Tipo: LÍNEA DE COMANDOS (CLI)
Entregable: Apartado 4 / 5 - Entrenamiento de Random Forest

Pipeline:
1. Carga 'data/train_numerico.csv'.
2. Conserva variables continuas y codificación de VisitorType.
3. Ajusta RandomForestClassifier con justificación técnica de hiperparámetros.
4. Mide tiempo de entrenamiento y evalúa con validación cruzada interna (5-Fold Stratified).
5. Imprime reporte en consola y guarda en:
   - 'artifacts/random_forest.joblib'
   - 'artifacts/rf_metadatos.json'
"""

import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_NUM_PATH, RF_MODEL_JOBLIB, ARTIFACTS_DIR, TARGET_COL, RANDOM_STATE


def cargar_datos_train():
    if not TRAIN_NUM_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {TRAIN_NUM_PATH}. Ejecuta primero src/util/01_cli_preparar_datos.py"
        )
    df = pd.read_csv(TRAIN_NUM_PATH)
    features = [c for c in df.columns if c != TARGET_COL]
    X = df[features]
    y = df[TARGET_COL].astype(bool)
    return X, y, features


def entrenar_random_forest():
    print("=" * 70)
    print(" 02. ENTRENAMIENTO DE RANDOM FOREST (CLI)")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[1/4] Cargando datos numéricos de entrenamiento...")
    X_train, y_train, feature_names = cargar_datos_train()
    print(f"      Muestras: {len(X_train)} filas x {len(feature_names)} características:")
    for f in feature_names:
        print(f"        - {f}")

    # 2. Configurar hiperparámetros y justificación
    print("\n[2/4] Configurando hiperparámetros justificados:")
    rf_params = {
        "n_estimators": 150,        # Número de árboles para estabilidad y reducción de varianza
        "max_depth": 8,             # Control de profundidad para prevenir memorización/sobreajuste
        "min_samples_split": 10,    # Mínimo de muestras para dividir un nodo interno
        "min_samples_leaf": 5,      # Mínimo de muestras en hojas para regularizar regiones ruidosas
        "class_weight": "balanced", # Compensación del desbalance (85% no compra vs 15% compra)
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    for k, v in rf_params.items():
        print(f"      {k}: {v}")

    # 3. Validación cruzada interna en train (5-Folds)
    print("\n[3/4] Evaluando robustez interna (5-Fold Stratified CV en Train)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_cv = RandomForestClassifier(**rf_params)
    scores = cross_validate(
        rf_cv, X_train, y_train, cv=cv,
        scoring=["precision", "recall", "f1", "roc_auc"]
    )
    print(f"      F1 promedio (CV):        {scores['test_f1'].mean():.4f} (+/- {scores['test_f1'].std():.4f})")
    print(f"      Sensibilidad / Recall:  {scores['test_recall'].mean():.4f} (+/- {scores['test_recall'].std():.4f})")
    print(f"      Precisión:              {scores['test_precision'].mean():.4f} (+/- {scores['test_precision'].std():.4f})")
    print(f"      ROC AUC:                {scores['test_roc_auc'].mean():.4f} (+/- {scores['test_roc_auc'].std():.4f})")

    # 4. Ajuste final sobre todo el conjunto de entrenamiento
    print("\n[4/4] Ajustando modelo final y midiendo tiempo...")
    t0 = time.time()
    rf_final = RandomForestClassifier(**rf_params)
    rf_final.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"      Tiempo de entrenamiento: {t_train:.4f} segundos.")

    # Importancia de características
    importancias = dict(zip(feature_names, [float(v) for v in rf_final.feature_importances_]))
    print("\n      Importancia de características:")
    for f_name, imp in sorted(importancias.items(), key=lambda x: x[1], reverse=True):
        print(f"        * {f_name:32s}: {imp * 100:6.2f}%")

    # Serializar modelo y metadatos
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_final, RF_MODEL_JOBLIB)
    print(f"\n[OK] Modelo guardado en: {RF_MODEL_JOBLIB.relative_to(BASE_DIR)}")

    metadatos = {
        "modelo": "RandomForestClassifier",
        "hiperparametros": rf_params,
        "tiempo_entrenamiento_seg": t_train,
        "caracteristicas": feature_names,
        "importancia_caracteristicas": importancias,
        "cv_5fold": {
            "f1_mean": float(scores['test_f1'].mean()),
            "recall_mean": float(scores['test_recall'].mean()),
            "precision_mean": float(scores['test_precision'].mean()),
            "roc_auc_mean": float(scores['test_roc_auc'].mean())
        }
    }
    meta_path = ARTIFACTS_DIR / "rf_metadatos.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=4, ensure_ascii=False)
    print(f"[OK] Metadatos guardados en: {meta_path.relative_to(BASE_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    entrenar_random_forest()
