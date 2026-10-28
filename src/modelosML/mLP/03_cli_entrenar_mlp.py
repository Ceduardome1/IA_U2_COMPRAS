#!/usr/bin/env python3
"""
03_cli_entrenar_mlp.py
Tipo: LÍNEA DE COMANDOS (CLI)
Entregable: Apartado 4 / 5 - Entrenamiento de Perceptrón Multicapa (MLP) con Early Stopping

Pipeline:
1. Carga 'data/train_numerico.csv'.
2. Ajusta StandardScaler ÚNICAMENTE sobre los datos de entrenamiento.
3. Serializa el escalador en 'artifacts/scaler_mlp.joblib'.
4. Configura MLPClassifier con dos capas ocultas, función ReLU, optimizador Adam,
   validación interna (15%) y parada temprana (early_stopping=True, n_iter_no_change=10).
5. Mide tiempo de entrenamiento, registra épocas hasta convergencia y curva de pérdida.
6. Serializa el modelo en 'artifacts/mlp_model.joblib' y sus metadatos en 'artifacts/mlp_metadatos.json'.
"""

import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_NUM_PATH, MLP_MODEL_JOBLIB, SCALER_MLP_JOBLIB, ARTIFACTS_DIR, TARGET_COL, RANDOM_STATE


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


def entrenar_mlp():
    print("=" * 70)
    print(" 03. ENTRENAMIENTO DE PERCEPTRÓN MULTICAPA (MLP) (CLI)")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[1/5] Cargando datos numéricos de entrenamiento...")
    X_train, y_train, feature_names = cargar_datos_train()
    print(f"      Muestras: {len(X_train)} filas x {len(feature_names)} características:")
    for f in feature_names:
        print(f"        - {f}")

    # 2. Ajustar StandardScaler solo en Train
    print("\n[2/5] Ajustando StandardScaler únicamente con datos de entrenamiento...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_MLP_JOBLIB)
    print(f"      Medias ajustadas: {dict(zip(feature_names, [round(float(m), 2) for m in scaler.mean_]))}")
    print(f"      Escalador serializado en: {SCALER_MLP_JOBLIB.relative_to(BASE_DIR)}")

    # 3. Configurar arquitectura e hiperparámetros
    print("\n[3/5] Configurando arquitectura de red y parámetros:")
    mlp_params = {
        "hidden_layer_sizes": (32, 16),    # Dos capas ocultas para extracción jerárquica de patrones
        "activation": "relu",              # Rectified Linear Unit para evitar desvanecimiento del gradiente
        "solver": "adam",                  # Optimización estocástica adaptativa
        "alpha": 0.01,                     # Regularización L2 (weight decay) para controlar sobreajuste
        "learning_rate_init": 0.005,       # Tasa de aprendizaje inicial
        "early_stopping": True,            # Monitoreo interno sobre validación
        "validation_fraction": 0.15,       # 15% de train reservado para validación interna
        "n_iter_no_change": 10,            # Detener si no mejora en 10 épocas consecutivas
        "max_iter": 350,                   # Límite superior de épocas
        "random_state": RANDOM_STATE
    }
    for k, v in mlp_params.items():
        print(f"      {k}: {v}")

    # 4. Ajustar modelo con validación y parada temprana
    print("\n[4/5] Entrenando MLPClassifier y monitoreando curva de pérdida...")
    mlp = MLPClassifier(**mlp_params)
    t0 = time.time()
    mlp.fit(X_train_scaled, y_train)
    t_train = time.time() - t0

    epocas = int(mlp.n_iter_)
    loss_final = float(mlp.loss_)
    val_score = float(mlp.best_validation_score_) if hasattr(mlp, "best_validation_score_") and mlp.best_validation_score_ is not None else None
    curva_perdida = [float(loss) for loss in mlp.loss_curve_]
    curva_val = [float(s) for s in mlp.validation_scores_] if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None else []

    print(f"\n      Resultado del ajuste:")
    print(f"      • Épocas hasta parada temprana: {epocas} (máx. 350)")
    print(f"      • Pérdida final en train:       {loss_final:.4f}")
    if val_score is not None:
        print(f"      • Mejor Score en validación:    {val_score:.4f}")
    print(f"      • Tiempo total de cómputo:       {t_train:.4f} segundos")

    # Evaluación interna sobre train
    y_pred_proba = mlp.predict_proba(X_train_scaled)[:, 1]
    roc_auc = float(roc_auc_score(y_train, y_pred_proba))
    print(f"      • ROC AUC sobre entrenamiento:   {roc_auc:.4f}")

    # 5. Serializar modelo y metadatos
    print("\n[5/5] Serializando modelo y metadatos...")
    joblib.dump(mlp, MLP_MODEL_JOBLIB)
    print(f"[OK] Modelo MLP guardado en: {MLP_MODEL_JOBLIB.relative_to(BASE_DIR)}")

    metadatos_mlp = {
        "modelo": "MLPClassifier",
        "arquitectura": {
            "nodos_entrada": len(feature_names),
            "capas_ocultas": list(mlp_params["hidden_layer_sizes"]),
            "nodos_salida": 2,
            "activacion": mlp_params["activation"],
            "optimizador": mlp_params["solver"]
        },
        "hiperparametros": {
            "alpha": mlp_params["alpha"],
            "learning_rate_init": mlp_params["learning_rate_init"],
            "early_stopping": mlp_params["early_stopping"],
            "n_iter_no_change": mlp_params["n_iter_no_change"],
            "validation_fraction": mlp_params["validation_fraction"]
        },
        "entrenamiento": {
            "tiempo_segundos": t_train,
            "epocas_convergidas": epocas,
            "perdida_final": loss_final,
            "mejor_score_validacion": val_score,
            "roc_auc_train": roc_auc,
            "curva_perdida": curva_perdida,
            "curva_validacion": curva_val
        },
        "caracteristicas": feature_names
    }

    meta_path = ARTIFACTS_DIR / "mlp_metadatos.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatos_mlp, f, indent=4, ensure_ascii=False)
    print(f"[OK] Metadatos guardados en: {meta_path.relative_to(BASE_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    entrenar_mlp()
