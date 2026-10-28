import os
from pathlib import Path

# Directorios principales
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATASET_DIR = BASE_DIR / "dataset"

# Archivos de datos
RAW_DATA_PATH = DATA_DIR / "online_shoppers_intention.csv"
TRAIN_CAT_PATH = DATA_DIR / "train_categorico.csv"
TEST_CAT_PATH = DATA_DIR / "test_categorico.csv"
TRAIN_NUM_PATH = DATA_DIR / "train_numerico.csv"
TEST_NUM_PATH = DATA_DIR / "test_numerico.csv"

# Archivos de artefactos
PERCENTILES_JSON = ARTIFACTS_DIR / "cortes_percentiles.json"
CPT_LAPLACE_JSON = ARTIFACTS_DIR / "tablas_cpt_laplace.json"
SCALER_MLP_JOBLIB = ARTIFACTS_DIR / "scaler_mlp.joblib"
RF_MODEL_JOBLIB = ARTIFACTS_DIR / "random_forest.joblib"
MLP_MODEL_JOBLIB = ARTIFACTS_DIR / "mlp_model.joblib"
METRICAS_JSON = ARTIFACTS_DIR / "metricas_evaluacion.json"

# Configuración del experimento
RANDOM_STATE = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50

# Variables de entrada y objetivo
TARGET_COL = "Revenue"
NUMERIC_COLS = ["ProductRelated", "ProductRelated_Duration"]
CATEGORICAL_COLS = ["VisitorType"]
SELECTED_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

# Categorías canónicas
CATEGORIES_DISCRETE = ["bajo", "medio", "alto"]
CATEGORIES_VISITOR = ["Returning_Visitor", "New_Visitor", "Other"]
REVENUE_CLASSES = [False, True]
