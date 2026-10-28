import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    SELECTED_FEATURES,
    CATEGORIES_VISITOR,
    CATEGORIES_DISCRETE,
    RANDOM_STATE,
    TEST_SIZE,
    PERCENTILES_JSON
)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset crudo asegurando las columnas del estudio."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos crudo en {path}")
    df = pd.read_csv(path)
    required_cols = SELECTED_FEATURES + [TARGET_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")
    return df[required_cols].copy()


def split_data(df: pd.DataFrame, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Partición estratificada 80% train / 20% test según Revenue."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL]
    )
    return train_df.copy(), test_df.copy()


def calculate_percentiles_train(train_df: pd.DataFrame) -> dict:
    """Calcula los percentiles P33 y P67 únicamente sobre el conjunto de entrenamiento."""
    p33_pr = float(np.percentile(train_df["ProductRelated"], 33))
    p67_pr = float(np.percentile(train_df["ProductRelated"], 67))
    p33_dur = float(np.percentile(train_df["ProductRelated_Duration"], 33))
    p67_dur = float(np.percentile(train_df["ProductRelated_Duration"], 67))

    cortes = {
        "ProductRelated": {
            "p33": p33_pr,
            "p67": p67_pr,
            "etiquetas": CATEGORIES_DISCRETE
        },
        "ProductRelated_Duration": {
            "p33": p33_dur,
            "p67": p67_dur,
            "etiquetas": CATEGORIES_DISCRETE
        }
    }
    return cortes


def save_percentiles(cortes: dict, file_path=PERCENTILES_JSON):
    """Guarda los percentiles en un archivo JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cortes, f, indent=4, ensure_ascii=False)


def load_percentiles(file_path=PERCENTILES_JSON) -> dict:
    """Carga los percentiles desde el JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discretize_value(val: float, p33: float, p67: float) -> str:
    """Discretiza un valor escalar continuo en 'bajo', 'medio' o 'alto'."""
    if pd.isna(val):
        return None
    if val <= p33:
        return "bajo"
    elif val <= p67:
        return "medio"
    else:
        return "alto"


def discretize_series(series: pd.Series, p33: float, p67: float) -> pd.Series:
    """Discretiza una serie de pandas usando los límites p33 y p67."""
    return series.apply(lambda v: discretize_value(v, p33, p67))


def prepare_categorical_representation(df: pd.DataFrame, percentiles: dict) -> pd.DataFrame:
    """
    Genera la versión categórica para Naive Bayes:
    ProductRelated y ProductRelated_Duration discretizados en bajo, medio, alto.
    """
    cat_df = pd.DataFrame()
    cat_df["VisitorType"] = df["VisitorType"].astype(str)
    
    pr_cortes = percentiles["ProductRelated"]
    cat_df["ProductRelated"] = discretize_series(
        df["ProductRelated"], pr_cortes["p33"], pr_cortes["p67"]
    )
    
    dur_cortes = percentiles["ProductRelated_Duration"]
    cat_df["ProductRelated_Duration"] = discretize_series(
        df["ProductRelated_Duration"], dur_cortes["p33"], dur_cortes["p67"]
    )
    
    cat_df[TARGET_COL] = df[TARGET_COL].astype(bool)
    return cat_df


def prepare_numeric_representation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera la versión numérica para ML (Random Forest y MLP):
    Conserva valores numéricos continuos y codifica VisitorType con One-Hot Encoding consistente.
    """
    num_df = pd.DataFrame()
    num_df["ProductRelated"] = df["ProductRelated"].astype(float)
    num_df["ProductRelated_Duration"] = df["ProductRelated_Duration"].astype(float)
    
    # One-hot encoding canónico para evitar dependencias del orden en test
    for v_type in CATEGORIES_VISITOR:
        col_name = f"VisitorType_{v_type}"
        num_df[col_name] = (df["VisitorType"] == v_type).astype(int)
        
    num_df[TARGET_COL] = df[TARGET_COL].astype(bool)
    return num_df
