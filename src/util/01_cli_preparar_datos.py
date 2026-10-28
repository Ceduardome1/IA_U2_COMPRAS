#!/usr/bin/env python3
"""
01_cli_preparar_datos.py
Preparación y Transformación de Datos para cada Modelo.

Acciones:
1. Lee el dataset crudo (data/online_shoppers_intention.csv).
2. Ejecuta la partición estratificada 80% train / 20% test con semilla fija (random_state=42).
3. Calcula los percentiles P33 y P67 únicamente en entrenamiento y los guarda en artifacts/cortes_percentiles.json.
4. Genera y exporta las representaciones de datos:
   - Versión Categórica (para Naive Bayes): ProductRelated y ProductRelated_Duration
     discretizados en 'bajo', 'medio' y 'alto'.
     -> data/train_categorico.csv y data/test_categorico.csv
   - Versión Numérica (para ML: Random Forest y MLP): valores continuos originales
     y VisitorType codificado numéricamente (One-Hot).
     -> data/train_numerico.csv y data/test_numerico.csv
"""

import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TRAIN_CAT_PATH,
    TEST_CAT_PATH,
    TRAIN_NUM_PATH,
    TEST_NUM_PATH,
    PERCENTILES_JSON,
    RANDOM_STATE,
    TEST_SIZE
)
from src.preprocessing import (
    load_raw_data,
    split_data,
    calculate_percentiles_train,
    save_percentiles,
    prepare_categorical_representation,
    prepare_numeric_representation
)


def preparar_datos():
    print("=" * 70)
    print(" PREPARACIÓN Y TRANSFORMACIÓN DE DATOS")
    print("=" * 70)

    # 1. Cargar dataset crudo
    print("\n[1/4] Leyendo dataset crudo...")
    df_raw = load_raw_data()
    print(f"      Total registros leídos: {len(df_raw)} filas x {df_raw.shape[1]} columnas.")

    # 2. Partición estratificada 80/20
    print(f"\n[2/4] Ejecutando partición estratificada (80% train / 20% test, random_state={RANDOM_STATE})...")
    train_df, test_df = split_data(df_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    train_rev = train_df["Revenue"].value_counts(normalize=True).to_dict()
    test_rev = test_df["Revenue"].value_counts(normalize=True).to_dict()
    print(f"      Entrenamiento: {len(train_df)} muestras (Revenue=True: {train_rev.get(True, 0):.2%}, False: {train_rev.get(False, 0):.2%})")
    print(f"      Prueba:        {len(test_df)} muestras (Revenue=True: {test_rev.get(True, 0):.2%}, False: {test_rev.get(False, 0):.2%})")

    # 3. Percentiles P33 y P67 ajustados SOLO en train
    print("\n[3/4] Calculando percentiles P33 y P67 únicamente en train...")
    percentiles = calculate_percentiles_train(train_df)
    save_percentiles(percentiles, PERCENTILES_JSON)
    for var, cuts in percentiles.items():
        print(f"      {var}: P33 = {cuts['p33']:.2f}, P67 = {cuts['p67']:.2f}")
    print(f"      Guardado en: {PERCENTILES_JSON.relative_to(BASE_DIR)}")

    # 4. Generación de las dos versiones
    print("\n[4/4] Generando formatos de datos para cada modelo...")

    # Formato Categórico (para Naive Bayes)
    train_cat = prepare_categorical_representation(train_df, percentiles)
    test_cat = prepare_categorical_representation(test_df, percentiles)
    train_cat.to_csv(TRAIN_CAT_PATH, index=False)
    test_cat.to_csv(TEST_CAT_PATH, index=False)
    print(f"      [Para Naive Bayes] Guardados:")
    print(f"         - {TRAIN_CAT_PATH.relative_to(BASE_DIR)} ({train_cat.shape})")
    print(f"         - {TEST_CAT_PATH.relative_to(BASE_DIR)} ({test_cat.shape})")

    # Formato Numérico (para Random Forest y MLP)
    train_num = prepare_numeric_representation(train_df)
    test_num = prepare_numeric_representation(test_df)
    train_num.to_csv(TRAIN_NUM_PATH, index=False)
    test_num.to_csv(TEST_NUM_PATH, index=False)
    print(f"      [Para Modelos ML]  Guardados:")
    print(f"         - {TRAIN_NUM_PATH.relative_to(BASE_DIR)} ({train_num.shape})")
    print(f"         - {TEST_NUM_PATH.relative_to(BASE_DIR)} ({test_num.shape})")

    print("\n[OK] Datos preparados y transformados con éxito para cada modelo.")
    print("=" * 70)


if __name__ == "__main__":
    preparar_datos()
