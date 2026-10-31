#!/usr/bin/env python3
"""
cli_resumen_datos.py
Tipo: LÍNEA DE COMANDOS (CLI)
Responsable: JAIME / CESAR
Entregable: Apartado 1 - Resumen Exploratorio de Datos por Consola

Objetivos y Especificaciones:
1. Inspección del archivo crudo 'data/online_shoppers_intention.csv'.
2. Conteo de valores nulos o atípicos en las 4 variables de interés.
3. Tabla de proporciones exactas de la clase objetivo 'Revenue'.
4. Estadísticas descriptivas de ProductRelated y ProductRelated_Duration.
5. Frecuencia de categorías de VisitorType.

Uso:
    python src/analisisDatos/cli_resumen_datos.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    TRAIN_CAT_PATH,
    TRAIN_NUM_PATH,
    TARGET_COL,
    SELECTED_FEATURES,
    PERCENTILES_JSON
)
from src.preprocessing import load_percentiles


def inspeccionar_datos_crudos():
    """Imprime resumen estadístico y balance de clases en consola."""
    print("=" * 78)
    print(" APARTADO 1: INSPECCIÓN EXPLORATORIA DEL DATASET ONLINE SHOPPERS")
    print("=" * 78)

    if not RAW_DATA_PATH.exists():
        print(f"[ERROR] No se localizó el dataset crudo en: {RAW_DATA_PATH}")
        sys.exit(1)

    df_raw = pd.read_csv(RAW_DATA_PATH)
    n_filas_total, n_cols_total = df_raw.shape
    print(f"\n[1] METADATOS GENERALES DEL ARCHIVO:")
    print(f"    - Ruta del archivo: {RAW_DATA_PATH}")
    print(f"    - Registros totales: {n_filas_total:,}")
    print(f"    - Columnas totales:  {n_cols_total}")

    cols_estudio = SELECTED_FEATURES + [TARGET_COL]
    print(f"\n[2] VARIABLES SELECCIONADAS PARA EL ESTUDIO ({len(cols_estudio)}):")
    for col in cols_estudio:
        tipo = df_raw[col].dtype
        n_nulos = int(df_raw[col].isna().sum())
        n_unicos = df_raw[col].nunique()
        print(f"    • {col:<26} | Tipo: {str(tipo):<10} | Nulos: {n_nulos:<4} | Únicos: {n_unicos}")

    # Balance de la clase objetivo
    print("\n[3] BALANCE DE LA CLASE OBJETIVO ('Revenue'):")
    rev_counts = df_raw[TARGET_COL].value_counts()
    rev_props = df_raw[TARGET_COL].value_counts(normalize=True) * 100
    print(f"    {'Clase (Revenue)':<22} {'Conteo':<12} {'Proporción (%)':<15}")
    print("    " + "-" * 50)
    for clase, conteo in rev_counts.items():
        etiqueta = "Compra (True)" if bool(clase) else "No Compra (False)"
        prop = rev_props[clase]
        print(f"    {etiqueta:<22} {conteo:<12} {prop:>6.2f}%")
    print(f"    Ratio de Desbalance: 1 Compra por cada {rev_counts[False] / rev_counts[True]:.2f} No Compras")

    # Frecuencia de VisitorType
    print("\n[4] FRECUENCIA DE CATEGORÍAS ('VisitorType'):")
    vis_counts = df_raw["VisitorType"].value_counts()
    vis_props = df_raw["VisitorType"].value_counts(normalize=True) * 100
    print(f"    {'Categoría':<22} {'Conteo':<12} {'Proporción (%)':<15} {'Tasa Compra (%)':<15}")
    print("    " + "-" * 66)
    for cat, conteo in vis_counts.items():
        prop = vis_props[cat]
        tasa_compra = (df_raw[df_raw["VisitorType"] == cat][TARGET_COL] == True).mean() * 100
        print(f"    {cat:<22} {conteo:<12} {prop:>6.2f}%         {tasa_compra:>6.2f}%")

    # Estadísticas descriptivas de variables numéricas
    print("\n[5] ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS:")
    num_cols = ["ProductRelated", "ProductRelated_Duration"]
    stats_df = pd.DataFrame(index=[
        "Conteo", "Media", "Desv. Estándar", "Mínimo",
        "Percentil 25 (P25)", "Percentil 33 (P33)", "Mediana (P50)",
        "Percentil 67 (P67)", "Percentil 75 (P75)", "Máximo"
    ])

    for col in num_cols:
        s = df_raw[col].dropna()
        stats_df[col] = [
            f"{len(s):,}",
            f"{s.mean():.2f}",
            f"{s.std():.2f}",
            f"{s.min():.2f}",
            f"{np.percentile(s, 25):.2f}",
            f"{np.percentile(s, 33):.2f}",
            f"{s.median():.2f}",
            f"{np.percentile(s, 67):.2f}",
            f"{np.percentile(s, 75):.2f}",
            f"{s.max():.2f}",
        ]

    header_str = f"    {'Métrica / Estadístico':<26} " + "".join([f"{col:<26}" for col in num_cols])
    print(header_str)
    print("    " + "-" * len(header_str.strip()))
    for idx, row in stats_df.iterrows():
        fila_str = f"    {idx:<26} " + "".join([f"{row[col]:<26}" for col in num_cols])
        print(fila_str)

    # Cortes de discretización guardados
    if PERCENTILES_JSON.exists():
        percentiles = load_percentiles(PERCENTILES_JSON)
        print("\n[6] CORTES DE DISCRETIZACIÓN (Calculados en Train 80%):")
        for var, d in percentiles.items():
            print(f"    • {var}:")
            print(f"      - Rango 'Bajo':  x <= {d['p33']:.2f}")
            print(f"      - Rango 'Medio': {d['p33']:.2f} < x <= {d['p67']:.2f}")
            print(f"      - Rango 'Alto':  x > {d['p67']:.2f}")
    else:
        print(f"\n[6] CORTES DE DISCRETIZACIÓN: Aún no generados. Ejecutar 'python src/util/01_cli_preparar_datos.py'.")

    print("\n" + "=" * 78)
    print(" [OK] Inspección exploratoria finalizada correctamente.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    inspeccionar_datos_crudos()
