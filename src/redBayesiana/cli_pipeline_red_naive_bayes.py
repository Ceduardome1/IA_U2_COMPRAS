#!/usr/bin/env python3
"""
cli_pipeline_red_naive_bayes.py
Tipo: LÍNEA DE COMANDOS (CLI)
Responsable: JAIME / CESAR
Entregable: Apartado 3 - Construcción de Red Naive Bayes y Tablas CPT con Laplace

Objetivos y Especificaciones:
1. Definición formal de la Red Bayesiana:
   - Nodo Padre: Revenue (Compra = True / False).
   - Nodos Hijos condicionalmente independientes dado Revenue:
     * VisitorType (Returning_Visitor, New_Visitor, Other) -> K=3
     * ProductRelated (bajo, medio, alto) -> K=3
     * ProductRelated_Duration (bajo, medio, alto) -> K=3
2. Cálculo de Tablas de Probabilidad Condicional (CPT) con Suavizado de Laplace:
   - Para cada variable X_i con cardinalidad de dominio K_i:
     P(X_i = x | Revenue = c) = [conteo(X_i = x, Revenue = c) + 1] / [conteo(Revenue = c) + K_i]
   - A priori de Revenue con Laplace:
     P(Revenue = c) = [conteo(Revenue = c) + 1] / [N + 2]
   - Justificación de Laplace: evitar probabilidades condicionales cero que anulen el producto a posteriori.
3. Persistencia y Almacenamiento:
   - Serializar las tablas y metadatos en 'artifacts/tablas_cpt_laplace.json'.
   - Exportar tabla plana de análisis estadístico en 'artifacts/analisis_estadistico_cpt.csv'.
4. Motor de Inferencia Progresiva:
   - Soporte para calcular P(Revenue = True | Evidencia).
   - Manejo de variables no proporcionadas o desconocidas mediante marginalización (factor neutro 1.0).

Uso:
    python src/redBayesiana/cli_pipeline_red_naive_bayes.py
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TRAIN_CAT_PATH,
    CPT_LAPLACE_JSON,
    TARGET_COL
)


def calcular_cpt_laplace(train_cat_path: Path = TRAIN_CAT_PATH) -> dict:
    """
    Calcula priors y CPTs con suavizado de Laplace (+1) para todas las categorías
    a partir del conjunto de entrenamiento categórico.
    """
    if not train_cat_path.exists():
        # Fallback de búsqueda robusta
        candidatos = [
            BASE_DIR / "data" / "train_categorico.csv",
            Path.cwd() / "data" / "train_categorico.csv",
            Path(__file__).resolve().parent.parent.parent / "data" / "train_categorico.csv"
        ]
        encontrado = None
        for c in candidatos:
            if c.exists():
                encontrado = c
                break
        if not encontrado:
            raise FileNotFoundError(f"No se encontró train_categorico.csv en ninguna de las rutas esperadas.")
        train_cat_path = encontrado

    df = pd.read_csv(train_cat_path)
    total_muestras = len(df)

    # Conteos a priori de Revenue
    conteo_false = int((df[TARGET_COL] == False).sum())
    conteo_true = int((df[TARGET_COL] == True).sum())

    prior_emp_false = conteo_false / total_muestras
    prior_emp_true = conteo_true / total_muestras

    # Suavizado de Laplace para priors (K=2 clases binarias)
    prior_lap_false = (conteo_false + 1) / (total_muestras + 2)
    prior_lap_true = (conteo_true + 1) / (total_muestras + 2)

    variables = ["VisitorType", "ProductRelated", "ProductRelated_Duration"]
    cpt_dict = {}
    tabla_plana = []
    resumen_comparativo = []

    for var in variables:
        cpt_dict[var] = {}
        categorias = sorted(list(df[var].astype(str).unique()))
        k_i = len(categorias)

        for cat in categorias:
            # Conteos
            n_cat_false = int(((df[var].astype(str) == cat) & (df[TARGET_COL] == False)).sum())
            n_cat_true = int(((df[var].astype(str) == cat) & (df[TARGET_COL] == True)).sum())

            # Probabilidades empíricas
            prob_emp_false = n_cat_false / conteo_false if conteo_false > 0 else 0.0
            prob_emp_true = n_cat_true / conteo_true if conteo_true > 0 else 0.0

            # Suavizado de Laplace (+1 / +K_i)
            prob_lap_false = (n_cat_false + 1) / (conteo_false + k_i)
            prob_lap_true = (n_cat_true + 1) / (conteo_true + k_i)

            # Razón de verosimilitud (Likelihood Ratio) y Log-Odds
            lr = prob_lap_true / prob_lap_false if prob_lap_false > 0 else 1.0
            log_odds = math.log(lr) if lr > 0 else 0.0

            if lr > 1.05:
                impacto = "Favorece Compra (+)"
            elif lr < 0.95:
                impacto = "Favorece No Compra (-)"
            else:
                impacto = "Neutro (~)"

            # Guardar en estructura CPT
            cpt_dict[var][cat] = {
                "conteo_false": n_cat_false,
                "conteo_true": n_cat_true,
                "conteo_total": n_cat_false + n_cat_true,
                "k_categorias": k_i,
                "prob_empirica_false": round(prob_emp_false, 6),
                "prob_empirica_true": round(prob_emp_true, 6),
                "prob_laplace_false": round(prob_lap_false, 6),
                "prob_laplace_true": round(prob_lap_true, 6),
                "razon_verosimilitud": round(lr, 4),
                "log_odds": round(log_odds, 4),
                "impacto": impacto
            }

            # Registro plano para análisis y exportación
            resumen_comparativo.append({
                "Variable": var,
                "Categoria": cat,
                "Conteo_No_Compra": n_cat_false,
                "Conteo_Compra": n_cat_true,
                "Total": n_cat_false + n_cat_true,
                "Prob_Empirica_No_Compra": round(prob_emp_false, 6),
                "Prob_Empirica_Compra": round(prob_emp_true, 6),
                "Prob_Laplace_No_Compra": round(prob_lap_false, 6),
                "Prob_Laplace_Compra": round(prob_lap_true, 6),
                "Razon_Verosimilitud": round(lr, 4),
                "Impacto": impacto
            })

    cpt_data = {
        "metadatos": {
            "fecha_generacion": datetime.now().isoformat(),
            "dataset_origen": str(train_cat_path),
            "total_muestras_train": total_muestras,
            "conteo_revenue_false": conteo_false,
            "conteo_revenue_true": conteo_true,
            "priors_empiricos": {
                "False": round(prior_emp_false, 6),
                "True": round(prior_emp_true, 6)
            },
            "priors_laplace": {
                "False": round(prior_lap_false, 6),
                "True": round(prior_lap_true, 6)
            },
            "variables_incluidas": variables,
            "cardinalidades": {var: len(cpt_dict[var]) for var in variables}
        },
        "cpt": cpt_dict,
        "tabla_plana": resumen_comparativo
    }

    return cpt_data


def guardar_tablas_cpt(cpt_data: dict, out_json: Path = CPT_LAPLACE_JSON):
    """Serializa las tablas calculadas en JSON y exporta un resumen CSV para análisis."""
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cpt_data, f, indent=4, ensure_ascii=False)

    # Exportar también el CSV de análisis estadístico
    out_csv = out_json.parent / "analisis_estadistico_cpt.csv"
    df_analisis = pd.DataFrame(cpt_data["tabla_plana"])
    df_analisis.to_csv(out_csv, index=False, encoding="utf-8")


def cargar_tablas_cpt(json_path: Path = CPT_LAPLACE_JSON, force_recalc: bool = False) -> dict:
    """
    Carga las tablas CPT persistidas desde el archivo JSON si existe.
    Si no existe o si force_recalc=True, calcula desde train_categorico.csv y persiste.
    """
    if not force_recalc and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cpt_data = json.load(f)
                return cpt_data
        except Exception:
            pass

    # Recalcular y persistir
    cpt_data = calcular_cpt_laplace()
    guardar_tablas_cpt(cpt_data, json_path)
    return cpt_data


def inferir_probabilidad_naive_bayes(cpt_data: dict, evidencia: dict) -> dict:
    """
    Motor de Inferencia Naive Bayes con Laplace:
    Calcula P(Revenue = True | Evidencia) y P(Revenue = False | Evidencia).
    Soporta evidencia parcial (variables ausentes o None se marginalizan con factor neutro 1.0).
    """
    p_true = cpt_data["metadatos"]["priors_laplace"]["True"]
    p_false = cpt_data["metadatos"]["priors_laplace"]["False"]

    cpt = cpt_data["cpt"]
    detalles = []

    puntaje_true = p_true
    puntaje_false = p_false

    for var, val in evidencia.items():
        if val is None or str(val).lower() in ["desconocido", "none", "nan", ""]:
            # Variable no proporcionada: factor neutro 1.0
            detalles.append({
                "variable": var,
                "valor": "Desconocido / Marginalizado",
                "factor_true": 1.0,
                "factor_false": 1.0
            })
            continue

        val_str = str(val)
        if var in cpt and val_str in cpt[var]:
            f_true = cpt[var][val_str]["prob_laplace_true"]
            f_false = cpt[var][val_str]["prob_laplace_false"]
        else:
            # Fallback suavizado si surge un valor fuera de vocabulario
            k_i = cpt_data["metadatos"]["cardinalidades"].get(var, 3)
            f_true = 1.0 / (cpt_data["metadatos"]["conteo_revenue_true"] + k_i)
            f_false = 1.0 / (cpt_data["metadatos"]["conteo_revenue_false"] + k_i)

        puntaje_true *= f_true
        puntaje_false *= f_false

        detalles.append({
            "variable": var,
            "valor": val_str,
            "factor_true": f_true,
            "factor_false": f_false
        })

    # Normalización para obtener probabilidades que sumen 1.0
    evidencia_total = puntaje_true + puntaje_false
    prob_true = (puntaje_true / evidencia_total) if evidencia_total > 0 else 0.5
    prob_false = 1.0 - prob_true

    return {
        "prob_true": prob_true,
        "prob_false": prob_false,
        "puntaje_crudo_true": puntaje_true,
        "puntaje_crudo_false": puntaje_false,
        "detalles": detalles,
        "decision_050": "Compra (True)" if prob_true >= 0.50 else "No Compra (False)"
    }


def evaluar_consultas_progresivas(cpt_data: dict, caso_ejemplo: dict):
    """Evalúa las 3 etapas sobre un caso de prueba para verificar el motor de inferencia."""
    print("\n" + "=" * 78)
    print(" EVALUACIÓN DE CONSULTAS PROGRESIVAS (MOTOR DE INFERENCIA NAIVE BAYES)")
    print("=" * 78)

    # Etapa 1: Solo VisitorType
    ev1 = {"VisitorType": caso_ejemplo.get("VisitorType")}
    res1 = inferir_probabilidad_naive_bayes(cpt_data, ev1)
    print(f"\n[Etapa 1] Evidencia: VisitorType = '{ev1['VisitorType']}'")
    print(f"          P(Compra | Evidencia)    = {res1['prob_true']:.4f} ({res1['prob_true']*100:.2f}%)")
    print(f"          P(No Compra | Evidencia) = {res1['prob_false']:.4f} ({res1['prob_false']*100:.2f}%)")
    print(f"          Decisión (Umbral 0.50):   {res1['decision_050']}")

    # Etapa 2: VisitorType + ProductRelated
    ev2 = {
        "VisitorType": caso_ejemplo.get("VisitorType"),
        "ProductRelated": caso_ejemplo.get("ProductRelated")
    }
    res2 = inferir_probabilidad_naive_bayes(cpt_data, ev2)
    print(f"\n[Etapa 2] Evidencia agregada: ProductRelated = '{ev2['ProductRelated']}'")
    print(f"          P(Compra | Evidencia)    = {res2['prob_true']:.4f} ({res2['prob_true']*100:.2f}%)")
    print(f"          P(No Compra | Evidencia) = {res2['prob_false']:.4f} ({res2['prob_false']*100:.2f}%)")
    print(f"          Decisión (Umbral 0.50):   {res2['decision_050']}")

    # Etapa 3: Evidencia Completa
    ev3 = {
        "VisitorType": caso_ejemplo.get("VisitorType"),
        "ProductRelated": caso_ejemplo.get("ProductRelated"),
        "ProductRelated_Duration": caso_ejemplo.get("ProductRelated_Duration")
    }
    res3 = inferir_probabilidad_naive_bayes(cpt_data, ev3)
    print(f"\n[Etapa 3] Evidencia completa: Duración = '{ev3['ProductRelated_Duration']}'")
    print(f"          P(Compra | Evidencia)    = {res3['prob_true']:.4f} ({res3['prob_true']*100:.2f}%)")
    print(f"          P(No Compra | Evidencia) = {res3['prob_false']:.4f} ({res3['prob_false']*100:.2f}%)")
    print(f"          Decisión (Umbral 0.50):   {res3['decision_050']}")


def main():
    """Genera las tablas CPT y muestra el resumen en consola."""
    print("=" * 78)
    print(" APARTADO 3: CONSTRUCCIÓN Y PERSISTENCIA DE TABLAS CPT CON LAPLACE (+1)")
    print("=" * 78)

    print("\n[1/3] Calculando probabilidades condicionales con Suavizado de Laplace...")
    cpt_data = calcular_cpt_laplace()
    meta = cpt_data["metadatos"]

    print(f"      • Dataset analizado:   {meta['dataset_origen']}")
    print(f"      • Muestras de Train:   {meta['total_muestras_train']:,}")
    print(f"      • Conteo No Compra:    {meta['conteo_revenue_false']:,} (Prior Laplace: {meta['priors_laplace']['False']:.4f})")
    print(f"      • Conteo Compra:       {meta['conteo_revenue_true']:,} (Prior Laplace: {meta['priors_laplace']['True']:.4f})")

    print("\n[2/3] Guardando persistencia de datos estadísticos...")
    guardar_tablas_cpt(cpt_data, CPT_LAPLACE_JSON)
    print(f"      [OK] JSON persistido en:  {CPT_LAPLACE_JSON}")
    print(f"      [OK] CSV de análisis en:  {CPT_LAPLACE_JSON.parent / 'analisis_estadistico_cpt.csv'}")

    print("\n[3/3] Resumen Tabular de las Tablas de Probabilidad Condicional (CPT):")
    header = f"    {'Variable':<24} {'Categoría':<20} {'P(x|¬C) Laplace':<18} {'P(x|C) Laplace':<18} {'LR':<8} {'Impacto':<20}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for row in cpt_data["tabla_plana"]:
        print(f"    {row['Variable']:<24} {row['Categoria']:<20} {row['Prob_Laplace_No_Compra']:<18.4f} {row['Prob_Laplace_Compra']:<18.4f} {row['Razon_Verosimilitud']:<8.3f} {row['Impacto']:<20}")

    # Demostración del motor con caso de prueba representativo
    caso_test = {
        "VisitorType": "Returning_Visitor",
        "ProductRelated": "alto",
        "ProductRelated_Duration": "alto"
    }
    evaluar_consultas_progresivas(cpt_data, caso_test)

    print("\n" + "=" * 78)
    print(" [OK] Pipeline del Apartado 3 finalizado exitosamente con datos persistidos.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
