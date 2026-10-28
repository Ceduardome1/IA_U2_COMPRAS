#!/usr/bin/env python3
"""
04_cli_evaluar_todos.py
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
   - Referencia Base (Baseline): clasificador mayoritario que siempre predice "No compra" (P(Compra) = 0.0 o frecuencia base).
3. Regla de Decisión Común (Umbral 0.50):
   - Para cada modelo, si P(Compra) >= 0.50 -> Predice Compra (True / 1).
   - Si P(Compra) < 0.50 -> Predice No Compra (False / 0).
4. Métricas de Evaluación Requeridas:
   - Matriz de confusión completa (TN, FP, FN, TP).
   - Precisión (Precision) para la clase Compra: TP / (TP + FP).
   - Sensibilidad (Recall) para la clase Compra: TP / (TP + FN).
   - Puntuación F1 para la clase Compra: 2 * (Prec * Rec) / (Prec + Rec).
   - Exactitud global (Accuracy).
   - Tiempos de cómputo: tiempo de entrenamiento (registrado previamente) y tiempo de inferencia en test.
5. Salida y Serialización:
   - Impresión en consola de tabla comparativa formateada.
   - Serialización de todas las métricas en 'artifacts/metricas_evaluacion.json'.

Uso:
    python src/comparacion/04_cli_evaluar_todos.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def cargar_modelos_y_datos_test():
    """Carga los conjuntos de prueba y los 3 modelos serializados."""
    pass


def evaluar_modelo(nombre, y_true, y_probas, umbral=0.50, tiempo_inferencia=0.0) -> dict:
    """Calcula matriz de confusión, precisión, sensibilidad, F1 y exactitud."""
    pass


def ejecutar_evaluacion_completa():
    """Ejecuta la evaluación simultánea de los 4 enfoques y guarda resultados."""
    pass


def main():
    """Punto de entrada de evaluación por consola."""
    pass


if __name__ == "__main__":
    main()
