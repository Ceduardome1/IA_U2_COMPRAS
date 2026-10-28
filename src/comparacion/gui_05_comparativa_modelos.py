#!/usr/bin/env python3
"""
gui_05_comparativa_modelos.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: CESAR
Entregable: Apartado 5 - Panel Comparativo de Modelos y Matrices de Confusión

Objetivos y Especificaciones:
1. Panel Visual de Resultados Finales:
   - Carga y visualización de 'artifacts/metricas_evaluacion.json' (generado por 04_cli_evaluar_todos.py).
   - Tabla comparativa interactiva con los 4 enfoques:
     * Naive Bayes
     * Random Forest
     * Perceptrón Multicapa (MLP)
     * Referencia Base (Mayoritaria "No Compra")
   - Columnas de la tabla:
     * Precisión (Compra)
     * Sensibilidad / Recall (Compra)
     * F1-Score (Compra)
     * Exactitud Global
     * Tiempo de Inferencia
2. Visualización Gráfica de Matrices de Confusión:
   - Representación visual tipo Heatmap (mapa de calor) de las 4 matrices de confusión:
     * Cuadrícula de 2x2 o selector dinámico de modelo.
     * Muestra de conteos absolutos y etiquetas (Verdaderos Negativos, Falsos Positivos, Falsos Negativos, Verdaderos Positivos).
3. Panel de Análisis de Supuestos y Errores:
   - Resumen interactivo de las diferencias estructurales:
     * Por qué Naive Bayes pierde información al discretizar en 3 categorías.
     * Comparación de errores entre RF y MLP ante el desbalance de clases.
     * Justificación del comportamiento del Baseline mayoritario frente a la clase minoritaria.

Uso:
    python src/comparacion/gui_05_comparativa_modelos.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def render_tabla_metricas():
    """Dibuja la tabla comparativa con formato de cuadrícula."""
    pass


def render_matrices_confusion():
    """Genera los mapas de calor de las matrices de confusión con matplotlib."""
    pass


def render_panel_discusion():
    """Muestra cuadro de texto enriquecido con el análisis de errores y supuestos."""
    pass


def main():
    """Inicia la ventana de usuario de GUI 05."""
    pass


if __name__ == "__main__":
    main()
