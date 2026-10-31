#!/usr/bin/env python3
"""
exploracion_mapa.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: JAIME / CESAR
Entregable: Apartado 1 - Exploración de Datos y Mapa Conceptual Integrado

Objetivos y Especificaciones:
1. Mapa Conceptual con Grafos NetworkX:
   - Representación gráfica de la red de conceptos del problema:
     * Concepto central: Intención de Compra en Sesión Web.
     * Enlaces y proposiciones:
       - "clasificada por" -> Paradigma probabilístico (Bayes / Naive Bayes) y Aprendizaje Supervisado (RF / MLP).
       - "caracterizada por entradas" -> Tipo de Visitante, Páginas de Producto, Duración.
       - "discretizada para" -> Naive Bayes (bajo, medio, alto mediante P33 y P67).
       - "escalada para" -> Red Neuronal MLP.
       - "estimada mediante" -> Probabilidad continua P(Compra).
       - "decidida bajo umbral" -> Regla de corte 0.50.
2. Exploración visual del dataset Online Shoppers Purchasing Intention:
   - Gráfico de barras del desbalance de la clase objetivo 'Revenue' (84.53% False vs 15.47% True).
   - Resumen estadístico y distribuciones de las 3 variables de entrada:
     * VisitorType (Returning_Visitor, New_Visitor, Other)
     * ProductRelated (conteos continuos y cortes P33=10, P67=29)
     * ProductRelated_Duration (segundos continuos y cortes P33=285.93, P67=1107.23)
3. Modos de Ejecución Separados y Autónomos:
   - Incluye acceso directo para lanzar en ventanas autónomas independientes:
     * python src/analisisDatos/mapa_conceptual.py
     * python src/analisisDatos/distribucion_datos.py

Uso:
    python src/analisisDatos/exploracion_mapa.py
"""

import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    PERCENTILES_JSON,
    ARTIFACTS_DIR
)
from src.preprocessing import load_percentiles
from src.analisisDatos.mapa_conceptual import MapaConceptualApp
from src.analisisDatos.distribucion_datos import DistribucionDatosApp


class ExploracionMapaIntegradoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apartado 1: Exploración de Datos y Mapa Conceptual (NetworkX)")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 720)

        self._configurar_estilos()
        self._crear_interfaz()

    def _configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Arial", 14, "bold"), foreground="#1e293b")
        style.configure("Subheader.TLabel", font=("Arial", 10), foreground="#64748b")
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), padding=6)
        style.configure("Action.TButton", font=("Arial", 9), padding=5)

    def _crear_interfaz(self):
        # Barra superior de navegación y lanzadores independientes
        top_bar = ttk.Frame(self.root, padding="10 8 10 8")
        top_bar.pack(fill=tk.X)

        info_box = ttk.Frame(top_bar)
        info_box.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            info_box,
            text="Apartado 1: Exploración del Dataset y Mapa Conceptual",
            style="Header.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            info_box,
            text="Red de Conceptos con NetworkX • Distribuciones Estadísticas • Lanzadores Autónomos",
            style="Subheader.TLabel"
        ).pack(anchor="w")

        btn_box = ttk.Frame(top_bar)
        btn_box.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(
            btn_box,
            text="Abrir Mapa NetworkX en Ventana Propia",
            style="Accent.TButton",
            command=self._lanzar_mapa_autonomo
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_box,
            text="Abrir Distribución en Ventana Propia",
            style="Action.TButton",
            command=self._lanzar_distribucion_autonoma
        ).pack(side=tk.LEFT, padx=5)

        # Notebook principal con las dos grandes áreas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Pestaña 1: Mapa Conceptual con NetworkX
        tab_mapa = ttk.Frame(self.notebook)
        self.notebook.add(tab_mapa, text="Mapa Conceptual del Problema (NetworkX)")
        self.sub_mapa = MapaConceptualApp(tab_mapa)

        # Pestaña 2: Distribución y Representación de Datos
        tab_dist = ttk.Frame(self.notebook)
        self.notebook.add(tab_dist, text="📊 Distribución y Representación de Datos")
        self.sub_dist = DistribucionDatosApp(tab_dist)

    def _lanzar_mapa_autonomo(self):
        script_mapa = BASE_DIR / "src" / "analisisDatos" / "mapa_conceptual.py"
        subprocess.Popen([sys.executable, str(script_mapa)])

    def _lanzar_distribucion_autonoma(self):
        script_dist = BASE_DIR / "src" / "analisisDatos" / "distribucion_datos.py"
        subprocess.Popen([sys.executable, str(script_dist)])


def render_distribucion_revenue():
    """Genera gráfico de barras de balance de clases de Revenue."""
    if not RAW_DATA_PATH.exists():
        print(f"Error: {RAW_DATA_PATH} no encontrado.")
        return
    df = pd.read_csv(RAW_DATA_PATH)
    counts = df[TARGET_COL].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(["No Compra (False)", "Compra (True)"], counts.values, color=["#ef4444", "#10b981"])
    plt.title("Balance de Clase Objetivo (Revenue)")
    plt.ylabel("Sesiones")
    plt.show()


def render_distribucion_variables():
    """Genera histogramas para las 3 variables predictoras."""
    if not RAW_DATA_PATH.exists():
        return
    df = pd.read_csv(RAW_DATA_PATH)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(df["ProductRelated"].clip(upper=100), bins=30, color="#3b82f6")
    axs[0].set_title("ProductRelated (Páginas)")
    axs[1].hist(df["ProductRelated_Duration"].clip(upper=4000), bins=30, color="#8b5cf6")
    axs[1].set_title("ProductRelated_Duration (Segundos)")
    plt.tight_layout()
    plt.show()


def render_mapa_conceptual():
    """Lanza la visualización del mapa conceptual de NetworkX."""
    root = tk.Tk()
    app = MapaConceptualApp(root)
    root.mainloop()


def main():
    root = tk.Tk()
    app = ExploracionMapaIntegradoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
