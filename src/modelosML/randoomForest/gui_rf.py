#!/usr/bin/env python3
"""
gui_rf.py
Tipo: INTERFAZ GRÁFICA (GUI)
Módulo: Random Forest - Diagnóstico e Inferencia Interactiva

Características:
1. Panel de Diagnóstico:
   - Visualización de la justificación de hiperparámetros.
   - Gráfico de barras embebido de Importancia de Características.
   - Tiempos de entrenamiento y métricas de validación interna.
2. Panel de Inferencia y Simulación:
   - Selectores interactivos para VisitorType, ProductRelated y ProductRelated_Duration.
   - Estimación en tiempo real de P(Compra) y P(No Compra).
   - Veredicto binario bajo la regla de umbral reglamentario 0.50.
"""

import sys
import json
import joblib
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import RF_MODEL_JOBLIB, ARTIFACTS_DIR, CATEGORIES_VISITOR


class VentanaRandomForest(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Random Forest - Intención de Compra")
        self.geometry("980x720")
        self.minsize(850, 600)

        self.modelo = None
        self.metadatos = None
        self.cargar_modelo_y_metadatos()

        self.crear_widgets()

    def cargar_modelo_y_metadatos(self):
        meta_path = ARTIFACTS_DIR / "rf_metadatos.json"
        if RF_MODEL_JOBLIB.exists() and meta_path.exists():
            try:
                self.modelo = joblib.load(RF_MODEL_JOBLIB)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadatos = json.load(f)
            except Exception as e:
                messagebox.showerror("Error de carga", f"No se pudo cargar el modelo RF:\n{e}")
        else:
            messagebox.showwarning(
                "Modelo no encontrado",
                "No se encontraron los artefactos de Random Forest.\n"
                "Ejecuta en consola: python src/modelosML/randoomForest/02_cli_entrenar_rf.py"
            )

    def crear_widgets(self):
        # Título superior
        header = ttk.Frame(self, padding=12)
        header.pack(fill=tk.X)
        lbl_title = ttk.Label(
            header,
            text="Modelo de Machine Learning: Random Forest",
            font=("Helvetica", 16, "bold")
        )
        lbl_title.pack(anchor=tk.W)
        lbl_sub = ttk.Label(
            header,
            text="Variables numéricas originales continuas + VisitorType One-Hot | Umbral común 0.50",
            font=("Helvetica", 10),
            foreground="#555"
        )
        lbl_sub.pack(anchor=tk.W)

        # Pestañas principales
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        tab_diagnostico = ttk.Frame(notebook, padding=10)
        tab_simulador = ttk.Frame(notebook, padding=10)

        notebook.add(tab_diagnostico, text="  Diagnóstico e Importancia  ")
        notebook.add(tab_simulador, text="  Simulador de Sesiones Web  ")

        self.construir_tab_diagnostico(tab_diagnostico)
        self.construir_tab_simulador(tab_simulador)

    def construir_tab_diagnostico(self, parent):
        frame_top = ttk.Frame(parent)
        frame_top.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Hiperparámetros y CV
        left_panel = ttk.LabelFrame(frame_top, text="Configuración y Justificación", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8), pady=4)
        left_panel.config(width=420)

        txt_info = tk.Text(left_panel, width=48, height=22, wrap=tk.WORD, font=("Courier", 9))
        txt_info.pack(fill=tk.BOTH, expand=True)

        if self.metadatos:
            params = self.metadatos.get("hiperparametros", {})
            cv_res = self.metadatos.get("cv_5fold", {})
            t_train = self.metadatos.get("tiempo_entrenamiento_seg", 0.0)

            contenido = (
                "--- HIPERPARÁMETROS AJUSTADOS ---\n"
                f"• n_estimators     : {params.get('n_estimators')} (estabilidad del ensamble)\n"
                f"• max_depth        : {params.get('max_depth')} (control de sobreajuste)\n"
                f"• min_samples_split: {params.get('min_samples_split')} (mín. para bifurcar)\n"
                f"• min_samples_leaf : {params.get('min_samples_leaf')} (mín. muestras en hoja)\n"
                f"• class_weight     : {params.get('class_weight')} (compensar desbalance 85/15)\n"
                f"• random_state     : {params.get('random_state')} (reproducibilidad)\n\n"
                "--- RENDIMIENTO DE ENTRENAMIENTO ---\n"
                f"• Tiempo de ajuste : {t_train:.4f} s\n\n"
                "--- VALIDACIÓN INTERNA (5-FOLD CV) ---\n"
                f"• Sensibilidad (Recall) : {cv_res.get('recall_mean', 0):.4f}\n"
                f"• Precisión             : {cv_res.get('precision_mean', 0):.4f}\n"
                f"• Puntuación F1         : {cv_res.get('f1_mean', 0):.4f}\n"
                f"• ROC AUC               : {cv_res.get('roc_auc_mean', 0):.4f}\n\n"
                "--- JUSTIFICACIÓN TÉCNICA ---\n"
                "Random Forest opera sobre las cantidades y\n"
                "duraciones continuas originales sin necesidad\n"
                "de discretización. El uso de class_weight\n"
                "'balanced' permite penalizar errores en la clase\n"
                "minoritaria de compra, alcanzando un recall\n"
                "superior al 75% en validación interna."
            )
        else:
            contenido = "No hay metadatos cargados. Entrena el modelo primero."

        txt_info.insert(tk.END, contenido)
        txt_info.config(state=tk.DISABLED)

        # Panel derecho: Gráfico de Importancia
        right_panel = ttk.LabelFrame(frame_top, text="Importancia Relativa de Características", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=4)

        if self.metadatos and "importancia_caracteristicas" in self.metadatos:
            importancias = self.metadatos["importancia_caracteristicas"]
            sorted_items = sorted(importancias.items(), key=lambda x: x[1])
            labels = [k.replace("VisitorType_", "VT_") for k, _ in sorted_items]
            vals = [v * 100 for _, v in sorted_items]

            fig = Figure(figsize=(5.5, 4.8), dpi=100)
            ax = fig.add_subplot(111)
            bars = ax.barh(labels, vals, color="#2b5c8f", edgecolor="#1a3b5c")
            ax.set_xlabel("Importancia (%)", fontsize=9)
            ax.set_title("Contribución al Árbol de Decisión", fontsize=10, fontweight="bold")
            ax.set_xlim(0, max(vals) * 1.15)

            for bar in bars:
                w = bar.get_width()
                ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                        va="center", ha="left", fontsize=8, color="#333")

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=right_panel)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            lbl_no_graph = ttk.Label(right_panel, text="Gráfico no disponible sin entrenamiento previo.")
            lbl_no_graph.pack(pady=40)

    def construir_tab_simulador(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Entradas del usuario
        input_frame = ttk.LabelFrame(container, text="Datos de la Sesión Web de Entrada", padding=15)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 1. Tipo de visitante
        ttk.Label(input_frame, text="Tipo de Visitante (VisitorType):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.var_visitor = tk.StringVar(value="Returning_Visitor")
        cb_visitor = ttk.Combobox(
            input_frame, textvariable=self.var_visitor, values=CATEGORIES_VISITOR, state="readonly", width=30
        )
        cb_visitor.pack(anchor=tk.W, pady=(0, 12))

        # 2. ProductRelated (Páginas)
        ttk.Label(input_frame, text="Cantidad de Páginas de Productos (ProductRelated):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.var_paginas = tk.DoubleVar(value=25.0)
        spin_paginas = ttk.Spinbox(input_frame, from_=0, to=1000, increment=1, textvariable=self.var_paginas, width=15)
        spin_paginas.pack(anchor=tk.W, pady=(0, 4))
        scale_paginas = ttk.Scale(input_frame, from_=0, to=150, variable=self.var_paginas, orient=tk.HORIZONTAL)
        scale_paginas.pack(fill=tk.X, pady=(0, 12))

        # 3. ProductRelated_Duration (Segundos)
        ttk.Label(input_frame, text="Duración en Páginas de Productos (Segundos):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.var_duracion = tk.DoubleVar(value=800.0)
        spin_dur = ttk.Spinbox(input_frame, from_=0, to=50000, increment=10, textvariable=self.var_duracion, width=15)
        spin_dur.pack(anchor=tk.W, pady=(0, 4))
        scale_dur = ttk.Scale(input_frame, from_=0, to=3000, variable=self.var_duracion, orient=tk.HORIZONTAL)
        scale_dur.pack(fill=tk.X, pady=(0, 15))

        # Botón predecir
        btn_predecir = ttk.Button(
            input_frame, text="Estimar Intención con Random Forest", command=self.ejecutar_prediccion
        )
        btn_predecir.pack(fill=tk.X, pady=10)

        # Panel derecho: Resultados
        self.res_frame = ttk.LabelFrame(container, text="Resultado de la Estimación", padding=15)
        self.res_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_proba_compra = ttk.Label(
            self.res_frame, text="P(Compra): -- %", font=("Helvetica", 13, "bold"), foreground="#1a5276"
        )
        self.lbl_proba_compra.pack(anchor=tk.W, pady=4)

        self.lbl_proba_nocompra = ttk.Label(
            self.res_frame, text="P(No Compra): -- %", font=("Helvetica", 11)
        )
        self.lbl_proba_nocompra.pack(anchor=tk.W, pady=2)

        ttk.Separator(self.res_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Label(self.res_frame, text="Regla de Decisión (Umbral 0.50):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.lbl_veredicto = ttk.Label(
            self.res_frame, text="SIN CONSULTAR", font=("Helvetica", 16, "bold"), foreground="#777"
        )
        self.lbl_veredicto.pack(anchor=tk.W, pady=8)

        self.txt_explicacion = tk.Text(self.res_frame, height=8, wrap=tk.WORD, font=("Helvetica", 9), relief=tk.FLAT)
        self.txt_explicacion.pack(fill=tk.BOTH, expand=True, pady=6)
        self.txt_explicacion.insert(tk.END, "Presiona 'Estimar Intención' para ejecutar la inferencia del modelo.")
        self.txt_explicacion.config(state=tk.DISABLED)

    def ejecutar_prediccion(self):
        if self.modelo is None:
            messagebox.showerror("Error", "El modelo Random Forest no está disponible.")
            return

        try:
            v_type = self.var_visitor.get()
            paginas = float(self.var_paginas.get())
            duracion = float(self.var_duracion.get())

            if paginas < 0 or duracion < 0:
                messagebox.showerror("Error de entrada", "Las páginas y la duración no pueden ser negativas.")
                return

            # Construir vector numérico
            features = self.metadatos["caracteristicas"]
            fila = {}
            fila["ProductRelated"] = paginas
            fila["ProductRelated_Duration"] = duracion
            for vt in CATEGORIES_VISITOR:
                col = f"VisitorType_{vt}"
                fila[col] = 1 if vt == v_type else 0

            df_input = pd.DataFrame([fila])[features]

            # Inferencia de probabilidades
            probas = self.modelo.predict_proba(df_input)[0]
            p_nocompra = probas[0]
            p_compra = probas[1]

            # Decisión con umbral 0.50
            decision = p_compra >= 0.50

            self.lbl_proba_compra.config(text=f"P(Compra): {p_compra * 100:.2f} %")
            self.lbl_proba_nocompra.config(text=f"P(No Compra): {p_nocompra * 100:.2f} %")

            if decision:
                self.lbl_veredicto.config(text="✓ CLASIFICACIÓN: COMPRA", foreground="#1e8449")
            else:
                self.lbl_veredicto.config(text="✗ CLASIFICACIÓN: NO COMPRA", foreground="#c0392b")

            self.txt_explicacion.config(state=tk.NORMAL)
            self.txt_explicacion.delete("1.0", tk.END)
            exp_text = (
                f"El ensamble de 150 árboles ponderó las variables continuas originales "
                f"({paginas:.0f} páginas y {duracion:.1f} segundos de sesión) para el visitante de tipo '{v_type}'.\n\n"
                f"Dado que la probabilidad estimada de compra ({p_compra:.4f}) es "
                f"{'>=' if decision else '<'} al umbral reglamentario de 0.50, "
                f"la decisión final es '{'Compra' if decision else 'No Compra'}'."
            )
            self.txt_explicacion.insert(tk.END, exp_text)
            self.txt_explicacion.config(state=tk.DISABLED)

        except ValueError as ex:
            messagebox.showerror("Entrada inválida", f"Error en los datos numéricos:\n{ex}")


def main():
    app = VentanaRandomForest()
    app.mainloop()


if __name__ == "__main__":
    main()
