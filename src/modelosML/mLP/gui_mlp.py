#!/usr/bin/env python3
"""
gui_mlp.py
Tipo: INTERFAZ GRÁFICA (GUI)
Módulo: Perceptrón Multicapa (MLP) - Arquitectura, Pérdida e Inferencia

Características:
1. Panel de Diagnóstico y Entrenamiento:
   - Gráfico interactivo embebido de la Curva de Pérdida (Loss Curve) a lo largo de las épocas.
   - Indicador visual del punto donde se activó el Early Stopping.
   - Diagrama y resumen de la arquitectura de la red (5 -> 32 -> 16 -> 2).
2. Panel de Inferencia y Escalado:
   - Selectores interactivos para la sesión web.
   - Demostración visual de la transformación del StandardScaler (valores originales vs Z-scores).
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

from src.config import MLP_MODEL_JOBLIB, SCALER_MLP_JOBLIB, ARTIFACTS_DIR, CATEGORIES_VISITOR


class VentanaMLP(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptrón Multicapa (MLP) - Intención de Compra")
        self.geometry("980x740")
        self.minsize(850, 620)

        self.modelo = None
        self.scaler = None
        self.metadatos = None
        self.cargar_artefactos()

        self.crear_widgets()

    def cargar_artefactos(self):
        meta_path = ARTIFACTS_DIR / "mlp_metadatos.json"
        if MLP_MODEL_JOBLIB.exists() and SCALER_MLP_JOBLIB.exists() and meta_path.exists():
            try:
                self.modelo = joblib.load(MLP_MODEL_JOBLIB)
                self.scaler = joblib.load(SCALER_MLP_JOBLIB)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadatos = json.load(f)
            except Exception as e:
                messagebox.showerror("Error de carga", f"No se pudo cargar el modelo MLP:\n{e}")
        else:
            messagebox.showwarning(
                "Artefactos no encontrados",
                "No se encontraron los artefactos del MLP.\n"
                "Ejecuta en consola: python src/modelosML/mLP/03_cli_entrenar_mlp.py"
            )

    def crear_widgets(self):
        # Encabezado
        header = ttk.Frame(self, padding=12)
        header.pack(fill=tk.X)
        lbl_title = ttk.Label(
            header,
            text="Red Neuronal Artificial: Perceptrón Multicapa (MLP)",
            font=("Helvetica", 16, "bold")
        )
        lbl_title.pack(anchor=tk.W)
        lbl_sub = ttk.Label(
            header,
            text="Entradas estandarizadas con StandardScaler | Parada Temprana (Early Stopping) | Umbral común 0.50",
            font=("Helvetica", 10),
            foreground="#555"
        )
        lbl_sub.pack(anchor=tk.W)

        # Cuaderno de pestañas
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        tab_diagnostico = ttk.Frame(notebook, padding=10)
        tab_simulador = ttk.Frame(notebook, padding=10)

        notebook.add(tab_diagnostico, text="  Arquitectura y Curva de Pérdida  ")
        notebook.add(tab_simulador, text="  Simulador con Escalado en Tiempo Real  ")

        self.construir_tab_diagnostico(tab_diagnostico)
        self.construir_tab_simulador(tab_simulador)

    def construir_tab_diagnostico(self, parent):
        frame_top = ttk.Frame(parent)
        frame_top.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Arquitectura y Metadatos
        left_panel = ttk.LabelFrame(frame_top, text="Arquitectura e Hiperparámetros", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8), pady=4)
        left_panel.config(width=420)

        txt_info = tk.Text(left_panel, width=48, height=24, wrap=tk.WORD, font=("Courier", 9))
        txt_info.pack(fill=tk.BOTH, expand=True)

        if self.metadatos:
            arq = self.metadatos.get("arquitectura", {})
            params = self.metadatos.get("hiperparametros", {})
            ent = self.metadatos.get("entrenamiento", {})

            contenido = (
                "--- TOPOLOGÍA DE LA RED NEURONAL ---\n"
                f"• Capa de Entrada   : {arq.get('nodos_entrada')} nodos (features)\n"
                f"• Capa Oculta 1     : {arq.get('capas_ocultas', [32])[0]} neuronas (ReLU)\n"
                f"• Capa Oculta 2     : {arq.get('capas_ocultas', [32, 16])[1]} neuronas (ReLU)\n"
                f"• Capa de Salida    : {arq.get('nodos_salida')} nodos (Softmax)\n\n"
                "--- OPTIMIZACIÓN Y REGULARIZACIÓN ---\n"
                f"• Optimizador       : {arq.get('optimizador', 'adam').upper()}\n"
                f"• Regularización L2 : alpha = {params.get('alpha')}\n"
                f"• Learning Rate Ini : {params.get('learning_rate_init')}\n"
                f"• Validación Interna: {params.get('validation_fraction') * 100:.0f}% de train\n"
                f"• Paciencia (No Chg): {params.get('n_iter_no_change')} épocas\n\n"
                "--- RESULTADOS DEL APRENDIZAJE ---\n"
                f"• Épocas entrenadas : {ent.get('epocas_convergidas')}\n"
                f"• Pérdida final     : {ent.get('perdida_final', 0):.4f}\n"
                f"• Tiempo de cómputo : {ent.get('tiempo_segundos', 0):.4f} s\n"
                f"• ROC AUC en Train  : {ent.get('roc_auc_train', 0):.4f}\n\n"
                "--- PARADA TEMPRANA (EARLY STOPPING) ---\n"
                "La red monitorea continuamente el score de\n"
                "validación interna. Al no detectar mejoras\n"
                "significativas tras 10 épocas consecutivas,\n"
                "interrumpe el ciclo para evitar el sobreajuste\n"
                "(overfitting) sobre los datos de entrenamiento."
            )
        else:
            contenido = "No hay metadatos cargados. Entrena el modelo primero."

        txt_info.insert(tk.END, contenido)
        txt_info.config(state=tk.DISABLED)

        # Panel derecho: Curva de Pérdida
        right_panel = ttk.LabelFrame(frame_top, text="Curva de Pérdida (Loss Curve)", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=4)

        if self.metadatos and "entrenamiento" in self.metadatos:
            curva = self.metadatos["entrenamiento"].get("curva_perdida", [])
            epocas = list(range(1, len(curva) + 1))

            fig = Figure(figsize=(5.5, 4.8), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(epocas, curva, marker="o", color="#b03a2e", linewidth=2, markersize=5, label="Pérdida (Cross-Entropy)")
            ax.set_xlabel("Época de Entrenamiento", fontsize=9)
            ax.set_ylabel("Valor de la Función de Pérdida", fontsize=9)
            ax.set_title("Convergencia del Perceptrón Multicapa", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=9)

            # Anotación de parada temprana
            if epocas:
                ax.annotate(
                    f"Early Stopping\n(Época {epocas[-1]})",
                    xy=(epocas[-1], curva[-1]),
                    xytext=(epocas[-1] - max(1, len(epocas) // 3), curva[-1] + 0.04),
                    arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.5, headwidth=6),
                    fontsize=8,
                    fontweight="bold",
                    color="#196f3d"
                )

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=right_panel)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            lbl_no_graph = ttk.Label(right_panel, text="Curva no disponible sin entrenamiento previo.")
            lbl_no_graph.pack(pady=40)

    def construir_tab_simulador(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Entradas
        input_frame = ttk.LabelFrame(container, text="Datos de Entrada de la Sesión", padding=15)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(input_frame, text="Tipo de Visitante (VisitorType):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.var_visitor = tk.StringVar(value="Returning_Visitor")
        cb_visitor = ttk.Combobox(
            input_frame, textvariable=self.var_visitor, values=CATEGORIES_VISITOR, state="readonly", width=30
        )
        cb_visitor.pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(input_frame, text="Cantidad de Páginas (ProductRelated):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.var_paginas = tk.DoubleVar(value=25.0)
        spin_paginas = ttk.Spinbox(input_frame, from_=0, to=1000, increment=1, textvariable=self.var_paginas, width=15)
        spin_paginas.pack(anchor=tk.W, pady=(0, 4))
        scale_paginas = ttk.Scale(input_frame, from_=0, to=150, variable=self.var_paginas, orient=tk.HORIZONTAL)
        scale_paginas.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(input_frame, text="Duración en Páginas (Segundos):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.var_duracion = tk.DoubleVar(value=800.0)
        spin_dur = ttk.Spinbox(input_frame, from_=0, to=50000, increment=10, textvariable=self.var_duracion, width=15)
        spin_dur.pack(anchor=tk.W, pady=(0, 4))
        scale_dur = ttk.Scale(input_frame, from_=0, to=3000, variable=self.var_duracion, orient=tk.HORIZONTAL)
        scale_dur.pack(fill=tk.X, pady=(0, 15))

        btn_predecir = ttk.Button(
            input_frame, text="Transformar con Scaler y Estimar con MLP", command=self.ejecutar_prediccion
        )
        btn_predecir.pack(fill=tk.X, pady=10)

        # Panel derecho: Escalado y Resultados
        self.res_frame = ttk.LabelFrame(container, text="Transformación de Datos y Resultado", padding=15)
        self.res_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(self.res_frame, text="Estandarización Z-Score (StandardScaler ajustado en Train):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.txt_scaler_info = tk.Text(self.res_frame, height=5, wrap=tk.WORD, font=("Courier", 8), bg="#f8f9f9")
        self.txt_scaler_info.pack(fill=tk.X, pady=4)
        self.txt_scaler_info.insert(tk.END, "Sin transformar. Presiona el botón para calcular los Z-scores.")
        self.txt_scaler_info.config(state=tk.DISABLED)

        ttk.Separator(self.res_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.lbl_proba_compra = ttk.Label(
            self.res_frame, text="P(Compra): -- %", font=("Helvetica", 13, "bold"), foreground="#78281f"
        )
        self.lbl_proba_compra.pack(anchor=tk.W, pady=4)

        self.lbl_proba_nocompra = ttk.Label(
            self.res_frame, text="P(No Compra): -- %", font=("Helvetica", 11)
        )
        self.lbl_proba_nocompra.pack(anchor=tk.W, pady=2)

        ttk.Separator(self.res_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(self.res_frame, text="Regla de Decisión (Umbral 0.50):", font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        self.lbl_veredicto = ttk.Label(
            self.res_frame, text="SIN CONSULTAR", font=("Helvetica", 16, "bold"), foreground="#777"
        )
        self.lbl_veredicto.pack(anchor=tk.W, pady=6)

        self.txt_explicacion = tk.Text(self.res_frame, height=5, wrap=tk.WORD, font=("Helvetica", 9), relief=tk.FLAT)
        self.txt_explicacion.pack(fill=tk.BOTH, expand=True, pady=4)
        self.txt_explicacion.insert(tk.END, "Presiona el botón para propagar los valores hacia adelante en el MLP.")
        self.txt_explicacion.config(state=tk.DISABLED)

    def ejecutar_prediccion(self):
        if self.modelo is None or self.scaler is None:
            messagebox.showerror("Error", "El modelo o el escalador del MLP no están disponibles.")
            return

        try:
            v_type = self.var_visitor.get()
            paginas = float(self.var_paginas.get())
            duracion = float(self.var_duracion.get())

            if paginas < 0 or duracion < 0:
                messagebox.showerror("Error de entrada", "Las páginas y la duración no pueden ser negativas.")
                return

            features = self.metadatos["caracteristicas"]
            fila = {}
            fila["ProductRelated"] = paginas
            fila["ProductRelated_Duration"] = duracion
            for vt in CATEGORIES_VISITOR:
                col = f"VisitorType_{vt}"
                fila[col] = 1 if vt == v_type else 0

            df_input = pd.DataFrame([fila])[features]

            # Escalado estricto con StandardScaler ajustado solo en train
            X_scaled = self.scaler.transform(df_input)

            # Mostrar Z-scores
            self.txt_scaler_info.config(state=tk.NORMAL)
            self.txt_scaler_info.delete("1.0", tk.END)
            scaler_txt = (
                f"• ProductRelated          : {paginas:.1f} -> Z = {X_scaled[0][0]:.4f}\n"
                f"• ProductRelated_Duration : {duracion:.1f} s -> Z = {X_scaled[0][1]:.4f}\n"
                f"• VisitorType Indicadores : [{X_scaled[0][2]:.2f}, {X_scaled[0][3]:.2f}, {X_scaled[0][4]:.2f}]"
            )
            self.txt_scaler_info.insert(tk.END, scaler_txt)
            self.txt_scaler_info.config(state=tk.DISABLED)

            # Propagación hacia adelante en MLP
            probas = self.modelo.predict_proba(X_scaled)[0]
            p_nocompra = probas[0]
            p_compra = probas[1]

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
                f"La red neuronal activó sus dos capas ocultas (32 y 16 neuronas ReLU) "
                f"sobre las entradas estandarizadas.\n"
                f"La probabilidad estimada de compra P(Compra) = {p_compra:.4f} es "
                f"{'>=' if decision else '<'} a 0.50, concluyendo: '{'Compra' if decision else 'No Compra'}'."
            )
            self.txt_explicacion.insert(tk.END, exp_text)
            self.txt_explicacion.config(state=tk.DISABLED)

        except ValueError as ex:
            messagebox.showerror("Entrada inválida", f"Error en los datos numéricos:\n{ex}")


def main():
    app = VentanaMLP()
    app.mainloop()


if __name__ == "__main__":
    main()
