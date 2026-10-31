#!/usr/bin/env python3
"""
gui_comparativa_predicciones.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: CESAR / JAIME
Entregable: Comparador de Inferencia entre los 3 Modelos con Manejo de Evidencia Incompleta

Objetivos y Especificaciones:
1. Comparación simultánea de predicciones sobre los mismos datos de entrada:
   - Red Naive Bayes (Modelo Gráfico Probabilístico con Laplace).
   - Random Forest (Ensamble Supervisado Continuo).
   - Perceptrón Multicapa MLP (Red Neuronal Artificial Escalada).
2. Regla Fundamental de Evidencia Parcial:
   - Red Bayesiana: Permite omitir campos mediante marginalización formal (factor neutro 1.0).
   - Random Forest y MLP: Arrojan un ERROR controlado y explícito si falta algún campo,
     explicando por qué sus arquitecturas supervisadas no admiten vectores incompletos.
3. Botones para cargar casos de prueba directos y análisis de contrastes.

Uso:
    python src/comparacion/gui_comparativa_predicciones.py
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    CPT_LAPLACE_JSON,
    PERCENTILES_JSON,
    RF_MODEL_JOBLIB,
    MLP_MODEL_JOBLIB,
    SCALER_MLP_JOBLIB
)
from src.preprocessing import load_percentiles, discretize_value
from src.redBayesiana.cli_pipeline_red_naive_bayes import (
    cargar_tablas_cpt,
    inferir_probabilidad_naive_bayes
)


class ComparativaPrediccionesApp:
    def __init__(self, root):
        self.root = root
        if hasattr(self.root, "title"):
            self.root.title("Comparativa de Predicciones: Naive Bayes vs Random Forest vs MLP")
        if hasattr(self.root, "geometry"):
            self.root.geometry("1260x860")
        if hasattr(self.root, "minsize"):
            self.root.minsize(1080, 720)

        self._configurar_estilos()
        self._cargar_recursos()
        self._crear_interfaz()
        self.ejecutar_comparacion()

    def _configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Arial", 14, "bold"), foreground="#1e293b")
        style.configure("Subheader.TLabel", font=("Arial", 10), foreground="#64748b")
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), padding=6)
        style.configure("Quick.TButton", font=("Arial", 9), padding=4)

    def _cargar_recursos(self):
        # 1. CPT Naive Bayes
        try:
            self.cpt_data = cargar_tablas_cpt(CPT_LAPLACE_JSON)
        except Exception as e:
            self.cpt_data = None
            messagebox.showerror("Error", f"No se pudo cargar la persistencia CPT de Naive Bayes:\n{e}")

        # 2. Percentiles para discretizar
        try:
            self.percentiles = load_percentiles(PERCENTILES_JSON)
        except Exception:
            self.percentiles = {
                "ProductRelated": {"p33": 10.0, "p67": 29.0},
                "ProductRelated_Duration": {"p33": 285.93, "p67": 1107.23}
            }

        # 3. Modelos ML
        self.rf_model = None
        self.mlp_model = None
        self.scaler = None

        if RF_MODEL_JOBLIB.exists():
            try:
                self.rf_model = joblib.load(RF_MODEL_JOBLIB)
            except Exception:
                pass

        if MLP_MODEL_JOBLIB.exists() and SCALER_MLP_JOBLIB.exists():
            try:
                self.mlp_model = joblib.load(MLP_MODEL_JOBLIB)
                self.scaler = joblib.load(SCALER_MLP_JOBLIB)
            except Exception:
                pass

    def _crear_interfaz(self):
        # Barra superior
        top_bar = ttk.Frame(self.root, padding="12 10 12 10")
        top_bar.pack(fill=tk.X)

        ttk.Label(
            top_bar,
            text="Comparativa de Inferencia Multimodelo (Mismos Datos de Entrada)",
            style="Header.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            top_bar,
            text="Regla: Naive Bayes permite omitir campos (Marginalización) • Random Forest y MLP arrojan Error ante datos faltantes",
            style="Subheader.TLabel"
        ).pack(anchor="w")

        # Barra de carga de casos predefinidos
        frame_quick = ttk.LabelFrame(self.root, text=" Casos Predefinidos de Prueba Rápida ", padding=8)
        frame_quick.pack(fill=tk.X, padx=12, pady=(0, 6))

        ttk.Button(
            frame_quick,
            text="✅ Caso Completo (Returning, 35 pág, 1500s) -> Los 3 modelos predicen",
            style="Quick.TButton",
            command=self._cargar_caso_completo
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            frame_quick,
            text="⚠️ Caso Parcial 1 (Solo VisitorType) -> Naive Bayes predice | RF y MLP ERROR",
            style="Quick.TButton",
            command=self._cargar_caso_solo_visitor
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            frame_quick,
            text="⚠️ Caso Parcial 2 (Sin Duración) -> Naive Bayes predice | RF y MLP ERROR",
            style="Quick.TButton",
            command=self._cargar_caso_sin_duracion
        ).pack(side=tk.LEFT, padx=6)

        # Panel de entrada de datos unificado
        frame_in = ttk.LabelFrame(self.root, text=" Formulario de Entrada Común para los 3 Modelos ", padding=10)
        frame_in.pack(fill=tk.X, padx=12, pady=4)

        self.var_visitor = tk.StringVar(value="Returning_Visitor")
        self.omitir_visitor = tk.BooleanVar(value=False)

        self.var_prod = tk.StringVar(value="35")
        self.omitir_prod = tk.BooleanVar(value=False)

        self.var_dur = tk.StringVar(value="1500.0")
        self.omitir_dur = tk.BooleanVar(value=False)

        # 1. VisitorType
        ttk.Label(frame_in, text="1. VisitorType:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="e", padx=6, pady=5)
        self.combo_vis = ttk.Combobox(
            frame_in,
            textvariable=self.var_visitor,
            values=["Returning_Visitor", "New_Visitor", "Other"],
            state="readonly",
            width=20
        )
        self.combo_vis.grid(row=0, column=1, sticky="w", padx=6, pady=5)
        ttk.Checkbutton(
            frame_in,
            text="Omitir campo (Desconocido)",
            variable=self.omitir_visitor,
            command=self._toggle_inputs
        ).grid(row=0, column=2, sticky="w", padx=10)

        # 2. ProductRelated
        ttk.Label(frame_in, text="2. ProductRelated (Páginas):", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="e", padx=6, pady=5)
        self.entry_prod = ttk.Entry(frame_in, textvariable=self.var_prod, width=22)
        self.entry_prod.grid(row=1, column=1, sticky="w", padx=6, pady=5)
        ttk.Checkbutton(
            frame_in,
            text="Omitir campo (Desconocido)",
            variable=self.omitir_prod,
            command=self._toggle_inputs
        ).grid(row=1, column=2, sticky="w", padx=10)
        self.lbl_disc_prod = ttk.Label(frame_in, text="Discretizado: alto", foreground="#047857", font=("Arial", 9, "bold"))
        self.lbl_disc_prod.grid(row=1, column=3, sticky="w", padx=10)

        # 3. ProductRelated_Duration
        ttk.Label(frame_in, text="3. Duration (Segundos):", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="e", padx=6, pady=5)
        self.entry_dur = ttk.Entry(frame_in, textvariable=self.var_dur, width=22)
        self.entry_dur.grid(row=2, column=1, sticky="w", padx=6, pady=5)
        ttk.Checkbutton(
            frame_in,
            text="Omitir campo (Desconocido)",
            variable=self.omitir_dur,
            command=self._toggle_inputs
        ).grid(row=2, column=2, sticky="w", padx=10)
        self.lbl_disc_dur = ttk.Label(frame_in, text="Discretizado: alto", foreground="#047857", font=("Arial", 9, "bold"))
        self.lbl_disc_dur.grid(row=2, column=3, sticky="w", padx=10)

        self.entry_prod.bind("<KeyRelease>", lambda e: self._actualizar_etiquetas_discretizacion())
        self.entry_dur.bind("<KeyRelease>", lambda e: self._actualizar_etiquetas_discretizacion())

        btn_eval = ttk.Button(
            frame_in,
            text="⚡ Comparar Predicciones Simultáneas",
            style="Accent.TButton",
            command=self.ejecutar_comparacion
        )
        btn_eval.grid(row=3, column=0, columnspan=4, pady=10)

        # Panel de las 3 Columnas de Resultados
        frame_cards = ttk.Frame(self.root)
        frame_cards.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Columna 1: Naive Bayes
        self.card_nb = ttk.LabelFrame(frame_cards, text=" 1. Red Naive Bayes (+ Laplace) ", padding=10)
        self.card_nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.txt_nb = tk.Text(self.card_nb, wrap="word", font=("Courier", 9), bg="#f8fafc", padx=8, pady=8)
        self.txt_nb.pack(fill=tk.BOTH, expand=True)

        # Columna 2: Random Forest
        self.card_rf = ttk.LabelFrame(frame_cards, text=" 2. Random Forest (Continuo) ", padding=10)
        self.card_rf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.txt_rf = tk.Text(self.card_rf, wrap="word", font=("Courier", 9), bg="#f8fafc", padx=8, pady=8)
        self.txt_rf.pack(fill=tk.BOTH, expand=True)

        # Columna 3: MLP
        self.card_mlp = ttk.LabelFrame(frame_cards, text=" 3. Red Neuronal MLP (Escalada) ", padding=10)
        self.card_mlp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.txt_mlp = tk.Text(self.card_mlp, wrap="word", font=("Courier", 9), bg="#f8fafc", padx=8, pady=8)
        self.txt_mlp.pack(fill=tk.BOTH, expand=True)

        # Panel Inferior de Conclusiones y Resumen Comparativo
        frame_concl = ttk.LabelFrame(self.root, text=" Resumen y Justificación Teórica de las Diferencias ", padding=8)
        frame_concl.pack(fill=tk.X, padx=12, pady=(4, 10))

        self.txt_concl = tk.Text(frame_concl, height=4, wrap="word", font=("Arial", 9), bg="#f1f5f9", relief="flat")
        self.txt_concl.pack(fill=tk.X)

    def _toggle_inputs(self):
        if self.omitir_visitor.get():
            self.combo_vis.config(state="disabled")
        else:
            self.combo_vis.config(state="readonly")

        if self.omitir_prod.get():
            self.entry_prod.config(state="disabled")
        else:
            self.entry_prod.config(state="normal")

        if self.omitir_dur.get():
            self.entry_dur.config(state="disabled")
        else:
            self.entry_dur.config(state="normal")

        self._actualizar_etiquetas_discretizacion()

    def _actualizar_etiquetas_discretizacion(self):
        if self.omitir_prod.get():
            self.lbl_disc_prod.config(text="[Omitido]", foreground="#64748b")
        else:
            try:
                v = float(self.var_prod.get())
                disc = discretize_value(v, self.percentiles["ProductRelated"]["p33"], self.percentiles["ProductRelated"]["p67"])
                self.lbl_disc_prod.config(text=f"Discretizado: {disc}", foreground="#047857")
            except Exception:
                self.lbl_disc_prod.config(text="Valor inválido", foreground="#dc2626")

        if self.omitir_dur.get():
            self.lbl_disc_dur.config(text="[Omitido]", foreground="#64748b")
        else:
            try:
                v = float(self.var_dur.get())
                disc = discretize_value(v, self.percentiles["ProductRelated_Duration"]["p33"], self.percentiles["ProductRelated_Duration"]["p67"])
                self.lbl_disc_dur.config(text=f"Discretizado: {disc}", foreground="#047857")
            except Exception:
                self.lbl_disc_dur.config(text="Valor inválido", foreground="#dc2626")

    def _cargar_caso_completo(self):
        self.omitir_visitor.set(False)
        self.var_visitor.set("Returning_Visitor")
        self.omitir_prod.set(False)
        self.var_prod.set("35")
        self.omitir_dur.set(False)
        self.var_dur.set("1500.0")
        self._toggle_inputs()
        self.ejecutar_comparacion()

    def _cargar_caso_solo_visitor(self):
        self.omitir_visitor.set(False)
        self.var_visitor.set("Returning_Visitor")
        self.omitir_prod.set(True)
        self.omitir_dur.set(True)
        self._toggle_inputs()
        self.ejecutar_comparacion()

    def _cargar_caso_sin_duracion(self):
        self.omitir_visitor.set(False)
        self.var_visitor.set("Returning_Visitor")
        self.omitir_prod.set(False)
        self.var_prod.set("35")
        self.omitir_dur.set(True)
        self._toggle_inputs()
        self.ejecutar_comparacion()

    def ejecutar_comparacion(self):
        # 1. Extraer estados de entrada
        vis_val = None if self.omitir_visitor.get() else self.var_visitor.get()

        prod_num = None
        error_prod = False
        if not self.omitir_prod.get():
            try:
                prod_num = float(self.var_prod.get())
                if prod_num < 0:
                    error_prod = True
            except ValueError:
                error_prod = True

        dur_num = None
        error_dur = False
        if not self.omitir_dur.get():
            try:
                dur_num = float(self.var_dur.get())
                if dur_num < 0:
                    error_dur = True
            except ValueError:
                error_dur = True

        campos_omitidos = []
        if self.omitir_visitor.get():
            campos_omitidos.append("VisitorType")
        if self.omitir_prod.get():
            campos_omitidos.append("ProductRelated")
        if self.omitir_dur.get():
            campos_omitidos.append("ProductRelated_Duration")

        hay_omision = len(campos_omitidos) > 0

        # =====================================================================
        # 1. EVALUAR RED NAIVE BAYES (PERMITE OMISIÓN MEDIANTE MARGINALIZACIÓN)
        # =====================================================================
        prod_cat = None
        if prod_num is not None and not error_prod:
            prod_cat = discretize_value(prod_num, self.percentiles["ProductRelated"]["p33"], self.percentiles["ProductRelated"]["p67"])

        dur_cat = None
        if dur_num is not None and not error_dur:
            dur_cat = discretize_value(dur_num, self.percentiles["ProductRelated_Duration"]["p33"], self.percentiles["ProductRelated_Duration"]["p67"])

        evidencia_nb = {
            "VisitorType": vis_val,
            "ProductRelated": prod_cat,
            "ProductRelated_Duration": dur_cat
        }

        res_nb = inferir_probabilidad_naive_bayes(self.cpt_data, evidencia_nb)
        p_nb_true = res_nb["prob_true"]
        p_nb_false = res_nb["prob_false"]
        dec_nb = res_nb["decision_050"]

        texto_nb = (
            "ESTADO: INFERENCIA EXITOSA [OK]\n"
            + ("• Capacidad: Marginalización formal de atributos ausentes activa.\n" if hay_omision else "• Capacidad: Inferencia con evidencia completa.\n")
            + "\n"
            "EVIDENCIA UTILIZADA:\n"
            f"• VisitorType: {vis_val if vis_val else '[OMITIDO - FACTOR 1.0]'}\n"
            f"• ProductRelated: {prod_cat if prod_cat else '[OMITIDO - FACTOR 1.0]'}\n"
            f"• Duration: {dur_cat if dur_cat else '[OMITIDO - FACTOR 1.0]'}\n\n"
            "RESULTADOS NAIVE BAYES:\n"
            f"• P(Compra | Evidencia):    {p_nb_true:.4f}  ({p_nb_true*100:.2f}%)\n"
            f"• P(No Compra | Evidencia): {p_nb_false:.4f}  ({p_nb_false*100:.2f}%)\n"
            f"• Decisión (Corte 0.50):    {dec_nb}\n\n"
            "DETALLE DE LA REGLA:\n"
            "Las variables no proporcionadas son integradas (marginalizadas)\n"
            "usando el elemento neutro multiplicativo (1.0). El modelo\n"
            "probabilístico estima la posterior sin requerir imputación."
        )

        self.txt_nb.config(state="normal")
        self.txt_nb.delete("1.0", tk.END)
        self.txt_nb.insert("1.0", texto_nb)
        self.txt_nb.config(state="disabled")

        # =====================================================================
        # 2. EVALUAR RANDOM FOREST (ARROJA ERROR SI HAY OMISIÓN)
        # =====================================================================
        if hay_omision or error_prod or error_dur or self.rf_model is None:
            razon_error = []
            if hay_omision:
                razon_error.append(f"Campos omitidos: {', '.join(campos_omitidos)}")
            if error_prod:
                razon_error.append("ProductRelated contiene un valor no numérico o negativo.")
            if error_dur:
                razon_error.append("ProductRelated_Duration contiene un valor no numérico o negativo.")
            if self.rf_model is None:
                razon_error.append("Modelo no localizado en artifacts/random_forest.joblib.")

            texto_rf = (
                "ESTADO: ❌ ERROR DE INFERENCIA\n\n"
                "DETALLE DEL ERROR:\n"
                "Random Forest RECHAZA la solicitud porque NO permite\n"
                "vectores de características incompletos.\n\n"
                f"• Causa: {' | '.join(razon_error)}\n\n"
                "JUSTIFICACIÓN ARQUITECTÓNICA:\n"
                "Un ensamble de árboles de decisión (Random Forest) evalúa\n"
                "umbrales deterministas en cada nodo (ej. x_j <= umbral).\n"
                "Si la variable x_j está ausente (NaN / omitida), el árbol\n"
                "no puede decidir qué rama seguir sin una imputación previa."
            )
        else:
            x_rf = pd.DataFrame([{
                "ProductRelated": float(prod_num),
                "ProductRelated_Duration": float(dur_num),
                "VisitorType_Returning_Visitor": 1.0 if vis_val == "Returning_Visitor" else 0.0,
                "VisitorType_New_Visitor": 1.0 if vis_val == "New_Visitor" else 0.0,
                "VisitorType_Other": 1.0 if vis_val == "Other" else 0.0
            }])
            p_rf_true = float(self.rf_model.predict_proba(x_rf)[0][1])
            dec_rf = "Compra (True)" if p_rf_true >= 0.50 else "No Compra (False)"

            texto_rf = (
                "ESTADO: INFERENCIA EXITOSA [OK]\n"
                "• Capacidad: Vector numérico completo evaluado en 100 árboles.\n\n"
                "EVIDENCIA UTILIZADA:\n"
                f"• Páginas continuas: {prod_num}\n"
                f"• Duración continua: {dur_num}s\n"
                f"• One-Hot: {vis_val}\n\n"
                "RESULTADOS RANDOM FOREST:\n"
                f"• P(Compra | X):         {p_rf_true:.4f}  ({p_rf_true*100:.2f}%)\n"
                f"• P(No Compra | X):      {1.0 - p_rf_true:.4f}  ({(1.0 - p_rf_true)*100:.2f}%)\n"
                f"• Decisión (Corte 0.50): {dec_rf}\n\n"
                "DETALLE DE LA REGLA:\n"
                "Promedio directo de los votos probabilísticos de todos\n"
                "los árboles con pesos compensados ante el desbalance."
            )

        self.txt_rf.config(state="normal")
        self.txt_rf.delete("1.0", tk.END)
        self.txt_rf.insert("1.0", texto_rf)
        self.txt_rf.config(state="disabled")

        # =====================================================================
        # 3. EVALUAR RED NEURONAL MLP (ARROJA ERROR SI HAY OMISIÓN)
        # =====================================================================
        if hay_omision or error_prod or error_dur or self.mlp_model is None or self.scaler is None:
            razon_mlp = []
            if hay_omision:
                razon_mlp.append(f"Campos omitidos: {', '.join(campos_omitidos)}")
            if error_prod or error_dur:
                razon_mlp.append("Valores no continuos o inválidos.")
            if self.mlp_model is None or self.scaler is None:
                razon_mlp.append("Pesos de MLP o StandardScaler no encontrados.")

            texto_mlp = (
                "ESTADO: ❌ ERROR DE INFERENCIA\n\n"
                "DETALLE DEL ERROR:\n"
                "El Perceptrón Multicapa (MLP) RECHAZA la solicitud porque\n"
                "exige dimensionalidad de entrada fija y completa.\n\n"
                f"• Causa: {' | '.join(razon_mlp)}\n\n"
                "JUSTIFICACIÓN ARQUITECTÓNICA:\n"
                "Una red neuronal artificial feedforward calcula:\n"
                "a = ReLU(W · x + b).\n"
                "Cada neurona de entrada requiere obligatoriamente un número real\n"
                "para multiplicar su peso sináptico. Si x contiene valores\n"
                "nulos, la operación tensorial arroja un error matemático."
            )
        else:
            x_mlp_raw = pd.DataFrame([{
                "ProductRelated": float(prod_num),
                "ProductRelated_Duration": float(dur_num),
                "VisitorType_Returning_Visitor": 1.0 if vis_val == "Returning_Visitor" else 0.0,
                "VisitorType_New_Visitor": 1.0 if vis_val == "New_Visitor" else 0.0,
                "VisitorType_Other": 1.0 if vis_val == "Other" else 0.0
            }])
            x_scaled = self.scaler.transform(x_mlp_raw)
            p_mlp_true = float(self.mlp_model.predict_proba(x_scaled)[0][1])
            dec_mlp = "Compra (True)" if p_mlp_true >= 0.50 else "No Compra (False)"

            texto_mlp = (
                "ESTADO: INFERENCIA EXITOSA [OK]\n"
                "• Capacidad: Vector estandarizado (μ=0, σ=1) propagado en capas.\n\n"
                "EVIDENCIA UTILIZADA:\n"
                f"• Páginas continuas: {prod_num}\n"
                f"• Duración continua: {dur_num}s\n"
                f"• One-Hot: {vis_val}\n\n"
                "RESULTADOS MLP NEURAL NET:\n"
                f"• P(Compra | X):         {p_mlp_true:.4f}  ({p_mlp_true*100:.2f}%)\n"
                f"• P(No Compra | X):      {1.0 - p_mlp_true:.4f}  ({(1.0 - p_mlp_true)*100:.2f}%)\n"
                f"• Decisión (Corte 0.50): {dec_mlp}\n\n"
                "DETALLE DE LA REGLA:\n"
                "Activación de salida Sigmoide/Softmax tras capas densas\n"
                "entrenadas con Backpropagation y Early Stopping."
            )

        self.txt_mlp.config(state="normal")
        self.txt_mlp.delete("1.0", tk.END)
        self.txt_mlp.insert("1.0", texto_mlp)
        self.txt_mlp.config(state="disabled")

        # Conclusiones dinámicas
        if hay_omision:
            concl = (
                f"🔍 DIAGNÓSTICO ANTE EVIDENCIA INCOMPLETA ({len(campos_omitidos)} atributo(s) omitido(s)):\n"
                f"• Red Naive Bayes: Pudo inferir de manera matemáticamente rigurosa mediante marginalización (P(Compra) = {p_nb_true*100:.2f}%).\n"
                "• Random Forest y MLP: Fallaron con rechazo explícito controlado, pues los modelos discriminativos estándar no toleran entradas ausentes sin imputación externa."
            )
        else:
            concl = (
                "🔍 DIAGNÓSTICO ANTE EVIDENCIA COMPLETA (Mismos Datos de Entrada):\n"
                f"• Naive Bayes: P(Compra) = {p_nb_true*100:.2f}% (Decisión: {dec_nb})\n"
                f"• Random Forest: P(Compra) = {p_rf_true*100:.2f}% (Decisión: {dec_rf})\n"
                f"• MLP Neural Net: P(Compra) = {p_mlp_true*100:.2f}% (Decisión: {dec_mlp})\n"
                "Random Forest eleva la probabilidad al aplicar balanceo de pesos ('balanced'), mitigando el desbalance de clases de la tienda."
            )

        self.txt_concl.config(state="normal")
        self.txt_concl.delete("1.0", tk.END)
        self.txt_concl.insert("1.0", concl)
        self.txt_concl.config(state="disabled")


def main():
    root = tk.Tk()
    app = ComparativaPrediccionesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
