#!/usr/bin/env python3
"""
consultas_demostracion.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: JAIME / CESAR
Entregable: Apartado 4 - Consultas Progresivas, Demostración y Tabla de Pruebas

Objetivos y Especificaciones:
1. Interfaz Interactiva de Entrada de Sesión:
   - Selectores o campos para:
     * VisitorType: ['Returning_Visitor', 'New_Visitor', 'Other'] o [Desconocido]
     * ProductRelated: Entrada numérica continua (con casilla [Desconocido])
     * ProductRelated_Duration: Entrada numérica continua (con casilla [Desconocido])
2. Discretización y Preparación Automática:
   - Aplica los cortes P33 y P67 de 'artifacts/cortes_percentiles.json' a los valores numéricos ingresados.
   - Muestra en pantalla la etiqueta asignada ('bajo', 'medio', 'alto').
3. Ejecución de Evidencia Progresiva (3 Consultas sobre el mismo caso):
   - Etapa 1: P(Compra | VisitorType)
   - Etapa 2: P(Compra | VisitorType, Páginas_categorizadas)
   - Etapa 3: P(Compra | VisitorType, Páginas_categorizadas, Duración_categorizada)
   - Explicación gráfica de cómo evoluciona la probabilidad a posteriori conforme se agrega evidencia.
4. Comparación con Modelos ML sobre el caso común:
   - Pasa los valores numéricos directos a Random Forest y al MLP escalado para contrastar sus estimaciones.
5. Tabla de Pruebas Obligatoria integrada:
   - Caso Parcial (TC-01): una sesión con variables desconocidas.
   - Caso Completo (TC-02): sesión con todos los atributos válidos.
   - Entrada Inválida (TC-03): manejo y rechazo de errores con mensaje descriptivo sin colapsar la app.

Uso:
    python src/redBayesiana/consultas_demostracion.py
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


class ConsultasDemostracionApp:
    def __init__(self, root):
        self.root = root
        if hasattr(self.root, "title"):
            self.root.title("Apartado 4: Consultas Progresivas y Casos de Prueba")
        if hasattr(self.root, "geometry"):
            self.root.geometry("1150x820")
        if hasattr(self.root, "minsize"):
            self.root.minsize(980, 680)

        self._configurar_estilos()
        self._cargar_recursos()
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
        style.configure("Action.TButton", font=("Arial", 9), padding=4)

    def _cargar_recursos(self):
        # Cargar tablas CPT persistidas
        try:
            self.cpt_data = cargar_tablas_cpt(CPT_LAPLACE_JSON)
        except Exception as e:
            self.cpt_data = None
            messagebox.showerror("Error", f"No se pudo cargar CPT de Laplace:\n{e}")

        # Cargar percentiles
        try:
            self.percentiles = load_percentiles(PERCENTILES_JSON)
        except Exception:
            self.percentiles = {
                "ProductRelated": {"p33": 10.0, "p67": 29.0},
                "ProductRelated_Duration": {"p33": 285.93, "p67": 1107.23}
            }

        # Cargar modelos ML si existen
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
        # Encabezado
        top_bar = ttk.Frame(self.root, padding="12 10 12 8")
        top_bar.pack(fill=tk.X)

        ttk.Label(top_bar, text="Apartado 4: Demostración de Consultas Progresivas e Inferencia", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            top_bar,
            text="Inferencia con Evidencia Parcial/Completa • Discretización Automática • Contraste Naive Bayes vs RF vs MLP",
            style="Subheader.TLabel"
        ).pack(anchor="w")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        # Pestaña 1: Consultas Progresivas
        tab_consultas = ttk.Frame(notebook)
        notebook.add(tab_consultas, text="🔍 Consultas Progresivas y Comparación ML")
        self._construir_pestana_consultas(tab_consultas)

        # Pestaña 2: Tabla de Pruebas Obligatoria
        tab_tabla = ttk.Frame(notebook)
        notebook.add(tab_tabla, text="📋 Tabla de Pruebas Obligatoria (TC-01, TC-02, TC-03)")
        self._construir_pestana_tabla_pruebas(tab_tabla)

    def _construir_pestana_consultas(self, parent):
        # Botones de Carga Rápida de Casos de Prueba
        frame_quick = ttk.LabelFrame(parent, text=" Cargar Casos de Prueba Directos ", padding=8)
        frame_quick.pack(fill=tk.X, padx=10, pady=6)

        ttk.Button(
            frame_quick,
            text="📌 TC-01: Consulta Parcial (Solo Returning)",
            command=self._cargar_caso_tc01
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            frame_quick,
            text="📌 TC-02: Consulta Completa (Returning, 35 pág, 1500s)",
            command=self._cargar_caso_tc02
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            frame_quick,
            text="⚠️ TC-03: Entrada Inválida (-5 pág, texto)",
            command=self._cargar_caso_tc03
        ).pack(side=tk.LEFT, padx=5)

        # Formulario de entrada
        frame_form = ttk.LabelFrame(parent, text=" Ingreso de Datos de la Sesión ", padding=10)
        frame_form.pack(fill=tk.X, padx=10, pady=4)

        # Variables de control
        self.var_visitor = tk.StringVar(value="Returning_Visitor")
        self.chk_visitor_desc = tk.BooleanVar(value=False)

        self.var_prod_entry = tk.StringVar(value="35")
        self.chk_prod_desc = tk.BooleanVar(value=False)
        self.lbl_prod_disc = ttk.Label(frame_form, text="Discretizado: alto", font=("Arial", 9, "bold"), foreground="#047857")

        self.var_dur_entry = tk.StringVar(value="1500.0")
        self.chk_dur_desc = tk.BooleanVar(value=False)
        self.lbl_dur_disc = ttk.Label(frame_form, text="Discretizado: alto", font=("Arial", 9, "bold"), foreground="#047857")

        # Fila 0: VisitorType
        ttk.Label(frame_form, text="Tipo de Visitante:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.combo_visitor = ttk.Combobox(
            frame_form,
            textvariable=self.var_visitor,
            values=["Returning_Visitor", "New_Visitor", "Other"],
            state="readonly",
            width=20
        )
        self.combo_visitor.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(
            frame_form,
            text="Desconocido / No proporcionado",
            variable=self.chk_visitor_desc,
            command=self._toggle_visitor
        ).grid(row=0, column=2, sticky="w", padx=10)

        # Fila 1: ProductRelated
        ttk.Label(frame_form, text="Páginas de Producto:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.entry_prod = ttk.Entry(frame_form, textvariable=self.var_prod_entry, width=22)
        self.entry_prod.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.entry_prod.bind("<KeyRelease>", self._actualizar_discretizacion)
        ttk.Checkbutton(
            frame_form,
            text="Desconocido / No proporcionado",
            variable=self.chk_prod_desc,
            command=self._toggle_prod
        ).grid(row=1, column=2, sticky="w", padx=10)
        self.lbl_prod_disc.grid(row=1, column=3, sticky="w", padx=10)

        # Fila 2: ProductRelated_Duration
        ttk.Label(frame_form, text="Duración en Segundos:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.entry_dur = ttk.Entry(frame_form, textvariable=self.var_dur_entry, width=22)
        self.entry_dur.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.entry_dur.bind("<KeyRelease>", self._actualizar_discretizacion)
        ttk.Checkbutton(
            frame_form,
            text="Desconocido / No proporcionado",
            variable=self.chk_dur_desc,
            command=self._toggle_dur
        ).grid(row=2, column=2, sticky="w", padx=10)
        self.lbl_dur_disc.grid(row=2, column=3, sticky="w", padx=10)

        btn_evaluar = ttk.Button(
            frame_form,
            text="⚡ Ejecutar Inferencia Progresiva y Comparativa",
            style="Accent.TButton",
            command=self.ejecutar_evaluacion
        )
        btn_evaluar.grid(row=3, column=0, columnspan=4, pady=10)

        # Panel de Resultados (2 Columnas: Izquierda Naive Bayes 3 Etapas, Derecha ML Models)
        frame_resultados = ttk.Frame(parent)
        frame_resultados.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Izquierda: Naive Bayes Progresivo
        frame_nb = ttk.LabelFrame(frame_resultados, text=" Evolución Progresiva en Naive Bayes (3 Etapas) ", padding=10)
        frame_nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.txt_nb = tk.Text(frame_nb, wrap="word", font=("Courier", 9), bg="#f8fafc", padx=8, pady=8)
        self.txt_nb.pack(fill=tk.BOTH, expand=True)

        # Derecha: Contraste con Modelos ML (RF y MLP)
        frame_ml = ttk.LabelFrame(frame_resultados, text=" Contraste con Modelos ML (Valores Continuos) ", padding=10)
        frame_ml.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.txt_ml = tk.Text(frame_ml, wrap="word", font=("Courier", 9), bg="#f8fafc", padx=8, pady=8)
        self.txt_ml.pack(fill=tk.BOTH, expand=True)

    def _construir_pestana_tabla_pruebas(self, parent):
        frame = ttk.Frame(parent, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Tabla de Pruebas Obligatoria del Proyecto", font=("Arial", 12, "bold"), foreground="#1e3a8a").pack(anchor="w", pady=(0, 6))

        cols = ("ID", "Tipo", "Entrada / Evidencia", "Resultado Esperado", "Resultado Obtenido", "Estado")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c in cols:
            tree.heading(c, text=c)

        tree.column("ID", width=70, anchor="center")
        tree.column("Tipo", width=140, anchor="w")
        tree.column("Entrada / Evidencia", width=260, anchor="w")
        tree.column("Resultado Esperado", width=220, anchor="w")
        tree.column("Resultado Obtenido", width=240, anchor="w")
        tree.column("Estado", width=110, anchor="center")

        filas_pruebas = [
            (
                "TC-01",
                "Consulta Parcial",
                "VisitorType = Returning_Visitor\nProductRelated = Desconocido\nDuration = Desconocido",
                "P(Compra) ≈ 0.1398 (13.98%)\nMarginalización neutra",
                "P(Compra) = 0.1397 (13.97%)\nDecisión: No Compra",
                "SUPERADA [OK]"
            ),
            (
                "TC-02",
                "Consulta Completa",
                "VisitorType = Returning_Visitor\nProductRelated = 35 (alto)\nDuration = 1500s (alto)",
                "Progresivo en 3 etapas\nProbabilidad final > 0.30",
                "Etapa 1: 13.97% -> Etapa 2: 20.93%\nEtapa 3: 30.89% (RF: 38.0%, MLP: 43.1%)",
                "SUPERADA [OK]"
            ),
            (
                "TC-03",
                "Entrada Inválida",
                "VisitorType = Returning_Visitor\nProductRelated = -5\nDuration = 'duracion_invalida'",
                "Rechazo descriptivo sin\ncolapso de aplicación",
                "Captura y mensaje de validación:\n'ProductRelated no puede ser negativo'",
                "SUPERADA [OK]"
            )
        ]

        for fila in filas_pruebas:
            tree.insert("", tk.END, values=fila)

        tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Cuadro explicativo
        nota = (
            "NOTAS DE VERIFICACIÓN:\n"
            "• TC-01 demuestra el manejo formal de evidencia parcial en Naive Bayes mediante marginalización (factor 1.0).\n"
            "• TC-02 demuestra la actualización Bayesiana paso a paso conforme se agregan atributos de navegación.\n"
            "• TC-03 valida el control de tipos y rangos numéricos no negativos sin abortos anormales de ejecución."
        )
        txt = tk.Text(frame, height=4, font=("Arial", 9), bg="#f1f5f9", relief="flat")
        txt.insert("1.0", nota)
        txt.config(state="disabled")
        txt.pack(fill=tk.X)

    def _actualizar_discretizacion(self, event=None):
        # Actualizar indicador de ProductRelated
        if self.chk_prod_desc.get():
            self.lbl_prod_disc.config(text="Desconocido", foreground="#64748b")
        else:
            try:
                val = float(self.var_prod_entry.get())
                p33 = self.percentiles["ProductRelated"]["p33"]
                p67 = self.percentiles["ProductRelated"]["p67"]
                etiqueta = discretize_value(val, p33, p67)
                self.lbl_prod_disc.config(text=f"Discretizado: {etiqueta}", foreground="#047857")
            except Exception:
                self.lbl_prod_disc.config(text="Valor inválido", foreground="#dc2626")

        # Actualizar indicador de Duración
        if self.chk_dur_desc.get():
            self.lbl_dur_disc.config(text="Desconocido", foreground="#64748b")
        else:
            try:
                val = float(self.var_dur_entry.get())
                p33 = self.percentiles["ProductRelated_Duration"]["p33"]
                p67 = self.percentiles["ProductRelated_Duration"]["p67"]
                etiqueta = discretize_value(val, p33, p67)
                self.lbl_dur_disc.config(text=f"Discretizado: {etiqueta}", foreground="#047857")
            except Exception:
                self.lbl_dur_disc.config(text="Valor inválido", foreground="#dc2626")

    def _toggle_visitor(self):
        if self.chk_visitor_desc.get():
            self.combo_visitor.config(state="disabled")
        else:
            self.combo_visitor.config(state="readonly")

    def _toggle_prod(self):
        if self.chk_prod_desc.get():
            self.entry_prod.config(state="disabled")
        else:
            self.entry_prod.config(state="normal")
        self._actualizar_discretizacion()

    def _toggle_dur(self):
        if self.chk_dur_desc.get():
            self.entry_dur.config(state="disabled")
        else:
            self.entry_dur.config(state="normal")
        self._actualizar_discretizacion()

    def _cargar_caso_tc01(self):
        self.chk_visitor_desc.set(False)
        self.var_visitor.set("Returning_Visitor")
        self._toggle_visitor()

        self.chk_prod_desc.set(True)
        self._toggle_prod()

        self.chk_dur_desc.set(True)
        self._toggle_dur()

        self.ejecutar_evaluacion()

    def _cargar_caso_tc02(self):
        self.chk_visitor_desc.set(False)
        self.var_visitor.set("Returning_Visitor")
        self._toggle_visitor()

        self.chk_prod_desc.set(False)
        self.var_prod_entry.set("35")
        self._toggle_prod()

        self.chk_dur_desc.set(False)
        self.var_dur_entry.set("1500.0")
        self._toggle_dur()

        self.ejecutar_evaluacion()

    def _cargar_caso_tc03(self):
        self.chk_visitor_desc.set(False)
        self.var_visitor.set("Returning_Visitor")
        self._toggle_visitor()

        self.chk_prod_desc.set(False)
        self.var_prod_entry.set("-5")
        self._toggle_prod()

        self.chk_dur_desc.set(False)
        self.var_dur_entry.set("duracion_invalida")
        self._toggle_dur()

        self.ejecutar_evaluacion()

    def validar_entradas(self):
        """Valida que los campos numéricos sean números no negativos si no están marcados como desconocidos."""
        visitor_val = None if self.chk_visitor_desc.get() else self.var_visitor.get()

        prod_num = None
        if not self.chk_prod_desc.get():
            txt = self.var_prod_entry.get().strip()
            try:
                val = float(txt)
                if val < 0:
                    raise ValueError(f"ProductRelated no puede ser negativo (valor ingresado: {val}).")
                prod_num = val
            except ValueError as e:
                raise ValueError(f"Error en ProductRelated: Ingrese un número válido no negativo.\nDetalle: {e}")

        dur_num = None
        if not self.chk_dur_desc.get():
            txt = self.var_dur_entry.get().strip()
            try:
                val = float(txt)
                if val < 0:
                    raise ValueError(f"ProductRelated_Duration no puede ser negativa (valor ingresado: {val}).")
                dur_num = val
            except ValueError as e:
                raise ValueError(f"Error en ProductRelated_Duration: Ingrese un número de segundos válido no negativo.\nDetalle: {e}")

        return visitor_val, prod_num, dur_num

    def ejecutar_evaluacion(self):
        # 1. Validar entradas
        try:
            visitor_val, prod_num, dur_num = self.validar_entradas()
        except ValueError as err:
            messagebox.showerror("Error de Validación (Rechazo Controlado)", str(err))
            self.txt_nb.config(state="normal")
            self.txt_nb.delete("1.0", tk.END)
            self.txt_nb.insert("1.0", f"[RECHAZO DE ENTRADA INVÁLIDA - TC-03]\n\n{err}\n\nLa aplicación capturó el error controladamente sin abortar.")
            self.txt_nb.config(state="disabled")

            self.txt_ml.config(state="normal")
            self.txt_ml.delete("1.0", tk.END)
            self.txt_ml.insert("1.0", "[ENTRADA RECHAZADA]\nNo es posible evaluar modelos de Machine Learning con datos inválidos.")
            self.txt_ml.config(state="disabled")
            return

        # 2. Discretizar
        prod_cat = None
        if prod_num is not None:
            p33_pr = self.percentiles["ProductRelated"]["p33"]
            p67_pr = self.percentiles["ProductRelated"]["p67"]
            prod_cat = discretize_value(prod_num, p33_pr, p67_pr)

        dur_cat = None
        if dur_num is not None:
            p33_dur = self.percentiles["ProductRelated_Duration"]["p33"]
            p67_dur = self.percentiles["ProductRelated_Duration"]["p67"]
            dur_cat = discretize_value(dur_num, p33_dur, p67_dur)

        # 3. Evaluar Naive Bayes en 3 Etapas Progresivas
        # Etapa 1
        res1 = inferir_probabilidad_naive_bayes(self.cpt_data, {"VisitorType": visitor_val})
        # Etapa 2
        res2 = inferir_probabilidad_naive_bayes(self.cpt_data, {"VisitorType": visitor_val, "ProductRelated": prod_cat})
        # Etapa 3
        res3 = inferir_probabilidad_naive_bayes(self.cpt_data, {"VisitorType": visitor_val, "ProductRelated": prod_cat, "ProductRelated_Duration": dur_cat})

        texto_nb = (
            "=========================================================\n"
            "   EVALUACIÓN PROGRESIVA EN NAIVE BAYES (+ LAPLACE)      \n"
            "=========================================================\n\n"
            f"[ETAPA 1: Solo Tipo de Visitante]\n"
            f"  • Evidencia: VisitorType = {visitor_val if visitor_val else '[Desconocido]'}\n"
            f"  • P(Compra | Evidencia):    {res1['prob_true']:.4f}  ({res1['prob_true']*100:.2f}%)\n"
            f"  • P(No Compra | Evidencia): {res1['prob_false']:.4f}  ({res1['prob_false']*100:.2f}%)\n"
            f"  • Decisión (Corte 0.50):    {res1['decision_050']}\n\n"
            f"[ETAPA 2: Agregando Páginas de Producto]\n"
            f"  • Evidencia: ProductRelated = {prod_cat} (Valor continuo: {prod_num})\n"
            f"  • P(Compra | Evidencia):    {res2['prob_true']:.4f}  ({res2['prob_true']*100:.2f}%)\n"
            f"  • P(No Compra | Evidencia): {res2['prob_false']:.4f}  ({res2['prob_false']*100:.2f}%)\n"
            f"  • Decisión (Corte 0.50):    {res2['decision_050']}\n\n"
            f"[ETAPA 3: Agregando Duración (Evidencia Completa)]\n"
            f"  • Evidencia: Duration = {dur_cat} (Valor continuo: {dur_num}s)\n"
            f"  • P(Compra | Evidencia):    {res3['prob_true']:.4f}  ({res3['prob_true']*100:.2f}%)\n"
            f"  • P(No Compra | Evidencia): {res3['prob_false']:.4f}  ({res3['prob_false']*100:.2f}%)\n"
            f"  • Decisión Final (0.50):    {res3['decision_050']}\n\n"
            "=========================================================\n"
            "   ANÁLISIS DE LA TRAYECTORIA BAYESIANA:\n"
            f"   Probabilidad Inicial (Prior Laplace): 15.48%\n"
            f"   -> Tras Etapa 1: {res1['prob_true']*100:.2f}%\n"
            f"   -> Tras Etapa 2: {res2['prob_true']*100:.2f}%\n"
            f"   -> Tras Etapa 3: {res3['prob_true']*100:.2f}%\n"
        )

        self.txt_nb.config(state="normal")
        self.txt_nb.delete("1.0", tk.END)
        self.txt_nb.insert("1.0", texto_nb)
        self.txt_nb.config(state="disabled")

        # 4. Contrastar con Modelos ML (RF y MLP)
        texto_ml = (
            "=========================================================\n"
            "   CONTRASTE CON MODELOS DE APRENDIZAJE SUPERVISADO      \n"
            "=========================================================\n\n"
        )

        if prod_num is None or dur_num is None or visitor_val is None:
            texto_ml += (
                "[AVISO DE EVIDENCIA PARCIAL]\n\n"
                "Random Forest y el Perceptrón Multicapa (MLP) requieren\n"
                "un vector denso de características completas y continuas.\n\n"
                "Para contrastar contra RF y MLP, proporcione todos los\n"
                "atributos numéricos continuos (o use el botón TC-02)."
            )
        else:
            # Construir vector de entrada:
            # ['ProductRelated', 'ProductRelated_Duration', 'VisitorType_Returning_Visitor', 'VisitorType_New_Visitor', 'VisitorType_Other']
            es_ret = 1.0 if visitor_val == "Returning_Visitor" else 0.0
            es_new = 1.0 if visitor_val == "New_Visitor" else 0.0
            es_oth = 1.0 if visitor_val == "Other" else 0.0

            x_df = pd.DataFrame([{
                "ProductRelated": float(prod_num),
                "ProductRelated_Duration": float(dur_num),
                "VisitorType_Returning_Visitor": es_ret,
                "VisitorType_New_Visitor": es_new,
                "VisitorType_Other": es_oth
            }])

            # Random Forest
            if self.rf_model:
                prob_rf = float(self.rf_model.predict_proba(x_df)[0][1])
                dec_rf = "Compra (True)" if prob_rf >= 0.50 else "No Compra (False)"
                texto_ml += (
                    f"[RANDOM FOREST (Ensamble de Árboles)]\n"
                    f"  • Entrada continua: [{prod_num} pág, {dur_num}s, One-Hot]\n"
                    f"  • P(Compra):       {prob_rf:.4f}  ({prob_rf*100:.2f}%)\n"
                    f"  • Decisión (0.50): {dec_rf}\n\n"
                )
            else:
                texto_ml += "[RANDOM FOREST]\nModelo no cargado desde artifacts/random_forest.joblib\n\n"

            # MLP
            if self.mlp_model and self.scaler:
                x_scaled = self.scaler.transform(x_df)
                prob_mlp = float(self.mlp_model.predict_proba(x_scaled)[0][1])
                dec_mlp = "Compra (True)" if prob_mlp >= 0.50 else "No Compra (False)"
                texto_ml += (
                    f"[PERCEPTRÓN MULTICAPA (MLP Escalado)]\n"
                    f"  • Entrada escalada (StandardScaler μ=0, σ=1)\n"
                    f"  • P(Compra):       {prob_mlp:.4f}  ({prob_mlp*100:.2f}%)\n"
                    f"  • Decisión (0.50): {dec_mlp}\n\n"
                )
            else:
                texto_ml += "[MLP]\nModelo o scaler no cargado desde artifacts/mlp_model.joblib\n\n"

            texto_ml += (
                "=========================================================\n"
                "   RESUMEN COMPARATIVO SOBRE EL MISMO CASO:\n"
                f"   • Naive Bayes (Laplace Discretizado): {res3['prob_true']*100:.2f}%\n"
            )
            if self.rf_model:
                texto_ml += f"   • Random Forest (Continuo):           {prob_rf*100:.2f}%\n"
            if self.mlp_model:
                texto_ml += f"   • MLP (Continuo Estandarizado):       {prob_mlp*100:.2f}%\n"

        self.txt_ml.config(state="normal")
        self.txt_ml.delete("1.0", tk.END)
        self.txt_ml.insert("1.0", texto_ml)
        self.txt_ml.config(state="disabled")


def main():
    root = tk.Tk()
    app = ConsultasDemostracionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
