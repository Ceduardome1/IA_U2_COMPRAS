#!/usr/bin/env python3
"""
clasificador_bayes.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: JAIME / CESAR
Entregable: Apartado 4 - Clasificador Interactivo Naive Bayes con Laplace

Uso:
    python src/redBayesiana/clasificador_bayes.py
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_CAT_PATH, CPT_LAPLACE_JSON
from src.redBayesiana.cli_pipeline_red_naive_bayes import (
    cargar_tablas_cpt,
    inferir_probabilidad_naive_bayes
)


class ClasificadorBayesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apartado 4: Clasificador Naive Bayes (Laplace)")
        self.root.geometry("560x520")
        self.root.minsize(480, 460)

        self._cargar_datos_y_persistencia()
        self._crear_interfaz()

    def _cargar_datos_y_persistencia(self):
        # Cargar persistencia CPT
        try:
            self.cpt_data = cargar_tablas_cpt(CPT_LAPLACE_JSON)
        except Exception as e:
            self.cpt_data = None
            messagebox.showerror("Error", f"No se pudo cargar la persistencia CPT:\n{e}")

        # Opciones canónicas
        self.opciones_visitor = ["Returning_Visitor", "New_Visitor", "Other"]
        self.opciones_product = ["bajo", "medio", "alto"]
        self.opciones_duration = ["bajo", "medio", "alto"]

    def _crear_interfaz(self):
        tk.Label(
            self.root,
            text="Predicción de Compra (Naive Bayes)",
            font=("Arial", 14, "bold"),
            fg="#1e3a8a"
        ).pack(pady=15)

        lbl_sub = tk.Label(
            self.root,
            text="Inferencia con Suavizado de Laplace (+1) basada en CPT persistida",
            font=("Arial", 9),
            fg="#64748b"
        )
        lbl_sub.pack(pady=(0, 10))

        frame_form = ttk.LabelFrame(self.root, text=" Evidencia de Navegación del Usuario ", padding=15)
        frame_form.pack(padx=20, pady=5, fill=tk.X)

        self.var_visitor = tk.StringVar(value=self.opciones_visitor[0])
        self.var_product = tk.StringVar(value=self.opciones_product[1])
        self.var_duration = tk.StringVar(value=self.opciones_duration[1])

        # Formularios desplegables
        ttk.Label(frame_form, text="Tipo de Visitante (VisitorType):", font=("Arial", 10)).grid(row=0, column=0, sticky="e", padx=10, pady=8)
        ttk.Combobox(frame_form, textvariable=self.var_visitor, values=self.opciones_visitor, state="readonly", width=22).grid(row=0, column=1, padx=10, pady=8)

        ttk.Label(frame_form, text="Páginas de Producto (ProductRelated):", font=("Arial", 10)).grid(row=1, column=0, sticky="e", padx=10, pady=8)
        ttk.Combobox(frame_form, textvariable=self.var_product, values=self.opciones_product, state="readonly", width=22).grid(row=1, column=1, padx=10, pady=8)

        ttk.Label(frame_form, text="Duración en Productos (Duration):", font=("Arial", 10)).grid(row=2, column=0, sticky="e", padx=10, pady=8)
        ttk.Combobox(frame_form, textvariable=self.var_duration, values=self.opciones_duration, state="readonly", width=22).grid(row=2, column=1, padx=10, pady=8)

        btn_predecir = ttk.Button(self.root, text="🔍 Clasificar Sesión de Usuario", command=self.predecir)
        btn_predecir.pack(pady=15)

        self.frame_res = ttk.LabelFrame(self.root, text=" Resultado de la Inferencia ", padding=12)
        self.frame_res.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

        self.lbl_resultado = tk.Label(self.frame_res, text="Seleccione los atributos y presione 'Clasificar'.", font=("Arial", 11), justify="left")
        self.lbl_resultado.pack(fill=tk.BOTH, expand=True)

    def predecir(self):
        if not self.cpt_data:
            messagebox.showerror("Error", "No hay datos CPT cargados.")
            return

        evidencia = {
            "VisitorType": self.var_visitor.get(),
            "ProductRelated": self.var_product.get(),
            "ProductRelated_Duration": self.var_duration.get()
        }

        res = inferir_probabilidad_naive_bayes(self.cpt_data, evidencia)
        prob_true = res["prob_true"]
        prob_false = res["prob_false"]
        decision = res["decision_050"]

        color = "#16a34a" if prob_true >= 0.50 else "#dc2626"

        texto = (
            f"• P(Compra | Evidencia):    {prob_true:.4f}  ({prob_true * 100:.2f}%)\n"
            f"• P(No Compra | Evidencia): {prob_false:.4f}  ({prob_false * 100:.2f}%)\n\n"
            f"• Decisión (Umbral 0.50):    {decision}\n"
            f"• Conclusión: El modelo clasifica que el usuario {decision.upper()}."
        )

        self.lbl_resultado.config(text=texto, fg=color, font=("Arial", 11, "bold"))


def main():
    root = tk.Tk()
    app = ClasificadorBayesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
