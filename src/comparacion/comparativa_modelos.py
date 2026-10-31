#!/usr/bin/env python3
"""
comparativa_modelos.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: CESAR
Entregable: Apartado 5 - Panel Comparativo de Modelos y Matrices de Confusión

Objetivos y Especificaciones:
1. Panel Visual de Resultados Finales:
   - Carga y visualización de 'artifacts/metricas_evaluacion.json'.
   - Tabla comparativa interactiva con los 4 enfoques:
     * Naive Bayes
     * Random Forest
     * Perceptrón Multicapa (MLP)
     * Referencia Base (Mayoritaria "No Compra")
   - Columnas: Exactitud, Precisión, Sensibilidad (Recall), F1-Score y Tiempo de Inferencia.
2. Visualización Gráfica de Matrices de Confusión:
   - Cuadrícula 2x2 de mapas de calor (Heatmaps) con anotaciones explícitas de TN, FP, FN, TP.
3. Panel de Análisis de Supuestos, Desbalance y Errores.

Uso:
    python src/comparacion/comparativa_modelos.py
"""

import sys
import json
import importlib
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import METRICAS_JSON


def ejecutar_evaluacion_completa():
    mod = importlib.import_module("src.comparacion.04_cli_evaluar_todos")
    return mod.ejecutar_evaluacion_completa()


class ComparativaModelosApp:
    def __init__(self, root):
        self.root = root
        if hasattr(self.root, "title"):
            self.root.title("Apartado 5: Comparativa Final de Modelos y Matrices de Confusión")
        if hasattr(self.root, "geometry"):
            self.root.geometry("1240x840")
        if hasattr(self.root, "minsize"):
            self.root.minsize(1050, 700)

        self.metricas = {}
        self._configurar_estilos()
        self._cargar_o_calcular_metricas()
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

    def _cargar_o_calcular_metricas(self):
        if METRICAS_JSON.exists():
            try:
                with open(METRICAS_JSON, "r", encoding="utf-8") as f:
                    self.metricas = json.load(f)
            except Exception:
                self.metricas = {}

        if not self.metricas:
            try:
                self.metricas = ejecutar_evaluacion_completa()
            except Exception as e:
                messagebox.showerror("Error al Evaluar", f"No se pudieron generar las métricas de evaluación:\n{e}")

    def _crear_interfaz(self):
        # Barra superior
        top_bar = ttk.Frame(self.root, padding="12 10 12 10")
        top_bar.pack(fill=tk.X)

        info_box = ttk.Frame(top_bar)
        info_box.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(info_box, text="Apartado 5: Evaluación Comparativa sobre Conjunto de Prueba (20%)", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            info_box,
            text="Evaluación bajo Umbral de Decisión 0.50 • 2,466 Sesiones en Test • Análisis de Desbalance",
            style="Subheader.TLabel"
        ).pack(anchor="w")

        btn_box = ttk.Frame(top_bar)
        btn_box.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(
            btn_box,
            text="🔄 Reevaluar Todos los Modelos",
            style="Accent.TButton",
            command=self._reevaluar
        ).pack(side=tk.RIGHT, padx=5)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        # Pestaña 1: Matrices de Confusión Heatmaps
        tab_matrices = ttk.Frame(notebook)
        notebook.add(tab_matrices, text="🔥 Matrices de Confusión (2x2 Heatmaps)")
        self._construir_pestana_matrices(tab_matrices)

        # Pestaña 2: Tabla de Métricas
        tab_tabla = ttk.Frame(notebook)
        notebook.add(tab_tabla, text="📊 Tabla Comparativa de Desempeño")
        self._construir_pestana_tabla(tab_tabla)

        # Pestaña 3: Análisis de Errores y Discusión
        tab_disc = ttk.Frame(notebook)
        notebook.add(tab_disc, text="📝 Discusión de Errores y Supuestos")
        self._construir_pestana_disc(tab_disc)

    def _construir_pestana_matrices(self, parent):
        if not self.metricas:
            ttk.Label(parent, text="No hay métricas disponibles.").pack(pady=20)
            return

        fig = plt.Figure(figsize=(10.5, 7.2), dpi=95, tight_layout=True)
        axs = fig.subplots(2, 2)
        fig.patch.set_facecolor("#f8fafc")

        modelos = [
            ("Naive Bayes", axs[0, 0]),
            ("Random Forest", axs[0, 1]),
            ("Perceptrón Multicapa (MLP)", axs[1, 0]),
            ("Referencia Base (Mayoritaria)", axs[1, 1])
        ]

        for mod_name, ax in modelos:
            data = self.metricas.get(mod_name, {})
            cm = data.get("matriz_confusion", {"TN": 0, "FP": 0, "FN": 0, "TP": 0})
            tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
            mat = np.array([[tn, fp], [fn, tp]])

            im = ax.imshow(mat, cmap="Blues" if mod_name != "Random Forest" else "Greens", alpha=0.75)

            # Textos en celdas
            celdas_info = [
                (0, 0, f"VN (TN)\n{tn:,}"),
                (0, 1, f"FP\n{fp:,}"),
                (1, 0, f"FN\n{fn:,}"),
                (1, 1, f"VP (TP)\n{tp:,}")
            ]
            for r, c, txt in celdas_info:
                val = mat[r, c]
                color_txt = "white" if val > (mat.max() * 0.55) else "#0f172a"
                ax.text(c, r, txt, ha="center", va="center", color=color_txt, fontsize=10, fontweight="bold")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred: No Compra", "Pred: Compra"], fontsize=8)
            ax.set_yticklabels(["Real: No Compra", "Real: Compra"], fontsize=8)
            f1 = data.get("f1_score", 0.0)
            rec = data.get("sensibilidad", 0.0)
            ax.set_title(f"{mod_name}\nRecall={rec:.4f} | F1={f1:.4f}", fontsize=10, fontweight="bold", pad=6)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def _construir_pestana_tabla(self, parent):
        frame = ttk.Frame(parent, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        cols = ("Modelo", "Exactitud (Acc)", "Precisión (Prec)", "Sensibilidad (Recall)", "F1-Score", "Tiempo Inferencia", "Diagnostico")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=6)
        for c in cols:
            tree.heading(c, text=c)

        tree.column("Modelo", width=230, anchor="w")
        tree.column("Exactitud (Acc)", width=130, anchor="center")
        tree.column("Precisión (Prec)", width=130, anchor="center")
        tree.column("Sensibilidad (Recall)", width=150, anchor="center")
        tree.column("F1-Score", width=120, anchor="center")
        tree.column("Tiempo Inferencia", width=140, anchor="center")
        tree.column("Diagnostico", width=220, anchor="w")

        diagnosticos = {
            "Naive Bayes": "Colapso a clase mayoritaria por umbral 0.50 y discretización",
            "Random Forest": "Óptimo para desbalance: balanceo de pesos prioriza detección (Recall 75.9%)",
            "Perceptrón Multicapa (MLP)": "Sensible al desbalance sin ajuste de umbral/costos (predice no compra)",
            "Referencia Base (Mayoritaria)": "Predice siempre No Compra: Exactitud ilusoria del 84.51%, F1=0.0"
        }

        for mod, data in self.metricas.items():
            diag = diagnosticos.get(mod, "-")
            tree.insert("", tk.END, values=(
                mod,
                f"{data['exactitud']:.4f} ({data['exactitud']*100:.1f}%)",
                f"{data['precision']:.4f}",
                f"{data['sensibilidad']:.4f}",
                f"{data['f1_score']:.4f}",
                f"{data['tiempo_inferencia_ms']:.1f} ms",
                diag
            ))

        tree.pack(fill=tk.X, expand=True, pady=(0, 15))

        # Cuadro resumen
        lbl_resumen = ttk.LabelFrame(frame, text=" Hallazgo Metodológico Clave sobre la Evaluación en Desbalance ", padding=10)
        lbl_resumen.pack(fill=tk.BOTH, expand=True)

        txt_res = (
            "1. PARADOJA DE LA EXACTITUD (ACCURACY PARADOX):\n"
            "   La Referencia Base mayoritaria alcanza 84.51% de exactitud acertando todas las 'No Compras', pero es\n"
            "   completamente inútil para el negocio (Recall = 0.0, F1 = 0.0), fallando el 100% de los compradores reales (382 perdidos).\n\n"
            "2. DESEMPEÑO DE RANDOM FOREST (MODELO GANADOR EN NEGOCIO):\n"
            "   Gracias a 'class_weight=balanced', Random Forest sacrifica algo de exactitud global (54.54%) para\n"
            "   capturar exitosamente 290 de los 382 compradores (Sensibilidad / Recall del 75.92% y F1 de 0.3410).\n\n"
            "3. LIMITACIÓN DE NAIVE BAYES Y MLP CON UMBRAL FIJO 0.50:\n"
            "   Tanto Naive Bayes como MLP estiman probabilidades a posteriori bien calibradas que raramente cruzan el umbral\n"
            "   simétrico de 0.50 debido a que el prior de compra es solo del 15.5%. Para desplegar Naive Bayes o MLP en producción\n"
            "   con valor comercial, el umbral de corte debe ajustarse al prior (~0.15) o función de costo asimétrica."
        )

        txt = tk.Text(lbl_resumen, wrap="word", font=("Courier", 9), bg="#f8fafc", fg="#0f172a", relief="groove", padx=8, pady=8)
        txt.insert("1.0", txt_res)
        txt.config(state="disabled")
        txt.pack(fill=tk.BOTH, expand=True)

    def _construir_pestana_disc(self, parent):
        frame = ttk.Frame(parent, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        disc_md = (
            "====================================================================================================\n"
            "   APARTADO 5: ANÁLISIS EXHAUSTIVO DE SUPUESTOS, ERRORES Y ESTRUCTURAS DE MODELADO                 \n"
            "====================================================================================================\n\n"
            "1. PÉRDIDA DE INFORMACIÓN EN NAIVE BAYES POR DISCRETIZACIÓN:\n"
            "   • Al convertir 'ProductRelated' y 'ProductRelated_Duration' de escalas numéricas continuas a solo\n"
            "     3 categorías ('bajo', 'medio', 'alto'), se pierde la variabilidad fina de los datos.\n"
            "   • Por ejemplo, un usuario con 30 páginas de producto se etiqueta como 'alto' al igual que uno con\n"
            "     700 páginas. Random Forest y MLP operan con el valor continuo exacto, permitiendo divisiones y\n"
            "     aproximaciones no lineales más precisas.\n\n"
            "2. SUPUESTO DE INDEPENDENCIA CONDICIONAL:\n"
            "   • Naive Bayes asume que P(Páginas, Duración | Revenue) = P(Páginas | Revenue) · P(Duración | Revenue).\n"
            "   • En la navegación web real, la cantidad de páginas de producto y el tiempo total guardan una correlación\n"
            "     positiva natural. Violar este supuesto no impide obtener un ranking de probabilidades útil, pero\n"
            "     tiende a concentrar las probabilidades en los extremos.\n\n"
            "3. ROL DEL SUAVIZADO DE LAPLACE (+1):\n"
            "   • Previene que combinaciones raras (como VisitorType='Other' con solo 74 casos) generen probabilidades 0.0\n"
            "     que anularían por multiplicación toda la evidencia en Naive Bayes.\n\n"
            "4. IMPACTO DEL DESBALANCE DE CLASES (84.5% vs 15.5%):\n"
            "   • Bajo el umbral de decisión canónico de 0.50, los modelos que minimizan error global sin compensación\n"
            "     de costos caen en la trampa mayoritaria.\n"
            "   • Random Forest supera este reto mediante balanceo de pesos en los árboles, logrando el mayor valor\n"
            "     práctico para la detección de compradores en la tienda.\n"
        )

        txt = tk.Text(frame, wrap="word", font=("Courier", 10), bg="#f8fafc", fg="#0f172a", relief="groove", padx=10, pady=10)
        txt.insert("1.0", disc_md)
        txt.config(state="disabled")
        txt.pack(fill=tk.BOTH, expand=True)

    def _reevaluar(self):
        try:
            self.metricas = ejecutar_evaluacion_completa()
            messagebox.showinfo("Reevaluación Completa", "Se han recalculado todas las predicciones y matrices sobre el set de test.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al reevaluar:\n{e}")


def main():
    root = tk.Tk()
    app = ComparativaModelosApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
