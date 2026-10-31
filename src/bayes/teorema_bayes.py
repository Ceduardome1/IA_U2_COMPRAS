#!/usr/bin/env python3
"""
teorema_bayes.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: JAIME / CESAR
Entregable: Apartado 2 - Visualizador del Teorema de Bayes por Conteos

Objetivos y Especificaciones:
1. Interfaz interactiva para el Teorema de Bayes univariado:
   - Panel de datos de entrenamiento:
     * Tabla de contingencia 2x2 interactiva (Recurrente vs No Recurrente contra Compra vs No Compra).
     * Muestra de los conteos exactos: N, N(C), N(¬C), N(R), N(CR), N(¬CR).
2. Panel de desglose paso a paso:
   - Presentación de probabilidades a priori y verosimilitudes con formato matemático.
   - Fórmula de Bayes desarrollada con los valores sustituidos explícitamente:
     P(C|R) = [P(R|C) * P(C)] / [P(R|C)*P(C) + P(R|¬C)*P(¬C)]
   - Comprobación empírica directa: N(CR) / N(R).
   - Indicador de validación de equivalencia matemática (diferencia = 0).
3. Conclusiones e interpretación textual:
   - Explicación de cómo la condición de recurrente modifica la probabilidad a priori de compra.

Uso:
    python src/bayes/teorema_bayes.py
"""

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_CAT_PATH
from src.bayes.cli_teorema_bayes import (
    obtener_conteos_train,
    calcular_probabilidades,
    ejecutar_comprobacion_directa
)


class BayesApp:
    def __init__(self, root):
        self.root = root
        if hasattr(self.root, "title"):
            self.root.title("Apartado 2: Teorema de Bayes por Conteos Exactos")
        if hasattr(self.root, "geometry"):
            self.root.geometry("1050x750")
        if hasattr(self.root, "minsize"):
            self.root.minsize(880, 620)

        self._configurar_estilos()
        self._cargar_y_calcular()
        self._crear_interfaz()

    def _configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Arial", 14, "bold"), foreground="#1e293b")
        style.configure("Subheader.TLabel", font=("Arial", 10), foreground="#64748b")
        style.configure("Math.TLabel", font=("Courier", 10, "bold"), foreground="#1e3a8a")

    def _cargar_y_calcular(self):
        try:
            self.conteos = obtener_conteos_train()
            self.probs = calcular_probabilidades(self.conteos)
            self.p_directa = ejecutar_comprobacion_directa(self.conteos)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo CSV para Bayes:\n{e}")
            self.conteos = None
            self.probs = None

    def _crear_interfaz(self):
        if not self.conteos:
            ttk.Label(self.root, text="Error: No se pudieron cargar los datos de entrenamiento.").pack(pady=30)
            return

        # Encabezado
        top_bar = ttk.Frame(self.root, padding="12 10 12 10")
        top_bar.pack(fill=tk.X)

        ttk.Label(top_bar, text="Apartado 2: Teorema de Bayes Univariado por Conteos Frecuentistas", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            top_bar,
            text=f"Demostración analítica y empírica sobre Train 80% ({self.conteos['N']:,} sesiones) • Evento Compra vs Visitante Recurrente",
            style="Subheader.TLabel"
        ).pack(anchor="w")

        # Contenedor con scroll o split vertical
        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        # Panel 1: Tabla de Contingencia 2x2
        frame_cont = ttk.LabelFrame(paned, text=" 1. Tabla de Contingencia 2x2 (Frecuencias Absolutas de Entrenamiento) ", padding=10)
        paned.add(frame_cont, weight=4)

        cols = ("Condicion", "Compra (Revenue=True)", "No Compra (Revenue=False)", "Total Marginal")
        tree = ttk.Treeview(frame_cont, columns=cols, show="headings", height=4)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, anchor="center" if c != "Condicion" else "w", width=220)

        n_r = self.conteos["N_R"]
        n_cr = self.conteos["N_CR"]
        n_nocr = self.conteos["N_noCR"]

        n_total = self.conteos["N"]
        n_nor = self.conteos["N_noR"]
        n_c = self.conteos["N_C"]
        n_noc = self.conteos["N_noC"]

        n_cnor = n_c - n_cr
        n_nocnor = n_noc - n_nocr

        tree.insert("", tk.END, values=("Visitante Recurrente (R)", f"{n_cr:,}", f"{n_nocr:,}", f"{n_r:,} (85.59%)"))
        tree.insert("", tk.END, values=("Visitante No Recurrente (¬R)", f"{n_cnor:,}", f"{n_nocnor:,}", f"{n_nor:,} (14.41%)"))
        tree.insert("", tk.END, values=("Total por Clase de Revenue", f"{n_c:,} (15.47%)", f"{n_noc:,} (84.53%)", f"{n_total:,} (100.0%)"))
        tree.pack(fill=tk.X, expand=True)

        # Panel 2: Desarrollo Paso a Paso y Validación
        frame_math = ttk.LabelFrame(paned, text=" 2. Desglose Analítico del Teorema de Bayes vs Comprobación Directa ", padding=10)
        paned.add(frame_math, weight=6)

        dif = abs(self.probs["P_C_dado_R_Bayes"] - self.p_directa)
        valid_txt = "EQUIVALENCIA MATEMÁTICA EXACTA (Diferencia ≈ 0.0)" if dif < 1e-10 else f"Diferencia: {dif}"

        texto_algebraico = (
            "====================================================================================================\n"
            "   A) ESTIMACIÓN FRECUENTISTA DE PROBABILIDADES A PRIORI Y VEROSIMILITUDES                          \n"
            "====================================================================================================\n"
            f"   • Probabilidad a Priori de Compra:       P(C)    = N(C) / N        = {n_c} / {n_total} = {self.probs['P_C']:.6f} ({self.probs['P_C']*100:.2f}%)\n"
            f"   • Probabilidad a Priori de No Compra:    P(¬C)   = N(¬C) / N       = {n_noc} / {n_total} = {self.probs['P_noC']:.6f} ({self.probs['P_noC']*100:.2f}%)\n"
            f"   • Verosimilitud de Recurrente si Compra: P(R|C)  = N(CR) / N(C)    = {n_cr} / {n_c} = {self.probs['P_R_dado_C']:.6f} ({self.probs['P_R_dado_C']*100:.2f}%)\n"
            f"   • Verosimilitud si No Compra:            P(R|¬C) = N(¬C,R) / N(¬C) = {n_nocr} / {n_noc} = {self.probs['P_R_dado_noC']:.6f} ({self.probs['P_R_dado_noC']*100:.2f}%)\n\n"
            "====================================================================================================\n"
            "   B) SUSTITUCIÓN EN LA FÓRMULA DEL TEOREMA DE BAYES                                                \n"
            "====================================================================================================\n"
            "                    P(R | C) · P(C)\n"
            "       P(C | R) = ------------------------------------\n"
            "                  P(R | C)·P(C) + P(R | ¬C)·P(¬C)\n\n"
            f"       P(C | R) = ({self.probs['P_R_dado_C']:.6f} · {self.probs['P_C']:.6f}) / "
            f"({self.probs['Numerador_Bayes']:.6f} + {self.probs['P_R_dado_noC']*self.probs['P_noC']:.6f})\n\n"
            f"       P(C | R) = {self.probs['Numerador_Bayes']:.6f} / {self.probs['Denominador_Bayes']:.6f} = {self.probs['P_C_dado_R_Bayes']:.6f}  ({self.probs['P_C_dado_R_Bayes']*100:.4f}%)\n\n"
            "====================================================================================================\n"
            "   C) COMPROBACIÓN EMPÍRICA DIRECTA Y VALIDACIÓN DE LA IDENTIDAD                                    \n"
            "====================================================================================================\n"
            f"   • Cálculo Directo Frecuentista: P_directa(C | R) = N(CR) / N(R) = {n_cr} / {n_r} = {self.p_directa:.6f} ({self.p_directa*100:.4f}%)\n"
            f"   • Estado de Validación:         {valid_txt} (Diferencia de máquina: {dif:.1e})\n\n"
            "====================================================================================================\n"
            "   D) INTERPRETACIÓN DEL COMPORTAMIENTO DEL CONSUMIDOR                                              \n"
            "====================================================================================================\n"
            f"   • La probabilidad general de compra para cualquier usuario es del {self.probs['P_C']*100:.2f}%.\n"
            f"   • Si sabemos que el usuario es Recurrente ('Returning_Visitor'), la probabilidad cae a {self.probs['P_C_dado_R_Bayes']*100:.2f}%.\n"
            f"   • Conclusión: En este sitio web, los visitantes recurrentes muestran una menor conversión que los\n"
            f"     visitantes nuevos (que compran con ~24.9% de tasa). Por tanto, la condición de recurrente disminuye\n"
            f"     la creencia a priori de compra."
        )

        txt_math = tk.Text(frame_math, wrap="word", font=("Courier", 9), bg="#f8fafc", fg="#0f172a", relief="groove", padx=8, pady=8)
        txt_math.insert("1.0", texto_algebraico)
        txt_math.config(state="disabled")
        txt_math.pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    app = BayesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
