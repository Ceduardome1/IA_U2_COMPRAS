#!/usr/bin/env python3
"""
distribucion_datos.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: JAIME / CESAR
Entregable: Apartado 1 - Exploración y Distribución de Datos

Objetivos y Especificaciones:
1. Vista especializada e interactiva de la distribución de las variables:
   - Balance de la clase objetivo 'Revenue' (84.53% False vs 15.47% True).
   - Distribución de ProductRelated con cortes percentiles P33 (10) y P67 (29).
   - Distribución de ProductRelated_Duration con cortes percentiles P33 (285.93 s) y P67 (1107.23 s).
   - Frecuencia y tasa de conversión según VisitorType.
2. Panel interactivo de estadísticas descriptivas con tabla detallada (Treeview).
3. Acceso directo al Mapa Conceptual mediante botón dedicado.

Uso:
    python src/analisisDatos/distribucion_datos.py
"""

import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    TRAIN_NUM_PATH,
    TRAIN_CAT_PATH,
    PERCENTILES_JSON,
    TARGET_COL,
    SELECTED_FEATURES
)
from src.preprocessing import load_percentiles


class DistribucionDatosApp:
    def __init__(self, root):
        self.root = root
        if hasattr(self.root, "title"):
            self.root.title("Apartado 1: Distribución y Exploración de Datos")
        if hasattr(self.root, "geometry"):
            self.root.geometry("1180x780")
        if hasattr(self.root, "minsize"):
            self.root.minsize(980, 680)

        self._configurar_estilos()
        self._cargar_datos()
        self._crear_interfaz()

    def _configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Arial", 14, "bold"), foreground="#1e293b")
        style.configure("Subheader.TLabel", font=("Arial", 10), foreground="#64748b")
        style.configure("Card.TFrame", background="#ffffff", relief="ridge")
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), padding=6)
        style.configure("Action.TButton", font=("Arial", 10), padding=5)

    def _cargar_datos(self):
        self.df_raw = None
        self.df_train_num = None
        self.df_train_cat = None
        self.percentiles = {}

        # Cargar raw data
        if RAW_DATA_PATH.exists():
            self.df_raw = pd.read_csv(RAW_DATA_PATH)

        # Cargar train sets si existen
        if TRAIN_NUM_PATH.exists():
            self.df_train_num = pd.read_csv(TRAIN_NUM_PATH)
        if TRAIN_CAT_PATH.exists():
            self.df_train_cat = pd.read_csv(TRAIN_CAT_PATH)

        # Cargar percentiles
        if PERCENTILES_JSON.exists():
            try:
                self.percentiles = load_percentiles(PERCENTILES_JSON)
            except Exception:
                pass

        if not self.percentiles and self.df_raw is not None:
            # Fallback en memoria si aún no se guardó el json
            p33_pr = float(np.percentile(self.df_raw["ProductRelated"], 33))
            p67_pr = float(np.percentile(self.df_raw["ProductRelated"], 67))
            p33_dur = float(np.percentile(self.df_raw["ProductRelated_Duration"], 33))
            p67_dur = float(np.percentile(self.df_raw["ProductRelated_Duration"], 67))
            self.percentiles = {
                "ProductRelated": {"p33": p33_pr, "p67": p67_pr},
                "ProductRelated_Duration": {"p33": p33_dur, "p67": p67_dur}
            }

    def _crear_interfaz(self):
        # Barra superior con título y botón para mapa conceptual
        top_bar = ttk.Frame(self.root, padding="12 10 12 10")
        top_bar.pack(fill=tk.X)

        info_box = ttk.Frame(top_bar)
        info_box.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(
            info_box,
            text="Exploración y Distribución de Variables del Problema",
            style="Header.TLabel"
        ).pack(anchor="w")
        total_txt = f"{len(self.df_raw):,} registros" if self.df_raw is not None else "Sin datos"
        ttk.Label(
            info_box,
            text=f"Dataset Online Shoppers Purchasing Intention (UCI) • {total_txt} • 4 Atributos de Estudio",
            style="Subheader.TLabel"
        ).pack(anchor="w")

        btn_box = ttk.Frame(top_bar)
        btn_box.pack(side=tk.RIGHT, fill=tk.Y)

        btn_mapa = ttk.Button(
            btn_box,
            text="🌐 Ver Mapa Conceptual (NetworkX)",
            style="Accent.TButton",
            command=self._abrir_mapa_conceptual
        )
        btn_mapa.pack(side=tk.RIGHT, padx=5)

        btn_recargar = ttk.Button(
            btn_box,
            text="🔄 Recargar Datos",
            style="Action.TButton",
            command=self._recargar_todo
        )
        btn_recargar.pack(side=tk.RIGHT, padx=5)

        # Panel principal con pestañas
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        # Pestaña 1: Visualización Gráfica de Distribuciones
        tab_graficas = ttk.Frame(notebook)
        notebook.add(tab_graficas, text="📊 Distribuciones y Cortes Percentiles")
        self._construir_pestana_graficas(tab_graficas)

        # Pestaña 2: Tabla Estadística Descriptiva y Cruces
        tab_tablas = ttk.Frame(notebook)
        notebook.add(tab_tablas, text="📋 Estadísticas Descriptivas y Tablas")
        self._construir_pestana_tablas(tab_tablas)

        # Pestaña 3: Guía y Metadatos de Atributos
        tab_info = ttk.Frame(notebook)
        notebook.add(tab_info, text="ℹ️ Definición de Atributos")
        self._construir_pestana_info(tab_info)

    def _construir_pestana_graficas(self, parent):
        if self.df_raw is None:
            ttk.Label(parent, text="Error: No se encontró el dataset en data/online_shoppers_intention.csv").pack(pady=30)
            return

        fig = plt.Figure(figsize=(11, 7.5), dpi=95, tight_layout=True)
        axs = fig.subplots(2, 2)

        # 1. Desbalance de la variable objetivo Revenue
        ax1 = axs[0, 0]
        conteo_rev = self.df_raw[TARGET_COL].value_counts().sort_index()
        etiquetas_rev = ["No Compra\n(False)", "Compra\n(True)"]
        colores_rev = ["#ef4444", "#10b981"]
        barras = ax1.bar(etiquetas_rev, conteo_rev.values, color=colores_rev, width=0.52, edgecolor="#334155", linewidth=1.1)
        total = len(self.df_raw)
        for b, v in zip(barras, conteo_rev.values):
            pct = (v / total) * 100
            ax1.text(b.get_x() + b.get_width() / 2, v + total * 0.015, f"{v:,}\n({pct:.2f}%)",
                     ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1e293b")
        ax1.set_title("Balance de Clase Objetivo (Revenue)", fontsize=11, fontweight="bold", pad=8)
        ax1.set_ylabel("Cantidad de Sesiones", fontsize=9)
        ax1.set_ylim(0, max(conteo_rev.values) * 1.20)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # 2. Distribución de ProductRelated con cortes P33 y P67
        ax2 = axs[0, 1]
        p33_pr = self.percentiles.get("ProductRelated", {}).get("p33", 10.0)
        p67_pr = self.percentiles.get("ProductRelated", {}).get("p67", 29.0)
        pr_data = self.df_raw["ProductRelated"].clip(upper=100)  # recortar visualmente para ver el núcleo
        ax2.hist(pr_data, bins=35, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.75, density=False)
        ax2.axvline(p33_pr, color="#d97706", linestyle="--", linewidth=2.0, label=f"P33 = {p33_pr:.0f} (Bajo ≤ {p33_pr:.0f})")
        ax2.axvline(p67_pr, color="#b91c1c", linestyle="--", linewidth=2.0, label=f"P67 = {p67_pr:.0f} (Alto > {p67_pr:.0f})")
        ax2.set_title("ProductRelated (Páginas de Producto)", fontsize=11, fontweight="bold", pad=8)
        ax2.set_xlabel("Número de Páginas (recorte visual a 100)", fontsize=9)
        ax2.set_ylabel("Frecuencia", fontsize=9)
        ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        # 3. Distribución de ProductRelated_Duration con cortes P33 y P67
        ax3 = axs[1, 0]
        p33_dur = self.percentiles.get("ProductRelated_Duration", {}).get("p33", 285.93)
        p67_dur = self.percentiles.get("ProductRelated_Duration", {}).get("p67", 1107.23)
        dur_data = self.df_raw["ProductRelated_Duration"].clip(upper=4000)
        ax3.hist(dur_data, bins=35, color="#8b5cf6", edgecolor="#6d28d9", alpha=0.75, density=False)
        ax3.axvline(p33_dur, color="#d97706", linestyle="--", linewidth=2.0, label=f"P33 = {p33_dur:.1f} s")
        ax3.axvline(p67_dur, color="#b91c1c", linestyle="--", linewidth=2.0, label=f"P67 = {p67_dur:.1f} s")
        ax3.set_title("ProductRelated_Duration (Tiempo en Segundos)", fontsize=11, fontweight="bold", pad=8)
        ax3.set_xlabel("Duración (segundos, recorte visual a 4000s)", fontsize=9)
        ax3.set_ylabel("Frecuencia", fontsize=9)
        ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax3.grid(axis="y", linestyle="--", alpha=0.4)

        # 4. Frecuencia y tasa de compra por VisitorType
        ax4 = axs[1, 1]
        vis_cats = ["Returning_Visitor", "New_Visitor", "Other"]
        conteos_vis = [int((self.df_raw["VisitorType"] == c).sum()) for c in vis_cats]
        tasas_compra = [
            float((self.df_raw[self.df_raw["VisitorType"] == c][TARGET_COL] == True).mean() * 100)
            if (self.df_raw["VisitorType"] == c).sum() > 0 else 0.0
            for c in vis_cats
        ]

        x_indices = np.arange(len(vis_cats))
        width = 0.36
        bars1 = ax4.bar(x_indices - width/2, conteos_vis, width, label="Sesiones Totales", color="#0284c7")
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x_indices + width/2, tasas_compra, width, label="Tasa Compra (%)", color="#10b981")

        for b, v in zip(bars1, conteos_vis):
            ax4.text(b.get_x() + b.get_width()/2, v + 150, f"{v:,}", ha="center", va="bottom", fontsize=8, color="#0369a1")
        for b, v in zip(bars2, tasas_compra):
            ax4_twin.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#047857")

        ax4.set_xticks(x_indices)
        ax4.set_xticklabels(["Returning", "New", "Other"], fontsize=9)
        ax4.set_title("VisitorType: Frecuencia vs Tasa de Conversión", fontsize=11, fontweight="bold", pad=8)
        ax4.set_ylabel("Cantidad de Sesiones", fontsize=9, color="#0284c7")
        ax4_twin.set_ylabel("Tasa de Compra (%)", fontsize=9, color="#10b981")
        ax4.set_ylim(0, max(conteos_vis) * 1.15)
        ax4_twin.set_ylim(0, max(tasas_compra) * 1.35)

        # Canvas Matplotlib en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def _construir_pestana_tablas(self, parent):
        if self.df_raw is None:
            return

        frame_split = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        frame_split.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel superior: Estadísticas de variables continuas
        frame_top = ttk.LabelFrame(frame_split, text=" Estadísticas Descriptivas de ProductRelated y Duración ", padding=8)
        frame_split.add(frame_top, weight=1)

        cols_stat = ("Métrica", "ProductRelated (Páginas)", "ProductRelated_Duration (Segundos)")
        tree_stat = ttk.Treeview(frame_top, columns=cols_stat, show="headings", height=8)
        tree_stat.heading("Métrica", text="Métrica / Estadístico")
        tree_stat.heading("ProductRelated (Páginas)", text="ProductRelated (Páginas)")
        tree_stat.heading("ProductRelated_Duration (Segundos)", text="ProductRelated_Duration (Segundos)")
        tree_stat.column("Métrica", width=220, anchor="w")
        tree_stat.column("ProductRelated (Páginas)", width=180, anchor="center")
        tree_stat.column("ProductRelated_Duration (Segundos)", width=220, anchor="center")

        pr = self.df_raw["ProductRelated"].dropna()
        dur = self.df_raw["ProductRelated_Duration"].dropna()
        metricas = [
            ("Conteo Total", f"{len(pr):,}", f"{len(dur):,}"),
            ("Media Aritmética (μ)", f"{pr.mean():.2f}", f"{dur.mean():.2f} s"),
            ("Desviación Estándar (σ)", f"{pr.std():.2f}", f"{dur.std():.2f} s"),
            ("Valor Mínimo", f"{pr.min():.2f}", f"{dur.min():.2f} s"),
            ("Percentil 25 (P25)", f"{np.percentile(pr, 25):.2f}", f"{np.percentile(dur, 25):.2f} s"),
            ("Percentil 33 (P33 - Límite Bajo)", f"{np.percentile(pr, 33):.2f}", f"{np.percentile(dur, 33):.2f} s"),
            ("Mediana (P50)", f"{pr.median():.2f}", f"{dur.median():.2f} s"),
            ("Percentil 67 (P67 - Límite Alto)", f"{np.percentile(pr, 67):.2f}", f"{np.percentile(dur, 67):.2f} s"),
            ("Percentil 75 (P75)", f"{np.percentile(pr, 75):.2f}", f"{np.percentile(dur, 75):.2f} s"),
            ("Valor Máximo", f"{pr.max():.2f}", f"{dur.max():.2f} s")
        ]
        for m in metricas:
            tree_stat.insert("", tk.END, values=m)
        tree_stat.pack(fill=tk.BOTH, expand=True)

        # Panel inferior: Segmentación y Tasa de Conversión
        frame_bot = ttk.LabelFrame(frame_split, text=" Segmentación de Categorías y Tasa Empírica de Compra ", padding=8)
        frame_split.add(frame_bot, weight=1)

        cols_seg = ("Variable", "Categoría / Intervalo", "Sesiones", "Proporción", "Compras", "Tasa Compra (%)")
        tree_seg = ttk.Treeview(frame_bot, columns=cols_seg, show="headings", height=8)
        for c in cols_seg:
            tree_seg.heading(c, text=c)
            tree_seg.column(c, anchor="center" if c != "Variable" and c != "Categoría / Intervalo" else "w")

        # Categorías de VisitorType
        total_ses = len(self.df_raw)
        for cat in ["Returning_Visitor", "New_Visitor", "Other"]:
            sub = self.df_raw[self.df_raw["VisitorType"] == cat]
            cnt = len(sub)
            compras = int((sub[TARGET_COL] == True).sum())
            tasa = (compras / cnt * 100) if cnt > 0 else 0
            tree_seg.insert("", tk.END, values=(
                "VisitorType", cat, f"{cnt:,}", f"{(cnt/total_ses)*100:.2f}%", f"{compras:,}", f"{tasa:.2f}%"
            ))

        # Categorías discretizadas de ProductRelated
        p33_pr = self.percentiles.get("ProductRelated", {}).get("p33", 10.0)
        p67_pr = self.percentiles.get("ProductRelated", {}).get("p67", 29.0)
        pr_bajo = self.df_raw[self.df_raw["ProductRelated"] <= p33_pr]
        pr_med = self.df_raw[(self.df_raw["ProductRelated"] > p33_pr) & (self.df_raw["ProductRelated"] <= p67_pr)]
        pr_alto = self.df_raw[self.df_raw["ProductRelated"] > p67_pr]

        for etiqueta, sub, rango in [("bajo", pr_bajo, f"≤ {p33_pr:.0f}"), ("medio", pr_med, f"{p33_pr:.0f} a {p67_pr:.0f}"), ("alto", pr_alto, f"> {p67_pr:.0f}")]:
            cnt = len(sub)
            compras = int((sub[TARGET_COL] == True).sum())
            tasa = (compras / cnt * 100) if cnt > 0 else 0
            tree_seg.insert("", tk.END, values=(
                "ProductRelated (P33/P67)", f"{etiqueta} ({rango})", f"{cnt:,}", f"{(cnt/total_ses)*100:.2f}%", f"{compras:,}", f"{tasa:.2f}%"
            ))

        tree_seg.pack(fill=tk.BOTH, expand=True)

    def _construir_pestana_info(self, parent):
        text_frame = ttk.Frame(parent, padding=15)
        text_frame.pack(fill=tk.BOTH, expand=True)

        info_md = (
            "========================================================================================\n"
            "   APARTADO 1: DEFINICIÓN DE ATRIBUTOS Y PREPARACIÓN DE REPRESENTACIONES DE DATOS       \n"
            "========================================================================================\n\n"
            "1. DESCRIPCIÓN DEL DOMINIO DEL PROBLEMA:\n"
            "   El dataset 'Online Shoppers Purchasing Intention' modela sesiones de navegación en una tienda\n"
            "   de comercio electrónico. El objetivo es predecir si una sesión finalizará con una transacción\n"
            "   de compra exitosa ('Revenue' = True) o si el usuario abandonará sin comprar ('Revenue' = False).\n\n"
            "2. ATRIBUTOS SELECCIONADOS:\n"
            "   • VisitorType (Categórica Nominal):\n"
            "     - Representa el tipo de visitante: 'Returning_Visitor' (recurrente), 'New_Visitor' (nuevo)\n"
            "       o 'Other' (desconocido/otro).\n"
            "     - Los visitantes nuevos muestran una tasa de conversión superior (~24.91%) frente a los recurrentes\n"
            "       (~13.93%), lo que evidencia el valor predictivo de esta variable en el Teorema de Bayes.\n\n"
            "   • ProductRelated (Numérica Discreta - Conteos):\n"
            "     - Cantidad de páginas de productos vistas durante la sesión.\n"
            "     - Distribución con asimetría positiva fuerte (sesgo a la derecha): mediana de 18 vs media de 31.73.\n"
            "     - Discretización mediante percentiles en Train: P33 = 10 y P67 = 29 ('bajo', 'medio', 'alto').\n\n"
            "   • ProductRelated_Duration (Numérica Continua - Segundos):\n"
            "     - Tiempo total acumulado por el usuario en páginas de productos.\n"
            "     - Mediana de ~598.9 segundos (~10 min) y percentiles P33 = 285.93 s y P67 = 1107.23 s.\n\n"
            "   • Revenue (Variable Objetivo - Booleana):\n"
            "     - Clase desbalanceada con un 84.53% False y 15.47% True (~5.5 visitas sin compra por cada compra).\n"
            "     - Requiere evaluación estratificada y métricas robustas al desbalance (Recall y F1-Score).\n\n"
            "3. REPRESENTACIONES DUALES:\n"
            "   • Para Modelos Probabilísticos (Naive Bayes):\n"
            "     - Requiere variables categóricas homogéneas discretizadas con percentiles P33/P67 en Train.\n"
            "   • Para Modelos de Aprendizaje Supervisado (Random Forest & MLP):\n"
            "     - Mantienen valores continuos y One-Hot Encoding para VisitorType.\n"
            "     - MLP requiere adicionalmente estandarización StandardScaler (μ=0, σ=1).\n"
        )

        txt = tk.Text(text_frame, wrap="word", font=("Courier", 10), bg="#f8fafc", fg="#0f172a", relief="groove")
        txt.insert("1.0", info_md)
        txt.config(state="disabled")
        txt.pack(fill=tk.BOTH, expand=True)

    def _abrir_mapa_conceptual(self):
        script_mapa = BASE_DIR / "src" / "analisisDatos" / "mapa_conceptual.py"
        if not script_mapa.exists():
            messagebox.showerror("Error", f"No se encontró el script del mapa conceptual en:\n{script_mapa}")
            return
        subprocess.Popen([sys.executable, str(script_mapa)])

    def _recargar_todo(self):
        self._cargar_datos()
        messagebox.showinfo("Datos Recargados", "Se han recargado las fuentes de datos y percentiles.")


def main():
    root = tk.Tk()
    app = DistribucionDatosApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
