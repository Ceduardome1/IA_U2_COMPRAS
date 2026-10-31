#!/usr/bin/env python3
"""
red_naive_bayes.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: JAIME / CESAR
Entregable: Apartado 3 - Grafo Naive Bayes y Tablas CPT con Suavizado de Laplace

Objetivos y Especificaciones:
1. Representación gráfica de la Red Naive Bayes mediante NetworkX:
   - Nodo Padre: Revenue (Compra = True / False).
   - Nodos Hijos condicionalmente independientes:
     * VisitorType (Returning_Visitor, New_Visitor, Other)
     * ProductRelated (bajo, medio, alto)
     * ProductRelated_Duration (bajo, medio, alto)
2. Tablas CPT con Suavizado de Laplace (+1):
   - Muestra las 18 probabilidades condicionales calculadas en entrenamiento.
   - Compara la probabilidad empírica directa frente a la probabilidad con Laplace.
   - Razón de verosimilitudes (LR) e impacto predictivo hacia Compra o No Compra.
3. Persistencia de Datos Estadísticos:
   - Carga instantánea desde 'artifacts/tablas_cpt_laplace.json' para evitar recálculos innecesarios.
   - Exportación de la tabla de análisis a 'artifacts/analisis_estadistico_cpt.csv'.
   - Botón de recálculo y actualización de la persistencia bajo demanda.
4. Visor especializado de Matrices CPT Formales:
   - Permite visualizar de forma matricial las tablas CPT calculadas para cada variable,
     con sumas axiomáticas unitarias y modos condicionales.
5. Filtros interactivos y visor de metadatos.

Uso:
    python src/redBayesiana/red_naive_bayes.py
"""

import sys
import json
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configurar path raíz y garantizar imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TRAIN_CAT_PATH,
    CPT_LAPLACE_JSON,
    ARTIFACTS_DIR
)
from src.redBayesiana.cli_pipeline_red_naive_bayes import (
    cargar_tablas_cpt,
    guardar_tablas_cpt,
    calcular_cpt_laplace
)


class RedNaiveBayesApp:
    def __init__(self, root):
        self.root = root
        if hasattr(self.root, "title"):
            self.root.title("Apartado 3: Red Naive Bayes y Tablas CPT con Laplace (+1)")
        if hasattr(self.root, "geometry"):
            self.root.geometry("1180x820")
        if hasattr(self.root, "minsize"):
            self.root.minsize(980, 680)

        self.cpt_data = None
        self.origen_datos_txt = ""

        self._configurar_estilos()
        self._cargar_o_calcular_cpt(forzar_recalculo=False)
        self._crear_interfaz()
        self._actualizar_tabla()

    def _configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Arial", 13, "bold"), foreground="#1e293b")
        style.configure("Subheader.TLabel", font=("Arial", 9), foreground="#64748b")
        style.configure("Status.TLabel", font=("Arial", 9, "bold"), foreground="#047857")
        style.configure("Accent.TButton", font=("Arial", 9, "bold"), padding=5)
        style.configure("Action.TButton", font=("Arial", 9), padding=4)

    def _cargar_o_calcular_cpt(self, forzar_recalculo=False):
        """Carga desde la persistencia JSON o calcula y persiste si no existe o se fuerza."""
        try:
            if not forzar_recalculo and CPT_LAPLACE_JSON.exists():
                self.cpt_data = cargar_tablas_cpt(CPT_LAPLACE_JSON, force_recalc=False)
                fecha = self.cpt_data.get("metadatos", {}).get("fecha_generacion", "Registrada")
                self.origen_datos_txt = f"💾 Cargado desde persistencia: '{CPT_LAPLACE_JSON.name}' (Generado: {fecha[:19].replace('T', ' ')})"
            else:
                self.cpt_data = cargar_tablas_cpt(CPT_LAPLACE_JSON, force_recalc=True)
                self.origen_datos_txt = f"🔄 Recalculado y persistido exitosamente en: '{CPT_LAPLACE_JSON.name}'"
        except Exception as e:
            messagebox.showerror("Error al Cargar Datos", f"No se pudo cargar o calcular las tablas CPT:\n{e}")
            self.cpt_data = None

    def _crear_interfaz(self):
        # Barra superior con título y estado de persistencia
        top_bar = ttk.Frame(self.root, padding="10 8 10 8")
        top_bar.pack(fill=tk.X)

        info_box = ttk.Frame(top_bar)
        info_box.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(info_box, text="Estructura de Red Naive Bayes y Tablas CPT con Suavizado de Laplace (+1)", style="Header.TLabel").pack(anchor="w")
        self.lbl_status = ttk.Label(info_box, text=self.origen_datos_txt, style="Status.TLabel")
        self.lbl_status.pack(anchor="w", pady=(2, 0))

        btn_box = ttk.Frame(top_bar)
        btn_box.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(
            btn_box,
            text="📑 Ver Matriz CPT Completa",
            style="Accent.TButton",
            command=self._ver_matriz_cpt_completa
        ).pack(side=tk.RIGHT, padx=4)

        ttk.Button(
            btn_box,
            text="🔄 Recalcular y Guardar",
            style="Action.TButton",
            command=self._forzar_recalculo
        ).pack(side=tk.RIGHT, padx=4)

        ttk.Button(
            btn_box,
            text="📋 Ver Metadatos JSON",
            style="Action.TButton",
            command=self._ver_metadatos_json
        ).pack(side=tk.RIGHT, padx=4)

        ttk.Button(
            btn_box,
            text="📊 Exportar CSV",
            style="Action.TButton",
            command=self._exportar_csv_analisis
        ).pack(side=tk.RIGHT, padx=4)

        # Contenedor dividido: Arriba Grafo NetworkX, Abajo Tablas CPT
        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Panel Superior: Grafo de la Red Naive Bayes
        frame_grafo = ttk.LabelFrame(paned, text=" Grafo Dirigido de la Red Bayesiana (Supuesto de Independencia Condicional) ", padding=6)
        paned.add(frame_grafo, weight=4)
        self._dibujar_grafo(frame_grafo)

        # Panel Inferior: Tablas CPT con filtros y explicaciones
        frame_cpt = ttk.LabelFrame(paned, text=" Tablas de Probabilidad Condicional (CPT) con Persistencia ", padding=6)
        paned.add(frame_cpt, weight=6)
        self._construir_panel_cpt(frame_cpt)

    def _dibujar_grafo(self, parent):
        fig, ax = plt.subplots(figsize=(8.5, 2.6), dpi=95)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")

        G = nx.DiGraph()
        
        nodo_padre = "Revenue\n(Compra = True / False)"
        hijos = [
            ("VisitorType\n(K=3 categorías)", (0.18, 0.15)),
            ("ProductRelated\n(K=3 terciles)", (0.50, 0.15)),
            ("ProductRelated_Duration\n(K=3 terciles)", (0.82, 0.15))
        ]

        pos = {nodo_padre: (0.50, 0.85)}
        edges = []
        edge_labels = {}

        for hijo_label, h_pos in hijos:
            pos[hijo_label] = h_pos
            edges.append((nodo_padre, hijo_label))
            edge_labels[(nodo_padre, hijo_label)] = "P(Atributo | Revenue)"

        G.add_edges_from(edges)

        # Dibujar aristas
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#0284c7",
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            width=2.0,
            connectionstyle="arc3,rad=0.0"
        )

        # Etiquetas en aristas
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=8, font_color="#0369a1",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#bae6fd", alpha=0.95)
        )

        # Nodo Padre (Revenue)
        ax.text(
            pos[nodo_padre][0], pos[nodo_padre][1],
            nodo_padre,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#1e3a8a", edgecolor="#0f172a", linewidth=1.5)
        )

        # Nodos Hijos (Atributos)
        colores_hijos = ["#059669", "#d97706", "#7c3aed"]
        for (hijo_label, h_pos), col in zip(hijos, colores_hijos):
            ax.text(
                h_pos[0], h_pos[1],
                hijo_label,
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="#ffffff",
                bbox=dict(boxstyle="round,pad=0.40", facecolor=col, edgecolor="#0f172a", linewidth=1.2)
            )

        ax.set_xlim(0.02, 0.98)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _construir_panel_cpt(self, parent):
        # Barra de filtros
        frame_filtros = ttk.Frame(parent)
        frame_filtros.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(frame_filtros, text="Filtrar Variable: ", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        self.var_filtro_var = tk.StringVar(value="Todas")
        combo_var = ttk.Combobox(
            frame_filtros,
            textvariable=self.var_filtro_var,
            values=["Todas", "VisitorType", "ProductRelated", "ProductRelated_Duration"],
            state="readonly",
            width=22
        )
        combo_var.bind("<<ComboboxSelected>>", lambda e: self._actualizar_tabla())
        combo_var.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(frame_filtros, text="Filtrar Clase: ", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        self.var_filtro_clase = tk.StringVar(value="Todas")
        combo_clase = ttk.Combobox(
            frame_filtros,
            textvariable=self.var_filtro_clase,
            values=["Todas", "Compra (True)", "No Compra (False)"],
            state="readonly",
            width=18
        )
        combo_clase.bind("<<ComboboxSelected>>", lambda e: self._actualizar_tabla())
        combo_clase.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(frame_filtros, text="Impacto: ", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 3))
        self.var_filtro_impacto = tk.StringVar(value="Todos")
        combo_impacto = ttk.Combobox(
            frame_filtros,
            textvariable=self.var_filtro_impacto,
            values=["Todos", "Favorece Compra (+)", "Favorece No Compra (-)"],
            state="readonly",
            width=22
        )
        combo_impacto.bind("<<ComboboxSelected>>", lambda e: self._actualizar_tabla())
        combo_impacto.pack(side=tk.LEFT, padx=(0, 10))

        # Tabla Treeview para las probabilidades condicionales
        cols = ("Clase", "Variable", "Categoria", "Conteo", "P_Empirica", "P_Laplace", "LR", "Impacto")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)

        self.tree.heading("Clase", text="Clase (Revenue)")
        self.tree.heading("Variable", text="Variable (Atributo)")
        self.tree.heading("Categoria", text="Categoría")
        self.tree.heading("Conteo", text="Conteo (N)")
        self.tree.heading("P_Empirica", text="P(x|C) Empírica")
        self.tree.heading("P_Laplace", text="P(x|C) Laplace (+1)")
        self.tree.heading("LR", text="Razón LR")
        self.tree.heading("Impacto", text="Impacto Predictivo")

        self.tree.column("Clase", width=120, anchor="center")
        self.tree.column("Variable", width=170, anchor="w")
        self.tree.column("Categoria", width=140, anchor="center")
        self.tree.column("Conteo", width=90, anchor="center")
        self.tree.column("P_Empirica", width=130, anchor="center")
        self.tree.column("P_Laplace", width=140, anchor="center")
        self.tree.column("LR", width=90, anchor="center")
        self.tree.column("Impacto", width=160, anchor="center")

        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Pie con nota matemática sobre Laplace
        lbl_formula = ttk.Label(
            parent,
            text="Fórmula Suavizado de Laplace: P_laplace(X_i = x | C = c) = [conteo(x, c) + 1] / [total(c) + K_i]   •   Priors: P(c) = [total(c) + 1] / [N + 2]",
            font=("Courier", 8, "bold"),
            foreground="#475569"
        )
        lbl_formula.pack(anchor="w", pady=(4, 0))

    def _actualizar_tabla(self):
        if not self.cpt_data:
            return

        # Limpiar
        for item in self.tree.get_children():
            self.tree.delete(item)

        filtro_var = self.var_filtro_var.get()
        filtro_clase = self.var_filtro_clase.get()
        filtro_impacto = self.var_filtro_impacto.get()

        cpt = self.cpt_data["cpt"]
        meta = self.cpt_data["metadatos"]

        # Recorrer ambas clases y variables
        for var, categorias_dict in cpt.items():
            if filtro_var != "Todas" and var != filtro_var:
                continue

            for cat, metricas in categorias_dict.items():
                lr = metricas["razon_verosimilitud"]
                impacto = metricas["impacto"]

                if filtro_impacto != "Todos" and impacto != filtro_impacto:
                    continue

                # Fila para Revenue = True
                if filtro_clase in ["Todas", "Compra (True)"]:
                    self.tree.insert("", tk.END, values=(
                        "Compra (True)",
                        var,
                        cat,
                        f"{metricas['conteo_true']:,}",
                        f"{metricas['prob_empirica_true']:.4f} ({metricas['prob_empirica_true']*100:.1f}%)",
                        f"{metricas['prob_laplace_true']:.4f} ({metricas['prob_laplace_true']*100:.1f}%)",
                        f"{lr:.3f}",
                        impacto
                    ))

                # Fila para Revenue = False
                if filtro_clase in ["Todas", "No Compra (False)"]:
                    self.tree.insert("", tk.END, values=(
                        "No Compra (False)",
                        var,
                        cat,
                        f"{metricas['conteo_false']:,}",
                        f"{metricas['prob_empirica_false']:.4f} ({metricas['prob_empirica_false']*100:.1f}%)",
                        f"{metricas['prob_laplace_false']:.4f} ({metricas['prob_laplace_false']*100:.1f}%)",
                        f"{lr:.3f}",
                        impacto
                    ))

    def _ver_matriz_cpt_completa(self):
        """Abre un visor formal de la tabla CPT calculada organizada por variables en formato matricial."""
        if not self.cpt_data:
            messagebox.showwarning("Aviso", "No hay datos CPT cargados.")
            return

        modal = tk.Toplevel(self.root)
        modal.title("Visor Formal de Tablas de Probabilidad Condicional (CPT)")
        modal.geometry("920x650")
        modal.minsize(800, 520)
        modal.transient(self.root)

        top_modal = ttk.Frame(modal, padding="12 10 12 6")
        top_modal.pack(fill=tk.X)

        ttk.Label(top_modal, text="Tablas de Probabilidad Condicional CPT Calculadas (+ Laplace)", font=("Arial", 13, "bold"), foreground="#1e3a8a").pack(anchor="w")
        ttk.Label(top_modal, text="Representación matricial de P(Atributo | Revenue) con suma axiomática unitaria por columna", font=("Arial", 9), foreground="#64748b").pack(anchor="w")

        # Selector de Modo: Laplace vs Empírico
        modo_frame = ttk.Frame(modal, padding="10 4 10 4")
        modo_frame.pack(fill=tk.X)

        ttk.Label(modo_frame, text="Modo de Visualización: ", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        var_modo = tk.StringVar(value="Laplace")
        combo_modo = ttk.Combobox(modo_frame, textvariable=var_modo, values=["Probabilidad Laplace (+1)", "Probabilidad Empírica", "Conteos Absolutos (N)"], state="readonly", width=26)
        combo_modo.pack(side=tk.LEFT, padx=6)

        notebook_cpt = ttk.Notebook(modal)
        notebook_cpt.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        meta = self.cpt_data["metadatos"]
        cpt = self.cpt_data["cpt"]

        tablas_dict = {}

        for var in ["VisitorType", "ProductRelated", "ProductRelated_Duration"]:
            tab = ttk.Frame(notebook_cpt, padding=8)
            notebook_cpt.add(tab, text=f"📌 {var}")

            cols_cpt = ("Categoria", "P(x | No Compra)", "P(x | Compra)", "Razon_LR", "Impacto")
            tree_mat = ttk.Treeview(tab, columns=cols_cpt, show="headings", height=6)
            tree_mat.heading("Categoria", text=f"Categoría ({var})")
            tree_mat.heading("P(x | No Compra)", text="P(x | Revenue=False)")
            tree_mat.heading("P(x | Compra)", text="P(x | Revenue=True)")
            tree_mat.heading("Razon_LR", text="Razón Verosimilitud (LR)")
            tree_mat.heading("Impacto", text="Impacto en Decisión")

            tree_mat.column("Categoria", width=180, anchor="w")
            tree_mat.column("P(x | No Compra)", width=170, anchor="center")
            tree_mat.column("P(x | Compra)", width=170, anchor="center")
            tree_mat.column("Razon_LR", width=150, anchor="center")
            tree_mat.column("Impacto", width=160, anchor="center")

            tree_mat.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
            tablas_dict[var] = tree_mat

        def _llenar_matrices():
            modo = var_modo.get()
            for var, tree_mat in tablas_dict.items():
                for item in tree_mat.get_children():
                    tree_mat.delete(item)

                suma_false = 0.0
                suma_true = 0.0

                for cat, d in cpt[var].items():
                    lr = d["razon_verosimilitud"]
                    imp = d["impacto"]

                    if "Laplace" in modo:
                        p_f = d["prob_laplace_false"]
                        p_t = d["prob_laplace_true"]
                        suma_false += p_f
                        suma_true += p_t
                        txt_f = f"{p_f:.6f} ({p_f*100:.2f}%)"
                        txt_t = f"{p_t:.6f} ({p_t*100:.2f}%)"
                    elif "Empírica" in modo:
                        p_f = d["prob_empirica_false"]
                        p_t = d["prob_empirica_true"]
                        suma_false += p_f
                        suma_true += p_t
                        txt_f = f"{p_f:.6f} ({p_f*100:.2f}%)"
                        txt_t = f"{p_t:.6f} ({p_t*100:.2f}%)"
                    else:
                        c_f = d["conteo_false"]
                        c_t = d["conteo_true"]
                        suma_false += c_f
                        suma_true += c_t
                        txt_f = f"{c_f:,} sesiones"
                        txt_t = f"{c_t:,} sesiones"

                    tree_mat.insert("", tk.END, values=(cat, txt_f, txt_t, f"{lr:.3f}", imp))

                # Fila de validación
                if "Conteos" in modo:
                    tree_mat.insert("", tk.END, values=("Σ TOTAL SESIONES", f"{int(suma_false):,}", f"{int(suma_true):,}", "-", "[VALIDADO]"))
                else:
                    tree_mat.insert("", tk.END, values=("Σ PROBABILIDADES", f"{suma_false:.4f} (100.0%)", f"{suma_true:.4f} (100.0%)", "-", "[AXIOMA OK]"))

        combo_modo.bind("<<ComboboxSelected>>", lambda e: _llenar_matrices())
        _llenar_matrices()

        # Nota al pie
        lbl_pie = ttk.Label(
            modal,
            text=f"Muestras de Train: {meta['total_muestras_train']:,}  •  Prior P(Compra) Laplace = {meta['priors_laplace']['True']:.4f} ({meta['priors_laplace']['True']*100:.2f}%)  •  Prior P(No Compra) Laplace = {meta['priors_laplace']['False']:.4f} ({meta['priors_laplace']['False']*100:.2f}%)",
            font=("Arial", 9, "bold"),
            foreground="#0f766e"
        )
        lbl_pie.pack(anchor="w", padx=12, pady=(0, 8))

    def _forzar_recalculo(self):
        if messagebox.askyesno(
            "Confirmar Recálculo",
            "¿Desea recalcular todas las tablas de probabilidad desde 'train_categorico.csv' y sobrescribir el archivo de persistencia?"
        ):
            self._cargar_o_calcular_cpt(forzar_recalculo=True)
            self.lbl_status.config(text=self.origen_datos_txt)
            self._actualizar_tabla()
            messagebox.showinfo("Éxito", "Tablas CPT recalculadas y persistidas correctamente.")

    def _ver_metadatos_json(self):
        if not self.cpt_data:
            return
        meta = self.cpt_data.get("metadatos", {})
        texto_meta = (
            f"• Fecha de Generación: {meta.get('fecha_generacion', 'N/A')}\n"
            f"• Dataset Origen:       {meta.get('dataset_origen', 'N/A')}\n"
            f"• Muestras de Train:    {meta.get('total_muestras_train', 0):,}\n\n"
            f"• Conteo Revenue=False: {meta.get('conteo_revenue_false', 0):,}\n"
            f"• Conteo Revenue=True:  {meta.get('conteo_revenue_true', 0):,}\n\n"
            f"• Prior Empírico False: {meta.get('priors_empiricos', {}).get('False', 0):.6f}\n"
            f"• Prior Empírico True:  {meta.get('priors_empiricos', {}).get('True', 0):.6f}\n\n"
            f"• Prior Laplace False:  {meta.get('priors_laplace', {}).get('False', 0):.6f}\n"
            f"• Prior Laplace True:   {meta.get('priors_laplace', {}).get('True', 0):.6f}\n\n"
            f"• Variables Analizadas: {', '.join(meta.get('variables_incluidas', []))}\n"
            f"• Cardinalidades K_i:   {meta.get('cardinalidades', {})}\n\n"
            f"• Archivo Persistido:   {CPT_LAPLACE_JSON}"
        )

        modal = tk.Toplevel(self.root)
        modal.title("Metadatos de Persistencia (JSON)")
        modal.geometry("580x420")
        modal.transient(self.root)

        txt = tk.Text(modal, wrap="word", font=("Courier", 10), bg="#f8fafc", padx=10, pady=10)
        txt.insert("1.0", texto_meta)
        txt.config(state="disabled")
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _exportar_csv_analisis(self):
        out_csv = ARTIFACTS_DIR / "analisis_estadistico_cpt.csv"
        if not out_csv.exists() and self.cpt_data:
            guardar_tablas_cpt(self.cpt_data, CPT_LAPLACE_JSON)

        messagebox.showinfo(
            "Archivo CSV de Análisis",
            f"El archivo tabular de análisis se encuentra disponible en:\n{out_csv}\n\nPuede abrirlo en Excel o procesarlo con Pandas."
        )


def main():
    root = tk.Tk()
    app = RedNaiveBayesApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
