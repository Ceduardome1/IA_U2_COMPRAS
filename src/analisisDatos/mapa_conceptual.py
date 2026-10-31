#!/usr/bin/env python3
"""
mapa_conceptual.py
Tipo: INTERFAZ GRÁFICA (GUI)
Responsable: JAIME / CESAR
Entregable: Apartado 1 - Mapa Conceptual con Grafos NetworkX

Objetivos y Especificaciones:
1. Representación formal del dominio del problema mediante grafos dirigidos con NetworkX:
   - Nodo central: Intención de Compra en Sesión Web (Problema de Clasificación Binaria).
   - Variables de entrada (VisitorType, ProductRelated, ProductRelated_Duration).
   - Transformaciones (Discretización P33/P67, One-Hot Encoding, StandardScaler).
   - Paradigmas y modelos (Teorema de Bayes univariado, Red Naive Bayes, Random Forest, MLP).
   - Inferencia y decisión (Probabilidad a posteriori, Umbral 0.50, Predicción final).
   - Evaluación y desempeño (Matriz de Confusión, Recall, F1-Score).
2. Proposiciones y conectores textuales legibles en las aristas.
3. Interfaz interactiva amigable:
   - Panel de visualización con Canvas Matplotlib embebido en Tkinter y barra de navegación (Zoom, Pan, Guardar).
   - Panel lateral con Ficha Técnica Detallada del concepto seleccionado (Definición teórica, Implementación técnica, Fórmulas y Conexiones).
   - Filtros por paradigma (Completo, Flujo Bayesiano, Flujo Machine Learning Supervisado).
   - Botón directo para lanzar la vista de Distribución de Datos.

Uso:
    python src/analisisDatos/mapa_conceptual.py
"""

import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import ARTIFACTS_DIR


class MapaConceptualApp:
    def __init__(self, root):
        self.root = root
        if hasattr(self.root, "title"):
            self.root.title("Apartado 1: Mapa Conceptual del Problema (NetworkX)")
        if hasattr(self.root, "geometry"):
            self.root.geometry("1280x820")
        if hasattr(self.root, "minsize"):
            self.root.minsize(1050, 700)

        self.filtro_actual = "Todos los Componentes"
        self.nodo_seleccionado = "CENTRAL"
        self.mostrar_etiquetas_aristas = tk.BooleanVar(value=True)

        self._configurar_estilos()
        self._construir_grafo()
        self._crear_interfaz()
        self._actualizar_ficha_nodo("CENTRAL")
        self._dibujar_grafo()

    def _configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Arial", 14, "bold"), foreground="#1e293b")
        style.configure("Subheader.TLabel", font=("Arial", 10), foreground="#64748b")
        style.configure("Category.TLabel", font=("Arial", 10, "bold"), foreground="#0f766e")
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), padding=6)
        style.configure("Action.TButton", font=("Arial", 9), padding=5)

    def _construir_grafo(self):
        self.G = nx.DiGraph()

        # Metadatos de cada nodo: label, categoria, color, pos, descripcion, tecnico, formula
        self.nodos_info = {
            "CENTRAL": {
                "label": "INTENCIÓN DE COMPRA\n(Sesión Web)",
                "categoria": "Problema Central",
                "color": "#1e3a8a",  # Azul oscuro
                "textColor": "#ffffff",
                "pos": (0.50, 1.00),
                "desc": "Problema central de clasificación binaria: predecir si una sesión de comercio electrónico terminará en una transacción efectiva ('Revenue' = True) o abandono sin compra ('Revenue' = False).",
                "tecnico": "Dataset UCI Online Shoppers (12,330 sesiones). Variable objetivo con desbalance 84.53% False vs 15.47% True. Partición estratificada 80/20.",
                "formula": "Clase = Revenue ∈ {True, False}"
            },
            "VAR_IN": {
                "label": "Variables Predictoras\n(Navegación)",
                "categoria": "Entradas del Sistema",
                "color": "#047857",  # Verde esmeralda
                "textColor": "#ffffff",
                "pos": (0.50, 0.84),
                "desc": "Conjunto seleccionado de 3 atributos que capturan la identidad del usuario y su comportamiento transaccional en la tienda.",
                "tecnico": "Definidas en 'src/config.py' como SELECTED_FEATURES: VisitorType, ProductRelated y ProductRelated_Duration.",
                "formula": "X = [VisitorType, ProductRelated, ProductRelated_Duration]"
            },
            "VISITOR": {
                "label": "VisitorType\n(Categórica)",
                "categoria": "Variables de Entrada",
                "color": "#059669",
                "textColor": "#ffffff",
                "pos": (0.16, 0.69),
                "desc": "Tipo de visitante: 'Returning_Visitor' (85.57%), 'New_Visitor' (13.74%), 'Other' (0.69%). Presenta tasas de conversión notablemente diferenciadas (24.91% nuevos vs 13.93% recurrentes).",
                "tecnico": "Analizada analíticamente en 'src/bayes/cli_teorema_bayes.py' y codificada mediante One-Hot para ML.",
                "formula": "VisitorType ∈ {Returning_Visitor, New_Visitor, Other}"
            },
            "PROD": {
                "label": "ProductRelated\n(Páginas de Producto)",
                "categoria": "Variables de Entrada",
                "color": "#059669",
                "textColor": "#ffffff",
                "pos": (0.50, 0.69),
                "desc": "Número entero de páginas de producto visualizadas en la sesión. Presenta fuerte sesgo positivo (media 31.7, mediana 18, máximo 705).",
                "tecnico": "Discretizada en 3 rangos por percentiles en train (P33=10, P67=29) para Naive Bayes. Continua para Random Forest y MLP.",
                "formula": "ProductRelated ∈ ℕ₀"
            },
            "DUR": {
                "label": "ProductRelated_Duration\n(Segundos)",
                "categoria": "Variables de Entrada",
                "color": "#059669",
                "textColor": "#ffffff",
                "pos": (0.84, 0.69),
                "desc": "Tiempo total en segundos en páginas de producto. Refleja el compromiso del usuario (mediana ~598s, P33=285.9s, P67=1107.2s).",
                "tecnico": "Discretizada para Naive Bayes. Estandarizada con StandardScaler para MLP.",
                "formula": "ProductRelated_Duration ∈ ℝ⁺"
            },
            "DISC": {
                "label": "Discretización P33/P67\n(Bajo / Medio / Alto)",
                "categoria": "Preprocesamiento",
                "color": "#d97706",  # Ámbar
                "textColor": "#ffffff",
                "pos": (0.33, 0.53),
                "desc": "Transformación no paramétrica por terciles calculada exclusivamente sobre el conjunto de entrenamiento (80%) para evitar fuga de información (data leakage).",
                "tecnico": "Genera 'data/train_categorico.csv' y 'data/test_categorico.csv'. Persistido en 'artifacts/cortes_percentiles.json'.",
                "formula": "x ≤ P33 → 'bajo' | P33 < x ≤ P67 → 'medio' | x > P67 → 'alto'"
            },
            "SCALER": {
                "label": "StandardScaler &\nOne-Hot Encoding",
                "categoria": "Preprocesamiento",
                "color": "#d97706",
                "textColor": "#ffffff",
                "pos": (0.75, 0.53),
                "desc": "Preprocesamiento numérico para algoritmos supervisados: VisitorType se expande en 3 variables dummy binarias y las variables continuas se centran a media 0 y varianza unitaria.",
                "tecnico": "Genera 'data/train_numerico.csv'. Escalador guardado en 'artifacts/scaler_mlp.joblib'.",
                "formula": "z = (x - μ_train) / σ_train"
            },
            "BAYES_UNI": {
                "label": "Teorema de Bayes\n(Conteos Univariados)",
                "categoria": "Modelos / Paradigmas",
                "color": "#7c3aed",  # Púrpura
                "textColor": "#ffffff",
                "pos": (0.08, 0.36),
                "desc": "Cálculo analítico directo de probabilidad condicional sobre VisitorType usando conteos exactos de la tabla de contingencia 2x2.",
                "tecnico": "Implementado en 'src/bayes/cli_teorema_bayes.py' y 'teorema_bayes.py'. Valida la identidad Bayesiana: analítica vs empírica directa.",
                "formula": "P(C|R) = [P(R|C) · P(C)] / P(R)"
            },
            "NAIVE_BAYES": {
                "label": "Red Naive Bayes\n(+ Suavizado Laplace)",
                "categoria": "Modelos / Paradigmas",
                "color": "#7c3aed",
                "textColor": "#ffffff",
                "pos": (0.33, 0.36),
                "desc": "Clasificador probabilístico multivariable que asume independencia condicional entre las 3 características dado el estado de compra, con corrección +1 de Laplace para probabilidades cero.",
                "tecnico": "Implementado en 'src/redBayesiana/cli_pipeline_red_naive_bayes.py' y 'red_naive_bayes.py'. Almacena CPTs en 'artifacts/tablas_cpt_laplace.json'.",
                "formula": "P(c | x) ∝ P(c) ∏ P(xᵢ | c) ; P_laplace(xᵢ|c) = (count + 1) / (N_c + |Kᵢ|)"
            },
            "RF": {
                "label": "Random Forest\n(Ensamble Árboles)",
                "categoria": "Modelos / Paradigmas",
                "color": "#4f46e5",  # Índigo
                "textColor": "#ffffff",
                "pos": (0.64, 0.36),
                "desc": "Ensamble supervisado de árboles de decisión con particionado recursivo no lineal, robusto al desbalance mediante balanceo de pesos de clase ('balanced').",
                "tecnico": "Entrenado en 'src/modelosML/randoomForest/cli_entrenar_rf.py'. Modelo persistido en 'artifacts/random_forest.joblib'.",
                "formula": "P(C | x) = (1 / B) ∑_{b=1}^B T_b(x)"
            },
            "MLP": {
                "label": "Perceptrón Multicapa\n(Red Neuronal MLP)",
                "categoria": "Modelos / Paradigmas",
                "color": "#4f46e5",
                "textColor": "#ffffff",
                "pos": (0.90, 0.36),
                "desc": "Red neuronal artificial feed-forward entrenada con Backpropagation, activación ReLU, regularización L2 y Early Stopping para evitar sobreajuste.",
                "tecnico": "Entrenado en 'src/modelosML/mLP/cli_entrenar_mlp.py'. Pesos y arquitectura guardados en 'artifacts/mlp_model.joblib'.",
                "formula": "y = σ( W₂ · ReLU(W₁ x + b₁) + b₂ )"
            },
            "PROB": {
                "label": "Probabilidad Posterior\nP(Revenue=True | Evidencia)",
                "categoria": "Inferencia & Decisión",
                "color": "#c026d3",  # Fucsia
                "textColor": "#ffffff",
                "pos": (0.50, 0.20),
                "desc": "Puntuación de confianza continua en [0.0, 1.0] estimada por cada paradigma predictivo a partir de la evidencia del usuario.",
                "tecnico": "Evaluada homogéneamente en los 4 enfoques para permitir una comparación justa en el set de prueba.",
                "formula": "Score_p = P(Revenue = True | x) ∈ [0, 1]"
            },
            "UMBRAL": {
                "label": "Regla de Decisión\n(Umbral de Corte = 0.50)",
                "categoria": "Inferencia & Decisión",
                "color": "#dc2626",  # Rojo
                "textColor": "#ffffff",
                "pos": (0.50, 0.08),
                "desc": "Regla de decisión fija de 0.50 según las especificaciones del proyecto: si la probabilidad supera 0.50 se predice Compra; en caso contrario, No Compra.",
                "tecnico": "Permite contrastar el impacto del desbalance severo en la sensibilidad (Recall) de cada modelo bajo un umbral simétrico.",
                "formula": "y_pred = 1 si P(Revenue=True | x) ≥ 0.50 sino 0"
            },
            "PRED": {
                "label": "Predicción Final\n(Compra vs No Compra)",
                "categoria": "Inferencia & Decisión",
                "color": "#b91c1c",
                "textColor": "#ffffff",
                "pos": (0.20, 0.04),
                "desc": "Etiqueta categórica final generada para la sesión: 'Compra (True)' o 'No Compra (False)'.",
                "tecnico": "Sujeta a validación en los casos de prueba de demostración (parcial, completa, inválida).",
                "formula": "Clase_Final ∈ {'Compra', 'No Compra'}"
            },
            "METRICAS": {
                "label": "Evaluación Comparativa\n(Test 20%: F1 / Recall / Matriz)",
                "categoria": "Evaluación y Desempeño",
                "color": "#0f766e",  # Teal oscuro
                "textColor": "#ffffff",
                "pos": (0.80, 0.04),
                "desc": "Módulo de evaluación en el 20% no visto (2,466 muestras) comparando los 4 modelos con una referencia base mayoritaria.",
                "tecnico": "Calculado en 'src/comparacion/cli_evaluar_todos.py' y visualizado en 'comparativa_modelos.py'.",
                "formula": "F1 = 2·(Prec · Rec) / (Prec + Rec) ; Acc = (TP + TN) / Total"
            }
        }

        for n_id, data in self.nodos_info.items():
            self.G.add_node(n_id, **data)

        # Aristas con sus proposiciones
        self.aristas = [
            ("CENTRAL", "VAR_IN", "se caracteriza por"),
            ("VAR_IN", "VISITOR", "comprende"),
            ("VAR_IN", "PROD", "comprende"),
            ("VAR_IN", "DUR", "comprende"),
            ("VISITOR", "BAYES_UNI", "analizado por"),
            ("VISITOR", "NAIVE_BAYES", "atributo nominal"),
            ("PROD", "DISC", "discretizado"),
            ("DUR", "DISC", "discretizado"),
            ("DISC", "NAIVE_BAYES", "categorías (bajo/med/alto)"),
            ("VISITOR", "SCALER", "one-hot"),
            ("PROD", "SCALER", "normalización continua"),
            ("DUR", "SCALER", "normalización continua"),
            ("SCALER", "RF", "vector de features"),
            ("SCALER", "MLP", "vector escalado"),
            ("BAYES_UNI", "PROB", "calcula"),
            ("NAIVE_BAYES", "PROB", "estima"),
            ("RF", "PROB", "predice"),
            ("MLP", "PROB", "estima"),
            ("PROB", "UMBRAL", "evaluada bajo"),
            ("UMBRAL", "PRED", "determina"),
            ("PRED", "METRICAS", "contrastada en test"),
            ("UMBRAL", "METRICAS", "regla fija 0.50")
        ]

        for u, v, label in self.aristas:
            self.G.add_edge(u, v, label=label)

    def _crear_interfaz(self):
        # Barra superior con título, filtros y acciones
        top_bar = ttk.Frame(self.root, padding="10 8 10 8")
        top_bar.pack(fill=tk.X)

        info_box = ttk.Frame(top_bar)
        info_box.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(info_box, text="Mapa Conceptual del Problema (Grafo NetworkX)", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            info_box,
            text="Red de Conceptos, Paradigmas de Estimación, Preprocesamiento y Reglas de Decisión",
            style="Subheader.TLabel"
        ).pack(anchor="w")

        btn_box = ttk.Frame(top_bar)
        btn_box.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(btn_box, text="Filtro de Flujo: ", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=3)
        self.combo_filtro = ttk.Combobox(
            btn_box,
            values=["Todos los Componentes", "Flujo Probabilístico (Bayes)", "Flujo Machine Learning (RF/MLP)", "Preprocesamiento y Entradas"],
            state="readonly",
            width=28
        )
        self.combo_filtro.set(self.filtro_actual)
        self.combo_filtro.bind("<<ComboboxSelected>>", self._al_cambiar_filtro)
        self.combo_filtro.pack(side=tk.LEFT, padx=5)

        chk_etiquetas = ttk.Checkbutton(
            btn_box,
            text="Ver Proposiciones en Aristas",
            variable=self.mostrar_etiquetas_aristas,
            command=self._dibujar_grafo
        )
        chk_etiquetas.pack(side=tk.LEFT, padx=8)

        btn_distribucion = ttk.Button(
            btn_box,
            text="📊 Ver Distribución de Datos",
            style="Action.TButton",
            command=self._abrir_distribucion_datos
        )
        btn_distribucion.pack(side=tk.LEFT, padx=4)

        btn_guardar = ttk.Button(
            btn_box,
            text="💾 Guardar PNG",
            style="Accent.TButton",
            command=self._guardar_imagen_grafo
        )
        btn_guardar.pack(side=tk.LEFT, padx=4)

        # Contenedor central dividido en panel izquierdo (Grafo) y panel derecho (Ficha Técnica)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Panel Izquierdo: Grafo Matplotlib
        frame_grafo_container = ttk.Frame(main_paned)
        main_paned.add(frame_grafo_container, weight=7)

        self.fig = plt.Figure(figsize=(8.5, 7.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor("#f8fafc")
        self.ax.set_facecolor("#f8fafc")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_grafo_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(frame_grafo_container)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Conectar eventos de clic en el lienzo
        self.canvas.mpl_connect("button_press_event", self._al_hacer_clic_lienzo)

        # Panel Derecho: Ficha Técnica y Selector de Conceptos
        frame_lateral = ttk.Frame(main_paned, width=380, padding="10 8 10 8")
        main_paned.add(frame_lateral, weight=3)

        ttk.Label(frame_lateral, text="Selector de Concepto:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 4))
        
        # Lista de conceptos seleccionables
        self.lista_conceptos = tk.Listbox(frame_lateral, height=6, font=("Arial", 9), selectmode=tk.SINGLE, exportselection=False)
        for n_id, n_data in self.nodos_info.items():
            nombre_limpio = n_data["label"].replace("\n", " ")
            self.lista_conceptos.insert(tk.END, f"{n_data['categoria']}: {nombre_limpio}")
        self.lista_conceptos.bind("<<ListboxSelect>>", self._al_seleccionar_lista)
        self.lista_conceptos.pack(fill=tk.X, pady=(0, 8))
        self.lista_conceptos.selection_set(0)

        # Cuadro de Ficha Técnica
        frame_ficha = ttk.LabelFrame(frame_lateral, text=" Ficha Técnica del Concepto ", padding=10)
        frame_ficha.pack(fill=tk.BOTH, expand=True)

        self.lbl_ficha_titulo = ttk.Label(frame_ficha, text="", font=("Arial", 12, "bold"), foreground="#1e3a8a", wraplength=340)
        self.lbl_ficha_titulo.pack(anchor="w", pady=(0, 2))

        self.lbl_ficha_categoria = ttk.Label(frame_ficha, text="", style="Category.TLabel")
        self.lbl_ficha_categoria.pack(anchor="w", pady=(0, 8))

        # Texto enriquecido para detalles
        self.txt_ficha = tk.Text(frame_ficha, wrap="word", font=("Arial", 9), bg="#ffffff", fg="#1e293b", relief="solid", borderwidth=1, padx=8, pady=8)
        self.txt_ficha.pack(fill=tk.BOTH, expand=True)

        # Leyenda de categorías al pie del panel lateral
        frame_leyenda = ttk.LabelFrame(frame_lateral, text=" Leyenda de Categorías ", padding=8)
        frame_leyenda.pack(fill=tk.X, pady=(8, 0))

        categorias_colores = [
            ("Problema Central", "#1e3a8a"),
            ("Variables de Entrada", "#059669"),
            ("Preprocesamiento", "#d97706"),
            ("Modelos / Paradigmas", "#7c3aed"),
            ("Inferencia & Decisión", "#dc2626"),
            ("Evaluación", "#0f766e")
        ]
        for cat, col in categorias_colores:
            row = ttk.Frame(frame_leyenda)
            row.pack(fill=tk.X, pady=1)
            tk.Canvas(row, width=14, height=14, bg=col, highlightthickness=0).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Label(row, text=cat, font=("Arial", 8)).pack(side=tk.LEFT)

    def _obtener_subgrafo_activo(self):
        filtro = self.combo_filtro.get()
        if filtro == "Flujo Probabilístico (Bayes)":
            nodos_permitidos = {"CENTRAL", "VAR_IN", "VISITOR", "PROD", "DUR", "DISC", "BAYES_UNI", "NAIVE_BAYES", "PROB", "UMBRAL", "PRED", "METRICAS"}
        elif filtro == "Flujo Machine Learning (RF/MLP)":
            nodos_permitidos = {"CENTRAL", "VAR_IN", "VISITOR", "PROD", "DUR", "SCALER", "RF", "MLP", "PROB", "UMBRAL", "PRED", "METRICAS"}
        elif filtro == "Preprocesamiento y Entradas":
            nodos_permitidos = {"CENTRAL", "VAR_IN", "VISITOR", "PROD", "DUR", "DISC", "SCALER"}
        else:
            nodos_permitidos = set(self.G.nodes)

        return self.G.subgraph(nodos_permitidos)

    def _dibujar_grafo(self):
        self.ax.clear()
        subG = self._obtener_subgrafo_activo()

        pos = {n: self.nodos_info[n]["pos"] for n in subG.nodes}

        # Dibujar aristas
        nx.draw_networkx_edges(
            subG,
            pos,
            ax=self.ax,
            edge_color="#94a3b8",
            arrows=True,
            arrowsize=14,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.03",
            width=1.5,
            alpha=0.85
        )

        # Dibujar etiquetas de aristas si está activado
        if self.mostrar_etiquetas_aristas.get():
            edge_labels = {(u, v): d.get("label", "") for u, v, d in subG.edges(data=True)}
            nx.draw_networkx_edge_labels(
                subG,
                pos,
                edge_labels=edge_labels,
                ax=self.ax,
                font_size=7,
                font_color="#334155",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9, linewidth=0.7)
            )

        # Dibujar cada nodo con su caja y texto
        for n in subG.nodes:
            info = self.nodos_info[n]
            x, y = pos[n]
            es_seleccionado = (n == self.nodo_seleccionado)

            borde_color = "#f59e0b" if es_seleccionado else "#0f172a"
            borde_ancho = 2.8 if es_seleccionado else 1.0

            bbox_props = dict(
                boxstyle="round,pad=0.45,rounding_size=0.25",
                facecolor=info["color"],
                edgecolor=borde_color,
                linewidth=borde_ancho,
                alpha=0.95
            )

            self.ax.text(
                x, y,
                info["label"],
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold" if es_seleccionado else "normal",
                color=info["textColor"],
                bbox=bbox_props
            )

        self.ax.set_xlim(-0.02, 1.02)
        self.ax.set_ylim(-0.02, 1.08)
        self.ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _al_cambiar_filtro(self, event=None):
        self._dibujar_grafo()

    def _al_seleccionar_lista(self, event=None):
        sel = self.lista_conceptos.curselection()
        if sel:
            idx = sel[0]
            node_key = list(self.nodos_info.keys())[idx]
            self.nodo_seleccionado = node_key
            self._actualizar_ficha_nodo(node_key)
            self._dibujar_grafo()

    def _al_hacer_clic_lienzo(self, event):
        if event.xdata is None or event.ydata is None:
            return

        click_x, click_y = event.xdata, event.ydata
        subG = self._obtener_subgrafo_activo()

        nodo_mas_cercano = None
        dist_min = float("inf")

        for n in subG.nodes:
            nx_pos, ny_pos = self.nodos_info[n]["pos"]
            # Distancia normalizada
            dist = ((click_x - nx_pos) ** 2 + ((click_y - ny_pos) * 1.5) ** 2) ** 0.5
            if dist < dist_min and dist < 0.09:
                dist_min = dist
                nodo_mas_cercano = n

        if nodo_mas_cercano:
            self.nodo_seleccionado = nodo_mas_cercano
            # Sincronizar listbox
            idx = list(self.nodos_info.keys()).index(nodo_mas_cercano)
            self.lista_conceptos.selection_clear(0, tk.END)
            self.lista_conceptos.selection_set(idx)
            self.lista_conceptos.see(idx)

            self._actualizar_ficha_nodo(nodo_mas_cercano)
            self._dibujar_grafo()

    def _actualizar_ficha_nodo(self, node_key):
        info = self.nodos_info.get(node_key, {})
        if not info:
            return

        self.lbl_ficha_titulo.config(text=info["label"].replace("\n", " "))
        self.lbl_ficha_categoria.config(text=f"Categoría: {info['categoria']}")

        # Encontrar conexiones entrantes y salientes
        entrantes = [f"• {self.nodos_info[u]['label'].replace(chr(10), ' ')} ({d.get('label', '')})" for u, v, d in self.G.in_edges(node_key, data=True)]
        salientes = [f"• {self.nodos_info[v]['label'].replace(chr(10), ' ')} ({d.get('label', '')})" for u, v, d in self.G.out_edges(node_key, data=True)]

        texto = (
            f"1. DEFINICIÓN TEÓRICA:\n{info['desc']}\n\n"
            f"2. IMPLEMENTACIÓN TÉCNICA:\n{info['tecnico']}\n\n"
            f"3. FORMULACIÓN MATEMÁTICA / REGLA:\n{info['formula']}\n\n"
            f"4. CONEXIONES ENTRANTES:\n" + ("\n".join(entrantes) if entrantes else "• (Nodo Raíz / Inicial)") + "\n\n"
            f"5. CONEXIONES SALIENTES:\n" + ("\n".join(salientes) if salientes else "• (Nodo Terminal / Salida)")
        )

        self.txt_ficha.config(state="normal")
        self.txt_ficha.delete("1.0", tk.END)
        self.txt_ficha.insert("1.0", texto)
        self.txt_ficha.config(state="disabled")

    def _abrir_distribucion_datos(self):
        script_dist = BASE_DIR / "src" / "analisisDatos" / "distribucion_datos.py"
        if not script_dist.exists():
            messagebox.showerror("Error", f"No se encontró el script de distribución en:\n{script_dist}")
            return
        subprocess.Popen([sys.executable, str(script_dist)])

    def _guardar_imagen_grafo(self):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ARTIFACTS_DIR / "mapa_conceptual_networkx.png"
        try:
            self.fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=self.fig.get_facecolor())
            messagebox.showinfo(
                "Exportación Exitosa",
                f"El mapa conceptual se ha guardado exitosamente en alta resolución en:\n{out_path}"
            )
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar la imagen:\n{e}")


def main():
    root = tk.Tk()
    app = MapaConceptualApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
