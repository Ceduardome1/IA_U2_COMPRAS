import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class ExploracionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apartado 1: Exploración y Mapa Conceptual")
        self.root.geometry("1000x700")

        # Intentar cargar los datos preparados por César
        ruta_datos = os.path.join("..", "data", "train_numerico.csv")
        try:
            self.df = pd.read_csv(ruta_datos)
        except FileNotFoundError:
            # Fallback por si ejecutas desde la raíz del proyecto
            ruta_datos = os.path.join("data", "train_numerico.csv")
            try:
                self.df = pd.read_csv(ruta_datos)
            except FileNotFoundError:
                self.df = None

        self.crear_interfaz()

    def crear_interfaz(self):
        # Panel con pestañas para organizar la ventana de forma limpia
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Pestana 1: Grafica y Resumen de Variables
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="1. Distribución y Variables")

        lbl_titulo = tk.Label(tab1, text="Exploración del Dataset (UCI Online Shoppers)", font=("Arial", 14, "bold"))
        lbl_titulo.pack(pady=10)

        # Marco superior para grafica y texto descriptivo
        frame_top = ttk.Frame(tab1)
        frame_top.pack(fill="both", expand=True, padx=10, pady=5)

        # Columna Izquierda: Grafica
        frame_grafica = ttk.Frame(frame_top)
        frame_grafica.pack(side="left", fill="both", expand=True)

        if self.df is not None and 'Revenue' in self.df.columns:
            fig, ax = plt.subplots(figsize=(5, 4))
            conteo = self.df['Revenue'].value_counts()
            ax.bar(["No Compra (False)", "Compra (True)"], conteo.values, color=['#e74c3c', '#2ecc71'])
            ax.set_ylabel("Cantidad de Sesiones")
            ax.set_title("Desbalance de la Clase Objetiva (Revenue)")
            
            # Mostrar porcentajes sobre las barras
            total = len(self.df)
            for i, v in enumerate(conteo.values):
                porcentaje = (v / total) * 100
                ax.text(i, v + (total * 0.01), f"{v} ({porcentaje:.1f}%)", ha='center', fontweight='bold')

            canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            tk.Label(frame_grafica, text="No se pudo cargar train_numerico.csv").pack()

        # Columna Derecha: Resumen de Variables de Entrada
        frame_info = ttk.LabelFrame(frame_top, text=" Variables de Entrada ")
        frame_info.pack(side="right", fill="both", expand=True, padx=10)

        texto_vars = (
            "• VisitorType (Categorica):\n"
            "  Tipo de usuario que navega (Returning_Visitor, New_Visitor, Other).\n\n"
            "• ProductRelated (Numerica Entera):\n"
            "  Cantidad total de paginas de productos visitadas durante la sesion.\n\n"
            "• ProductRelated_Duration (Numerica Continua):\n"
            "  Tiempo total (en segundos) que el usuario duro en paginas de productos.\n\n"
            "• Revenue (Variable Objetivo / Salida):\n"
            "  Indica si la sesion de navegacion concluyo en una compra (True / False)."
        )
        lbl_info = tk.Label(frame_info, text=texto_vars, justify="left", font=("Arial", 10), anchor="nw")
        lbl_info.pack(fill="both", expand=True, padx=10, pady=10)

        # Pestana 2: Mapa Conceptual
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="2. Mapa Conceptual del Problema")

        lbl_titulo_mapa = tk.Label(tab2, text="Representación Conceptual del Dominio", font=("Arial", 14, "bold"))
        lbl_titulo_mapa.pack(pady=10)

        texto_mapa = (
            "[ SESIÓN DE NAVEGACIÓN EN TIENDA EN LÍNEA ]\n"
            "       │\n"
            "       ├── (se caracteriza por) ──► [ VARIABLES DE ENTRADA ]\n"
            "       │                                  │\n"
            "       │                                  ├──► Tipo de Visitante (VisitorType)\n"
            "       │                                  ├──► Páginas de Producto (ProductRelated)\n"
            "       │                                  └──► Duración en Productos (ProductRelated_Duration)\n"
            "       │\n"
            "       ├── (se procesan mediante) ──► [ PARADIGMAS DE ESTIMACIÓN / ML ]\n"
            "       │                                  │\n"
            "       │                                  ├──► Teorema de Bayes (Cálculo univariado directo)\n"
            "       │                                  ├──► Red Naive Bayes (Atributos discretizados: bajo/medio/alto)\n"
            "       │                                  ├──► Random Forest (Atributos numéricos continuos)\n"
            "       │                                  └──► Red Neuronal MLP (Atributos numéricos escalados)\n"
            "       │\n"
            "       └── (permiten estimar) ────► [ INTENCIÓN DE COMPRA (Revenue) ] ──► (Clasificación: Compra / No compra)\n"
        )

        lbl_mapa = tk.Label(tab2, text=texto_mapa, justify="left", font=("Courier", 10, "bold"), bg="#f8f9fa", relief="sunken")
        lbl_mapa.pack(fill="both", expand=True, padx=20, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = ExploracionApp(root)
    root.mainloop()