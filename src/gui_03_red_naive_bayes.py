import tkinter as tk
from tkinter import ttk
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RedNaiveBayesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apartado 3: Red Naive Bayes y Suavizado de Laplace")
        self.root.geometry("850x650")

        # Cargar datos
        ruta_datos = os.path.join("..", "data", "train_categorico.csv")
        try:
            self.df = pd.read_csv(ruta_datos)
        except FileNotFoundError:
            try:
                ruta_datos = os.path.join("data", "train_categorico.csv")
                self.df = pd.read_csv(ruta_datos)
            except FileNotFoundError:
                tk.Label(self.root, text="Error: No se encontró train_categorico.csv").pack()
                return

        # Frame superior para el grafo
        self.frame_grafo = tk.Frame(self.root, bg="white", height=300)
        self.frame_grafo.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(self.frame_grafo, text="Estructura de la Red Naive Bayes", font=("Arial", 14, "bold"), bg="white").pack()
        self.dibujar_grafo()

        # Frame inferior para las Tablas CPT
        self.frame_tablas = tk.Frame(self.root)
        self.frame_tablas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(self.frame_tablas, text="Tabla de Probabilidad Condicional (CPT) con Suavizado de Laplace (+1)", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.tree = ttk.Treeview(self.frame_tablas, columns=("Clase (Revenue)", "Variable", "Valor", "Probabilidad"), show='headings', height=10)
        self.tree.heading("Clase (Revenue)", text="Clase (Revenue)")
        self.tree.heading("Variable", text="Variable (Atributo)")
        self.tree.heading("Valor", text="Categoría")
        self.tree.heading("Probabilidad", text="P(Valor | Clase) Laplace")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.calcular_laplace()

    def dibujar_grafo(self):
        fig, ax = plt.subplots(figsize=(6, 3))
        G = nx.DiGraph()
        
        # En Naive Bayes, la clase (Revenue) apunta a los atributos
        edges = [
            ("Revenue\n(Compra)", "VisitorType"),
            ("Revenue\n(Compra)", "ProductRelated"),
            ("Revenue\n(Compra)", "ProductRelated_\nDuration")
        ]
        G.add_edges_from(edges)
        
        # Posiciones fijas para mejor visualización
        pos = {
            "Revenue\n(Compra)": (0.5, 1),
            "VisitorType": (0.2, 0),
            "ProductRelated": (0.5, 0),
            "ProductRelated_\nDuration": (0.8, 0)
        }
        
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='#87CEFA', 
                node_size=3000, font_size=10, font_weight='bold', 
                arrows=True, arrowsize=20, edge_color='gray')
        
        # Integrar matplotlib en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafo)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def calcular_laplace(self):
        # Suavizado de Laplace: (conteo + 1) / (total_clase + numero_categorias)
        clases = self.df['Revenue'].unique()
        variables = ['VisitorType', 'ProductRelated'] # Muestra 2 para ejemplo interactivo
        
        for c in clases:
            df_clase = self.df[self.df['Revenue'] == c]
            total_c = len(df_clase)
            
            for var in variables:
                categorias = self.df[var].unique()
                num_categorias = len(categorias)
                
                for cat in categorias:
                    # Conteo de (Variable=cat AND Revenue=c)
                    conteo_interseccion = len(df_clase[df_clase[var] == cat])
                    
                    # Aplicar fórmula Laplace (+1)
                    prob_laplace = (conteo_interseccion + 1) / (total_c + num_categorias)
                    
                    self.tree.insert("", tk.END, values=(
                        f"{c}", 
                        var, 
                        cat, 
                        f"{prob_laplace:.4f} ({(prob_laplace*100):.2f}%)"
                    ))

if __name__ == "__main__":
    root = tk.Tk()
    app = RedNaiveBayesApp(root)
    root.mainloop()