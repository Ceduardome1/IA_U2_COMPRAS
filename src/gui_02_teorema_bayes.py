import tkinter as tk
from tkinter import ttk
import pandas as pd
import os

class BayesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apartado 2: Teorema de Bayes por Conteos")
        self.root.geometry("800x600")

        ruta_datos = os.path.join("..", "data", "train_numerico.csv")
        try:
            self.df = pd.read_csv(ruta_datos)
        except FileNotFoundError:
            ruta_datos = os.path.join("data", "train_numerico.csv")
            try:
                self.df = pd.read_csv(ruta_datos)
            except FileNotFoundError:
                tk.Label(self.root, text="Error: No se encontró train_numerico.csv").pack()
                return

        self.calcular_probabilidades()
        self.crear_interfaz()

    def calcular_probabilidades(self):
        # Total de sesiones
        self.N = len(self.df)
        
        # C = Compra (Revenue == True o 1)
        self.C = len(self.df[self.df['Revenue'] == True])
        
        # R = Visitante Recurrente. Detectamos el valor (suele ser el más frecuente)
        val_recurrente = self.df['VisitorType'].mode()[0]
        self.R = len(self.df[self.df['VisitorType'] == val_recurrente])
        
        # C intersección R (Compras hechas por recurrentes)
        self.C_and_R = len(self.df[(self.df['Revenue'] == True) & (self.df['VisitorType'] == val_recurrente)])

        # Probabilidades
        self.P_C = self.C / self.N
        self.P_R = self.R / self.N
        self.P_R_dado_C = self.C_and_R / self.C if self.C > 0 else 0
        
        # Bayes
        self.P_C_dado_R_bayes = (self.P_R_dado_C * self.P_C) / self.P_R if self.P_R > 0 else 0
        
        # Comprobación empírica (Conteo directo)
        self.P_C_dado_R_directo = self.C_and_R / self.R if self.R > 0 else 0

    def crear_interfaz(self):
        lbl_titulo = tk.Label(self.root, text="Cálculo del Teorema de Bayes", font=("Arial", 16, "bold"))
        lbl_titulo.pack(pady=15)

        frame_conteos = ttk.LabelFrame(self.root, text=" Tabla de Frecuencias (Conteos) ")
        frame_conteos.pack(fill="x", padx=20, pady=10)

        texto_conteos = (
            f"N (Total de sesiones en train): {self.N}\n"
            f"C (Sesiones con Compra): {self.C}\n"
            f"R (Sesiones de Visitantes Recurrentes): {self.R}\n"
            f"C ∩ R (Compras por Recurrentes): {self.C_and_R}"
        )
        tk.Label(frame_conteos, text=texto_conteos, font=("Courier", 12), justify="left").pack(padx=10, pady=10)

        frame_bayes = ttk.LabelFrame(self.root, text=" Sustitución Algebraica de Bayes ")
        frame_bayes.pack(fill="x", padx=20, pady=10)

        texto_bayes = (
            f"Fórmula: P(C|R) = [ P(R|C) * P(C) ] / P(R)\n\n"
            f"1. P(C) [Probabilidad a priori de compra] = {self.C} / {self.N} = {self.P_C:.4f}\n"
            f"2. P(R) [Probabilidad marginal de recurrente] = {self.R} / {self.N} = {self.P_R:.4f}\n"
            f"3. P(R|C) [Verosimilitud] = {self.C_and_R} / {self.C} = {self.P_R_dado_C:.4f}\n\n"
            f"Sustitución:\n"
            f"P(C|R) = ({self.P_R_dado_C:.4f} * {self.P_C:.4f}) / {self.P_R:.4f} = {self.P_C_dado_R_bayes:.4f}"
        )
        tk.Label(frame_bayes, text=texto_bayes, font=("Courier", 11), justify="left").pack(padx=10, pady=10)

        frame_comprobacion = ttk.LabelFrame(self.root, text=" Comprobación Directa Empírica ")
        frame_comprobacion.pack(fill="x", padx=20, pady=10)

        texto_comprobacion = (
            f"Cálculo: N(C ∩ R) / N(R) = {self.C_and_R} / {self.R} = {self.P_C_dado_R_directo:.4f}\n\n"
            f"Conclusión: El cálculo de Bayes coincide exactamente con la división empírica \n"
            f"directa, demostrando que de todos los visitantes recurrentes, el \n"
            f"{self.P_C_dado_R_directo*100:.2f}% terminó realizando una compra."
        )
        tk.Label(frame_comprobacion, text=texto_comprobacion, font=("Courier", 11), justify="left", fg="#2c3e50").pack(padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = BayesApp(root)
    root.mainloop()