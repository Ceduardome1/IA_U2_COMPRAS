import tkinter as tk
from tkinter import ttk
import pandas as pd
import os

class ClasificadorBayesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apartado 4: Clasificador Naive Bayes")
        self.root.geometry("500x450")

        # Cargar datos categorizados
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

        tk.Label(self.root, text="Predicción de Compra (Naive Bayes)", font=("Arial", 14, "bold")).pack(pady=15)

        frame_form = tk.Frame(self.root)
        frame_form.pack(pady=10)

        # Extraer categorías únicas para los menús
        opciones_visitor = list(self.df['VisitorType'].unique())
        opciones_product = list(self.df['ProductRelated'].unique())
        opciones_duration = list(self.df['ProductRelated_Duration'].unique())

        self.var_visitor = tk.StringVar(value=opciones_visitor[0])
        self.var_product = tk.StringVar(value=opciones_product[0])
        self.var_duration = tk.StringVar(value=opciones_duration[0])

        # Formularios desplegables
        tk.Label(frame_form, text="Tipo de Visitante:").grid(row=0, column=0, sticky="e", padx=10, pady=10)
        ttk.Combobox(frame_form, textvariable=self.var_visitor, values=opciones_visitor, state="readonly").grid(row=0, column=1, padx=10, pady=10)

        tk.Label(frame_form, text="Interacción con Productos:").grid(row=1, column=0, sticky="e", padx=10, pady=10)
        ttk.Combobox(frame_form, textvariable=self.var_product, values=opciones_product, state="readonly").grid(row=1, column=1, padx=10, pady=10)

        tk.Label(frame_form, text="Duración de Interacción:").grid(row=2, column=0, sticky="e", padx=10, pady=10)
        ttk.Combobox(frame_form, textvariable=self.var_duration, values=opciones_duration, state="readonly").grid(row=2, column=1, padx=10, pady=10)

        ttk.Button(self.root, text="Clasificar Usuario", command=self.predecir).pack(pady=20)

        self.lbl_resultado = tk.Label(self.root, text="", font=("Arial", 12), justify="left")
        self.lbl_resultado.pack(pady=10)

    def calcular_probabilidad(self, clase_valor, atributo, atributo_valor):
        # Utiliza la misma fórmula de Laplace del script 3
        df_clase = self.df[self.df['Revenue'] == clase_valor]
        total_c = len(df_clase)
        num_categorias = len(self.df[atributo].unique())
        conteo = len(df_clase[df_clase[atributo] == atributo_valor])
        return (conteo + 1) / (total_c + num_categorias)

    def predecir(self):
        v_visitor = self.var_visitor.get()
        v_product = self.var_product.get()
        v_duration = self.var_duration.get()

        total_registros = len(self.df)
        p_true = len(self.df[self.df['Revenue'] == True]) / total_registros
        p_false = len(self.df[self.df['Revenue'] == False]) / total_registros

        # Puntaje si SÍ compra
        p_v_true = self.calcular_probabilidad(True, 'VisitorType', v_visitor)
        p_p_true = self.calcular_probabilidad(True, 'ProductRelated', v_product)
        p_d_true = self.calcular_probabilidad(True, 'ProductRelated_Duration', v_duration)
        puntaje_true = p_true * p_v_true * p_p_true * p_d_true

        # Puntaje si NO compra
        p_v_false = self.calcular_probabilidad(False, 'VisitorType', v_visitor)
        p_p_false = self.calcular_probabilidad(False, 'ProductRelated', v_product)
        p_d_false = self.calcular_probabilidad(False, 'ProductRelated_Duration', v_duration)
        puntaje_false = p_false * p_v_false * p_p_false * p_d_false

        if puntaje_true > puntaje_false:
            prediccion = "SÍ COMPRARÁ (True)"
            color = "green"
        else:
            prediccion = "NO COMPRARÁ (False)"
            color = "red"

        resultado_texto = (
            f"Puntaje de Compra: {puntaje_true:.6f}\n"
            f"Puntaje de NO Compra: {puntaje_false:.6f}\n\n"
            f"Conclusión de la IA:\nEl usuario {prediccion}"
        )
        self.lbl_resultado.config(text=resultado_texto, fg=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = ClasificadorBayesApp(root)
    root.mainloop()