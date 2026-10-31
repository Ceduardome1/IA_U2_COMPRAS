#!/usr/bin/env python3
"""
main_dashboard.py
Tipo: INTERFAZ GRÁFICA MAESTRA (GUI)
Responsable: CESAR / JAIME
Entregable: Script Maestro de Navegación del Proyecto

Objetivos y Especificaciones:
1. Panel centralizado de navegación para todos los módulos e interfaces de la Unidad 2:
   - Apartado 1: Exploración de Datos, Distribuciones y Mapa Conceptual con NetworkX.
   - Apartado 2: Teorema de Bayes por Conteos Univariados y Validación de Identidad.
   - Apartado 3: Red Naive Bayes y Tablas CPT con Suavizado de Laplace (+1) y Persistencia.
   - Apartado 4: Inferencia Progresiva en 3 Etapas y Casos de Prueba Obligatorios.
   - Apartado 5: Comparativa Final de Modelos, Matrices de Confusión Heatmaps y
     Comparador en Vivo de Predicciones Multimodelo (con manejo de campos omitidos).
2. Verificación en tiempo real del estado de los datos y artefactos del pipeline.
3. Lanzamiento independiente de subprocesos sin bloquear la ventana maestra.

Uso:
    python src/main_dashboard.py
    python src/main.py
"""

import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# Configuración de path base
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    TRAIN_CAT_PATH,
    TRAIN_NUM_PATH,
    PERCENTILES_JSON,
    CPT_LAPLACE_JSON,
    RF_MODEL_JOBLIB,
    MLP_MODEL_JOBLIB,
    METRICAS_JSON,
    ARTIFACTS_DIR
)


class MasterDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Panel Maestro: Sistema de Inferencia y ML - Online Shoppers (Unidad 2)")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)

        self._configurar_estilos()
        self._crear_interfaz()

    def _configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", font=("Arial", 16, "bold"), foreground="#1e3a8a")
        style.configure("Sub.TLabel", font=("Arial", 10), foreground="#64748b")
        style.configure("Section.TLabel", font=("Arial", 11, "bold"), foreground="#0f766e")
        style.configure("Primary.TButton", font=("Arial", 9, "bold"), padding=6)
        style.configure("Secondary.TButton", font=("Arial", 9), padding=5)

    def _crear_interfaz(self):
        # 1. ENCABEZADO PRINCIPAL
        header_frame = ttk.Frame(self.root, padding="15 12 15 10")
        header_frame.pack(fill=tk.X)

        ttk.Label(
            header_frame,
            text="Panel Maestro de Navegación del Proyecto (Unidad 2)",
            style="Title.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            header_frame,
            text="Inteligencia Artificial • Predicción de Intención de Compra en Comercio Electrónico (UCI Dataset)",
            style="Sub.TLabel"
        ).pack(anchor="w")

        # 2. PANEL DE ESTADO DE ARTEFACTOS Y PIPELINE
        frame_status = ttk.LabelFrame(self.root, text=" Estado del Pipeline y Persistencia de Artefactos ", padding=10)
        frame_status.pack(fill=tk.X, padx=15, pady=(0, 10))

        artefactos_check = [
            ("Dataset Crudo", RAW_DATA_PATH),
            ("Datos Train/Test", TRAIN_CAT_PATH),
            ("Cortes P33/P67", PERCENTILES_JSON),
            ("CPT Laplace JSON", CPT_LAPLACE_JSON),
            ("Modelo RF", RF_MODEL_JOBLIB),
            ("Modelo MLP", MLP_MODEL_JOBLIB),
            ("Métricas Test JSON", METRICAS_JSON)
        ]

        for i, (nombre, path_obj) in enumerate(artefactos_check):
            col_idx = i % 4
            row_idx = i // 4
            existe = path_obj.exists()
            icono = "✅" if existe else "⚠️"
            color_txt = "#047857" if existe else "#b45309"
            lbl = ttk.Label(frame_status, text=f"{icono} {nombre}", font=("Arial", 9, "bold" if existe else "normal"), foreground=color_txt)
            lbl.grid(row=row_idx, column=col_idx, sticky="w", padx=12, pady=3)

        btn_refresh = ttk.Button(
            frame_status,
            text="🔄 Actualizar Estado",
            command=self._actualizar_estado
        )
        btn_refresh.grid(row=1, column=3, sticky="e", padx=10, pady=3)

        # 3. CONTENEDOR CON TRES COLUMNAS O TARJETAS POR APARTADOS
        main_cards = ttk.Frame(self.root, padding=10)
        main_cards.pack(fill=tk.BOTH, expand=True, padx=5)

        # Columna Izquierda: Apartado 1 y 2
        col_izq = ttk.Frame(main_cards)
        col_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        # Tarjeta Apartado 1
        card1 = ttk.LabelFrame(col_izq, text=" Apartado 1: Exploración y Representación de Datos ", padding=10)
        card1.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        ttk.Label(card1, text="Visualización de distribuciones, cortes percentiles y mapa de conceptos.", font=("Arial", 9), wraplength=320).pack(anchor="w", pady=(0, 8))

        ttk.Button(
            card1,
            text="🌐 1.1 Mapa Conceptual (NetworkX)",
            style="Primary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "analisisDatos" / "mapa_conceptual.py")
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            card1,
            text="📊 1.2 Distribución de Datos y Estadísticas",
            style="Secondary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "analisisDatos" / "distribucion_datos.py")
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            card1,
            text="🔀 1.3 Suite Integrada (Exploración + Mapa)",
            style="Secondary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "analisisDatos" / "exploracion_mapa.py")
        ).pack(fill=tk.X, pady=3)

        # Tarjeta Apartado 2
        card2 = ttk.LabelFrame(col_izq, text=" Apartado 2: Teorema de Bayes por Conteos ", padding=10)
        card2.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        ttk.Label(card2, text="Cálculo analítico frecuentista directo P(C|R), matriz 2x2 y validación de identidad.", font=("Arial", 9), wraplength=320).pack(anchor="w", pady=(0, 8))

        ttk.Button(
            card2,
            text="📐 2.1 Visualizador del Teorema de Bayes",
            style="Primary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "bayes" / "teorema_bayes.py")
        ).pack(fill=tk.X, pady=3)

        # Columna Central: Apartado 3 y 4
        col_cen = ttk.Frame(main_cards)
        col_cen.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        # Tarjeta Apartado 3
        card3 = ttk.LabelFrame(col_cen, text=" Apartado 3: Red Naive Bayes (+ Laplace) ", padding=10)
        card3.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        ttk.Label(card3, text="Grafo dirigido Naive Bayes, cálculo y persistencia de tablas CPT y visor matricial.", font=("Arial", 9), wraplength=320).pack(anchor="w", pady=(0, 8))

        ttk.Button(
            card3,
            text="🕸️ 3.1 Red Naive Bayes y Tablas CPT",
            style="Primary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "redBayesiana" / "red_naive_bayes.py")
        ).pack(fill=tk.X, pady=3)

        # Tarjeta Apartado 4
        card4 = ttk.LabelFrame(col_cen, text=" Apartado 4: Demostración y Pruebas ", padding=10)
        card4.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        ttk.Label(card4, text="Consultas progresivas en 3 etapas, casos de prueba obligatorios y clasificador interactivo.", font=("Arial", 9), wraplength=320).pack(anchor="w", pady=(0, 8))

        ttk.Button(
            card4,
            text="🧪 4.1 Inferencia Progresiva y Pruebas (TC-01, TC-02, TC-03)",
            style="Primary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "redBayesiana" / "consultas_demostracion.py")
        ).pack(fill=tk.X, pady=3)

        ttk.Button(
            card4,
            text="🤖 4.2 Clasificador Simple Naive Bayes",
            style="Secondary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "redBayesiana" / "clasificador_bayes.py")
        ).pack(fill=tk.X, pady=3)

        # Columna Derecha: Apartado 5 y Comparador Multimodelo
        col_der = ttk.Frame(main_cards)
        col_der.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        # Tarjeta Apartado 5
        card5 = ttk.LabelFrame(col_der, text=" Apartado 5: Comparativa y Evaluación Final ", padding=10)
        card5.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        ttk.Label(card5, text="Resultados finales en test, matrices de confusión Heatmaps y métricas F1 / Recall.", font=("Arial", 9), wraplength=320).pack(anchor="w", pady=(0, 8))

        ttk.Button(
            card5,
            text="📈 5.1 Panel Comparativo y Heatmaps (Test)",
            style="Primary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "comparacion" / "comparativa_modelos.py")
        ).pack(fill=tk.X, pady=3)

        # Tarjeta Comparativa de Predicciones en Vivo
        card_comp = ttk.LabelFrame(col_der, text=" Comparador de Predicciones Multimodelo ", padding=10)
        card_comp.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        ttk.Label(
            card_comp,
            text="Comparación simultánea con los mismos datos: Naive Bayes (permite omitir) vs RF y MLP (arrojan error si faltan campos).",
            font=("Arial", 9),
            wraplength=320
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(
            card_comp,
            text="⚖️ Comparativa en Vivo (Manejo de Omisión)",
            style="Primary.TButton",
            command=lambda: self._lanzar_script(BASE_DIR / "src" / "comparacion" / "gui_comparativa_predicciones.py")
        ).pack(fill=tk.X, pady=3)

        # 4. BARRA INFERIOR DE ACCIONES RÁPIDAS
        bottom_bar = ttk.Frame(self.root, padding="10 8 10 10")
        bottom_bar.pack(fill=tk.X)

        ttk.Button(
            bottom_bar,
            text="📂 Abrir Carpeta de Artefactos",
            command=self._abrir_carpeta_artefactos
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            bottom_bar,
            text="⚡ Ejecutar Pipeline CLI de Evaluación Completa",
            command=self._ejecutar_evaluacion_cli
        ).pack(side=tk.LEFT, padx=6)

        ttk.Button(
            bottom_bar,
            text="❌ Salir del Dashboard",
            command=self.root.quit
        ).pack(side=tk.RIGHT, padx=6)

    def _lanzar_script(self, script_path: Path):
        """Lanza el script en un subproceso independiente."""
        if not script_path.exists():
            messagebox.showerror("Error", f"No se encontró el script en:\n{script_path}")
            return
        try:
            subprocess.Popen([sys.executable, str(script_path)])
        except Exception as e:
            messagebox.showerror("Error al Lanzar", f"No se pudo ejecutar el script:\n{e}")

    def _actualizar_estado(self):
        # Destruir y reconstruir interfaz para refrescar estados
        for widget in self.root.winfo_children():
            widget.destroy()
        self._crear_interfaz()
        messagebox.showinfo("Estado Actualizado", "Se ha actualizado la verificación de persistencia y artefactos.")

    def _abrir_carpeta_artefactos(self):
        try:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["xdg-open", str(ARTIFACTS_DIR)])
        except Exception:
            messagebox.showinfo("Carpeta de Artefactos", f"La carpeta de artefactos es:\n{ARTIFACTS_DIR}")

    def _ejecutar_evaluacion_cli(self):
        script_eval = BASE_DIR / "src" / "comparacion" / "cli_evaluar_todos.py"
        try:
            subprocess.Popen([sys.executable, str(script_eval)])
            messagebox.showinfo("Pipeline CLI", "Se ha iniciado la evaluación comparativa en terminal.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar el pipeline CLI:\n{e}")


def main():
    root = tk.Tk()
    app = MasterDashboardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
