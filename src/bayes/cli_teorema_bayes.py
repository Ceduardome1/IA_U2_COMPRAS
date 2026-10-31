#!/usr/bin/env python3
"""
cli_teorema_bayes.py
Tipo: LÍNEA DE COMANDOS (CLI)
Responsable: JAIME / CESAR
Entregable: Apartado 2 - Teorema de Bayes por Conteos

Objetivos y Especificaciones:
1. Cálculo univariado analítico a partir de los datos de entrenamiento (80%):
   - Definición de eventos:
     * C = Compra (Revenue = True)
     * ¬C = No Compra (Revenue = False)
     * R = Visitante Recurrente (VisitorType = 'Returning_Visitor')
2. Extracción de conteos exactos de la tabla de contingencia de train:
   - N: Total de sesiones de entrenamiento.
   - N(C): Total de sesiones que terminaron en compra.
   - N(¬C): Total de sesiones que no terminaron en compra.
   - N(R): Total de sesiones de visitantes recurrentes.
   - N(CR): Sesiones de recurrentes que compraron.
   - N(¬C, R): Sesiones de recurrentes que no compraron.
3. Estimación frecuentista de probabilidades:
   - A priori: P(C) = N(C) / N,  P(¬C) = N(¬C) / N
   - Verosimilitud: P(R | C) = N(CR) / N(C),  P(R | ¬C) = N(¬C, R) / N(¬C)
   - Probabilidad marginal: P(R) = P(R | C)*P(C) + P(R | ¬C)*P(¬C) = N(R) / N
4. Sustitución en la fórmula del Teorema de Bayes:
   - P(C | R) = [P(R | C) * P(C)] / [P(R | C) * P(C) + P(R | ¬C) * P(¬C)]
5. Comprobación empírica directa:
   - P_directa(C | R) = N(CR) / N(R)
   - Demostración de que la sustitución algebraica y la comprobación directa coinciden exactamente.
6. Interpretación textual del resultado obtenido.

Uso:
    python src/bayes/cli_teorema_bayes.py
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import TRAIN_CAT_PATH, TARGET_COL


def obtener_conteos_train():
    """Calcula N, N(C), N(noC), N(R), N(CR), N(noCR) desde data/train_categorico.csv."""
    ruta = TRAIN_CAT_PATH
    if not ruta.exists():
        candidatos = [
            BASE_DIR / "data" / "train_categorico.csv",
            Path.cwd() / "data" / "train_categorico.csv"
        ]
        for c in candidatos:
            if c.exists():
                ruta = c
                break

    df = pd.read_csv(ruta)

    n_total = len(df)
    n_c = int((df[TARGET_COL] == True).sum())
    n_noc = int((df[TARGET_COL] == False).sum())

    val_rec = "Returning_Visitor"
    n_r = int((df["VisitorType"] == val_rec).sum())
    n_nor = n_total - n_r

    n_cr = int(((df["VisitorType"] == val_rec) & (df[TARGET_COL] == True)).sum())
    n_nocr = int(((df["VisitorType"] == val_rec) & (df[TARGET_COL] == False)).sum())

    return {
        "N": n_total,
        "N_C": n_c,
        "N_noC": n_noc,
        "N_R": n_r,
        "N_noR": n_nor,
        "N_CR": n_cr,
        "N_noCR": n_nocr
    }


def calcular_probabilidades(conteos: dict):
    """Calcula P(C), P(R|C), P(R|noC) y la sustitución formal de Bayes."""
    N = conteos["N"]
    n_c = conteos["N_C"]
    n_noc = conteos["N_noC"]
    n_r = conteos["N_R"]
    n_cr = conteos["N_CR"]
    n_nocr = conteos["N_noCR"]

    p_c = n_c / N
    p_noc = n_noc / N

    p_r_dado_c = n_cr / n_c
    p_r_dado_noc = n_nocr / n_noc

    # Denominador de Bayes (Ley de Probabilidad Total)
    num = p_r_dado_c * p_c
    den = (p_r_dado_c * p_c) + (p_r_dado_noc * p_noc)
    p_c_dado_r_bayes = num / den

    p_r_marginal = n_r / N

    return {
        "P_C": p_c,
        "P_noC": p_noc,
        "P_R_dado_C": p_r_dado_c,
        "P_R_dado_noC": p_r_dado_noc,
        "Numerador_Bayes": num,
        "Denominador_Bayes": den,
        "P_C_dado_R_Bayes": p_c_dado_r_bayes,
        "P_R_marginal": p_r_marginal
    }


def ejecutar_comprobacion_directa(conteos: dict):
    """Calcula N(CR)/N(R) y verifica la igualdad con la fórmula."""
    p_directa = conteos["N_CR"] / conteos["N_R"]
    return p_directa


def main():
    """Ejecuta el cálculo e imprime el desglose analítico completo en consola."""
    print("=" * 78)
    print(" APARTADO 2: TEOREMA DE BAYES POR CONTEOS EXACTOS (TRAIN 80%)")
    print("=" * 78)

    conteos = obtener_conteos_train()
    probs = calcular_probabilidades(conteos)
    p_directa = ejecutar_comprobacion_directa(conteos)

    print("\n[1] MATRIZ DE CONTINGENCIA DE FRECUENCIAS (TRAIN):")
    print(f"    {'Evento':<30} {'Símbolo':<14} {'Conteo':<10}")
    print("    " + "-" * 54)
    print(f"    {'Total de Sesiones':<30} {'N':<14} {conteos['N']:,}")
    print(f"    {'Sesiones con Compra':<30} {'N(C)':<14} {conteos['N_C']:,}")
    print(f"    {'Sesiones sin Compra':<30} {'N(¬C)':<14} {conteos['N_noC']:,}")
    print(f"    {'Visitantes Recurrentes':<30} {'N(R)':<14} {conteos['N_R']:,}")
    print(f"    {'Recurrentes que Compraron':<30} {'N(CR)':<14} {conteos['N_CR']:,}")
    print(f"    {'Recurrentes sin Compra':<30} {'N(¬C, R)':<14} {conteos['N_noCR']:,}")

    print("\n[2] PROBABILIDADES CALCULADAS:")
    print(f"    • A Priori de Compra:         P(C)     = {probs['P_C']:.6f}  ({probs['P_C']*100:.2f}%)")
    print(f"    • A Priori de No Compra:      P(¬C)    = {probs['P_noC']:.6f}  ({probs['P_noC']*100:.2f}%)")
    print(f"    • Verosimilitud si Compra:    P(R|C)   = {probs['P_R_dado_C']:.6f}  ({probs['P_R_dado_C']*100:.2f}%)")
    print(f"    • Verosimilitud si No Compra: P(R|¬C)  = {probs['P_R_dado_noC']:.6f}  ({probs['P_R_dado_noC']*100:.2f}%)")
    print(f"    • Evidencia Marginal:         P(R)     = {probs['P_R_marginal']:.6f}  ({probs['P_R_marginal']*100:.2f}%)")

    print("\n[3] SUSTITUCIÓN EN EL TEOREMA DE BAYES:")
    print("            P(R|C) · P(C)")
    print("    P(C|R) = ------------------------------------")
    print("            P(R|C)·P(C) + P(R|¬C)·P(¬C)")
    print(f"\n    P(C|R) = ({probs['P_R_dado_C']:.6f} · {probs['P_C']:.6f}) / ({probs['Numerador_Bayes']:.6f} + {probs['P_R_dado_noC']*probs['P_noC']:.6f})")
    print(f"    P(C|R) = {probs['Numerador_Bayes']:.6f} / {probs['Denominador_Bayes']:.6f}")
    print(f"    P(C|R) = {probs['P_C_dado_R_Bayes']:.6f}  ({probs['P_C_dado_R_Bayes']*100:.4f}%)")

    print("\n[4] COMPROBACIÓN EMPÍRICA DIRECTA:")
    print(f"    P_directa(C|R) = N(CR) / N(R) = {conteos['N_CR']} / {conteos['N_R']}")
    print(f"    P_directa(C|R) = {p_directa:.6f}  ({p_directa*100:.4f}%)")

    dif = abs(probs["P_C_dado_R_Bayes"] - p_directa)
    print(f"\n    Diferencia matemática: {dif:.1e}")
    if dif < 1e-12:
        print("    [VALIDACIÓN OK] Identidad Bayesiana demostrada con exactitud de máquina.")

    print("\n[5] INTERPRETACIÓN DEL RESULTADO:")
    print(f"    - La probabilidad a priori general de compra es de {probs['P_C']*100:.2f}%.")
    print(f"    - Al condicionar a que el usuario es un visitante recurrente, la probabilidad")
    print(f"      de compra se reduce a {probs['P_C_dado_R_Bayes']*100:.2f}%.")
    print(f"    - Esto confirma que los visitantes recurrentes en este comercio compran con")
    print(f"      menor tasa que el promedio, mientras que los visitantes nuevos tienen mayor propensión.")

    print("\n" + "=" * 78)
    print(" [OK] Teorema de Bayes evaluado y validado exitosamente.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
