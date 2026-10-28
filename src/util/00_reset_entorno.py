#!/usr/bin/env python3
"""
00_reset_entorno.py
Script de Purga y Reinicio del Entorno.

Acciones:
- Elimina los archivos derivados de data/ (train_*.csv, test_*.csv).
- Limpia todos los binarios y metadatos de artifacts/ (.joblib, .json).
- Borra las carpetas __pycache__ del entorno.
"""

import sys
import shutil
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import DATA_DIR, ARTIFACTS_DIR


def purge_environment():
    print("=" * 60)
    print(" INICIANDO PURGA Y REINICIO DEL ENTORNO")
    print("=" * 60)

    # 1. Limpiar archivos derivados en data/ (mantener online_shoppers_intention.csv)
    derivados_data = list(DATA_DIR.glob("train_*.csv")) + list(DATA_DIR.glob("test_*.csv"))
    if derivados_data:
        for f in derivados_data:
            f.unlink()
            print(f"[data/] Eliminado archivo derivado: {f.name}")
    else:
        print("[data/] No se encontraron archivos derivados train_*.csv o test_*.csv.")

    # 2. Limpiar artifacts/ (.joblib y .json)
    if ARTIFACTS_DIR.exists():
        artefactos = list(ARTIFACTS_DIR.glob("*.joblib")) + list(ARTIFACTS_DIR.glob("*.json"))
        if artefactos:
            for f in artefactos:
                f.unlink()
                print(f"[artifacts/] Eliminado artefacto: {f.name}")
        else:
            print("[artifacts/] No hay archivos .joblib o .json para eliminar.")
    else:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        print("[artifacts/] Carpeta creada.")

    # 3. Limpiar __pycache__
    pycache_dirs = list(BASE_DIR.rglob("__pycache__"))
    if pycache_dirs:
        for p in pycache_dirs:
            shutil.rmtree(p, ignore_errors=True)
            print(f"[cache] Eliminada carpeta: {p.relative_to(BASE_DIR)}")
    else:
        print("[cache] No se encontraron carpetas __pycache__.")

    print("\n[OK] Entorno reiniciado exitosamente a su estado base.")
    print("=" * 60)


if __name__ == "__main__":
    purge_environment()
