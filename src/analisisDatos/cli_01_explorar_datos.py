#!/usr/bin/env python3
"""
cli_01_explorar_datos.py
Tipo: LÍNEA DE COMANDOS (CLI)
Responsable: JAIME / CESAR
Entregable: Apartado 1 - Resumen Exploratorio de Datos por Consola

Objetivos y Especificaciones:
1. Inspección del archivo crudo 'data/online_shoppers_intention.csv'.
2. Conteo de valores nulos o atípicos en las 4 variables de interés.
3. Tabla de proporciones exactas de la clase objetivo 'Revenue'.
4. Estadísticas descriptivas de ProductRelated y ProductRelated_Duration.
5. Frecuencia de categorías de VisitorType.

Uso:
    python src/analisisDatos/cli_01_explorar_datos.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def inspeccionar_datos_crudos():
    """Imprime resumen estadístico y balance de clases en consola."""
    pass


if __name__ == "__main__":
    inspeccionar_datos_crudos()
