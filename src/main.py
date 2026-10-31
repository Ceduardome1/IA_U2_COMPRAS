#!/usr/bin/env python3
"""
main.py
Alias directo al Panel Maestro de Navegación (main_dashboard.py).

Uso:
    python src/main.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.main_dashboard import main

if __name__ == "__main__":
    main()
