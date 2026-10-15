"""
logger.py — Mensajes de progreso con hora y nombre del módulo.

En lugar de usar print() en todos lados, cada módulo pide su propio "logger".
Así cada mensaje dice a qué hora ocurrió y qué parte del código lo escribió:

    10:31:02 | download_data | Procesando Jun-2025.csv ...
"""

import logging
import sys


def get_logger(nombre: str) -> logging.Logger:
    """Devuelve un logger configurado para escribir en pantalla."""
    logger = logging.getLogger(nombre)            # un logger por módulo (por nombre)
    if not logger.handlers:                       # evita duplicar mensajes si se llama dos veces
        manejador = logging.StreamHandler(sys.stdout)            # escribir en la salida estándar
        formato = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
        manejador.setFormatter(formato)           # aplicar el formato "hora | módulo | mensaje"
        logger.addHandler(manejador)
        logger.setLevel(logging.INFO)             # mostrar mensajes de nivel INFO o superior
        logger.propagate = False                  # no repetir el mensaje en el logger raíz
    return logger
