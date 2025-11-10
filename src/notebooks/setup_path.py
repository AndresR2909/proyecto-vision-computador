"""
Configuración automática del path para notebooks
Ejecuta este archivo al inicio de cualquier notebook
"""
from __future__ import annotations

import os
import sys

def setup_project_path():
    """
    Configura el path de Python para permitir importaciones desde src
    """
    # Obtener el directorio raíz del proyecto (dos niveles arriba de notebooks)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # Agregar al path si no está presente
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ Path configurado: {project_root}")
    else:
        print(f"✅ Path ya configurado: {project_root}")

    return project_root

# Ejecutar automáticamente al importar
if __name__ == '__main__':
    setup_project_path()
else:
    # Si se importa como módulo, configurar automáticamente
    setup_project_path()
