"""
Módulo para preprocesamiento de datos: carga, validación y normalización.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


class DataPreprocessor:
    """
    Clase para preprocesar datos del dataset.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Inicializa el preprocesador.

        Args:
            base_path: Ruta base para resolver rutas relativas de frames
        """
        self.base_path = base_path

    def load_dataset(self, csv_path: str, frame_paths_col: str = 'frame_paths') -> pd.DataFrame:
        """
        Carga el dataset desde un CSV.

        Args:
            csv_path: Ruta al archivo CSV
            frame_paths_col: Nombre de la columna con las rutas de frames

        Returns:
            DataFrame cargado
        """
        print(f"Cargando dataset desde {csv_path}...")
        df = pd.read_csv(csv_path)

        # Convertir frame_paths de string a lista si es necesario
        if frame_paths_col in df.columns:
            if isinstance(df[frame_paths_col].iloc[0], str):
                df[frame_paths_col] = df[frame_paths_col].apply(ast.literal_eval)

        print(f"Dataset cargado: {len(df)} videos")
        return df

    def validate_frame_paths(self, frame_paths: List[str]) -> List[str]:
        """
        Valida que las rutas de frames existan.

        Args:
            frame_paths: Lista de rutas a frames

        Returns:
            Lista de rutas válidas
        """
        valid_paths = []

        for path in frame_paths:
            # Resolver ruta relativa si es necesario
            if self.base_path and not os.path.isabs(path):
                full_path = os.path.join(self.base_path, path)
            else:
                full_path = path

            if os.path.exists(full_path):
                valid_paths.append(path)
            else:
                print(f"Advertencia: Frame no encontrado: {path}")

        return valid_paths

    def validate_dataset(self, df: pd.DataFrame, frame_paths_col: str = 'frame_paths') -> pd.DataFrame:
        """
        Valida todas las rutas de frames en el dataset.

        Args:
            df: DataFrame a validar
            frame_paths_col: Nombre de la columna con las rutas de frames

        Returns:
            DataFrame con solo frames válidos
        """
        print('Validando rutas de frames...')

        def validate_row(row):
            frame_paths = row[frame_paths_col]
            if not frame_paths or len(frame_paths) == 0:
                return []

            valid_paths = self.validate_frame_paths(frame_paths)
            return valid_paths

        df[frame_paths_col] = df.apply(validate_row, axis=1)
        df['n_frames_validos'] = df[frame_paths_col].apply(len)

        # Filtrar videos sin frames válidos
        df = df[df['n_frames_validos'] > 0].copy()

        print(f"Videos con frames válidos: {len(df)}")
        return df

    def normalize_image(self, image_path: str, target_size: tuple = (224, 224)) -> Optional[np.ndarray]:
        """
        Normaliza una imagen para el modelo.

        Args:
            image_path: Ruta a la imagen
            target_size: Tamaño objetivo (alto, ancho)

        Returns:
            Array numpy normalizado o None si hay error
        """
        try:
            # Cargar imagen
            img = Image.open(image_path).convert('RGB')

            # Redimensionar
            img = img.resize(target_size)

            # Convertir a array y normalizar a [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0

            return img_array
        except Exception as e:
            print(f"Error normalizando imagen {image_path}: {e}")
            return None

    def prepare_video_data(
        self,
        row: pd.Series,
        frame_paths_col: str = 'frame_paths',
    ) -> Dict:
        """
        Prepara los datos de un video para procesamiento.

        Args:
            row: Fila del DataFrame con información del video
            frame_paths_col: Nombre de la columna con las rutas de frames

        Returns:
            Diccionario con datos del video
        """
        frame_paths = row[frame_paths_col]

        # Validar rutas
        valid_paths = self.validate_frame_paths(frame_paths)

        video_data = {
            'video_id': row.get('video_id', 'unknown'),
            'channel_name': row.get('channel_name', row.get('channel', 'unknown')),
            'frame_paths': valid_paths,
            'n_frames': len(valid_paths),
            'duration': row.get('duration', 0),
            'video_date': row.get('video_date', 'unknown'),
        }

        return video_data

    def get_video_index(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Crea un índice de videos por video_id.

        Args:
            df: DataFrame con videos

        Returns:
            Diccionario mapping video_id -> índice en DataFrame
        """
        return {row['video_id']: idx for idx, row in df.iterrows()}
