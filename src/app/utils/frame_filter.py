"""
Módulo para filtrar frames duplicados usando SSIM (Structural Similarity Index).
"""
from __future__ import annotations

import os
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


class FrameFilter:
    """
    Clase para filtrar frames duplicados usando SSIM.
    """

    def __init__(self, ssim_threshold: float = 0.98):
        """
        Inicializa el filtro de frames.

        Args:
            ssim_threshold: Threshold de SSIM para considerar frames como duplicados.
                          Valores más altos (0.95-0.98) son más estrictos.
        """
        self.ssim_threshold = ssim_threshold

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Carga una imagen y la convierte a escala de grises.

        Args:
            image_path: Ruta al archivo de imagen

        Returns:
            Array de numpy con la imagen en escala de grises
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray
        except Exception as e:
            print(f"Error cargando imagen {image_path}: {e}")
            return None

    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calcula el SSIM entre dos imágenes.

        Args:
            img1: Primera imagen
            img2: Segunda imagen

        Returns:
            Score SSIM entre 0 y 1
        """
        if img1 is None or img2 is None:
            return 0.0

        # Asegurar que las imágenes tienen el mismo tamaño
        if img1.shape != img2.shape:
            # Redimensionar la imagen más pequeña
            h, w = img2.shape
            img1 = cv2.resize(img1, (w, h))

        try:
            score = ssim(img1, img2, data_range=255)
            return score
        except Exception as e:
            print(f"Error calculando SSIM: {e}")
            return 0.0

    def filter_duplicate_frames(
        self,
        frame_paths: List[str],
        preserve_temporal_order: bool = True,
    ) -> Tuple[List[str], List[float], dict]:
        """
        Filtra frames duplicados manteniendo el orden temporal.

        Args:
            frame_paths: Lista de rutas a los frames
            preserve_temporal_order: Si True, solo compara frames consecutivos.
                                   Si False, compara todos los frames.

        Returns:
            Tuple con:
            - Lista de frame_paths filtrados
            - Lista de scores SSIM (None para frames mantenidos, score para frames filtrados)
            - Diccionario con estadísticas de filtrado
        """
        if len(frame_paths) <= 1:
            return frame_paths, [None] * len(frame_paths), {
                'original_count': len(frame_paths),
                'filtered_count': len(frame_paths),
                'reduction_percentage': 0.0,
            }

        filtered_paths = []
        ssim_scores = []
        original_count = len(frame_paths)

        # Cargar primera imagen
        prev_img = self.load_image(frame_paths[0])
        filtered_paths.append(frame_paths[0])
        ssim_scores.append(None)  # Primer frame siempre se mantiene

        # Comparar frames consecutivos
        for i in range(1, len(frame_paths)):
            current_img = self.load_image(frame_paths[i])

            if current_img is None:
                # Si no se puede cargar, mantener el frame anterior
                continue

            # Calcular SSIM con el frame anterior
            similarity = self.calculate_ssim(prev_img, current_img)

            if similarity >= self.ssim_threshold:
                # Frame muy similar, filtrarlo
                ssim_scores.append(similarity)
            else:
                # Frame diferente, mantenerlo
                filtered_paths.append(frame_paths[i])
                ssim_scores.append(None)
                prev_img = current_img  # Actualizar referencia

        filtered_count = len(filtered_paths)
        reduction_percentage = ((original_count - filtered_count) / original_count) * 100

        stats = {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'reduction_percentage': reduction_percentage,
            'frames_removed': original_count - filtered_count,
        }

        return filtered_paths, ssim_scores, stats

    def filter_dataset(
        self,
        input_csv_path: str,
        output_csv_path: str,
        frame_paths_col: str = 'frame_paths',
        ssim_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Filtra frames duplicados para todo el dataset.

        Args:
            input_csv_path: Ruta al CSV con el dataset original
            output_csv_path: Ruta donde guardar el dataset filtrado
            frame_paths_col: Nombre de la columna con las rutas de frames
            ssim_threshold: Threshold SSIM (si None, usa el del constructor)

        Returns:
            DataFrame con el dataset filtrado
        """
        if ssim_threshold is not None:
            self.ssim_threshold = ssim_threshold

        # Cargar dataset
        print(f"Cargando dataset desde {input_csv_path}...")
        df = pd.read_csv(input_csv_path)

        # Convertir frame_paths de string a lista si es necesario
        import ast
        if isinstance(df[frame_paths_col].iloc[0], str):
            df[frame_paths_col] = df[frame_paths_col].apply(ast.literal_eval)

        # Nuevas columnas para el dataset filtrado
        df['frame_paths_filtrado'] = None
        df['n_frames_original'] = df[frame_paths_col].apply(len)
        df['n_frames_filtrado'] = 0
        df['reduction_percentage'] = 0.0

        # Filtrar frames para cada video
        print(f"Filtrando frames duplicados (SSIM threshold: {self.ssim_threshold})...")
        all_stats = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc='Procesando videos'):
            frame_paths = row[frame_paths_col]

            if not frame_paths or len(frame_paths) == 0:
                df.at[idx, 'frame_paths_filtrado'] = []
                df.at[idx, 'n_frames_filtrado'] = 0
                continue

            # Filtrar frames duplicados
            filtered_paths, ssim_scores, stats = self.filter_duplicate_frames(
                frame_paths,
                preserve_temporal_order=True,
            )

            # Guardar resultados
            df.at[idx, 'frame_paths_filtrado'] = filtered_paths
            df.at[idx, 'n_frames_filtrado'] = stats['filtered_count']
            df.at[idx, 'reduction_percentage'] = stats['reduction_percentage']

            all_stats.append(stats)

        # Reemplazar columna original con la filtrada
        df['frame_paths'] = df['frame_paths_filtrado']
        df = df.drop(columns=['frame_paths_filtrado'])

        # Estadísticas generales
        total_original = sum(s['original_count'] for s in all_stats)
        total_filtered = sum(s['filtered_count'] for s in all_stats)
        avg_reduction = np.mean([s['reduction_percentage'] for s in all_stats])

        print(f"\n=== Estadísticas de Filtrado ===")
        print(f"Total frames originales: {total_original:,}")
        print(f"Total frames filtrados: {total_filtered:,}")
        print(f"Frames removidos: {total_original - total_filtered:,}")
        print(f"Reducción promedio: {avg_reduction:.2f}%")
        print(f"Reducción total: {((total_original - total_filtered) / total_original * 100):.2f}%")

        # Guardar dataset filtrado
        print(f"\nGuardando dataset filtrado en {output_csv_path}...")
        df.to_csv(output_csv_path, index=False)
        print('¡Dataset filtrado guardado exitosamente!')

        return df
