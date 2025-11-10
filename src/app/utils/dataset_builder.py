"""
Módulo para construir datasets con resultados de clustering.
"""
from __future__ import annotations

import os
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd


class DatasetBuilder:
    """
    Clase para construir datasets con resultados de clustering y categorización.
    """

    def __init__(self):
        """Inicializa el constructor de datasets."""
        pass

    def build_clusters_dataset(
        self,
        video_data: List[Dict],
        cluster_labels_per_video: Dict[str, np.ndarray],
        keyframes_per_video: Dict[str, List[str]],
        category_labels_per_keyframe: Optional[Dict[str, int]] = None,
        ssim_scores: Optional[Dict[str, List[float]]] = None,
    ) -> pd.DataFrame:
        """
        Construye el dataset con asignación de clusters por frame.

        Args:
            video_data: Lista de diccionarios con información de cada video
            cluster_labels_per_video: Diccionario mapping video_id -> cluster_labels array
            keyframes_per_video: Diccionario mapping video_id -> lista de key frame paths
            category_labels_per_keyframe: Diccionario opcional mapping frame_path -> category_id
            ssim_scores: Diccionario opcional mapping video_id -> lista de SSIM scores

        Returns:
            DataFrame con información de frames y clusters
        """
        rows = []

        for video_info in video_data:
            video_id = video_info['video_id']
            channel_name = video_info.get('channel_name', 'unknown')
            frame_paths = video_info['frame_paths']
            duration = video_info.get('duration', 0)
            video_date = video_info.get('video_date', 'unknown')

            # Obtener cluster labels para este video
            cluster_labels = cluster_labels_per_video.get(video_id, np.array([]))

            # Obtener key frames para este video
            keyframes = keyframes_per_video.get(video_id, [])

            # Obtener SSIM scores si están disponibles
            video_ssim_scores = ssim_scores.get(video_id, [None] * len(frame_paths)) if ssim_scores else [None] * len(frame_paths)

            # Crear filas para cada frame
            for idx, frame_path in enumerate(frame_paths):
                # Obtener cluster_id
                cluster_id = int(cluster_labels[idx]) if idx < len(cluster_labels) else -1

                # Verificar si es key frame
                is_keyframe = frame_path in keyframes

                # Obtener category_id si está disponible
                category_id = category_labels_per_keyframe.get(frame_path, -1) if category_labels_per_keyframe else -1

                # Obtener SSIM score
                ssim_score = video_ssim_scores[idx] if idx < len(video_ssim_scores) else None

                # Calcular timestamp aproximado (asumiendo 2 segundos por frame)
                timestamp = idx * 2.0

                row = {
                    'frame_path': frame_path,
                    'video_id': video_id,
                    'channel_name': channel_name,
                    'cluster_id': cluster_id,
                    'category_id': category_id,
                    'is_keyframe': is_keyframe,
                    'timestamp': timestamp,
                    'duration': duration,
                    'video_date': video_date,
                    'ssim_score': ssim_score,
                    'frame_index': idx,
                }

                rows.append(row)

        df = pd.DataFrame(rows)

        return df

    def build_summary_dataset(
        self,
        df_clusters: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construye un dataset resumen con estadísticas por video.

        Args:
            df_clusters: DataFrame con información de frames y clusters

        Returns:
            DataFrame resumen con estadísticas por video
        """
        summary_rows = []

        for video_id in df_clusters['video_id'].unique():
            video_df = df_clusters[df_clusters['video_id'] == video_id]

            summary = {
                'video_id': video_id,
                'channel_name': video_df['channel_name'].iloc[0],
                'duration': video_df['duration'].iloc[0],
                'video_date': video_df['video_date'].iloc[0],
                'n_frames_total': len(video_df),
                'n_keyframes': video_df['is_keyframe'].sum(),
                'n_clusters': video_df['cluster_id'].nunique(),
                'n_categories': video_df[video_df['category_id'] >= 0]['category_id'].nunique(),
                'reduction_percentage': (1 - video_df['is_keyframe'].sum() / len(video_df)) * 100 if len(video_df) > 0 else 0,
            }

            summary_rows.append(summary)

        df_summary = pd.DataFrame(summary_rows)

        return df_summary

    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: str,
        index: bool = False,
    ):
        """
        Guarda el dataset en CSV.

        Args:
            df: DataFrame a guardar
            output_path: Ruta donde guardar el CSV
            index: Si guardar el índice
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=index)
        print(f"Dataset guardado en {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
