"""
Módulo para evaluación y métricas del pipeline.
"""
from __future__ import annotations

import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import silhouette_score


class PipelineEvaluator:
    """
    Clase para evaluar el pipeline de key frame extraction.
    """

    def __init__(self):
        """Inicializa el evaluador."""
        pass

    def calculate_clustering_metrics(
        self,
        features: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> Dict:
        """
        Calcula métricas de clustering.

        Args:
            features: Array numpy con features
            cluster_labels: Labels de clusters

        Returns:
            Diccionario con métricas
        """
        metrics = {}

        # Silhouette score
        if len(np.unique(cluster_labels)) > 1 and len(features) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(features, cluster_labels)
            except:
                metrics['silhouette_score'] = None
        else:
            metrics['silhouette_score'] = None

        # Número de clusters
        metrics['n_clusters'] = len(np.unique(cluster_labels))

        # Distribución de clusters
        unique, counts = np.unique(cluster_labels, return_counts=True)
        metrics['cluster_distribution'] = dict(zip(unique.astype(int), counts.astype(int)))
        metrics['cluster_sizes'] = counts.tolist()

        # Estadísticas de tamaño de clusters
        if len(counts) > 0:
            metrics['mean_cluster_size'] = float(np.mean(counts))
            metrics['std_cluster_size'] = float(np.std(counts))
            metrics['min_cluster_size'] = int(np.min(counts))
            metrics['max_cluster_size'] = int(np.max(counts))
        else:
            metrics['mean_cluster_size'] = 0
            metrics['std_cluster_size'] = 0
            metrics['min_cluster_size'] = 0
            metrics['max_cluster_size'] = 0

        return metrics

    def calculate_diversity_metrics(
        self,
        keyframes: List[str],
        all_frames: List[str],
    ) -> Dict:
        """
        Calcula métricas de diversidad de key frames.

        Args:
            keyframes: Lista de rutas de key frames
            all_frames: Lista de rutas de todos los frames

        Returns:
            Diccionario con métricas de diversidad
        """
        metrics = {}

        metrics['n_keyframes'] = len(keyframes)
        metrics['n_total_frames'] = len(all_frames)
        metrics['reduction_ratio'] = len(keyframes) / len(all_frames) if len(all_frames) > 0 else 0
        metrics['compression_ratio'] = (1 - metrics['reduction_ratio']) * 100

        return metrics

    def calculate_temporal_coverage(
        self,
        keyframe_paths: List[str],
        all_frame_paths: List[str],
        frame_interval_sec: float = 2.0,
    ) -> Dict:
        """
        Calcula cobertura temporal de los key frames.

        Args:
            keyframe_paths: Lista de rutas de key frames
            all_frame_paths: Lista de rutas de todos los frames
            frame_interval_sec: Intervalo entre frames en segundos

        Returns:
            Diccionario con métricas de cobertura temporal
        """
        # Obtener índices de key frames en la lista completa
        keyframe_indices = [
            all_frame_paths.index(kf) for kf in keyframe_paths
            if kf in all_frame_paths
        ]

        if len(keyframe_indices) == 0:
            return {
                'coverage_percentage': 0.0,
                'time_span_covered': 0.0,
                'total_time_span': 0.0,
                'gaps': [],
            }

        # Calcular tiempo total
        total_time = len(all_frame_paths) * frame_interval_sec

        # Calcular tiempo cubierto (aproximado)
        time_covered = len(keyframe_indices) * frame_interval_sec

        # Calcular gaps (intervalos sin key frames)
        sorted_indices = sorted(keyframe_indices)
        gaps = []
        for i in range(len(sorted_indices) - 1):
            gap = sorted_indices[i + 1] - sorted_indices[i]
            gaps.append(gap * frame_interval_sec)

        metrics = {
            'coverage_percentage': (time_covered / total_time * 100) if total_time > 0 else 0.0,
            'time_span_covered': time_covered,
            'total_time_span': total_time,
            'mean_gap': float(np.mean(gaps)) if len(gaps) > 0 else 0.0,
            'max_gap': float(np.max(gaps)) if len(gaps) > 0 else 0.0,
            'min_gap': float(np.min(gaps)) if len(gaps) > 0 else 0.0,
            'n_gaps': len(gaps),
        }

        return metrics

    def plot_cluster_distribution(
        self,
        cluster_labels: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """
        Plotea la distribución de clusters.

        Args:
            cluster_labels: Labels de clusters
            save_path: Ruta opcional para guardar el gráfico
        """
        unique, counts = np.unique(cluster_labels, return_counts=True)

        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts, alpha=0.7)
        plt.xlabel('Cluster ID')
        plt.ylabel('Número de Frames')
        plt.title('Distribución de Frames por Cluster')
        plt.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_category_distribution(
        self,
        category_labels: np.ndarray,
        channel_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Plotea la distribución de categorías.

        Args:
            category_labels: Labels de categorías
            channel_names: Lista opcional de nombres de canales
            save_path: Ruta opcional para guardar el gráfico
        """
        if channel_names:
            df = pd.DataFrame({
                'channel': channel_names,
                'category': category_labels,
            })

            # Crear tabla de contingencia
            contingency = pd.crosstab(df['channel'], df['category'])

            plt.figure(figsize=(12, 6))
            contingency.plot(kind='bar', stacked=True, alpha=0.7)
            plt.xlabel('Canal')
            plt.ylabel('Número de Frames')
            plt.title('Distribución de Categorías por Canal')
            plt.legend(title='Categoría')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        else:
            unique, counts = np.unique(category_labels, return_counts=True)

            plt.figure(figsize=(10, 6))
            plt.bar(unique, counts, alpha=0.7)
            plt.xlabel('Categoría ID')
            plt.ylabel('Número de Frames')
            plt.title('Distribución de Frames por Categoría')
            plt.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_keyframe_mosaic(
        self,
        keyframe_paths: List[str],
        n_samples: int = 16,
        save_path: Optional[str] = None,
    ):
        """
        Crea un mosaico con key frames representativos.

        Args:
            keyframe_paths: Lista de rutas de key frames
            n_samples: Número de frames a mostrar
            save_path: Ruta opcional para guardar el mosaico
        """
        # Seleccionar muestras
        n_samples = min(n_samples, len(keyframe_paths))
        selected_paths = np.random.choice(keyframe_paths, n_samples, replace=False)

        # Calcular grid
        cols = int(np.ceil(np.sqrt(n_samples)))
        rows = int(np.ceil(n_samples / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, path in enumerate(selected_paths):
            row = idx // cols
            col = idx % cols

            try:
                img = Image.open(path)
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                axes[row, col].set_title(f'KF {idx+1}', fontsize=8)
            except Exception as e:
                axes[row, col].text(
                    0.5, 0.5, f'Error\n{path}',
                    ha='center', va='center', fontsize=6,
                )
                axes[row, col].axis('off')

        # Ocultar ejes vacíos
        for idx in range(n_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.suptitle(
            f'Mosaico de Key Frames (muestra de {n_samples})',
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_evaluation_report(
        self,
        metrics_dict: Dict,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Genera un reporte de evaluación en texto.

        Args:
            metrics_dict: Diccionario con todas las métricas
            save_path: Ruta opcional para guardar el reporte

        Returns:
            String con el reporte
        """
        report = '=' * 60 + '\n'
        report += 'REPORTE DE EVALUACIÓN - KEY FRAME EXTRACTION\n'
        report += '=' * 60 + '\n\n'

        # Métricas de clustering
        if 'clustering_metrics' in metrics_dict:
            report += 'MÉTRICAS DE CLUSTERING\n'
            report += '-' * 60 + '\n'
            cm = metrics_dict['clustering_metrics']
            report += f"Número de clusters: {cm.get('n_clusters', 'N/A')}\n"
            report += f"Silhouette Score: {cm.get('silhouette_score', 'N/A'):.4f}\n" if cm.get('silhouette_score') else 'Silhouette Score: N/A\n'
            report += f"Tamaño promedio de cluster: {cm.get('mean_cluster_size', 'N/A'):.2f}\n"
            report += '\n'

        # Métricas de diversidad
        if 'diversity_metrics' in metrics_dict:
            report += 'MÉTRICAS DE DIVERSIDAD\n'
            report += '-' * 60 + '\n'
            dm = metrics_dict['diversity_metrics']
            report += f"Número de key frames: {dm.get('n_keyframes', 'N/A')}\n"
            report += f"Número total de frames: {dm.get('n_total_frames', 'N/A')}\n"
            report += f"Ratio de reducción: {dm.get('reduction_ratio', 'N/A'):.4f}\n"
            report += f"Compresión: {dm.get('compression_ratio', 'N/A'):.2f}%\n"
            report += '\n'

        # Métricas de cobertura temporal
        if 'temporal_coverage' in metrics_dict:
            report += 'COBERTURA TEMPORAL\n'
            report += '-' * 60 + '\n'
            tc = metrics_dict['temporal_coverage']
            report += f"Porcentaje de cobertura: {tc.get('coverage_percentage', 'N/A'):.2f}%\n"
            report += f"Tiempo cubierto: {tc.get('time_span_covered', 'N/A'):.2f} segundos\n"
            report += f"Tiempo total: {tc.get('total_time_span', 'N/A'):.2f} segundos\n"
            report += f"Gap promedio: {tc.get('mean_gap', 'N/A'):.2f} segundos\n"
            report += '\n'

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Reporte guardado en {save_path}")

        return report
