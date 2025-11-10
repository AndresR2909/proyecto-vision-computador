"""
Módulo para guardar y comparar resultados de diferentes configuraciones de clustering.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultsComparator:
    """
    Clase para guardar y comparar resultados de diferentes configuraciones.
    """

    def __init__(self, results_dir: str = 'results_comparison'):
        """
        Inicializa el comparador de resultados.

        Args:
            results_dir: Directorio donde se guardan los resultados
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.configs = []

    def save_configuration_results(
        self,
        config_name: str,
        clustering_method: str,
        distance_metric: str,
        normalize_features: bool,
        keyframes_per_video: Dict,
        cluster_labels_per_video: Dict,
        clustering_stats_per_video: Dict,
        all_video_data: List[Dict],
        all_frame_paths: Dict,
        additional_params: Optional[Dict] = None,
    ) -> str:
        """
        Guarda resultados de una configuración específica.

        Args:
            config_name: Nombre identificador de la configuración (ej: 'kmeans_cosine', 'agg_euclidean')
            clustering_method: Método de clustering usado
            distance_metric: Métrica de distancia usada
            normalize_features: Si se normalizaron features
            keyframes_per_video: Diccionario con key frames por video
            cluster_labels_per_video: Diccionario con labels de clusters por video
            clustering_stats_per_video: Diccionario con estadísticas por video
            all_video_data: Lista con información de videos
            all_frame_paths: Diccionario con paths de frames por video
            additional_params: Parámetros adicionales a guardar

        Returns:
            Path del archivo de configuración guardado
        """
        config_data = {
            'config_name': config_name,
            'timestamp': datetime.now().isoformat(),
            'clustering_method': clustering_method,
            'distance_metric': distance_metric,
            'normalize_features': normalize_features,
            'additional_params': additional_params or {},
            'summary': {
                'n_videos': len(keyframes_per_video),
                'total_keyframes': sum(len(kfs) for kfs in keyframes_per_video.values()),
                'total_frames': sum(len(fps) for fps in all_frame_paths.values()),
                'avg_keyframes_per_video': sum(len(kfs) for kfs in keyframes_per_video.values()) / len(keyframes_per_video) if len(keyframes_per_video) > 0 else 0,
                'avg_clusters_per_video': np.mean([stats.get('n_clusters', 0) for stats in clustering_stats_per_video.values()]) if len(clustering_stats_per_video) > 0 else 0,
                'avg_silhouette_score': np.mean([stats.get('silhouette_score', 0) or 0 for stats in clustering_stats_per_video.values()]) if len(clustering_stats_per_video) > 0 else 0,
            },
        }

        # Guardar configuración
        config_path = os.path.join(self.results_dir, f"{config_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Guardar dataset de clusters para esta configuración
        dataset_path = os.path.join(self.results_dir, f"{config_name}_clusters.csv")
        self._save_clusters_dataset(
            dataset_path,
            all_video_data,
            cluster_labels_per_video,
            keyframes_per_video,
            all_frame_paths,
        )

        # Guardar estadísticas por video
        stats_path = os.path.join(self.results_dir, f"{config_name}_stats.csv")
        self._save_video_stats(
            stats_path,
            all_video_data,
            keyframes_per_video,
            clustering_stats_per_video,
        )

        # Guardar keyframes por video
        keyframes_path = os.path.join(self.results_dir, f"{config_name}_keyframes.csv")
        self._save_keyframes_list(
            keyframes_path,
            keyframes_per_video,
        )

        self.configs.append(config_name)
        print(f"✅ Configuración '{config_name}' guardada en {self.results_dir}/")

        return config_path

    def _save_clusters_dataset(
        self,
        output_path: str,
        all_video_data: List[Dict],
        cluster_labels_per_video: Dict,
        keyframes_per_video: Dict,
        all_frame_paths: Dict,
    ):
        """Guarda dataset con clusters para una configuración."""
        rows = []

        for video_data in all_video_data:
            video_id = video_data['video_id']
            frame_paths = all_frame_paths.get(video_id, [])
            cluster_labels = cluster_labels_per_video.get(video_id, np.array([]))
            keyframes = keyframes_per_video.get(video_id, [])

            for idx, frame_path in enumerate(frame_paths):
                cluster_id = int(cluster_labels[idx]) if idx < len(cluster_labels) else -1
                is_keyframe = frame_path in keyframes

                rows.append({
                    'frame_path': frame_path,
                    'video_id': video_id,
                    'channel_name': video_data.get('channel_name', 'unknown'),
                    'cluster_id': cluster_id,
                    'is_keyframe': is_keyframe,
                    'frame_index': idx,
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def _save_video_stats(
        self,
        output_path: str,
        all_video_data: List[Dict],
        keyframes_per_video: Dict,
        clustering_stats_per_video: Dict,
    ):
        """Guarda estadísticas por video."""
        rows = []

        for video_data in all_video_data:
            video_id = video_data['video_id']
            keyframes = keyframes_per_video.get(video_id, [])
            stats = clustering_stats_per_video.get(video_id, {})

            rows.append({
                'video_id': video_id,
                'channel_name': video_data.get('channel_name', 'unknown'),
                'n_frames': video_data.get('n_frames', 0),
                'n_keyframes': len(keyframes),
                'n_clusters': stats.get('n_clusters', 0),
                'silhouette_score': stats.get('silhouette_score', None),
                'reduction_percentage': stats.get('reduction_percentage', 0),
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def _save_keyframes_list(
        self,
        output_path: str,
        keyframes_per_video: Dict,
    ):
        """Guarda lista de keyframes por video."""
        rows = []

        for video_id, keyframes in keyframes_per_video.items():
            for idx, kf_path in enumerate(keyframes):
                rows.append({
                    'video_id': video_id,
                    'keyframe_index': idx,
                    'keyframe_path': kf_path,
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def compare_configurations(
        self,
        config_names: Optional[List[str]] = None,
        save_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Compara diferentes configuraciones guardadas.

        Args:
            config_names: Lista de nombres de configuraciones a comparar (si None, compara todas)
            save_plot: Si True, guarda gráfico de comparación

        Returns:
            DataFrame con comparación de configuraciones
        """
        if config_names is None:
            config_names = self.configs

        if len(config_names) == 0:
            print('⚠️  No hay configuraciones guardadas para comparar.')
            return pd.DataFrame()

        comparison_data = []

        for config_name in config_names:
            config_path = os.path.join(self.results_dir, f"{config_name}_config.json")

            if not os.path.exists(config_path):
                print(f"⚠️  Configuración '{config_name}' no encontrada.")
                continue

            with open(config_path, 'r') as f:
                config_data = json.load(f)

            summary = config_data['summary']
            comparison_data.append({
                'config_name': config_name,
                'clustering_method': config_data['clustering_method'],
                'distance_metric': config_data['distance_metric'],
                'normalize_features': config_data['normalize_features'],
                'n_videos': summary['n_videos'],
                'total_keyframes': summary['total_keyframes'],
                'total_frames': summary['total_frames'],
                'avg_keyframes_per_video': summary['avg_keyframes_per_video'],
                'avg_clusters_per_video': summary['avg_clusters_per_video'],
                'avg_silhouette_score': summary['avg_silhouette_score'],
                'reduction_percentage': (1 - summary['total_keyframes'] / summary['total_frames']) * 100 if summary['total_frames'] > 0 else 0,
                'timestamp': config_data['timestamp'],
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Guardar comparación
        comparison_path = os.path.join(self.results_dir, 'comparison_summary.csv')
        df_comparison.to_csv(comparison_path, index=False)

        print(f"\n📊 Comparación de {len(df_comparison)} configuraciones:")
        print('=' * 80)
        print(df_comparison.to_string(index=False))
        print(f"\n✅ Comparación guardada en {comparison_path}")

        # Crear visualización comparativa
        if save_plot and len(df_comparison) > 0:
            self._plot_comparison(df_comparison)

        return df_comparison

    def _plot_comparison(self, df_comparison: pd.DataFrame):
        """Crea gráficos comparativos de las configuraciones."""
        n_configs = len(df_comparison)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Promedio de key frames por video
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(n_configs), df_comparison['avg_keyframes_per_video'])
        ax1.set_xlabel('Configuración')
        ax1.set_ylabel('Promedio Key Frames por Video')
        ax1.set_title('Promedio de Key Frames por Video')
        ax1.set_xticks(range(n_configs))
        ax1.set_xticklabels(df_comparison['config_name'], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(df_comparison['avg_keyframes_per_video']):
            ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom')

        # 2. Promedio de clusters por video
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(n_configs), df_comparison['avg_clusters_per_video'])
        ax2.set_xlabel('Configuración')
        ax2.set_ylabel('Promedio Clusters por Video')
        ax2.set_title('Promedio de Clusters por Video')
        ax2.set_xticks(range(n_configs))
        ax2.set_xticklabels(df_comparison['config_name'], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        for i, v in enumerate(df_comparison['avg_clusters_per_video']):
            ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom')

        # 3. Silhouette Score promedio
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(n_configs), df_comparison['avg_silhouette_score'])
        ax3.set_xlabel('Configuración')
        ax3.set_ylabel('Silhouette Score Promedio')
        ax3.set_title('Calidad de Clustering (Silhouette Score)')
        ax3.set_xticks(range(n_configs))
        ax3.set_xticklabels(df_comparison['config_name'], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Límite (0)')
        for i, v in enumerate(df_comparison['avg_silhouette_score']):
            ax3.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top')

        # 4. Porcentaje de reducción
        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(n_configs), df_comparison['reduction_percentage'])
        ax4.set_xlabel('Configuración')
        ax4.set_ylabel('Porcentaje de Reducción (%)')
        ax4.set_title('Compresión de Frames')
        ax4.set_xticks(range(n_configs))
        ax4.set_xticklabels(df_comparison['config_name'], rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        for i, v in enumerate(df_comparison['reduction_percentage']):
            ax4.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

        plt.tight_layout()

        plot_path = os.path.join(self.results_dir, 'comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráficos comparativos guardados en {plot_path}")
        plt.show()

    def load_configuration(
        self,
        config_name: str,
    ) -> Dict:
        """
        Carga una configuración guardada.

        Args:
            config_name: Nombre de la configuración a cargar

        Returns:
            Diccionario con datos de la configuración
        """
        config_path = os.path.join(self.results_dir, f"{config_name}_config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuración '{config_name}' no encontrada.")

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Cargar datasets
        clusters_path = os.path.join(self.results_dir, f"{config_name}_clusters.csv")
        stats_path = os.path.join(self.results_dir, f"{config_name}_stats.csv")
        keyframes_path = os.path.join(self.results_dir, f"{config_name}_keyframes.csv")

        config_data['df_clusters'] = pd.read_csv(clusters_path) if os.path.exists(clusters_path) else None
        config_data['df_stats'] = pd.read_csv(stats_path) if os.path.exists(stats_path) else None
        config_data['df_keyframes'] = pd.read_csv(keyframes_path) if os.path.exists(keyframes_path) else None

        return config_data
