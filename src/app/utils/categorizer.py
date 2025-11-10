"""
Módulo para categorización global de key frames.
"""
from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class FrameCategorizer:
    """
    Clase para categorizar key frames a nivel global.
    """

    def __init__(
        self,
        n_categories: Optional[int] = None,
        random_state: int = 42,
        clustering_method: str = 'kmeans',
        distance_metric: str = 'euclidean',
        normalize_features: bool = False,
    ):
        """
        Inicializa el categorizador.

        Args:
            n_categories: Número de categorías (si None, se calcula automáticamente)
            random_state: Semilla para reproducibilidad
            clustering_method: Método de clustering ('kmeans', 'agglomerative')
            distance_metric: Métrica de distancia ('euclidean', 'cosine')
            normalize_features: Si True, normaliza features L2 antes de clustering (útil para cosine)
        """
        self.n_categories = n_categories
        self.random_state = random_state
        self.clustering_method = clustering_method
        self.distance_metric = distance_metric
        self.normalize_features = normalize_features
        self.kmeans_model = None

    def find_optimal_categories(
        self,
        features: np.ndarray,
        max_categories: int = 20,
        min_categories: int = 2,
    ) -> int:
        """
        Encuentra el número óptimo de categorías usando silhouette.

        Args:
            features: Array numpy con features de todos los key frames
            max_categories: Número máximo de categorías a probar
            min_categories: Número mínimo de categorías

        Returns:
            Número óptimo de categorías
        """
        n_samples = len(features)

        # Limitar max_categories según número de muestras
        max_categories = min(max_categories, n_samples - 1)
        max_categories = max(max_categories, min_categories)

        if n_samples <= min_categories:
            return min_categories

        # Normalizar features si se requiere
        features_to_use = features.copy()
        if self.normalize_features or self.distance_metric == 'cosine':
            features_to_use = normalize(features, norm='l2')

        # Calcular silhouette para diferentes números de categorías
        silhouettes = []
        k_range = range(min_categories, max_categories + 1)

        for k in k_range:
            # Clustear según método
            if self.clustering_method == 'kmeans':
                if self.distance_metric == 'cosine':
                    # Para cosine, usar precomputed similarity matrix
                    similarity_matrix = cosine_similarity(features_to_use)
                    distance_matrix = 1 - similarity_matrix
                    clustering = AgglomerativeClustering(
                        n_clusters=k,
                        linkage='average',
                        metric='precomputed',
                    )
                    labels = clustering.fit_predict(distance_matrix)
                else:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = kmeans.fit_predict(features_to_use)
            elif self.clustering_method == 'agglomerative':
                if self.distance_metric == 'cosine':
                    linkage = 'average'
                    affinity = 'cosine'
                else:
                    linkage = 'ward'
                    affinity = 'euclidean'
                clustering = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=linkage,
                    metric=affinity,
                )
                labels = clustering.fit_predict(features_to_use)
            else:
                # Fallback a kmeans
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(features_to_use)

            # Calcular silhouette (solo si hay al menos 2 categorías y más de 2 muestras)
            if k > 1 and n_samples > 2:
                try:
                    # Para cosine, usar metric='cosine' en silhouette
                    metric = 'cosine' if self.distance_metric == 'cosine' else 'euclidean'
                    sil_score = silhouette_score(features_to_use, labels, metric=metric)
                    silhouettes.append(sil_score)
                except:
                    silhouettes.append(0)
            else:
                silhouettes.append(0)

        # Seleccionar k que maximiza silhouette
        if len(silhouettes) > 0 and max(silhouettes) > 0:
            optimal_k = list(k_range)[np.argmax(silhouettes)]
        else:
            optimal_k = min_categories

        # Si no se especificó n_categories, usar el óptimo
        if self.n_categories is None:
            self.n_categories = optimal_k

        return optimal_k

    def categorize_keyframes(
        self,
        features: np.ndarray,
        n_categories: Optional[int] = None,
    ) -> Tuple[np.ndarray, object]:
        """
        Categoriza key frames usando clustering global.

        Args:
            features: Array numpy con features de todos los key frames
            n_categories: Número de categorías (si None, usa el del constructor o calcula óptimo)

        Returns:
            Tuple con:
            - Array con labels de categorías (n_samples,)
            - Modelo de clustering entrenado (KMeans, AgglomerativeClustering, o wrapper compatible)
        """
        if n_categories is None:
            if self.n_categories is None:
                n_categories = self.find_optimal_categories(features)
            else:
                n_categories = self.n_categories
        else:
            n_categories = min(n_categories, len(features))

        # Ajustar n_categories si hay muy pocas muestras
        n_categories = min(n_categories, len(features))

        # Normalizar features si se requiere
        features_to_use = features.copy()
        if self.normalize_features or self.distance_metric == 'cosine':
            features_to_use = normalize(features, norm='l2')

        # Clustear según método
        if self.clustering_method == 'kmeans':
            if self.distance_metric == 'cosine':
                # Para cosine, usar precomputed similarity matrix
                similarity_matrix = cosine_similarity(features_to_use)
                distance_matrix = 1 - similarity_matrix
                clustering = AgglomerativeClustering(
                    n_clusters=n_categories,
                    linkage='average',
                    metric='precomputed',
                )
                labels = clustering.fit_predict(distance_matrix)
                # Crear un objeto similar a KMeans para compatibilidad
                class CosineClusterModel:
                    def __init__(self, labels, features, n_clusters):
                        self.labels_ = labels
                        self.n_clusters = n_clusters
                        self.cluster_centers_ = np.array([
                            features[labels == i].mean(axis=0)
                            for i in range(n_clusters)
                        ])
                model = CosineClusterModel(labels, features_to_use, n_categories)
            else:
                kmeans = KMeans(
                    n_clusters=n_categories,
                    random_state=self.random_state,
                    n_init=10,
                )
                labels = kmeans.fit_predict(features_to_use)
                model = kmeans
        elif self.clustering_method == 'agglomerative':
            if self.distance_metric == 'cosine':
                linkage = 'average'
                affinity = 'cosine'
            else:
                linkage = 'ward'
                affinity = 'euclidean'
            clustering = AgglomerativeClustering(
                n_clusters=n_categories,
                linkage=linkage,
                metric=affinity,
            )
            labels = clustering.fit_predict(features_to_use)
            # Crear objeto con centroides para compatibilidad
            class AgglomerativeClusterModel:
                def __init__(self, labels, features, n_clusters):
                    self.labels_ = labels
                    self.n_clusters = n_clusters
                    self.cluster_centers_ = np.array([
                        features[labels == i].mean(axis=0)
                        for i in range(n_clusters)
                    ])
            model = AgglomerativeClusterModel(labels, features_to_use, n_categories)
        else:
            # Fallback a kmeans
            kmeans = KMeans(
                n_clusters=n_categories,
                random_state=self.random_state,
                n_init=10,
            )
            labels = kmeans.fit_predict(features_to_use)
            model = kmeans

        self.kmeans_model = model

        return labels, model

    def analyze_categories(
        self,
        category_labels: np.ndarray,
        video_ids: List[str],
        channel_names: List[str],
    ) -> pd.DataFrame:
        """
        Analiza la distribución de categorías por video y canal.

        Args:
            category_labels: Labels de categorías para cada key frame
            video_ids: Lista de video IDs
            channel_names: Lista de nombres de canales

        Returns:
            DataFrame con análisis de categorías
        """
        df = pd.DataFrame({
            'video_id': video_ids,
            'channel_name': channel_names,
            'category_id': category_labels,
        })

        return df

    def get_category_distribution(
        self,
        category_labels: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Obtiene la distribución de categorías.

        Args:
            category_labels: Labels de categorías
            channel_names: Lista opcional de nombres de canales

        Returns:
            Diccionario con estadísticas de distribución
        """
        unique_categories = np.unique(category_labels)
        category_counts = {
            int(cat): int(np.sum(category_labels == cat))
            for cat in unique_categories
        }

        stats = {
            'n_categories': len(unique_categories),
            'category_counts': category_counts,
            'total_frames': len(category_labels),
        }

        # Distribución por canal si se proporciona
        if channel_names:
            df = pd.DataFrame({
                'channel': channel_names,
                'category': category_labels,
            })
            channel_dist = df.groupby(['channel', 'category']).size().to_dict()
            stats['channel_distribution'] = channel_dist

        return stats
