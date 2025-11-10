"""
Módulo para selección de key frames desde clusters.
"""
from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class KeyFrameSelector:
    """
    Clase para seleccionar key frames desde clusters.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        random_state: int = 42,
        min_clusters_per_video: int = 5,
        max_clusters_per_video: int = 50,
        min_keyframe_ratio: float = 0.01,
        max_keyframe_ratio: float = 0.1,
        frames_per_second: float = 0.0,
        clustering_method: str = 'kmeans',
        distance_metric: str = 'euclidean',
        normalize_features: bool = False,
    ):
        """
        Inicializa el selector de key frames.

        Args:
            n_clusters: Número de clusters (si None, se calcula automáticamente)
            random_state: Semilla para reproducibilidad
            min_clusters_per_video: Mínimo número de clusters por video
            max_clusters_per_video: Máximo número de clusters por video
            min_keyframe_ratio: Ratio mínimo de key frames respecto al total (0.01 = 1%)
            max_keyframe_ratio: Ratio máximo de key frames respecto al total (0.1 = 10%)
            frames_per_second: (Opcional, desactivado por defecto) Frames por segundo del video
                             para calcular mínimo de clusters basado en duración.
                             Desactivado por defecto (0) porque después del filtrado SSIM
                             los frames ya no tienen frecuencia constante.
            clustering_method: Método de clustering ('kmeans', 'dbscan', 'agglomerative')
            distance_metric: Métrica de distancia ('euclidean', 'cosine')
            normalize_features: Si True, normaliza features L2 antes de clustering (útil para cosine)
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.min_clusters_per_video = min_clusters_per_video
        self.max_clusters_per_video = max_clusters_per_video
        self.min_keyframe_ratio = min_keyframe_ratio
        self.max_keyframe_ratio = max_keyframe_ratio
        self.frames_per_second = frames_per_second
        self.clustering_method = clustering_method
        self.distance_metric = distance_metric
        self.normalize_features = normalize_features

    def find_optimal_clusters(
        self,
        features: np.ndarray,
        max_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        n_frames: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> int:
        """
        Encuentra el número óptimo de clusters usando método del codo y silhouette.

        Args:
            features: Array numpy con features (n_samples, n_features)
            max_clusters: Número máximo de clusters a probar (si None, usa self.max_clusters_per_video)
            min_clusters: Número mínimo de clusters (si None, calcula basado en ratio mínimo)
            n_frames: Número total de frames (para calcular mínimo basado en ratio)
            duration: Duración del video en segundos (para calcular mínimo temporal)

        Returns:
            Número óptimo de clusters
        """
        n_samples = len(features)

        # Calcular mínimo de clusters basado en diferentes criterios
        min_clusters_computed = self.min_clusters_per_video

        # Criterio 1: Ratio mínimo de key frames
        if n_frames is not None:
            min_clusters_from_ratio = max(1, int(n_frames * self.min_keyframe_ratio))
            min_clusters_computed = max(min_clusters_computed, min_clusters_from_ratio)

        # Criterio 2: Duración del video (OPCIONAL - desactivado por defecto)
        # Nota: Este criterio se desactiva porque después del filtrado SSIM,
        # los frames ya no tienen frecuencia temporal constante, así que
        # usar la duración del video original no es apropiado.
        # Si quieres activarlo, establece frames_per_second > 0
        if duration is not None and self.frames_per_second > 0:
            min_clusters_from_duration = max(1, int(duration * self.frames_per_second))
            min_clusters_computed = max(min_clusters_computed, min_clusters_from_duration)

        # Usar mínimo proporcionado o calculado
        if min_clusters is None:
            min_clusters = min_clusters_computed
        else:
            min_clusters = max(min_clusters, min_clusters_computed)

        # Calcular máximo de clusters
        if max_clusters is None:
            max_clusters = self.max_clusters_per_video

        # Limitar max_clusters según número de muestras y ratio máximo
        if n_frames is not None:
            max_clusters_from_ratio = min(max_clusters, int(n_frames * self.max_keyframe_ratio))
            max_clusters = min(max_clusters, max_clusters_from_ratio)

        max_clusters = min(max_clusters, n_samples - 1)
        max_clusters = max(max_clusters, min_clusters)

        if n_samples <= min_clusters:
            return min_clusters

        # Normalizar features si se requiere (útil para cosine similarity)
        features_to_use = features.copy()
        if self.normalize_features or self.distance_metric == 'cosine':
            features_to_use = normalize(features, norm='l2')

        # Calcular inercia y silhouette para diferentes números de clusters
        inertias = []
        silhouettes = []
        k_range = range(min_clusters, max_clusters + 1)

        for k in k_range:
            if self.clustering_method == 'kmeans':
                # Configurar KMeans según métrica
                if self.distance_metric == 'cosine':
                    # Para cosine, usar precomputed similarity matrix
                    similarity_matrix = cosine_similarity(features_to_use)
                    # Convertir similitud a distancia (1 - similarity)
                    distance_matrix = 1 - similarity_matrix
                    # Usar MiniBatchKMeans con precomputed no es directo, usar agglomerative
                    clustering = AgglomerativeClustering(
                        n_clusters=k,
                        linkage='average',
                        metric='precomputed',
                    )
                    labels = clustering.fit_predict(distance_matrix)
                    # Calcular inercia manualmente para cosine
                    inertia = 0
                    for cluster_id in range(k):
                        cluster_mask = labels == cluster_id
                        if cluster_mask.sum() > 0:
                            cluster_features = features_to_use[cluster_mask]
                            centroid = cluster_features.mean(axis=0)
                            centroid = normalize(centroid.reshape(1, -1), norm='l2')
                            similarities = cosine_similarity(cluster_features, centroid)
                            distances = 1 - similarities.flatten()
                            inertia += distances.sum()
                else:
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = kmeans.fit_predict(features_to_use)
                    inertia = kmeans.inertia_
            else:
                # Para otros métodos, usar directamente
                labels, inertia = self._cluster_with_method(features_to_use, k)

            inertias.append(inertia)

            # Calcular silhouette (solo si hay al menos 2 clusters y más de 2 muestras)
            if k > 1 and n_samples > 2:
                try:
                    # Para cosine, usar metric='cosine' en silhouette
                    metric = 'cosine' if self.distance_metric == 'cosine' else 'euclidean'
                    sil_score = silhouette_score(features_to_use, labels, metric=metric)
                    silhouettes.append(sil_score)
                except Exception:
                    silhouettes.append(0)
            else:
                silhouettes.append(0)

        # Método del codo mejorado: detectar el punto de máxima curvatura
        optimal_k = self._find_elbow_point(inertias, k_range, min_clusters)

        # Si silhouette sugiere un k más bajo, considerar ese también
        if len(silhouettes) > 0 and max(silhouettes) > 0:
            silhouette_optimal = list(k_range)[np.argmax(silhouettes)]
            # Preferir el menor entre codo y silhouette (más conservador)
            optimal_k = min(optimal_k, silhouette_optimal)

        # Asegurar que no exceda el máximo permitido por ratio
        if n_frames is not None:
            max_k_from_ratio = int(n_frames * self.max_keyframe_ratio)
            optimal_k = min(optimal_k, max_k_from_ratio)

        # Asegurar que está dentro del rango válido
        optimal_k = max(min_clusters, min(optimal_k, max_clusters))

        # Si no se especificó n_clusters, usar el óptimo
        if self.n_clusters is None:
            self.n_clusters = optimal_k

        return optimal_k

    def _find_elbow_point(
        self,
        inertias: List[float],
        k_range: range,
        min_clusters: int,
    ) -> int:
        """
        Encuentra el punto del codo usando el método de la segunda derivada.

        Args:
            inertias: Lista de inercias para cada k
            k_range: Rango de valores de k probados
            min_clusters: Número mínimo de clusters

        Returns:
            Número óptimo de clusters (punto del codo)
        """
        if len(inertias) < 3:
            return min_clusters

        # Normalizar inercias (opcional, pero ayuda con la visualización)
        inertias = np.array(inertias)
        k_values = np.array(list(k_range))

        # Calcular primera derivada (tasa de cambio de inercia)
        first_derivative = np.diff(inertias)

        # Calcular segunda derivada (tasa de cambio de la primera derivada)
        # El punto del codo es donde la segunda derivada es máxima (máxima curvatura)
        if len(first_derivative) > 1:
            second_derivative = np.diff(first_derivative)

            # Encontrar el índice donde la segunda derivada es máxima
            # Esto indica el punto donde la curva de inercia cambia más abruptamente
            if len(second_derivative) > 0:
                # Usar valor absoluto de segunda derivada para encontrar el punto de inflexión
                elbow_idx = np.argmax(np.abs(second_derivative))

                # El índice en second_derivative corresponde a k_values[elbow_idx + 2]
                # porque: k_values[0], k_values[1] -> first_derivative[0]
                #         k_values[1], k_values[2] -> first_derivative[1], second_derivative[0]
                elbow_k = k_values[elbow_idx + 2] if elbow_idx + 2 < len(k_values) else k_values[-1]

                # Método alternativo: buscar el punto donde la reducción de inercia se estabiliza
                # Calcular la reducción porcentual de inercia
                stable_k = None
                if len(inertias) > 1:
                    reduction_rates = np.abs(first_derivative) / (inertias[:-1] + 1e-10)  # Evitar división por cero

                    # Encontrar el punto donde la reducción se vuelve pequeña (menos del 5% del valor anterior)
                    threshold = 0.05  # 5% de reducción
                    stable_points = np.where(reduction_rates < threshold)[0]

                    if len(stable_points) > 0:
                        # Tomar el primer punto donde la reducción se estabiliza
                        stable_k = k_values[stable_points[0] + 1]
                        # Usar el menor entre el método de segunda derivada y estabilización
                        if stable_k is not None:
                            elbow_k = min(elbow_k, stable_k)

                return int(elbow_k)

        # Fallback: usar el punto donde la reducción de inercia es mínima
        if len(first_derivative) > 0:
            min_reduction_idx = np.argmin(np.abs(first_derivative))
            return int(k_values[min_reduction_idx + 1])

        return min_clusters

    def _cluster_with_method(self, features: np.ndarray, n_clusters: int):
        """Método auxiliar para clustering con diferentes métodos."""
        if self.clustering_method == 'kmeans':
            if self.distance_metric == 'cosine':
                similarity_matrix = cosine_similarity(features)
                distance_matrix = 1 - similarity_matrix
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average',
                    metric='precomputed',
                )
                labels = clustering.fit_predict(distance_matrix)
                # Calcular inercia aproximada
                inertia = 0
                for cluster_id in range(n_clusters):
                    cluster_mask = labels == cluster_id
                    if cluster_mask.sum() > 0:
                        cluster_features = features[cluster_mask]
                        centroid = cluster_features.mean(axis=0)
                        centroid = normalize(centroid.reshape(1, -1), norm='l2')
                        similarities = cosine_similarity(cluster_features, centroid)
                        distances = 1 - similarities.flatten()
                        inertia += distances.sum()
                return labels, inertia
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(features)
                return labels, kmeans.inertia_
        elif self.clustering_method == 'agglomerative':
            if self.distance_metric == 'cosine':
                linkage = 'average'
                affinity = 'cosine'
            else:
                linkage = 'ward'
                affinity = 'euclidean'
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric=affinity,
            )
            labels = clustering.fit_predict(features)
            # Calcular inercia aproximada
            inertia = 0
            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_features = features[cluster_mask]
                    centroid = cluster_features.mean(axis=0)
                    if self.distance_metric == 'cosine':
                        centroid = normalize(centroid.reshape(1, -1), norm='l2')
                        similarities = cosine_similarity(cluster_features, centroid)
                        distances = 1 - similarities.flatten()
                    else:
                        distances = np.linalg.norm(cluster_features - centroid, axis=1)
                    inertia += distances.sum()
            return labels, inertia
        else:
            raise ValueError(f"Método de clustering desconocido: {self.clustering_method}")

    def cluster_frames(
        self,
        features: np.ndarray,
        n_clusters: Optional[int] = None,
    ) -> Tuple[np.ndarray, object]:
        """
        Clusterea frames usando el método configurado.

        Args:
            features: Array numpy con features (n_samples, n_features)
            n_clusters: Número de clusters (si None, usa el del constructor o calcula óptimo)

        Returns:
            Tuple con:
            - Array con labels de clusters (n_samples,)
            - Modelo de clustering entrenado (o dict con info si no es sklearn)
        """
        if n_clusters is None:
            if self.n_clusters is None:
                n_clusters = self.find_optimal_clusters(features)
            else:
                n_clusters = self.n_clusters
        else:
            n_clusters = min(n_clusters, len(features))

        # Ajustar n_clusters si hay muy pocas muestras
        n_clusters = min(n_clusters, len(features))

        # Validar que hay suficientes muestras para clustering
        if len(features) < 2:
            # Si hay menos de 2 frames, retornar todos como un solo cluster
            labels = np.zeros(len(features), dtype=int)
            class SingleClusterModel:
                def __init__(self, labels, features):
                    self.labels_ = labels
                    self.n_clusters = 1
                    self.cluster_centers_ = features.mean(axis=0, keepdims=True) if len(features) > 0 else features
            return labels, SingleClusterModel(labels, features)

        # Normalizar features si se requiere
        features_to_use = features.copy()
        if self.normalize_features or self.distance_metric == 'cosine':
            features_to_use = normalize(features, norm='l2')

        # Clustear según método
        if self.clustering_method == 'kmeans':
            if self.distance_metric == 'cosine':
                # Para cosine, necesitamos al menos 2 frames para crear matriz de similitud
                if len(features_to_use) < 2:
                    labels = np.zeros(len(features_to_use), dtype=int)
                    class SingleClusterModel:
                        def __init__(self, labels, features):
                            self.labels_ = labels
                            self.n_clusters = 1
                            self.cluster_centers_ = features.mean(axis=0, keepdims=True) if len(features) > 0 else features
                    return labels, SingleClusterModel(labels, features_to_use)

                similarity_matrix = cosine_similarity(features_to_use)
                distance_matrix = 1 - similarity_matrix

                # Asegurar que n_clusters no exceda el número de frames
                n_clusters = min(n_clusters, len(features_to_use))

                if n_clusters >= len(features_to_use):
                    # Si n_clusters >= número de frames, cada frame es su propio cluster
                    labels = np.arange(len(features_to_use))
                    class SingleClusterModel:
                        def __init__(self, labels, features):
                            self.labels_ = labels
                            self.n_clusters = len(labels)
                            self.cluster_centers_ = features
                    return labels, SingleClusterModel(labels, features_to_use)

                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average',
                    metric='precomputed',
                )
                labels = clustering.fit_predict(distance_matrix)
                # Crear un objeto similar a KMeans para compatibilidad
                class CosineClusterModel:
                    def __init__(self, labels, features, n_clusters):
                        self.labels_ = labels
                        self.n_clusters = n_clusters
                        # Calcular centroides como promedio de features normalizados
                        self.cluster_centers_ = np.array([
                            features[labels == i].mean(axis=0)
                            for i in range(n_clusters)
                        ])
                model = CosineClusterModel(labels, features_to_use, n_clusters)
            else:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    n_init=10,
                )
                labels = kmeans.fit_predict(features_to_use)
                model = kmeans
        elif self.clustering_method == 'agglomerative':
            # Validar que hay suficientes muestras (mínimo 2)
            if len(features_to_use) < 2:
                labels = np.zeros(len(features_to_use), dtype=int)
                class SingleClusterModel:
                    def __init__(self, labels, features):
                        self.labels_ = labels
                        self.n_clusters = 1
                        self.cluster_centers_ = features.mean(axis=0, keepdims=True) if len(features) > 0 else features
                return labels, SingleClusterModel(labels, features_to_use)

            # Asegurar que n_clusters no exceda el número de frames
            n_clusters = min(n_clusters, len(features_to_use))

            if self.distance_metric == 'cosine':
                linkage = 'average'
                affinity = 'cosine'
            else:
                linkage = 'ward'
                affinity = 'euclidean'

            # Si n_clusters >= número de frames, cada frame es su propio cluster
            if n_clusters >= len(features_to_use):
                labels = np.arange(len(features_to_use))
                class SingleClusterModel:
                    def __init__(self, labels, features):
                        self.labels_ = labels
                        self.n_clusters = len(labels)
                        self.cluster_centers_ = features
                return labels, SingleClusterModel(labels, features_to_use)

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
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
            model = AgglomerativeClusterModel(labels, features_to_use, n_clusters)
        else:
            raise ValueError(f"Método de clustering desconocido: {self.clustering_method}")

        return labels, model

    def select_keyframes_from_clusters(
        self,
        features: np.ndarray,
        frame_paths: List[str],
        cluster_labels: np.ndarray,
        centroids: np.ndarray,
    ) -> List[str]:
        """
        Selecciona key frames: el frame más cercano al centroide de cada cluster.

        Args:
            features: Array numpy con features
            frame_paths: Lista de rutas de frames
            cluster_labels: Labels de clusters para cada frame
            centroids: Centroides de los clusters

        Returns:
            Lista de rutas de key frames seleccionados
        """
        keyframes = []
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            # Obtener índices de frames en este cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                continue

            # Obtener features del cluster
            cluster_features = features[cluster_indices]

            # Obtener centroide del cluster
            centroid = centroids[cluster_id]

            # Calcular distancias al centroide según métrica configurada
            if self.distance_metric == 'cosine':
                # Normalizar centroide y features si es necesario
                centroid_norm = normalize(centroid.reshape(1, -1), norm='l2')
                cluster_features_norm = normalize(cluster_features, norm='l2')
                similarities = cosine_similarity(cluster_features_norm, centroid_norm)
                distances = 1 - similarities.flatten()
            else:
                distances = cdist(cluster_features, centroid.reshape(1, -1), metric='euclidean').flatten()

            # Encontrar el frame más cercano al centroide
            closest_idx = np.argmin(distances)
            frame_idx = cluster_indices[closest_idx]

            keyframes.append(frame_paths[frame_idx])

        return keyframes

    def select_keyframes(
        self,
        features: np.ndarray,
        frame_paths: List[str],
        n_clusters: Optional[int] = None,
        n_frames: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> Tuple[List[str], np.ndarray, KMeans, Dict]:
        """
        Selecciona key frames usando clustering.

        Args:
            features: Array numpy con features
            frame_paths: Lista de rutas de frames
            n_clusters: Número de clusters (si None, calcula óptimo)
            n_frames: Número total de frames (para calcular mínimo)
            duration: Duración del video en segundos (para calcular mínimo)

        Returns:
            Tuple con:
            - Lista de rutas de key frames
            - Labels de clusters
            - Modelo KMeans entrenado
            - Diccionario con estadísticas
        """
        # Si n_frames no se proporciona, usar len(frame_paths)
        if n_frames is None:
            n_frames = len(frame_paths)

        # Manejar casos especiales con muy pocos frames
        if len(features) == 0 or len(frame_paths) == 0:
            # Sin frames, retornar vacío
            return [], np.array([]), None, {
                'n_clusters': 0,
                'n_keyframes': 0,
                'n_frames': 0,
                'reduction_percentage': 0,
                'silhouette_score': None,
            }

        if len(features) == 1:
            # Solo 1 frame, retornarlo como único keyframe
            return [frame_paths[0]], np.array([0]), None, {
                'n_clusters': 1,
                'n_keyframes': 1,
                'n_frames': 1,
                'reduction_percentage': 0,
                'silhouette_score': None,
            }

        # Calcular número óptimo de clusters si no se especifica
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(
                features,
                n_frames=n_frames,
                duration=duration,
            )

        # Asegurar que n_clusters no sea mayor que el número de frames
        n_clusters = min(n_clusters, len(features))

        # Si después de ajustar solo hay 1 cluster posible, manejar caso especial
        if n_clusters >= len(features):
            # Todos los frames son keyframes
            labels = np.arange(len(features))
            class SingleClusterModel:
                def __init__(self, labels, features):
                    self.labels_ = labels
                    self.n_clusters = 1
                    self.cluster_centers_ = features.mean(axis=0, keepdims=True)
            model = SingleClusterModel(labels, features)
            keyframes = self.select_keyframes_from_clusters(
                features, frame_paths, labels, model.cluster_centers_,
            )
            stats = {
                'n_clusters': 1,
                'n_keyframes': len(keyframes),
                'n_frames': len(frame_paths),
                'reduction_percentage': (1 - len(keyframes) / len(frame_paths)) * 100 if len(frame_paths) > 0 else 0,
                'silhouette_score': None,
            }
            return keyframes, labels, model, stats

        # Clustear frames
        cluster_labels, kmeans_model = self.cluster_frames(features, n_clusters)

        # Seleccionar key frames
        keyframes = self.select_keyframes_from_clusters(
            features,
            frame_paths,
            cluster_labels,
            kmeans_model.cluster_centers_,
        )

        # Estadísticas
        stats = {
            'n_clusters': len(np.unique(cluster_labels)),
            'n_keyframes': len(keyframes),
            'n_frames': len(frame_paths),
            'reduction_percentage': (1 - len(keyframes) / len(frame_paths)) * 100 if len(frame_paths) > 0 else 0,
        }

        # Calcular silhouette si es posible
        if len(features) > 1 and stats['n_clusters'] > 1:
            try:
                stats['silhouette_score'] = silhouette_score(
                    features, cluster_labels,
                )
            except Exception:
                stats['silhouette_score'] = None
        else:
            stats['silhouette_score'] = None

        return keyframes, cluster_labels, kmeans_model, stats
