"""
Módulo para gestionar el cache del pipeline de procesamiento de videos.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class PipelineCacheManager:
    """
    Gestor de cache para el pipeline de procesamiento de videos.
    """

    def __init__(self, cache_dir: str):
        """
        Inicializa el gestor de cache.

        Args:
            cache_dir: Directorio donde se guardará el cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, video_id: str, cache_type: str) -> Path:
        """
        Obtiene la ruta del archivo de cache.

        Args:
            video_id: ID del video
            cache_type: Tipo de cache (frames_filtered, embeddings, keyframes, etc.)

        Returns:
            Ruta al archivo de cache
        """
        return self.cache_dir / f"{video_id}_{cache_type}.pkl"

    def get_metadata_path(self, video_id: str) -> Path:
        """
        Obtiene la ruta del archivo de metadata del cache.

        Args:
            video_id: ID del video

        Returns:
            Ruta al archivo de metadata
        """
        return self.cache_dir / f"{video_id}_metadata.json"

    def save_cache(
        self,
        video_id: str,
        cache_type: str,
        data: Any,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Guarda datos en cache.

        Args:
            video_id: ID del video
            cache_type: Tipo de cache
            data: Datos a guardar
            metadata: Metadata adicional (opcional)

        Returns:
            True si se guardó exitosamente
        """
        try:
            cache_path = self.get_cache_path(video_id, cache_type)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)

            # Guardar metadata si se proporciona
            if metadata:
                metadata_path = self.get_metadata_path(video_id)
                existing_metadata = self.load_metadata(video_id) or {}
                existing_metadata[cache_type] = metadata
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"Error guardando cache: {e}")
            return False

    def load_cache(
        self,
        video_id: str,
        cache_type: str,
    ) -> Optional[Any]:
        """
        Carga datos desde cache.

        Args:
            video_id: ID del video
            cache_type: Tipo de cache

        Returns:
            Datos cargados o None si no existe
        """
        cache_path = self.get_cache_path(video_id, cache_type)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error cargando cache: {e}")
            return None

    def load_metadata(self, video_id: str) -> Optional[Dict]:
        """
        Carga metadata del cache.

        Args:
            video_id: ID del video

        Returns:
            Metadata o None si no existe
        """
        metadata_path = self.get_metadata_path(video_id)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando metadata: {e}")
            return None

    def cache_exists(self, video_id: str, cache_type: str) -> bool:
        """
        Verifica si existe cache para un video y tipo específico.

        Args:
            video_id: ID del video
            cache_type: Tipo de cache

        Returns:
            True si existe el cache
        """
        cache_path = self.get_cache_path(video_id, cache_type)
        return cache_path.exists()

    def delete_cache(self, video_id: str, cache_type: Optional[str] = None) -> bool:
        """
        Elimina cache de un video.

        Args:
            video_id: ID del video
            cache_type: Tipo de cache a eliminar (None para eliminar todo)

        Returns:
            True si se eliminó exitosamente
        """
        try:
            if cache_type:
                # Eliminar cache específico
                cache_path = self.get_cache_path(video_id, cache_type)
                if cache_path.exists():
                    cache_path.unlink()
            else:
                # Eliminar todo el cache del video
                pattern = f"{video_id}_*.pkl"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()

                # Eliminar metadata
                metadata_path = self.get_metadata_path(video_id)
                if metadata_path.exists():
                    metadata_path.unlink()

            return True
        except Exception as e:
            print(f"Error eliminando cache: {e}")
            return False

    def clear_all_cache(self) -> bool:
        """
        Elimina todo el cache.

        Returns:
            True si se eliminó exitosamente
        """
        try:
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            for metadata_file in self.cache_dir.glob('*.json'):
                metadata_file.unlink()
            return True
        except Exception as e:
            print(f"Error eliminando todo el cache: {e}")
            return False

    def get_cache_info(self, video_id: str) -> Dict[str, bool]:
        """
        Obtiene información sobre qué tipos de cache existen para un video.

        Args:
            video_id: ID del video

        Returns:
            Diccionario con tipos de cache y si existen
        """
        cache_types = [
            'frames_filtered',
            'embeddings',
            'keyframes',
            'keyframe_stats',
            'classified_frames',
            'frame_categories',
        ]

        return {
            cache_type: self.cache_exists(video_id, cache_type)
            for cache_type in cache_types
        }
