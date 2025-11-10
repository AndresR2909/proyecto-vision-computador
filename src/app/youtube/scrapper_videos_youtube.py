"""
Módulo para descargar videos de YouTube y extraer frames.
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import pandas as pd
import yt_dlp
from tqdm import tqdm


class ScrapperVideosYoutube:
    """
    Clase para descargar videos de YouTube y extraer frames.
    """

    def __init__(
        self,
        downloads_dir: str = 'videos_youtube',
        frames_dir: str = 'frames',
        frame_interval_sec: int = 2,
        max_retries: int = 3,
        video_format: str = '135/136/137',
        quiet: bool = False,
    ):
        """
        Inicializa el scrapper de videos de YouTube.

        Args:
            downloads_dir: Directorio donde se guardan los videos descargados
            frames_dir: Directorio donde se guardan los frames extraídos
            frame_interval_sec: Intervalo en segundos entre frames extraídos
            max_retries: Número máximo de reintentos para descargas fallidas
            video_format: Formato de video para yt_dlp (por defecto: '135/136/137')
            quiet: Si True, suprime la salida de yt_dlp
        """
        self.downloads_dir = downloads_dir
        self.frames_dir = frames_dir
        self.frame_interval_sec = frame_interval_sec
        self.max_retries = max_retries
        self.video_format = video_format
        self.quiet = quiet

        # Crear directorios si no existen
        os.makedirs(self.downloads_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

    def _video_already_downloaded(
        self,
        video_id: str,
        output_dir: str,
    ) -> Optional[str]:
        """
        Verifica si un video ya fue descargado previamente.

        Args:
            video_id: ID del video de YouTube
            output_dir: Directorio donde buscar el video

        Returns:
            Ruta al video si existe, None si no existe
        """
        if not os.path.exists(output_dir):
            return None

        # Buscar en todos los subdirectorios (canales)
        for channel_dir in os.listdir(output_dir):
            channel_path = os.path.join(output_dir, channel_dir)
            if not os.path.isdir(channel_path):
                continue

            # Buscar archivo de video que contenga el video_id
            for video_file in os.listdir(channel_path):
                if video_file.endswith(('.mp4', '.webm', '.mkv')) and video_id in video_file:
                    video_path = os.path.join(channel_path, video_file)
                    if os.path.exists(video_path):
                        return video_path

        return None

    def download_videos_robust(
        self,
        video_urls: List[str],
        output_dir: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Descarga videos con manejo de errores y reintentos.
        Verifica si el video ya existe antes de descargarlo.

        Args:
            video_urls: Lista de URLs de videos de YouTube
            output_dir: Directorio de salida (por defecto usa self.downloads_dir)

        Returns:
            Tupla con (lista de URLs exitosas, lista de URLs fallidas)
        """
        if output_dir is None:
            output_dir = self.downloads_dir

        successful_downloads = []
        failed_downloads = []
        skipped_downloads = []

        for i, url in enumerate(video_urls):
            print(f"\n--- Procesando video {i+1}/{len(video_urls)} ---")
            print(f"URL: {url}")

            # Extraer video_id de la URL
            video_id = None
            try:
                import re
                patterns = [
                    r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
                    r'youtu\.be\/([0-9A-Za-z_-]{11})',
                ]
                for pattern in patterns:
                    match = re.search(pattern, url)
                    if match:
                        video_id = match.group(1)
                        break
            except Exception:
                pass

            # Verificar si el video ya existe
            if video_id:
                existing_video = self._video_already_downloaded(
                    video_id, output_dir,
                )
                if existing_video:
                    print(
                        f"⏭️  Video ya descargado: {existing_video}",
                    )
                    skipped_downloads.append(url)
                    successful_downloads.append(url)
                    continue

            # Intentar descargar el video
            for attempt in range(self.max_retries):
                try:
                    ydl_opts = {
                        'outtmpl': os.path.join(
                            output_dir + '/' + '%(uploader)s',
                            '%(id)s_%(upload_date)s.%(ext)s',
                        ),
                        # Formato flexible: intenta el formato solicitado,
                        # si no está disponible, usa el mejor formato <= 720p
                        'format': (
                            f'{self.video_format}/'
                            'best[height<=720]/'
                            'bestvideo[height<=720]+bestaudio/best[height<=720]/'
                            'best'
                        ),
                        'noplaylist': True,
                        'quiet': self.quiet,
                        'retries': 3,
                        'fragment_retries': 3,
                        'socket_timeout': 30,
                        'http_chunk_size': 10485760,  # 10MB chunks
                        # Opciones para evitar bloqueo HTTP 403
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['android', 'web'],
                                'player_skip': ['webpage', 'configs'],
                            },
                        },
                        'user_agent': (
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                            'AppleWebKit/537.36 (KHTML, like Gecko) '
                            'Chrome/120.0.0.0 Safari/537.36'
                        ),
                        'referer': 'https://www.youtube.com/',
                        'no_warnings': False,
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])

                    print(f"✅ Descarga exitosa: {url}")
                    successful_downloads.append(url)
                    break

                except Exception as e:
                    print(f"❌ Intento {attempt + 1} falló: {str(e)}")
                    if attempt < self.max_retries - 1:
                        wait_time = random.uniform(5, 15)  # Espera aleatoria entre 5-15 segundos
                        print(f"⏳ Esperando {wait_time:.1f} segundos antes del siguiente intento...")
                        time.sleep(wait_time)
                    else:
                        print(f"💥 Falló definitivamente: {url}")
                        failed_downloads.append(url)

        print(f"\n📊 RESUMEN DE DESCARGAS:")
        print(f"✅ Exitosas: {len(successful_downloads)}")
        if skipped_downloads:
            print(f"⏭️  Omitidas (ya existían): {len(skipped_downloads)}")
        print(f"❌ Fallidas: {len(failed_downloads)}")

        if failed_downloads:
            print(f"\n🔴 Videos que fallaron:")
            for url in failed_downloads:
                print(f"  - {url}")

        return successful_downloads, failed_downloads

    def extract_frames_from_video(
        self,
        video_path: str,
        frames_dir: Optional[str] = None,
        step_sec: Optional[int] = None,
    ) -> Tuple[float, List[str]]:
        """
        Extrae frames de un video a intervalos regulares.

        Args:
            video_path: Ruta al archivo de video
            frames_dir: Directorio donde guardar los frames (por defecto usa self.frames_dir)
            step_sec: Intervalo en segundos entre frames (por defecto usa self.frame_interval_sec)

        Returns:
            Tupla con (duración del video en segundos, lista de rutas de frames guardados)
        """
        if frames_dir is None:
            frames_dir = self.frames_dir
        if step_sec is None:
            step_sec = self.frame_interval_sec

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        frame_paths = []
        success, image = cap.read()
        count = 0
        sec = 0
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        while success:
            if int(cap.get(cv2.CAP_PROP_POS_MSEC)) >= sec * 1000:
                frame_path = os.path.join(frames_dir, f"{base_name}_frame{count}.jpg")
                cv2.imwrite(frame_path, image)
                frame_paths.append(frame_path)
                sec += step_sec
            success, image = cap.read()
            count += 1

        cap.release()
        return duration, frame_paths

    def process_all_videos(
        self,
        downloads_dir: Optional[str] = None,
        frames_dir: Optional[str] = None,
        frame_interval_sec: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Procesa todos los videos descargados y extrae frames, creando un DataFrame
        con la información de cada video.

        Args:
            downloads_dir: Directorio con videos descargados (por defecto usa self.downloads_dir)
            frames_dir: Directorio donde guardar frames (por defecto usa self.frames_dir)
            frame_interval_sec: Intervalo en segundos entre frames (por defecto usa self.frame_interval_sec)

        Returns:
            DataFrame con información de videos procesados (video_id, channel, video_date,
            video_filename, duration, n_frames, frame_paths)
        """
        if downloads_dir is None:
            downloads_dir = self.downloads_dir
        if frames_dir is None:
            frames_dir = self.frames_dir
        if frame_interval_sec is None:
            frame_interval_sec = self.frame_interval_sec

        patron = r'^(.+)_(\d{8})\.mp4$'

        # Filtrar archivos ocultos como .DS_Store
        archivos_filtrados = list(
            filter(lambda x: x != '.DS_Store', os.listdir(downloads_dir)),
        )

        eda_data = []

        for channel in tqdm(archivos_filtrados, desc='Procesando canales'):
            path_channel = os.path.join(downloads_dir, channel)
            path_frames = os.path.join(frames_dir, channel)
            os.makedirs(path_frames, exist_ok=True)

            videos = [v for v in os.listdir(path_channel) if v.endswith('.mp4')]

            for video in tqdm(videos, desc=f"Procesando videos de {channel}", leave=False):
                video_path = os.path.join(path_channel, video)
                duration, frame_paths = self.extract_frames_from_video(
                    video_path, path_frames, frame_interval_sec,
                )

                match = re.match(patron, video)
                if match:
                    video_id = match.group(1)
                    fecha = match.group(2)

                    eda_data.append({
                        'video_id': video_id,
                        'channel': channel,
                        'video_date': fecha,
                        'video_filename': video,
                        'duration': duration,
                        'n_frames': len(frame_paths),
                        'frame_paths': frame_paths,
                    })
                else:
                    print(f"⚠️  Advertencia: No se pudo parsear el nombre del video: {video}")

        df_videos = pd.DataFrame(eda_data)
        return df_videos

    def create_youtube_links_from_dataframe(
        self,
        df: pd.DataFrame,
        video_id_column: str = 'video_id',
    ) -> List[str]:
        """
        Crea una lista de links de YouTube a partir de un DataFrame con video_ids.

        Args:
            df: DataFrame con información de videos
            video_id_column: Nombre de la columna que contiene los video_ids

        Returns:
            Lista de URLs de YouTube
        """
        video_ids = df[video_id_column].unique()
        youtube_links = [
            f"https://www.youtube.com/watch?v={video_id}"
            for video_id in video_ids
        ]
        return youtube_links

    def save_download_results(
        self,
        successful: List[str],
        failed: List[str],
        output_path_successful: str = 'links_successful.csv',
        output_path_failed: str = 'links_failed.csv',
    ) -> None:
        """
        Guarda los resultados de las descargas en archivos CSV.

        Args:
            successful: Lista de URLs exitosas
            failed: Lista de URLs fallidas
            output_path_successful: Ruta para guardar URLs exitosas
            output_path_failed: Ruta para guardar URLs fallidas
        """
        if successful:
            links_successful = pd.DataFrame(successful, columns=['url_successful'])
            links_successful.to_csv(output_path_successful, index=False)
            print(f"✅ URLs exitosas guardadas en: {output_path_successful}")

        if failed:
            links_failed = pd.DataFrame(failed, columns=['url_failed'])
            links_failed.to_csv(output_path_failed, index=False)
            print(f"❌ URLs fallidas guardadas en: {output_path_failed}")
