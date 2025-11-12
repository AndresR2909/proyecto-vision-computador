from __future__ import annotations

import datetime
import json
import logging
import os
import re
import warnings
from pathlib import Path

import isodate
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv('API_KEY_YOUTUBE')
CHANNELS = [
    'Bolsas hoy | Invierte y Crece',
    'ARENA ALFA',
    'Bitcoin hoy Oficial',
    'Bolsas hoy | Esteban Pérez',
]
BASE_URL = os.getenv('BASE_URL_YOUTUBE')

# Configurar directorio de cache
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_EXPIRY_HOURS = 1  # Cache válido por 1 hora


class YoutubeIngest:
    """Class to ingest transcriptions and metadata from videos Youtube"""

    def __init__(self):
        self.list_metadata = []
        self.channels = CHANNELS
        self.url = BASE_URL

    def ingest_youtube_videos_from_channels(
        self, channels: list = CHANNELS, daysback: int = 1,
    ) -> list:
        list_ingest = []
        for canal in channels:
            logging.info(f"Ingesting data from {canal}")
            channel_id = self.get_channel_id(canal)
            if channel_id:
                videos = self.get_metadata_videos_delta(channel_id, daysback=daysback)
                if len(videos) > 0:
                    for video in videos:
                        video_id = video['videoId']
                        video['transcript'] = self.get_transcript_by_id(video_id)
                        list_ingest.append(video)
            else:
                logging.error(f"No se pudo obtener el channel_id de {canal}")
        self.list_metadata = list_ingest
        return list_ingest

    def get_last_videos_metadata_from_channels(
        self, channel: str, daysback: int = 1,
    ) -> list:
        """
        Obtiene metadatos de videos de un canal con cache.

        Args:
            channel: Nombre del canal
            daysback: Días hacia atrás para buscar videos

        Returns:
            Lista de metadatos de videos
        """
        # Generar nombre de archivo de cache
        cache_key = f"{channel}_{daysback}".replace(' ', '_').replace('|', '_')
        cache_file = CACHE_DIR / f"{cache_key}.json"

        # Verificar si existe cache válido
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                # Verificar si el cache no ha expirado
                cache_time = datetime.datetime.fromisoformat(
                    cache_data.get('timestamp', '2000-01-01T00:00:00'),
                )
                expiry_time = cache_time + datetime.timedelta(
                    hours=CACHE_EXPIRY_HOURS,
                )

                if datetime.datetime.now() < expiry_time:
                    logging.info(
                        f"✅ Usando cache para {channel} "
                        f"(válido hasta {expiry_time})",
                    )
                    return cache_data.get('videos', [])
                else:
                    logging.info(
                        f"⏰ Cache expirado para {channel}, "
                        f"actualizando desde API...",
                    )
            except Exception as e:
                logging.warning(
                    f"⚠️ Error leyendo cache para {channel}: {e}. "
                    f"Actualizando desde API...",
                )

        # Obtener videos desde API
        video_list = []
        logging.info(f"Ingesting data from {channel}")
        channel_id = self.get_channel_id(channel)
        if channel_id:
            videos = self.get_metadata_videos_delta(channel_id, daysback=daysback)
            if len(videos) > 0:
                for video in videos:
                    video_id = video['videoId']
                    video_list.append(video)
                    logging.info(f"Ingesting data from {video_id}")
        else:
            logging.error(f"No se pudo obtener el channel_id de {channel}")

        # Guardar en cache
        try:
            cache_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'channel': channel,
                'daysback': daysback,
                'videos': video_list,
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Cache guardado para {channel}")
        except Exception as e:
            logging.warning(f"⚠️ Error guardando cache para {channel}: {e}")

        return video_list

    @staticmethod
    def extract_video_id_from_url(url: str) -> str:
        """Extract the video ID from a YouTube URL"""
        # Handles various YouTube URL formats
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # standard and embed URLs
            r'youtu\.be\/([0-9A-Za-z_-]{11})',  # short URLs
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def get_transcript_by_id(video_id, language='es') -> str:
        """Get the transcript of a video by its id"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[language],
            )
            full_transcript = ' '.join([entry['text'] for entry in transcript])
            return full_transcript
        except Exception as e:
            logging.error(f"Error al obtener el transcript del video {video_id}: {e}")
            return None

    @staticmethod
    def get_channel_id(channel_name: str) -> str:
        """Get the channel id from the channel name"""
        url = (
            f"{BASE_URL}search?"
            f"key={API_KEY}&"
            f"q={channel_name}&"
            f"part=snippet&type=channel&maxResults=1"
        )
        response = requests.get(url)
        if response.status_code == 200:
            items = response.json().get('items', [])
            if items:
                return items[0]['snippet']['channelId']
        return None

    def get_metadata_videos_delta(self, channel_id, daysback=1):
        """Get the metadata of the videos from a channel"""
        delta = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        ) - datetime.timedelta(days=daysback)
        delta_str = delta.strftime('%Y-%m-%dT%H:%M:%SZ')
        video_list = []
        url = (
            f"{BASE_URL}search?"
            f"key={API_KEY}&"
            f"channelId={channel_id}&"
            f"part=snippet&id&"
            f"order=date&"
            f"publishedAfter={delta_str}&"
            f"maxResults=99"
        )

        response = requests.get(url)
        if response.status_code == 200:
            videos = response.json().get('items', [])
            for video in videos:
                video_id = video['id'].get('videoId')
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_data = {
                    'videoId': video_id,
                    'title': video['snippet'].get('title'),
                    'publishTime': video['snippet'].get('publishTime'),
                    'videoUrl': video_url,
                    'kind': video['id'].get('kind'),
                    'channelId': video['snippet'].get('channelId'),
                    'channelTitle': video['snippet'].get('channelTitle'),
                }
                duration = self._get_content_details(video_id)
                video_data['duration'] = self.iso8601_to_minutes(duration)
                video_list.append(video_data)
            logging.info(f"Se encontraron {len(video_list)} videos")
        else:
            logging.DEBUG(
                'Error al obtener los datos:', response.status_code, response,#.text,
            )
        return video_list

    @staticmethod
    def _get_content_details(video_id):
        """Get the duration of a video by its id"""
        url = (
            f"{BASE_URL}videos?"
            f"key={API_KEY}&"
            f"id={video_id}&"
            f"part=contentDetails"
        )
        response = requests.get(url)
        if response.status_code == 200:
            items = response.json().get('items', [])
            return items[0]['contentDetails']['duration']
        else:
            logging.DEBUG(
                'Error al obtener la duracion del video:',
                response.status_code,
                response.text,
            )
            return None

    @staticmethod
    def iso8601_to_minutes(iso_duration):
        """Convert an ISO8601 duration to minutes"""
        try:
            duracion = isodate.parse_duration(iso_duration)
            total_segundos = duracion.total_seconds()
            minutos = total_segundos // 60
            return minutos
        except Exception as e:
            logging.error(f"Error al convertir la duracion {iso_duration} a minutos")
            return 0
