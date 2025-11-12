# app/main_interface.py
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='📚 Pipeline de Procesamiento de Videos',
    layout='wide',
)

# fmt: off
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.youtube.youtube_ingest import YoutubeIngest
from app.youtube.scrapper_videos_youtube import ScrapperVideosYoutube
from app.utils.preprocessor import DataPreprocessor
from app.utils.frame_filter import FrameFilter
from app.utils.feature_extractor import FeatureExtractor
from app.utils.keyframe_selector import KeyFrameSelector
from app.llm.llm import FrameDescriptionLlm
from app.utils.cache_manager import PipelineCacheManager

# fmt: on

# Configurar rutas dentro de app/data para no mezclarse
# con archivos de entrenamiento
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, 'data')
VIDEOS_DIR = os.path.join(DATA_DIR, 'videos_youtube')
FRAMES_DIR = os.path.join(DATA_DIR, 'frames')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
CACHE_DIR = os.path.join(DATA_DIR, 'pipeline_cache')

# Crear directorios si no existen
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Inicializar gestor de cache
cache_manager = PipelineCacheManager(CACHE_DIR)

# Inicializar componentes
@st.cache_resource
def init_components():
    """Inicializa los componentes del pipeline"""
    youtube_ingest = YoutubeIngest()
    scrapper = ScrapperVideosYoutube(
        downloads_dir=VIDEOS_DIR,
        frames_dir=FRAMES_DIR,
        frame_interval_sec=2,
    )
    preprocessor = DataPreprocessor()
    frame_filter = FrameFilter(ssim_threshold=0.98)
    feature_extractor = FeatureExtractor(
        model_name='resnet50',
        device='cpu',
        batch_size=32,
    )
    keyframe_selector = KeyFrameSelector(
        clustering_method='kmeans',
        distance_metric='euclidean',
        normalize_features=False,
    )
    return {
        'youtube_ingest': youtube_ingest,
        'scrapper': scrapper,
        'preprocessor': preprocessor,
        'frame_filter': frame_filter,
        'feature_extractor': feature_extractor,
        'keyframe_selector': keyframe_selector,
    }

components = init_components()

# Inicializar session_state
if 'videos_metadata' not in st.session_state:
    st.session_state['videos_metadata'] = {}
if 'selected_video' not in st.session_state:
    st.session_state['selected_video'] = None
if 'frame_paths' not in st.session_state:
    st.session_state['frame_paths'] = []
if 'filtered_frames' not in st.session_state:
    st.session_state['filtered_frames'] = []
if 'features' not in st.session_state:
    st.session_state['features'] = None
if 'keyframes' not in st.session_state:
    st.session_state['keyframes'] = []
if 'keyframe_stats' not in st.session_state:
    st.session_state['keyframe_stats'] = None
if 'classified_frames' not in st.session_state:
    st.session_state['classified_frames'] = None
if 'frame_descriptions' not in st.session_state:
    st.session_state['frame_descriptions'] = {}
if 'frame_categories' not in st.session_state:
    st.session_state['frame_categories'] = {}

# Barra lateral de opciones
st.sidebar.title('🎬 Pipeline de Videos')
modo = st.sidebar.radio(
    'Selecciona una sección:',
    [
        '1️⃣ Cargar Videos',
        '2️⃣ Descargar y Extraer Frames',
        '3️⃣ Preprocesamiento',
        '4️⃣ Selección de Keyframes',
        '5️⃣ Clasificación',
        '6️⃣ Descripción Textual',
    ],
)

###################################################
# Sección 1: Cargar videos de últimos 3 días
###################################################
if modo == '1️⃣ Cargar Videos':
    daysback = 7
    st.title(f'1️⃣ Cargar Videos de Últimos {daysback} Días')

    youtube_ingest = components['youtube_ingest']


    st.subheader(
        f"Videos disponibles por canal (últimos {daysback} días)",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button('🔄 Cargar videos recientes de los canales'):
            st.session_state['videos_metadata'] = {}
            with st.spinner('Cargando videos (usando cache si está disponible)...'):
                for channel_name in youtube_ingest.channels:
                    try:
                        videos = (
                            youtube_ingest.get_last_videos_metadata_from_channels(
                                channel_name, daysback=daysback,
                            )
                        )
                        if videos:
                            df = pd.DataFrame(videos)
                            st.session_state['videos_metadata'][
                                channel_name
                            ] = df
                            st.success(
                                f"✅ {len(videos)} videos cargados de "
                                f"{channel_name}",
                            )
                    except Exception as e:
                        st.error(
                            f"❌ Error cargando videos de {channel_name}: {e}",
                        )

    with col2:
        # Botón para limpiar cache
        if st.button('🗑️ Limpiar Cache'):
            from pathlib import Path
            cache_dir = Path(__file__).parent / 'data' / 'cache'
            if cache_dir.exists():
                import shutil
                try:
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    st.success('✅ Cache limpiado')
                    st.session_state['videos_metadata'] = {}
                except Exception as e:
                    st.error(f"❌ Error limpiando cache: {e}")
            else:
                st.info('ℹ️ No hay cache para limpiar')

    # Mostrar los datos si ya están cargados
    if st.session_state['videos_metadata']:
        sel_columns = [
            'videoId', 'title', 'publishTime', 'videoUrl', 'duration',
        ]

        # Recolectar todos los videos
        all_videos = []
        for channel_name, df in st.session_state['videos_metadata'].items():
            if len(df) > 0:
                st.markdown(f"### 📺 {channel_name}")
                if all(col in df.columns for col in sel_columns):
                    display_df = df[sel_columns]
                else:
                    display_df = df
                st.dataframe(display_df, use_container_width=True)

                # Agregar videos a la lista
                for _, row in df.iterrows():
                    all_videos.append({
                        'channel': channel_name,
                        'videoId': row.get('videoId', ''),
                        'title': row.get('title', 'Sin título'),
                        'videoUrl': row.get('videoUrl', ''),
                        'duration': row.get('duration', 0),
                    })

        # Selección de video
        if all_videos:
            st.markdown('---')
            st.subheader('🎯 Seleccionar Video')

            # Crear lista de opciones para el selectbox
            video_options = [
                f"{v['channel']} - {v['title']} ({v['videoId']})"
                for v in all_videos
            ]
            selected_index = st.selectbox(
                'Selecciona un video:',
                range(len(video_options)),
                format_func=lambda x: (
                    video_options[x] if x < len(video_options) else ''
                ),
            )

            if selected_index is not None and selected_index < len(all_videos):
                selected_video = all_videos[selected_index]
                st.session_state['selected_video'] = selected_video

                st.success(f"✅ Video seleccionado: {selected_video['title']}")
                st.json({
                    'channel': selected_video['channel'],
                    'videoId': selected_video['videoId'],
                    'title': selected_video['title'],
                    'videoUrl': selected_video['videoUrl'],
                    'duration': f"{selected_video['duration']} minutos",
                })

    # También permitir entrada manual de URL
    st.markdown('---')
    st.subheader('🔗 O ingresar URL manualmente')
    manual_url = st.text_input('Ingresa el enlace del video de YouTube:')

    if manual_url:
        try:
            video_id = YoutubeIngest.extract_video_id_from_url(manual_url)
            if video_id:
                st.session_state['selected_video'] = {
                    'videoId': video_id,
                    'videoUrl': manual_url,
                    'title': 'Video manual',
                    'channel': 'Manual',
                    'duration': 0,
                }
                st.success(
                    f"✅ URL procesada. Video ID: {video_id}",
                )
            else:
                st.warning(
                    '⚠️ No se pudo extraer el video ID de la URL',
                )
        except Exception as e:
            st.error(f"❌ Error procesando URL: {e}")

###################################################
# Sección 2: Descargar video y extraer frames
###################################################
elif modo == '2️⃣ Descargar y Extraer Frames':
    st.title('2️⃣ Descargar Video y Extraer Frames')

    if st.session_state['selected_video'] is None:
        st.warning(
            '⚠️ Por favor selecciona un video en la sección '
            "'1️⃣ Cargar Videos'",
        )
        st.stop()

    selected_video = st.session_state['selected_video']
    scrapper = components['scrapper']

    st.subheader(f"Video: {selected_video['title']}")
    st.json({
        'videoId': selected_video['videoId'],
        'videoUrl': selected_video['videoUrl'],
        'channel': selected_video['channel'],
    })

    # Verificar si el video ya existe
    video_id = selected_video.get('videoId', '')
    if video_id:
        existing_video = scrapper._video_already_downloaded(
            video_id, scrapper.downloads_dir,
        )
        if existing_video:
            st.info(
                f"ℹ️ El video ya está descargado: {existing_video}\n"
                f"Puedes proceder directamente a extraer frames.",
            )

    if st.button('⬇️ Descargar Video'):
        with st.spinner(
            'Descargando video... Esto puede tomar varios minutos.',
        ):
            try:
                successful, failed = scrapper.download_videos_robust(
                    [selected_video['videoUrl']],
                )
                if successful:
                    # Verificar si fue descargado o ya existía
                    if video_id:
                        existing = scrapper._video_already_downloaded(
                            video_id, scrapper.downloads_dir,
                        )
                        if existing:
                            st.success(
                                f"✅ Video ya existía: {existing}\n"
                                f"No fue necesario descargarlo nuevamente.",
                            )
                        else:
                            st.success('✅ Video descargado exitosamente')
                    else:
                        st.success('✅ Video descargado exitosamente')
                else:
                    st.error(f"❌ Error descargando video: {failed}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.button('🎞️ Extraer Frames'):
        with st.spinner('Extrayendo frames del video...'):
            try:
                # Buscar el video descargado
                downloads_dir = scrapper.downloads_dir
                video_found = False

                for channel_dir in os.listdir(downloads_dir):
                    channel_path = os.path.join(downloads_dir, channel_dir)
                    if os.path.isdir(channel_path):
                        for video_file in os.listdir(channel_path):
                            if (
                                video_file.endswith('.mp4') and
                                selected_video['videoId'] in video_file
                            ):
                                video_path = os.path.join(
                                    channel_path, video_file,
                                )
                                frames_dir = os.path.join(
                                    scrapper.frames_dir, channel_dir,
                                )
                                os.makedirs(frames_dir, exist_ok=True)

                                duration, frame_paths = (
                                    scrapper.extract_frames_from_video(
                                        video_path,
                                        frames_dir,
                                        scrapper.frame_interval_sec,
                                    )
                                )

                                st.session_state['frame_paths'] = (
                                    frame_paths
                                )
                                video_found = True

                                st.success(
                                    f"✅ {len(frame_paths)} frames extraídos",
                                )
                                st.info(
                                    f"📊 Duración: {duration:.2f} segundos | "
                                    f"Frames: {len(frame_paths)}",
                                )
                                break

                        if video_found:
                            break

                if not video_found:
                    st.warning(
                        '⚠️ No se encontró el video descargado. '
                        'Por favor descarga el video primero.',
                    )

            except Exception as e:
                st.error(f"❌ Error extrayendo frames: {e}")

    # Mostrar frames extraídos
    if st.session_state['frame_paths']:
        num_frames = len(st.session_state['frame_paths'])
        st.subheader(f"📸 Frames Extraídos ({num_frames} frames)")

        # Mostrar muestra de frames
        num_samples = min(10, len(st.session_state['frame_paths']))
        sample_indices = np.linspace(
            0,
            len(st.session_state['frame_paths']) - 1,
            num_samples,
            dtype=int,
        )

        cols = st.columns(5)
        for idx, col in enumerate(cols[:num_samples]):
            if idx < len(sample_indices):
                frame_idx = sample_indices[idx]
                frame_path = st.session_state['frame_paths'][frame_idx]
                try:
                    col.image(
                        frame_path,
                        caption=f"Frame {frame_idx + 1}",
                        use_container_width=True,
                    )
                except Exception as e:
                    col.error(f"Error cargando frame: {e}")

###################################################
# Sección 3: Preprocesamiento y generación de embeddings
###################################################
elif modo == '3️⃣ Preprocesamiento':
    st.title('3️⃣ Preprocesamiento y Generación de Embeddings')

    if not st.session_state['frame_paths']:
        st.warning(
            '⚠️ Por favor extrae frames en la sección '
            "'2️⃣ Descargar y Extraer Frames'",
        )
        st.stop()

    frame_paths = st.session_state['frame_paths']
    frame_filter = components['frame_filter']
    feature_extractor = components['feature_extractor']

    # Obtener video_id para cache
    video_id = (
        st.session_state.get('selected_video', {})
        .get('videoId', 'unknown')
    )

    st.subheader(f"Frames de Entrada: {len(frame_paths)} frames")

    # Verificar cache
    cache_info = cache_manager.get_cache_info(video_id)
    if any(cache_info.values()):
        st.markdown('### 💾 Estado del Cache')
        cache_status = []
        for cache_type, exists in cache_info.items():
            status = '✅' if exists else '❌'
            cache_status.append(f"{status} {cache_type}")
        st.info(' | '.join(cache_status))

    # Botones para limpiar cache
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🗑️ Limpiar Cache de este Video'):
            if cache_manager.delete_cache(video_id):
                st.success('✅ Cache eliminado exitosamente')
                st.rerun()
            else:
                st.error('❌ Error eliminando cache')
    with col2:
        if st.button('🗑️ Limpiar Todo el Cache'):
            if cache_manager.clear_all_cache():
                st.success('✅ Todo el cache eliminado exitosamente')
                st.rerun()
            else:
                st.error('❌ Error eliminando cache')

    # Mostrar muestra de frames originales
    st.markdown('### 📸 Frames Originales (Muestra)')
    num_samples = min(10, len(frame_paths))
    sample_indices = np.linspace(0, len(frame_paths) - 1, num_samples, dtype=int)

    cols = st.columns(5)
    for idx, col in enumerate(cols[:num_samples]):
        if idx < len(sample_indices):
            frame_idx = sample_indices[idx]
            try:
                col.image(
                    frame_paths[frame_idx],
                    caption=f"Frame {frame_idx + 1}",
                    use_container_width=True,
                )
            except Exception as e:
                col.error(f"Error: {e}")

    # Filtrado SSIM
    # Verificar cache primero
    cached_filtered = cache_manager.load_cache(video_id, 'frames_filtered')
    if cached_filtered:
        st.info('💾 Cargando frames filtrados desde cache...')
        st.session_state['filtered_frames'] = cached_filtered
        st.success('✅ Frames filtrados cargados desde cache')
    else:
        if st.button('🔍 Filtrar Frames Duplicados (SSIM)'):
            with st.spinner(
                'Filtrando frames duplicados... '
                'Esto puede tomar varios minutos.',
            ):
                try:
                    filtered_paths, ssim_scores, filter_stats = (
                        frame_filter.filter_duplicate_frames(
                            frame_paths, preserve_temporal_order=True,
                        )
                    )

                    st.session_state['filtered_frames'] = filtered_paths

                    # Guardar en cache
                    cache_manager.save_cache(
                        video_id,
                        'frames_filtered',
                        filtered_paths,
                        metadata={'filter_stats': filter_stats},
                    )

                    st.success('✅ Filtrado completado y guardado en cache')
                    st.info(
                        f"📊 Estadísticas de Filtrado:\n"
                        f"- Frames originales: "
                        f"{filter_stats['original_count']}\n"
                        f"- Frames filtrados: "
                        f"{filter_stats['filtered_count']}\n"
                        f"- Frames eliminados: "
                        f"{filter_stats['frames_removed']}\n"
                        f"- Reducción: "
                        f"{filter_stats['reduction_percentage']:.2f}%",
                    )
                except Exception as e:
                    st.error(f"❌ Error filtrando frames: {e}")

    # Mostrar frames filtrados
    if st.session_state['filtered_frames']:
        filtered_paths = st.session_state['filtered_frames']
        st.markdown('### ✨ Frames Filtrados (Muestra)')

        num_samples = min(10, len(filtered_paths))
        sample_indices = np.linspace(0, len(filtered_paths) - 1, num_samples, dtype=int)

        cols = st.columns(5)
        for idx, col in enumerate(cols[:num_samples]):
            if idx < len(sample_indices):
                frame_idx = sample_indices[idx]
                try:
                    col.image(
                        filtered_paths[frame_idx],
                        caption=f"Frame {frame_idx + 1}",
                        use_container_width=True,
                    )
                except Exception as e:
                    col.error(f"Error: {e}")

    # Generación de embeddings
    if st.session_state['filtered_frames']:
        frames_to_process = st.session_state['filtered_frames']
    else:
        frames_to_process = frame_paths

    # Verificar cache de embeddings
    cached_embeddings = cache_manager.load_cache(video_id, 'embeddings')
    if cached_embeddings:
        st.info('💾 Cargando embeddings desde cache...')
        if isinstance(cached_embeddings, dict):
            st.session_state['features'] = cached_embeddings.get('features')
            if 'valid_paths' in cached_embeddings:
                st.session_state['filtered_frames'] = (
                    cached_embeddings['valid_paths']
                )
        else:
            st.session_state['features'] = cached_embeddings
        st.success('✅ Embeddings cargados desde cache')
    else:
        if st.button('🧠 Generar Embeddings'):
            with st.spinner(
                'Generando embeddings... Esto puede tomar varios minutos.',
            ):
                try:
                    # Ruta para guardar embeddings
                    embeddings_path = os.path.join(
                        EMBEDDINGS_DIR,
                        f"{video_id}_features.pkl",
                    )

                    features, valid_paths = (
                        feature_extractor.extract_features_from_paths(
                            frames_to_process,
                            save_path=embeddings_path,
                        )
                    )

                    st.session_state['features'] = features
                    # Actualizar frame_paths con los válidos
                    if len(valid_paths) != len(frames_to_process):
                        st.warning(
                            f"⚠️ Solo se procesaron {len(valid_paths)} de "
                            f"{len(frames_to_process)} frames",
                        )
                        st.session_state['filtered_frames'] = valid_paths

                    # Guardar en cache
                    cache_manager.save_cache(
                        video_id,
                        'embeddings',
                        {
                            'features': features,
                            'valid_paths': valid_paths,
                        },
                    )

                    st.success(
                        f"✅ Embeddings generados y guardados en cache: "
                        f"{features.shape}",
                    )
                    st.info(
                        f"📊 Dimensiones: {features.shape[0]} frames × "
                        f"{features.shape[1]} features\n"
                        f"💾 Guardado en: {embeddings_path}",
                    )
                except Exception as e:
                    st.error(f"❌ Error generando embeddings: {e}")

    # Mostrar información de embeddings
    if st.session_state['features'] is not None:
        features = st.session_state['features']
        st.markdown('### 📊 Información de Embeddings')
        st.json({
            'shape': list(features.shape),
            'dtype': str(features.dtype),
            'model': 'ResNet-50',
            'feature_dim': features.shape[1],
        })

###################################################
# Sección 4: Selección de keyframes
###################################################
elif modo == '4️⃣ Selección de Keyframes':
    st.title('4️⃣ Selección de Keyframes')

    if st.session_state['features'] is None:
        st.warning(
            '⚠️ Por favor genera embeddings en la sección '
            "'3️⃣ Preprocesamiento'",
        )
        st.stop()

    features = st.session_state['features']
    if st.session_state['filtered_frames']:
        frames_to_use = st.session_state['filtered_frames']
    else:
        frames_to_use = st.session_state['frame_paths']

    if len(frames_to_use) != features.shape[0]:
        st.warning(
            f"⚠️ Número de frames ({len(frames_to_use)}) no coincide "
            f"con features ({features.shape[0]})",
        )
        frames_to_use = frames_to_use[:features.shape[0]]

    # Obtener video_id para cache
    video_id = (
        st.session_state.get('selected_video', {})
        .get('videoId', 'unknown')
    )

    st.subheader(f"Frames de Entrada: {len(frames_to_use)} frames")

    # Selección de método
    method = st.radio(
        'Selecciona método de selección:',
        ['kmeans', 'cosine_similarity'],
        help=(
            'KMeans: Clustering con KMeans. '
            'Cosine Similarity: Usa distancia coseno para clustering.'
        ),
    )

    # Configuración avanzada de K-Means (solo para método kmeans)
    if method == 'kmeans':
        st.markdown('### ⚙️ Configuración Avanzada de K-Means')

        col1, col2 = st.columns(2)
        with col1:
            n_init = st.slider(
                'Número de inicializaciones (n_init):',
                min_value=1,
                max_value=100,
                value=10,
                help='Más inicializaciones = mejor calidad pero más lento',
            )
            max_iter = st.slider(
                'Máximo de iteraciones (max_iter):',
                min_value=100,
                max_value=1000,
                value=300,
                step=50,
                help='Más iteraciones = mejor convergencia pero más lento',
            )

        with col2:
            elbow_threshold = st.slider(
                'Umbral método del codo (elbow_method_threshold):',
                min_value=0.01,
                max_value=0.1,
                value=0.05,
                step=0.01,
                help='Umbral para detectar estabilización en método del codo',
            )
            optimization_method = st.selectbox(
                'Método de optimización (optimization_method):',
                ['both', 'elbow', 'silhouette'],
                help='Método para encontrar k óptimo',
            )

        st.markdown('### 🎯 Configuración de Selección de Keyframes')

        col3, col4, col5 = st.columns(3)
        with col3:
            keyframe_selection_method = st.selectbox(
                'Método de selección (keyframe_selection_method):',
                ['closest_to_centroid', 'furthest_from_centroid'],
                help='Cómo seleccionar el keyframe de cada cluster',
            )
        with col4:
            temporal_weight = st.slider(
                'Peso temporal (temporal_weight):',
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help='Peso para balancear distancia espacial vs temporal (0= solo espacial, 1= solo temporal)',
            )
        with col5:
            min_frames_per_cluster = st.slider(
                'Mínimo frames por cluster (min_frames_per_cluster):',
                min_value=1,
                max_value=10,
                value=1,
                help='Mínimo de frames requeridos para generar un keyframe',
            )
    else:
        # Valores por defecto para cosine_similarity
        n_init = 10
        max_iter = 300
        elbow_threshold = 0.05
        optimization_method = 'both'
        keyframe_selection_method = 'closest_to_centroid'
        temporal_weight = 0.0
        min_frames_per_cluster = 1

    # Verificar cache de keyframes
    cache_key = f'keyframes_{method}'
    cached_keyframes = cache_manager.load_cache(video_id, cache_key)
    cached_stats = cache_manager.load_cache(video_id, 'keyframe_stats')

    if cached_keyframes:
        st.info('💾 Cargando keyframes desde cache...')
        st.session_state['keyframes'] = cached_keyframes
        if cached_stats:
            st.session_state['keyframe_stats'] = cached_stats
        st.success('✅ Keyframes cargados desde cache')

    # Crear nuevo selector con parámetros configurados
    if method == 'cosine_similarity':
        keyframe_selector = KeyFrameSelector(
            clustering_method='kmeans',
            distance_metric='cosine',
            normalize_features=True,
            n_init=n_init,
            max_iter=max_iter,
            elbow_method_threshold=elbow_threshold,
            optimization_method=optimization_method,
            keyframe_selection_method=keyframe_selection_method,
            temporal_weight=temporal_weight,
            min_frames_per_cluster=min_frames_per_cluster,
        )
    else:
        keyframe_selector = KeyFrameSelector(
            clustering_method='kmeans',
            distance_metric='euclidean',
            normalize_features=False,
            n_init=n_init,
            max_iter=max_iter,
            elbow_method_threshold=elbow_threshold,
            optimization_method=optimization_method,
            keyframe_selection_method=keyframe_selection_method,
            temporal_weight=temporal_weight,
            min_frames_per_cluster=min_frames_per_cluster,
        )

    if not cached_keyframes:
        if st.button('🎯 Seleccionar Keyframes'):
            with st.spinner('Seleccionando keyframes...'):
                try:
                    keyframes, labels, model, stats = (
                        keyframe_selector.select_keyframes(
                            features,
                            frames_to_use,
                            n_frames=len(frames_to_use),
                        )
                    )

                    st.session_state['keyframes'] = keyframes
                    st.session_state['keyframe_stats'] = stats

                    # Guardar en cache
                    cache_manager.save_cache(
                        video_id,
                        cache_key,
                        keyframes,
                    )
                    cache_manager.save_cache(
                        video_id,
                        'keyframe_stats',
                        stats,
                    )

                    st.success(
                        f"✅ {len(keyframes)} keyframes seleccionados "
                        f"y guardados en cache",
                    )
                    st.info(
                        f"📊 Estadísticas de Selección:\n"
                        f"- Frames de entrada: {stats['n_frames']}\n"
                        f"- Keyframes seleccionados: "
                        f"{stats['n_keyframes']}\n"
                        f"- Número de clusters: {stats['n_clusters']}\n"
                        f"- Reducción: "
                        f"{stats['reduction_percentage']:.2f}%\n"
                        f"- Silhouette score: "
                        f"{stats.get('silhouette_score', 'N/A')}",
                    )
                except Exception as e:
                    st.error(f"❌ Error seleccionando keyframes: {e}")

    # Mostrar keyframes
    if st.session_state['keyframes']:
        keyframes = st.session_state['keyframes']
        stats = st.session_state['keyframe_stats']

        st.markdown('### 🎯 Keyframes Seleccionados')

        # Mostrar todos los keyframes en grid
        num_cols = 5
        num_keyframes = len(keyframes)

        for i in range(0, num_keyframes, num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < num_keyframes:
                    try:
                        col.image(
                            keyframes[idx],
                            caption=f"Keyframe {idx + 1}",
                            use_container_width=True,
                        )
                    except Exception as e:
                        col.error(f"Error: {e}")

###################################################
# Sección 5: Clasificación
###################################################
elif modo == '5️⃣ Clasificación':
    st.title('5️⃣ Clasificación de Keyframes')

    if not st.session_state['keyframes']:
        st.warning(
            '⚠️ Por favor selecciona keyframes en la sección '
            "'4️⃣ Selección de Keyframes'",
        )
        st.stop()

    keyframes = st.session_state['keyframes']

    # Obtener video_id para cache
    video_id = (
        st.session_state.get('selected_video', {})
        .get('videoId', 'unknown')
    )

    st.subheader(f"Keyframes a Clasificar: {len(keyframes)} frames")

    # Verificar cache de clasificación
    cached_classified = cache_manager.load_cache(
        video_id, 'classified_frames',
    )
    cached_categories = cache_manager.load_cache(
        video_id, 'frame_categories',
    )

    if cached_classified:
        st.info('💾 Cargando clasificación desde cache...')
        st.session_state['classified_frames'] = cached_classified
        if cached_categories:
            st.session_state['frame_categories'].update(cached_categories)
        st.success('✅ Clasificación cargada desde cache')

    # Configuración del modelo
    st.markdown('### ⚙️ Configuración del Modelo')
    model_path = st.text_input(
        'Ruta del modelo de clasificación:',
        value=(
            '/Users/andrestrepo/Documents/repos_personal/'
            'proyecto-vision-computador/src/notebooks/models/'
            'classifier_resnet50_class'
        ),
        help='Ruta al modelo TabularPredictor entrenado',
    )

    if not cached_classified:
        if st.button('🔮 Clasificar Keyframes'):
            with st.spinner('Clasificando keyframes...'):
                try:
                    # Cargar TabularPredictor
                    try:
                        from autogluon.tabular import TabularPredictor
                        predictor = TabularPredictor.load(
                            model_path,
                            require_version_match=False,
                        )
                    except ImportError:
                        st.error(
                            '❌ AutoGluon no está instalado. '
                            'Por favor instala: pip install autogluon.tabular',
                        )
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Error cargando modelo: {e}")
                        st.stop()

                    # Extraer embeddings de los keyframes usando FeatureExtractor
                    feature_extractor = components['feature_extractor']

                    st.info('📊 Extrayendo embeddings de keyframes...')
                    embeddings, valid_paths = (
                        feature_extractor.extract_features_from_paths(keyframes)
                    )

                    if len(embeddings) == 0:
                        st.error('❌ No se pudieron extraer embeddings')
                        st.stop()

                    # Convertir embeddings a DataFrame para TabularPredictor
                    # El modelo espera columnas: feature_0, feature_1, ..., feature_2047
                    feature_cols = [
                        f'feature_{i}' for i in range(embeddings.shape[1])
                    ]
                    embeddings_df = pd.DataFrame(embeddings, columns=feature_cols)

                    # Realizar predicciones
                    st.info('🔮 Clasificando embeddings...')
                    predictions = predictor.predict(embeddings_df)
                    probabilities = predictor.predict_proba(embeddings_df)

                    # Organizar resultados por categoría
                    classified = {}
                    frame_categories = {}
                    for frame_path, pred in zip(valid_paths, predictions):
                        category = str(pred)
                        if category not in classified:
                            classified[category] = []
                        classified[category].append(frame_path)
                        frame_categories[frame_path] = category

                    st.session_state['classified_frames'] = classified
                    st.session_state['frame_categories'].update(frame_categories)

                    # Guardar en cache
                    cache_manager.save_cache(
                        video_id,
                        'classified_frames',
                        classified,
                    )
                    cache_manager.save_cache(
                        video_id,
                        'frame_categories',
                        frame_categories,
                    )

                    # Mostrar estadísticas
                    st.success(
                        '✅ Clasificación completada y guardada en cache',
                    )
                    st.info(
                        f"📊 Estadísticas de Clasificación:\n"
                        f"- Total keyframes: {len(keyframes)}\n"
                        f"- Keyframes procesados: {len(valid_paths)}\n"
                        f"- Categorías encontradas: {len(classified)}",
                    )

                    # Mostrar distribución
                    category_counts = {
                        cat: len(frames)
                        for cat, frames in classified.items()
                    }
                    st.bar_chart(category_counts)

                except Exception as e:
                    st.error(f"❌ Error clasificando: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Mostrar keyframes por categoría
    if st.session_state['classified_frames']:
        classified = st.session_state['classified_frames']

        st.markdown('### 📂 Keyframes por Categoría')

        for category, frames in classified.items():
            st.markdown(
                f"#### 🏷️ Categoría: {category} ({len(frames)} frames)",
            )

            # Mostrar frames en grid
            num_cols = 5
            for i in range(0, len(frames), num_cols):
                cols = st.columns(num_cols)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(frames):
                        try:
                            col.image(
                                frames[idx],
                                caption=f"Frame {idx + 1}",
                                use_container_width=True,
                            )
                        except Exception as e:
                            col.error(f"Error: {e}")

###################################################
# Sección 6: Descripción Textual de Keyframes
###################################################
elif modo == '6️⃣ Descripción Textual':
    st.title('6️⃣ Descripción Textual de Keyframes')

    # Verificar si hay keyframes disponibles
    if not st.session_state['keyframes']:
        st.warning(
            '⚠️ Por favor selecciona keyframes en la sección '
            "'4️⃣ Selección de Keyframes'",
        )
        st.stop()

    keyframes = st.session_state['keyframes']
    frame_categories = st.session_state.get('frame_categories', {})

    st.subheader(f"Keyframes Disponibles: {len(keyframes)} frames")

    # Mostrar información sobre categorías disponibles
    if frame_categories:
        categories_available = set(frame_categories.values())
        st.info(
            f"📊 Categorías detectadas: {', '.join(sorted(categories_available))}",
        )
        st.info(
            '💡 Se usará automáticamente el prompt específico '
            'para cada categoría si está disponible.',
        )

    # Configuración del LLM
    st.markdown('### ⚙️ Configuración del LLM')

    # Selección de modelo de vision
    vision_model = st.selectbox(
        'Modelo de Vision:',
        ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1'],
        index=0,
        help='Selecciona el modelo de OpenAI para análisis de imágenes. '
             'gpt-4o-mini es más rápido y económico, gpt-4o es más preciso, '
             'gpt-4.1 es la versión más reciente.',
    )

    # Selección de keyframes a describir
    st.markdown('### 🎯 Seleccionar Keyframes para Describir')

    # Mostrar keyframes disponibles para selección con categoría si está disponible
    def format_frame_option(idx):
        frame_path = keyframes[idx]
        category = frame_categories.get(frame_path, 'Sin categoría')
        return f"Keyframe {idx + 1} ({category})"

    selected_frames = st.multiselect(
        'Selecciona los keyframes a describir:',
        options=range(len(keyframes)),
        format_func=format_frame_option,
        help='Puedes seleccionar uno o más keyframes para describir. '
             'Se usará el prompt específico de la categoría si está disponible.',
    )

    # Opción para usar prompt genérico o específico por categoría
    use_category_prompts = st.checkbox(
        'Usar prompts específicos por categoría',
        value=True,
        help='Si está marcado, usará el prompt específico de cada categoría. '
             'Si no está marcado, usará el prompt genérico.',
    )

    # Configuración del prompt genérico (solo si no se usan prompts por categoría)
    if not use_category_prompts:
        default_prompt_path = os.path.join(
            os.path.dirname(__file__),
            'prompts',
            'frame_description.txt',
        )
        prompt_path = st.text_input(
            'Ruta del prompt genérico (opcional):',
            value=default_prompt_path,
            help='Ruta al archivo de prompt para descripción genérica',
        )
    else:
        prompt_path = None

    # Botón para generar descripciones
    if st.button('📝 Generar Descripciones'):
        if not selected_frames:
            st.warning('⚠️ Por favor selecciona al menos un keyframe')
        else:
            with st.spinner(
                f'Generando descripciones para {len(selected_frames)} '
                f'keyframes... Esto puede tomar varios minutos.',
            ):
                try:
                    # Inicializar LLM para descripción
                    # Determinar prompt_path a usar
                    final_prompt_path = None
                    if not use_category_prompts and prompt_path:
                        # Usar prompt genérico si está especificado y existe
                        if os.path.exists(prompt_path):
                            final_prompt_path = prompt_path
                    # Si use_category_prompts está activado, se usará None
                    # y FrameDescriptionLlm usará el prompt por categoría

                    frame_desc_llm = FrameDescriptionLlm(
                        config={
                            'type': 'openai',
                            'model': vision_model,  # Usar el modelo seleccionado
                            'temperature': 0,
                        },
                        prompt_path=final_prompt_path,
                    )

                    descriptions = {}
                    progress_bar = st.progress(0)

                    for idx, frame_idx in enumerate(selected_frames):
                        frame_path = keyframes[frame_idx]

                        # Obtener categoría del frame si está disponible
                        category = None
                        if use_category_prompts:
                            category = frame_categories.get(frame_path)
                            if category:
                                st.info(
                                    f"📌 Usando prompt específico para "
                                    f"categoría: {category}",
                                )

                        try:
                            # Generar descripción con categoría si está disponible
                            description = frame_desc_llm.describe_image(
                                frame_path,
                                category=category if use_category_prompts else None,
                            )
                            descriptions[frame_path] = description

                            progress_bar.progress((idx + 1) / len(selected_frames))
                            category_info = f" ({category})" if category else ''
                            st.success(
                                f"✅ Descripción generada para "
                                f"Keyframe {frame_idx + 1}{category_info}",
                            )
                        except Exception as e:
                            st.error(
                                f"❌ Error describiendo Keyframe "
                                f"{frame_idx + 1}: {e}",
                            )
                            descriptions[frame_path] = None

                    # Guardar descripciones en session_state
                    st.session_state['frame_descriptions'].update(descriptions)

                    st.success(
                        f"✅ {len([d for d in descriptions.values() if d])} "
                        f"descripciones generadas exitosamente",
                    )

                except Exception as e:
                    st.error(f"❌ Error generando descripciones: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Mostrar descripciones generadas
    if st.session_state['frame_descriptions']:
        descriptions = st.session_state['frame_descriptions']

        st.markdown('### 📝 Descripciones Generadas')

        # Filtrar solo las descripciones válidas
        valid_descriptions = {
            path: desc
            for path, desc in descriptions.items()
            if desc is not None and path in keyframes
        }

        if valid_descriptions:
            for frame_path, description in valid_descriptions.items():
                # Encontrar índice del keyframe
                frame_idx = keyframes.index(frame_path)

                st.markdown(f"#### 🖼️ Keyframe {frame_idx + 1}")

                # Mostrar imagen y descripción lado a lado
                col1, col2 = st.columns([1, 2])

                with col1:
                    try:
                        st.image(frame_path, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error cargando imagen: {e}")

                with col2:
                    st.markdown('**Descripción:**')
                    st.write(description)
                    st.markdown('---')
        else:
            st.info('ℹ️ No hay descripciones válidas para mostrar')
