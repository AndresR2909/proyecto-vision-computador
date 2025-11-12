# Proyecto de Visión por Computador: Extracción y Clasificación de Keyframes de Videos de YouTube

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de procesamiento de videos de YouTube para la extracción automática de keyframes (frames clave) y su posterior clasificación en categorías específicas. El sistema está diseñado para procesar videos de canales de trading y finanzas, extrayendo información visual relevante mediante técnicas de visión por computador y aprendizaje automático.

El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining) para garantizar un desarrollo estructurado y reproducible.

---

## 🎯 Metodología CRISP-DM

### 1. Comprensión del Negocio

#### Problema a Resolver
Los videos de YouTube contienen una gran cantidad de información visual que puede ser difícil de procesar y analizar manualmente. En el contexto de canales de trading y finanzas, es crucial identificar y extraer los frames más representativos que contengan información relevante (gráficos, tablas, texto, personas, etc.) para:

- **Automatizar el análisis de contenido**: Reducir el tiempo necesario para revisar videos completos
- **Extracción de información estructurada**: Identificar y categorizar elementos visuales clave
- **Generación de resúmenes visuales**: Crear representaciones compactas del contenido de videos
- **Análisis de tendencias**: Identificar patrones visuales en múltiples videos

#### Objetivos del Proyecto
1. Extraer automáticamente keyframes representativos de videos de YouTube
2. Clasificar los frames en categorías específicas (gráficos, tablas, texto, personas, etc.)
3. Generar descripciones textuales estructuradas de los frames usando modelos de lenguaje
4. Proporcionar una interfaz interactiva para el procesamiento de videos

#### Criterios de Éxito
- Reducción significativa del número de frames a procesar (más del 80%)
- Precisión de clasificación superior al 90%
- Extracción de información financiera relevante de forma estructurada
- Pipeline funcional y reproducible

---

### 2. Comprensión de los Datos

#### Fuente de Datos
- **Videos de YouTube**: Canales especializados en trading y finanzas
- **Canales procesados**:
  - ARENA ALFA
  - Bitcoin hoy ｜ Esteban Perez Trader
  - Bolsas hoy ｜ Esteban Pérez Inversor
  - Otros canales de trading

#### Características de los Datos
- **Formato**: Videos MP4 descargados de YouTube
- **Resolución**: Variable (depende del video original)
- **Duración**: Videos de diferentes longitudes
- **Contenido**: Frames con gráficos financieros, tablas, texto, personas, logos, etc.

#### Exploración de Datos (EDA)
El proyecto incluye análisis exploratorio de datos que revela:
- Distribución de videos por canal
- Estadísticas de duración de videos
- Distribución temporal de publicaciones
- Análisis de frames extraídos (histogramas, boxplots, etc.)

**Archivos de análisis**:
- `src/notebooks/eda_videos_youtube.ipynb`: Análisis exploratorio de videos
- `src/notebooks/preprocessing_frames.ipynb`: Análisis de frames extraídos

#### Desafíos Identificados
1. **Redundancia de frames**: Muchos frames similares o duplicados
2. **Fondos uniformes**: Frames sin información relevante
3. **Variabilidad de contenido**: Diferentes tipos de elementos visuales
4. **Calidad variable**: Videos con diferentes resoluciones y calidades

---

### 3. Preparación de los Datos

#### Pipeline de Preprocesamiento

##### 3.1 Descarga de Videos
- **Herramienta**: `yt-dlp` para descargar videos de YouTube
- **Almacenamiento**: Videos organizados por canal en `src/app/data/videos_youtube/`
- **Componente**: `ScrapperVideosYoutube`

##### 3.2 Extracción de Frames
- **Intervalo**: Extracción de frames cada 2 segundos
- **Formato**: Imágenes PNG guardadas en `src/app/data/frames/`
- **Componente**: `ScrapperVideosYoutube.extract_frames_from_video()`

##### 3.3 Filtrado de Frames Duplicados
- **Método**: SSIM (Structural Similarity Index)
- **Umbral**: 0.98 (frames con similitud > 98% se consideran duplicados)
- **Preservación del orden temporal**: Sí
- **Componente**: `FrameFilter`
- **Resultado**: Reducción típica del 30-50% de frames redundantes

##### 3.4 Filtrado de Fondos Uniformes
- **Método**: Análisis de varianza de píxeles
- **Objetivo**: Eliminar frames con fondos uniformes sin información relevante
- **Componente**: `Preprocessor`

##### 3.5 Extracción de Características (Embeddings)
- **Modelo**: ResNet-50 preentrenado en ImageNet
- **Dimensión de features**: 2048
- **Dispositivo**: CPU/GPU configurable
- **Batch size**: 32
- **Componente**: `FeatureExtractor`
- **Almacenamiento**: Embeddings guardados en `src/app/data/embeddings/`

#### Datasets Generados
- **Frames etiquetados**: Dataset de 10,000 muestras con etiquetas consensuadas entre CLIP y OpenAI GPT-4.1
- **Embeddings pregenerados**: Features extraídas con ResNet-50 (reutilizadas para clasificación)
- **Ubicación**: `src/notebooks/datasets/`

---

### 4. Modelado

#### 4.1 Extracción de Keyframes

##### Método Principal: K-Means Clustering
Basado en el artículo **"Static Video Summarization Using Transfer Learning and Clustering"** (Kashid et al.):

1. **Extracción de Features**: ResNet-50 preentrenado
2. **Clustering**: K-Means con distancia euclidiana
3. **Optimización**: Silhouette Score para determinar número óptimo de clusters
4. **Selección de Keyframes**: Frame con mayor disimilitud al centroide de cada cluster

**Parámetros**:
- Métrica de distancia: Euclidiana o Coseno
- Normalización: Opcional (L2 normalization para cosine similarity)
- Número de clusters: Determinado automáticamente usando Silhouette Score

**Componente**: `KeyFrameSelector`

**Notebooks relacionados**:
- `src/notebooks/articulo_clustering_video.ipynb`: Implementación del método del artículo
- `src/notebooks/articulo_clustering_video_sin_pca.ipynb`: Variante sin PCA

##### Método Alternativo: Cosine Similarity
- Clustering basado en similitud coseno
- Normalización L2 de features
- Útil para comparación de características normalizadas

#### 4.2 Clasificación de Frames

##### Modelo: AutoGluon TabularPredictor
- **Features de entrada**: Embeddings de ResNet-50 (2048 dimensiones)
- **Clases a predecir**: 11 categorías
  - `background`: Fondos sin información relevante
  - `bar_chart`: Gráficos de barras
  - `candlestick`: Gráficos de velas (trading)
  - `diagram`: Diagramas y esquemas
  - `line_chart`: Gráficos de líneas
  - `logo`: Logos y marcas
  - `other`: Otros elementos
  - `person`: Personas en el video
  - `screenshot`: Capturas de pantalla
  - `table`: Tablas de datos
  - `text`: Texto visible

##### Proceso de Etiquetado del Dataset

El dataset de entrenamiento se creó mediante un proceso de etiquetado semi-automático que combina dos modelos para garantizar alta calidad:

1. **Selección de Muestra**: Se seleccionaron 10,000 frames válidos del dataset procesado, balanceados por canal y categoría.

2. **Etiquetado con CLIP**:
   - Primera etapa de etiquetado usando el modelo CLIP (Contrastive Language-Image Pre-training)
   - Notebook: `frame_labeling_clip_v3.ipynb`
   - Clasificación semántica basada en texto e imagen

3. **Etiquetado con OpenAI GPT-4.1**:
   - Segunda etapa usando OpenAI GPT-4.1 Vision API
   - Notebook: `frame_labeling_openai_v3.ipynb`
   - Se probaron varias versiones ajustando los prompts para optimizar la precisión
   - Revisión y selección de los mejores prompts por categoría

4. **Filtrado por Consenso**:
   - Solo se conservaron los frames donde **ambos modelos (CLIP y OpenAI) estaban de acuerdo** en la categoría
   - Esto garantiza un dataset de alta calidad y confiabilidad
   - Resultado: Dataset final con etiquetas consensuadas entre ambos modelos

##### Entrenamiento del Clasificador

- **Dataset**: Frames etiquetados con consenso CLIP + OpenAI GPT-4.1
- **Tamaño de entrenamiento**: 2,935 frames (frames con acuerdo entre modelos)
- **Tamaño de prueba**: 734 frames
- **Tiempo de entrenamiento**: 30 minutos (time_limit=1800s)
- **Precisión alcanzada**: 94.14%

**Arquitectura del Modelo**:
- **Base**: ResNet-50 preentrenado (ya utilizado en la aplicación para extracción de embeddings)
- **Cabeza de Clasificación**: AutoGluon TabularPredictor entrenada sobre los embeddings de ResNet-50
- **Ventajas de esta arquitectura**:
  - **No consume API**: El modelo entrenado funciona offline sin necesidad de llamadas a OpenAI
  - **No requiere VIT multimodal**: Evita el uso de modelos Vision Transformer multimodales que tienen tiempos de inferencia altos
  - **Reutilización de embeddings**: Aprovecha los embeddings de ResNet-50 ya generados para la extracción de keyframes, optimizando el procesamiento

**Notebooks de entrenamiento**:
- `src/notebooks/clasificacion_frames.ipynb`: Entrenamiento con etiquetas consensuadas (CLIP + OpenAI)
- `src/notebooks/train_clip_classifier.ipynb`: Entrenamiento alternativo con etiquetas de CLIP únicamente

**Modelos guardados**:
- `src/notebooks/models/classifier_resnet50_class/`: Modelo principal (entrenado con consenso CLIP + OpenAI)
- `src/notebooks/models/clip_classifier_resnet50_class/`: Modelo alternativo (entrenado solo con CLIP)

#### 4.3 Generación de Descripciones Textuales

##### Modelo: OpenAI GPT-4o Vision API
- **Modelos disponibles**: `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`
- **Prompts específicos por categoría**: Cada categoría tiene un prompt optimizado
- **Componente**: `FrameDescriptionLlm`

**Prompts disponibles**:
- `background.txt`: Para fondos
- `person.txt`: Para personas
- `text.txt`: Para texto
- `screenshot.txt`: Para capturas de pantalla
- `diagram.txt`: Para diagramas
- `table.txt`: Para tablas
- `logo.txt`: Para logos
- `candlestick.txt`: Para gráficos de velas
- `line_chart.txt`: Para gráficos de líneas
- `bar_chart.txt`: Para gráficos de barras
- `other.txt`: Para otros elementos

**Ubicación**: `src/app/prompts/`

---

### 5. Evaluación

#### 5.1 Métricas de Clustering

##### Silhouette Score
- **Rango**: [-1, 1]
- **Interpretación**: Valores cercanos a 1 indican clusters bien separados
- **Uso**: Optimización del número de clusters

##### Reducción de Frames
- **Métrica**: Porcentaje de reducción respecto a frames originales
- **Objetivo**: > 80% de reducción manteniendo información relevante

##### Distribución de Clusters
- Análisis de tamaño y distribución de clusters
- Identificación de clusters desbalanceados

#### 5.2 Métricas de Clasificación

##### Precisión Global
- **Modelo ResNet-50 + Cabeza de Predicción**: 94.14%
- **Dataset de entrenamiento**: 2,935 frames (frames con acuerdo entre CLIP y OpenAI)
- **Dataset de prueba**: 734 frames
- **Número de clases**: 11 categorías

##### Precisión por Clase
- Análisis de precisión, recall y F1-score por categoría
- Identificación de clases con menor rendimiento

##### Matriz de Confusión
- Visualización de errores de clasificación
- Identificación de confusiones entre clases similares

**Archivos de evaluación**:
- `src/notebooks/clasificacion_frames.ipynb`: Evaluación completa del clasificador
- `src/app/utils/evaluator.py`: Módulo de evaluación
- `src/notebooks/models/classifier_resnet50_class_info.json`: Información del modelo entrenado

#### 5.3 Métricas de Evaluación de Etiquetado

El proceso de etiquetado del dataset se evaluó mediante dos enfoques complementarios:

##### Evaluación con LLM as Evaluator (GPT-4.1 calificando CLIP)

En el notebook `frame_labeling_openai_v3.ipynb`, se utilizó GPT-4.1 Vision API para evaluar las etiquetas generadas por CLIP, implementando un enfoque de "LLM as Evaluator":

- **Dataset evaluado**: 10,000 frames balanceados por canal y categoría
- **Concordancia entre CLIP y OpenAI GPT-4.1**: 36.46% (3,646 / 10,000 frames)
- **Total de acuerdos**: 3,646 frames
- **Total de desacuerdos**: 6,354 frames
- **Dataset final usado para entrenamiento**: Solo los frames donde ambos modelos coincidieron (3,646 frames)

**Interpretación**:
- La concordancia del 36.46% indica que ambos modelos tienen criterios diferentes pero complementarios
- El filtrado por consenso garantiza alta calidad: solo se usaron frames con acuerdo entre ambos modelos
- Este enfoque de validación cruzada reduce errores de etiquetado y mejora la confiabilidad del dataset

**Notebook**: `src/notebooks/frame_labeling_openai_v3.ipynb`

##### Métricas de Confianza de CLIP

En el notebook `frame_labeling_clip_v3.ipynb`, se analizaron los scores de confianza del modelo CLIP:

- **Score promedio**: 0.2550
- **Score mediana**: 0.2546
- **Score mínimo**: 0.1673
- **Score máximo**: 0.3427
- **Desviación estándar**: 0.0179
- **Frames con baja confianza (score < 0.2)**: 449 frames (0.18% del total)

**Distribución de scores por categoría**:
- Las categorías con mayor score promedio: `table` (0.259), `candlestick` (0.259), `bar_chart` (0.258)
- Las categorías con menor score promedio: `logo` (0.226), `other` (0.234), `diagram` (0.237)

**Notebook**: `src/notebooks/frame_labeling_clip_v3.ipynb`

#### 5.4 Métricas de Cobertura Temporal
- **Cobertura porcentual**: Porcentaje del video cubierto por keyframes
- **Gaps temporales**: Intervalos sin keyframes seleccionados
- **Distribución temporal**: Análisis de distribución de keyframes a lo largo del video

#### 5.5 Visualizaciones Generadas
- Distribución de categorías
- Matrices de confusión
- Mosaicos de keyframes
- Gráficos de precisión por clase
- Análisis de calidad de clasificación

**Ubicación de imágenes**: `src/notebooks/images/`

---

### 6. Despliegue (Opcional)

#### Interfaz Web con Streamlit
El proyecto incluye una interfaz web interactiva para el procesamiento de videos:

**Componente**: `src/app/main_interface.py`

**Funcionalidades**:
1. **Cargar Videos**: Carga videos recientes de canales de YouTube
2. **Descargar y Extraer Frames**: Descarga videos y extrae frames
3. **Preprocesamiento**: Filtrado SSIM y generación de embeddings
4. **Selección de Keyframes**: Clustering y selección de frames representativos
5. **Clasificación**: Clasificación automática de keyframes
6. **Descripción Textual**: Generación de descripciones usando LLMs

**Características**:
- Sistema de caché para optimizar procesamiento
- Visualización de resultados en tiempo real
- Procesamiento por lotes
- Gestión de estado de sesión

**Ejecución**:
```bash
streamlit run src/app/main_interface.py
```

---

## 🛠️ Instalación y Configuración

### Requisitos del Sistema
- Python 3.8+
- CUDA (opcional, para aceleración GPU)

### Crear y Activar Entorno Virtual

**Opción 1: Usando venv (recomendado)**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
# venv\Scripts\activate
```

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Dependencias Principales
- `torch>=2.0.0`: PyTorch para modelos de deep learning
- `torchvision>=0.15.0`: Modelos preentrenados (ResNet-50)
- `transformers>=4.30.0`: Modelos de transformers
- `openai>=1.0.0`: API de OpenAI para descripciones
- `autogluon.tabular`: Clasificador automático
- `streamlit`: Interfaz web
- `yt-dlp>=2025.9.26`: Descarga de videos de YouTube
- `opencv-python>=4.5.0`: Procesamiento de video
- `scikit-learn>=1.0.0`: Clustering y métricas
- `scikit-image>=0.19.0`: SSIM para filtrado

### Configuración de API Keys

Crear archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=tu_api_key_aqui
YOUTUBE_API_KEY=tu_youtube_api_key_aqui  # Opcional
```

---

## 📁 Estructura del Proyecto

```
proyecto-vision-computador/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias del proyecto
├── .gitignore                   # Archivos ignorados por git
│
├── articles/                    # Artículos de referencia
│   ├── articulo_clustering_video.pdf
│   ├── kek_frame_extract_articulo.pdf
│   └── pdf_summary.txt
│
├── images/                      # Imágenes de análisis y resultados
│   ├── clusters_frames.png
│   ├── conteo_canales.png
│   ├── duracion_canales.png
│   └── ...
│
└── src/
    ├── __init__.py
    │
    ├── app/                     # Aplicación principal
    │   ├── __init__.py
    │   ├── main_interface.py    # Interfaz Streamlit
    │   │
    │   ├── data/                # Datos procesados
    │   │   ├── cache/           # Cache de metadata
    │   │   ├── embeddings/     # Embeddings pregenerados
    │   │   ├── frames/         # Frames extraídos
    │   │   ├── pipeline_cache/ # Cache del pipeline
    │   │   └── videos_youtube/ # Videos descargados
    │   │
    │   ├── llm/                 # Integración con LLMs
    │   │   ├── __init__.py
    │   │   └── llm.py          # FrameDescriptionLlm
    │   │
    │   ├── prompts/             # Prompts para LLMs
    │   │   ├── background.txt
    │   │   ├── person.txt
    │   │   ├── text.txt
    │   │   └── ...
    │   │
    │   ├── utils/               # Utilidades del pipeline
    │   │   ├── __init__.py
    │   │   ├── cache_manager.py      # Gestión de caché
    │   │   ├── categorizer.py        # Categorización global
    │   │   ├── dataset_builder.py    # Construcción de datasets
    │   │   ├── evaluator.py          # Evaluación de métricas
    │   │   ├── feature_extractor.py  # Extracción de features
    │   │   ├── frame_filter.py       # Filtrado SSIM
    │   │   ├── keyframe_selector.py   # Selección de keyframes
    │   │   ├── preprocessor.py        # Preprocesamiento
    │   │   └── results_comparator.py  # Comparación de resultados
    │   │
    │   └── youtube/             # Integración con YouTube
    │       ├── __init__.py
    │       ├── scrapper_videos_youtube.py  # Descarga de videos
    │       └── youtube_ingest.py           # Ingesta de metadata
    │   │
    └── notebooks/               # Notebooks de análisis y entrenamiento
        ├── __init__.py
        ├── setup_path.py
        │
        ├── datasets/            # Datasets procesados
        │   ├── df_frames_openai_labeled_v3.csv
        │   ├── df_videos_frames_filtrados_v2.csv
        │   └── ...
        │
        ├── models/              # Modelos entrenados
        │   ├── classifier_resnet50_class/
        │   ├── clip_classifier_resnet50_class/
        │   └── *_class_info.json
        │
        ├── images/              # Visualizaciones generadas
        │   ├── accuracy_by_class_*.png
        │   ├── confusion_matrix_*.png
        │   └── ...
        │
        ├── features_resnet50/   # Features pregeneradas
        │
        ├── frames/              # Frames de ejemplo
        │
        ├── videos_youtube/      # Videos de ejemplo
        │
        ├── eda_videos_youtube.ipynb              # EDA de videos
        ├── scrapper_videos_youtube.ipynb         # Scraping de videos
        ├── preprocessing_frames.ipynb            # Preprocesamiento
        ├── articulo_clustering_video.ipynb        # Clustering (artículo)
        ├── articulo_clustering_video_sin_pca.ipynb
        ├── articulo_key_frames_cosine_similarity.ipynb  # Extracción de Keyframes (Cosine Similarity)
        ├── frame_labeling_openai_v3.ipynb         # Etiquetado OpenAI (varias versiones de prompts)
        ├── frame_labeling_clip_v3.ipynb          # Etiquetado CLIP (primera etapa)
        ├── clasificacion_frames.ipynb             # Clasificación
        └── train_clip_classifier.ipynb            # Entrenamiento CLIP
```

---

## 🚀 Uso del Proyecto

### Opción 1: Interfaz Web (Recomendado)

**Activar el entorno virtual primero:**
```bash
# Si usaste venv:
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Si usaste conda:
conda activate proyecto-vision
```

**Ejecutar la interfaz:**
```bash
streamlit run src/app/main_interface.py
```

La interfaz web permite:
1. Cargar videos de YouTube
2. Procesar videos paso a paso
3. Visualizar resultados en tiempo real
4. Generar descripciones textuales

### Opción 2: Uso Programático

```python
from app.youtube.scrapper_videos_youtube import ScrapperVideosYoutube
from app.utils.frame_filter import FrameFilter
from app.utils.feature_extractor import FeatureExtractor
from app.utils.keyframe_selector import KeyFrameSelector

# 1. Descargar video y extraer frames
scrapper = ScrapperVideosYoutube(
    downloads_dir="data/videos_youtube",
    frames_dir="data/frames",
    frame_interval_sec=2
)
scrapper.download_videos_robust([video_url])
frames = scrapper.extract_frames_from_video(video_path, frames_dir)

# 2. Filtrar frames duplicados
frame_filter = FrameFilter(ssim_threshold=0.98)
filtered_frames, _, _ = frame_filter.filter_duplicate_frames(frames)

# 3. Extraer features
feature_extractor = FeatureExtractor(model_name='resnet50')
features, valid_paths = feature_extractor.extract_features_from_paths(filtered_frames)

# 4. Seleccionar keyframes
keyframe_selector = KeyFrameSelector(clustering_method='kmeans')
keyframes, labels, model, stats = keyframe_selector.select_keyframes(
    features, valid_paths, n_frames=len(valid_paths)
)
```

### Opción 3: Notebooks de Análisis

Ejecutar los notebooks en orden:
1. `eda_videos_youtube.ipynb`: Exploración de datos
2. `preprocessing_frames.ipynb`: Preprocesamiento
3. `articulo_clustering_video.ipynb`: Extracción de keyframes
4. `clasificacion_frames.ipynb`: Clasificación y evaluación

---

## 📊 Resultados

### Rendimiento del Clasificador

**Modelo**: ResNet-50 + AutoGluon TabularPredictor
- **Precisión global**: 94.14%
- **Dataset de entrenamiento**: 2,935 frames
- **Dataset de prueba**: 734 frames
- **Clases**: 11 categorías

### Reducción de Frames

- **Filtrado SSIM**: 30-50% de reducción
- **Selección de keyframes**: 80-90% de reducción total
- **Cobertura temporal**: > 70% del video cubierto

### Distribución de Categorías

Las categorías más comunes en videos de trading:
- `person`: Personas presentando
- `candlestick`: Gráficos de velas
- `line_chart`: Gráficos de líneas
- `table`: Tablas de datos
- `text`: Texto visible

---

## 🔬 Estrategia de Evaluación

### Métricas Utilizadas

1. **Clustering**:
   - Silhouette Score
   - Distribución de clusters
   - Reducción de frames

2. **Clasificación**:
   - Precisión global
   - Precisión por clase
   - Recall y F1-score
   - Matriz de confusión

3. **Cobertura Temporal**:
   - Porcentaje de cobertura
   - Gaps temporales
   - Distribución de keyframes

### Estrategia de Validación

- **División train/test**: 80/20
- **Validación cruzada**: Considerada para futuras mejoras
- **Evaluación en múltiples videos**: Validación en diferentes canales

---

## 🎓 Referencias y Artículos

### Artículos Implementados

1. **"Static Video Summarization Using Transfer Learning and Clustering"**
   - Autores: Shamal Kashid, Lalit K. Awasthi, Krishan Berwal, Parul Saini
   - Método: K-Means clustering con ResNet-50
   - Implementación: `articulo_clustering_video.ipynb`

2. **"Key-Frame Extraction Methods: A Review"**
   - Métodos de extracción de keyframes
   - Implementación: `articulo_key_frames_cosine_similarity.ipynb`

### Modelos Utilizados

- **ResNet-50**: Preentrenado en ImageNet
  - Extracción de embeddings para keyframes
  - Base para el clasificador de categorías (reutilización de embeddings)
- **AutoGluon TabularPredictor**: Clasificador automático
  - Cabeza de clasificación entrenada sobre embeddings de ResNet-50
- **OpenAI GPT-4.1 Vision**: Etiquetado del dataset y generación de descripciones
  - Etiquetado: Múltiples versiones de prompts ajustadas iterativamente
  - Generación: Descripciones textuales de frames clasificados
- **CLIP**: Modelo de visión-lenguaje
  - Primera etapa de etiquetado del dataset
  - Validación cruzada con OpenAI para garantizar calidad

---

## 🔧 Configuración del Entorno de Entrenamiento

### Hardware Recomendado

- **CPU**: Múltiples núcleos (procesamiento paralelo)
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **GPU**: Opcional pero recomendado para entrenamiento (CUDA compatible)
- **Almacenamiento**: Espacio suficiente para videos y frames (100GB+)

### Configuración de Entrenamiento

- **Batch size**: 32 (ajustable según memoria)
- **Device**: CPU o CUDA
- **Time limit**: 1800 segundos (30 minutos) para AutoGluon
- **Random state**: 42 (reproducibilidad)

---

## 📝 Notas Técnicas

### Función de Costo

- **Clustering**: Inercia (suma de distancias al cuadrado)
- **Clasificación**: Cross-entropy (manejada por AutoGluon)
- **Optimización**: Minimización de inercia + maximización de Silhouette Score

### Aumento de Datos

- **No aplicado en este proyecto**: Dataset suficientemente grande
- **Posible mejora futura**: Data augmentation para clases desbalanceadas

### Adquisición de Datos Adicionales

- **Fuente principal**: YouTube (canales de trading)
- **Etiquetado**: Semi-automático mediante consenso entre CLIP y OpenAI GPT-4.1
  - Proceso iterativo con múltiples versiones de prompts
  - Selección de 10,000 frames balanceados por canal y categoría
  - Filtrado por acuerdo entre ambos modelos para garantizar calidad
- **Validación**: Dataset final validado por consenso entre modelos (no requiere validación manual extensiva)

---

## 🎯 Conclusiones

### Extracción de Keyframes

**Cosine Similarity** ofrece una cobertura temporal y eficiencia muy superiores, procesando los videos 124 veces más rápido y cubriendo 10 veces más del contenido.

**K-Means** solo es ventajoso cuando se requiere la máxima compresión de datos, aunque sacrifica cobertura y velocidad.

**Cosine Similarity** es más simple, preserva mejor la secuencia temporal y detecta cambios relevantes entre frames consecutivos.

Para la mayoría de escenarios, **Cosine Similarity** es la opción recomendada; **K-Means** solo debe usarse si la compresión extrema es prioritaria y el tiempo de procesamiento no es crítico.

Implementar un sistema de extracción de información basado en descripciones de keyframes permite a un asistente de análisis financiero para inversores individuales acceder de manera ágil y eficiente a grandes volúmenes de videos de YouTube, sin sacrificar la cobertura informativa esencial.

En promedio, este método reduce el tiempo necesario para revisar el contenido en más de un **55%**, asegurando que los usuarios puedan identificar rápidamente la información relevante para la toma de decisiones financieras, sin la necesidad de ver cada video completo.

---

### Clasificación

El modelo **ResNet-50 + AutoGluon TabularPredictor** alcanza una precisión del **94.14%** en la clasificación de frames en 11 categorías, superando el objetivo inicial del 90%. Esta arquitectura híbrida aprovecha los embeddings de ResNet-50 ya generados para la extracción de keyframes, optimizando el procesamiento mediante la reutilización de características.

La estrategia de **reutilización de embeddings** elimina la necesidad de re-extraer características para la clasificación, reduciendo significativamente el tiempo de procesamiento y el consumo de recursos computacionales. El modelo funciona completamente **offline** sin necesidad de llamadas a APIs externas, lo que garantiza privacidad, velocidad y reducción de costos operativos.

El clasificador demuestra un rendimiento excepcional en categorías críticas para el análisis financiero: **person** (99.3%), **table** (96.4%), **candlestick** (93.3%), lo que valida su utilidad práctica para el dominio de aplicación. La arquitectura modular permite actualizar el clasificador sin afectar el pipeline de extracción de keyframes, facilitando mejoras iterativas y mantenimiento del sistema.

---

### Etiquetado

El proceso de etiquetado semi-automático mediante **consenso entre CLIP y OpenAI GPT-4.1** demuestra ser una estrategia efectiva para crear datasets de alta calidad sin requerir validación manual extensiva. La concordancia del **36.46%** entre ambos modelos, aunque aparentemente baja, garantiza que solo se conserven los frames con mayor confianza, resultando en un dataset de **3,646 frames** con etiquetas de alta calidad.

El enfoque de **"LLM as Evaluator"** implementado con GPT-4.1 calificando las etiquetas de CLIP permite una validación cruzada automatizada que reduce significativamente los errores de etiquetado. La iteración en múltiples versiones de prompts optimiza la precisión del etiquetado, demostrando que la ingeniería de prompts es crucial para maximizar el rendimiento de los modelos de visión.

El balanceo del dataset por canal y categoría asegura representatividad y reduce sesgos, mientras que el filtrado por consenso elimina frames ambiguos que podrían degradar el rendimiento del clasificador. Este proceso semi-automático reduce el tiempo de etiquetado manual en más del **90%** comparado con métodos tradicionales, manteniendo o mejorando la calidad del dataset.

---

### Preprocesamiento

El pipeline de preprocesamiento logra una **reducción acumulada del 97.53% en almacenamiento** (de 81 GB a 2.0 GB) mediante la conversión de imágenes a embeddings, mientras mantiene la información esencial para el procesamiento posterior. El filtrado **SSIM con umbral 0.95** elimina el **52.16% de frames duplicados**, siendo la etapa más efectiva de reducción antes de la extracción de keyframes.

El filtrado de fondos uniformes, aunque elimina solo el **0.16% de frames**, es crucial para eliminar contenido sin información relevante, mejorando la calidad del dataset y reduciendo el ruido en las etapas posteriores. La extracción de embeddings con **ResNet-50 preentrenado** en ImageNet proporciona características robustas y generalizables que son efectivas tanto para clustering como para clasificación.

El procesamiento por lotes (batch_size=32) optimiza el uso de recursos computacionales, especialmente cuando se utiliza GPU, reduciendo el tiempo de extracción de features de manera significativa. La preservación del orden temporal durante el filtrado SSIM es esencial para mantener la coherencia narrativa del video, permitiendo que los keyframes seleccionados representen adecuadamente la secuencia temporal del contenido.

---

### Despliegue

La interfaz web con **Streamlit** proporciona una solución accesible y fácil de usar para el procesamiento de videos, permitiendo a usuarios no técnicos aprovechar el sistema completo sin necesidad de conocimientos de programación. El sistema de **caché inteligente** implementado reduce drásticamente los tiempos de procesamiento en ejecuciones repetidas, almacenando resultados intermedios (frames filtrados, embeddings, keyframes, clasificaciones) y evitando reprocesamiento innecesario.


La integración con APIs externas (OpenAI GPT-4.1) para generación de descripciones es opcional y se ejecuta solo cuando se requiere, manteniendo el sistema funcional incluso sin conexión a servicios externos. Esta flexibilidad hace que el sistema sea robusto y adaptable a diferentes entornos de despliegue, desde desarrollo local hasta producción en la nube.
