"""
Módulo para extracción de features usando modelos preentrenados.
Soporta: ResNet-50, GoogLeNet (Inception v1), y VGG16.
"""
from __future__ import annotations

import os
import pickle
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import googlenet
from torchvision.models import GoogLeNet_Weights
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
from tqdm import tqdm


class FeatureExtractor:
    """
    Clase para extraer features de imágenes usando modelos preentrenados.
    Soporta: ResNet-50, GoogLeNet, y VGG16.
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Inicializa el extractor de features.

        Args:
            model_name: Nombre del modelo a usar ('resnet50', 'googlenet', 'vgg16')
            device: Dispositivo a usar ('cuda', 'cpu', o None para auto-detectar)
            batch_size: Tamaño del lote para procesamiento
        """
        self.model_name = model_name.lower()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # Cargar modelo según el nombre especificado
        print(f"Cargando {model_name.upper()} en {self.device}...")
        self.model, self.feature_dim = self._load_model(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        # Transformaciones de imagen (estándar para ImageNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"Modelo {model_name.upper()} cargado exitosamente")
        print(f"Dimensión de features: {self.feature_dim}")

    def _load_model(self, model_name: str) -> Tuple[nn.Module, int]:
        """
        Carga el modelo especificado y retorna el modelo sin capa de clasificación.

        Args:
            model_name: Nombre del modelo

        Returns:
            Tuple con (modelo, dimensión_de_features)
        """
        if model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Remover la capa de clasificación (avgpool + fc)
            model = nn.Sequential(*list(model.children())[:-1])
            feature_dim = 2048
            return model, feature_dim

        elif model_name == 'googlenet':
            model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1, transform_input=False)
            # GoogLeNet: extraer features hasta avgpool (sin clasificador)
            # Usar Sequential para simplificar
            class GoogLeNetFeatureExtractor(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    # Copiar todas las capas hasta avgpool
                    self.conv1 = base_model.conv1
                    self.maxpool1 = base_model.maxpool1
                    self.conv2 = base_model.conv2
                    self.conv3 = base_model.conv3
                    self.maxpool2 = base_model.maxpool2
                    self.inception3a = base_model.inception3a
                    self.inception3b = base_model.inception3b
                    self.maxpool3 = base_model.maxpool3
                    self.inception4a = base_model.inception4a
                    self.inception4b = base_model.inception4b
                    self.inception4c = base_model.inception4c
                    self.inception4d = base_model.inception4d
                    self.inception4e = base_model.inception4e
                    self.maxpool4 = base_model.maxpool4
                    self.inception5a = base_model.inception5a
                    self.inception5b = base_model.inception5b
                    self.avgpool = base_model.avgpool

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.maxpool1(x)
                    x = self.conv2(x)
                    x = self.conv3(x)
                    x = self.maxpool2(x)
                    x = self.inception3a(x)
                    x = self.inception3b(x)
                    x = self.maxpool3(x)
                    x = self.inception4a(x)
                    x = self.inception4b(x)
                    x = self.inception4c(x)
                    x = self.inception4d(x)
                    x = self.inception4e(x)
                    x = self.maxpool4(x)
                    x = self.inception5a(x)
                    x = self.inception5b(x)
                    x = self.avgpool(x)
                    # Flatten para obtener vector de features
                    x = x.view(x.size(0), -1)
                    return x

            feature_extractor = GoogLeNetFeatureExtractor(model)
            feature_dim = 1024  # GoogLeNet avgpool produce 1024 features
            return feature_extractor, feature_dim

        elif model_name == 'vgg16':
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            # Remover las capas de clasificación (classifier)
            # Usar solo hasta avgpool
            features_only = model.features
            avgpool = model.avgpool

            class VGG16FeatureExtractor(nn.Module):
                def __init__(self, features, avgpool):
                    super().__init__()
                    self.features = features
                    self.avgpool = avgpool

                def forward(self, x):
                    x = self.features(x)
                    x = self.avgpool(x)
                    return x

            feature_extractor = VGG16FeatureExtractor(features_only, avgpool)
            feature_dim = 512 * 7 * 7  # 25088, pero se puede usar un adaptive pooling
            # Usar adaptive pooling para reducir a 4096
            class VGG16FeatureExtractorFixed(nn.Module):
                def __init__(self, features, avgpool):
                    super().__init__()
                    self.features = features
                    self.avgpool = avgpool
                    self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

                def forward(self, x):
                    x = self.features(x)
                    x = self.adaptive_pool(x)
                    return x

            feature_extractor = VGG16FeatureExtractorFixed(features_only, avgpool)
            feature_dim = 512  # Después del adaptive pooling
            return feature_extractor, feature_dim

        else:
            raise ValueError(
                f"Modelo '{model_name}' no soportado. "
                f"Modelos disponibles: 'resnet50', 'googlenet', 'vgg16'",
            )

    def load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Carga y preprocesa una imagen.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Tensor de la imagen preprocesada o None si hay error
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            return img_tensor
        except Exception as e:
            print(f"Error cargando imagen {image_path}: {e}")
            return None

    def extract_features_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Extrae features de un lote de imágenes.

        Args:
            image_paths: Lista de rutas a imágenes

        Returns:
            Array numpy con features (n_images, feature_dim)
            feature_dim depende del modelo: ResNet-50=2048, GoogLeNet=1024, VGG16=512
        """
        # Cargar imágenes
        images = []
        valid_indices = []

        for idx, path in enumerate(image_paths):
            img_tensor = self.load_image(path)
            if img_tensor is not None:
                images.append(img_tensor)
                valid_indices.append(idx)

        if not images:
            return np.array([])

        # Apilar imágenes en un batch
        batch = torch.cat(images, dim=0).to(self.device)

        # Extraer features
        with torch.no_grad():
            features = self.model(batch)
            features = features.squeeze()  # Remover dimensiones extra

            # Si solo hay una imagen, agregar dimensión de batch
            if len(features.shape) == 1:
                features = features.unsqueeze(0)

            # Flatten features
            features = features.view(features.size(0), -1)

            # Convertir a numpy
            features_np = features.cpu().numpy()

        return features_np

    def extract_features_from_paths(
        self,
        image_paths: List[str],
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extrae features de una lista de imágenes.

        Args:
            image_paths: Lista de rutas a imágenes
            save_path: Ruta opcional para guardar features

        Returns:
            Tuple con:
            - Array numpy con features (n_images, feature_dim)
            - Lista de rutas válidas (puede ser menor que image_paths)
        """
        all_features = []
        valid_paths = []

        # Procesar en lotes
        n_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(n_batches), desc='Extrayendo features'):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]

            # Extraer features del lote
            batch_features = self.extract_features_batch(batch_paths)

            if len(batch_features) > 0:
                # Obtener índices válidos del lote
                for j, path in enumerate(batch_paths):
                    if os.path.exists(path):
                        valid_paths.append(path)

                all_features.append(batch_features)

        if not all_features:
            return np.array([]), []

        # Concatenar todos los features
        features_array = np.vstack(all_features)

        # Ajustar valid_paths si hay diferencias
        if len(valid_paths) != features_array.shape[0]:
            # Recalcular valid_paths basado en features extraídos
            valid_paths = [path for path in image_paths if os.path.exists(path)]
            valid_paths = valid_paths[:features_array.shape[0]]

        # Guardar si se especifica
        if save_path:
            self.save_features(features_array, valid_paths, save_path)

        return features_array, valid_paths

    def save_features(
        self,
        features: np.ndarray,
        image_paths: List[str],
        save_path: str,
    ):
        """
        Guarda features y sus rutas asociadas.

        Args:
            features: Array numpy con features
            image_paths: Lista de rutas asociadas
            save_path: Ruta donde guardar
        """
        data = {
            'features': features,
            'image_paths': image_paths,
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
        }

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Features guardados en {save_path} (modelo: {self.model_name}, dim: {self.feature_dim})")

    def load_features(self, load_path: str) -> Tuple[np.ndarray, List[str], Optional[dict]]:
        """
        Carga features guardados.

        Args:
            load_path: Ruta al archivo de features

        Returns:
            Tuple con (features, rutas asociadas, metadata)
            metadata contiene: model_name, feature_dim (si están guardados)
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        features = data['features']
        image_paths = data['image_paths']

        # Extraer metadata si está disponible
        metadata = {
            'model_name': data.get('model_name', 'unknown'),
            'feature_dim': data.get('feature_dim', features.shape[1] if len(features.shape) > 1 else None),
        }

        return features, image_paths, metadata
