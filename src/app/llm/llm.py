from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

default_config = {
    'type': 'ollama',
    'model': 'phi4:latest',
    'base_url': 'http://localhost:11434',
}


class Llm:
    def __init__(self, config: dict = default_config):
        self.type = config.get('type')
        if self.type == 'ollama':
            self.base_url = config.get('base_url')
            self.model = config.get('model')
            self.llm = ChatOllama(base_url=self.base_url, model=self.model)
        elif self.type == 'openai':
            self.api_key = config.get('api_key')
            self.model = config.get('model', 'gpt-4o-mini')
            self.temperature = config.get('temperature', 0)
            self.max_completion_tokens = config.get('max_tokens')
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                timeout=120,
                max_retries=2,
                # api_key="...",
                # base_url="...",
            )
        else:
            raise ValueError(
                'Unsupported LLM type. Supported types are: ollama, openai',
            )

    def get_llm(self):
        return self.llm

    def load_prompt(self, prompt_path: str):
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
        with open(prompt_path) as f:
            prompt_text = f.read()
        return prompt_text


class SummaryLlm(Llm):
    def __init__(
        self,
        config: dict = default_config,
        prompt_name: str = 'v1_summary_expert',
    ):
        super().__init__(config)
        self.prompt_name = prompt_name
        self.summary_prompt_template = self.load_prompt(
            prompt_path=f'app/prompts/{prompt_name}.txt',
        )

    def summarize(self, context):
        template = PromptTemplate(
            template=self.summary_prompt_template,
            input_variables=['context'],
        )

        qna_chain = template | self.llm | StrOutputParser()
        return qna_chain.invoke({'context': context})


class FrameDescriptionLlm(Llm):
    """
    Clase para describir frames usando OpenAI Vision API.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        prompt_path: Optional[str] = None,
    ):
        """
        Inicializa el LLM para descripción de frames.

        Args:
            config: Configuración del LLM (si None, usa OpenAI por defecto)
            prompt_path: Ruta al archivo de prompt
                       (si None, usa el por defecto)
        """
        if config is None:
            config = {
                'type': 'openai',
                'model': 'gpt-4o-mini',  # Modelo que soporta vision
                'temperature': 0,
            }
        super().__init__(config)

        # Cargar prompt
        if prompt_path is None:
            prompt_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'prompts',
                'frame_description.txt',
            )
        self.prompt_template = self.load_prompt(prompt_path)

    def describe_image(
        self,
        image_path: str,
        category: Optional[str] = None,
    ) -> str:
        """
        Describe una imagen usando OpenAI Vision API.

        Args:
            image_path: Ruta a la imagen a describir
            category: Categoría del frame para usar prompt específico

        Returns:
            Descripción textual de la imagen
        """
        import base64
        import openai
        from dotenv import load_dotenv

        # Usar prompt específico por categoría si está disponible
        prompt_to_use = self.prompt_template
        if category:
            category_prompt_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'prompts',
                f'{category}.txt',
            )
            if os.path.exists(category_prompt_path):
                prompt_to_use = self.load_prompt(category_prompt_path)

        # Leer y codificar imagen en base64
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                base64_image = base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Error leyendo imagen {image_path}: {e}")

        # Usar directamente la API de OpenAI para vision
        # ya que langchain puede tener problemas con el formato de imágenes
        load_dotenv()

        client = openai.OpenAI()

        # Asegurarse de usar un modelo que soporte vision
        vision_model = self.model
        # Validar que el modelo sea compatible con vision
        valid_vision_models = ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1']
        if vision_model not in valid_vision_models:
            # Si el modelo no es compatible, usar gpt-4o-mini por defecto
            vision_model = 'gpt-4o-mini'

        try:
            response = client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': prompt_to_use,
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': (
                                        f"data:image/jpeg;base64,"
                                        f"{base64_image}"
                                    ),
                                },
                            },
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=2000,  # Aumentar tokens para descripciones
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(
                f"Error generando descripción con OpenAI Vision API: {e}",
            )
