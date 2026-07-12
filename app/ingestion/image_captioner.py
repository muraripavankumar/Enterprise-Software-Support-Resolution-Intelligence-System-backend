import base64
from pathlib import Path
from typing import Any, Dict

from openai import AzureOpenAI, OpenAIError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ImageCaptioner:
    """Creates retrieval-friendly captions for extracted images."""

    def __init__(self) -> None:
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

    def caption_image(self, image: Dict[str, Any]) -> str:
        image_path = Path(str(image["image_path"]))
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        try:
            response = self.client.chat.completions.create(
                model=settings.azure_openai_llm_deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Caption this support-documentation image for search retrieval. Be concise.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded}"},
                            },
                        ],
                    }
                ],
                max_tokens=160,
            )
            return response.choices[0].message.content or self._fallback(image)
        except OpenAIError as exc:
            logger.warning("Image captioning failed; using fallback caption: %s", exc)
            return self._fallback(image)

    def _fallback(self, image: Dict[str, Any]) -> str:
        return f"Image caption from page {image.get('page_number')}, image {image.get('image_index')}."
