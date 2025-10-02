import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath(str(Path(__file__).resolve().parents[2])))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, OcrOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.models.easyocr_model import EasyOcrOptions
from docling.pipeline.vlm_pipeline import VlmPipeline

from .config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL_NAME, VLM_SYSTEM_PROMPT

def create_ocr_options() -> OcrOptions:
    """OCR 옵션을 설정하고 반환합니다."""
    ocr_options = OcrOptions(
        ocr_engine="easyocr",
        lang=['ko', 'en'],
        ocr_options=EasyOcrOptions()
    )
    print("✅ OCR 옵션 설정 완료 (EasyOCR)")
    return ocr_options

def create_vlm_options() -> Optional[ApiVlmOptions]:
    """OpenRouter용 ApiVlmOptions를 생성하고 반환합니다."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        print("❌ OpenRouter API 키가 설정되지 않았습니다. 환경 변수 OPENROUTER_API_KEY를 설정하거나 config.py를 수정해주세요.")
        return None

    headers = {
        "HTTP-Referer": "http://localhost",
        "X-Title": "Docling OCR-VLM Hybrid",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }

    params = {
        "model": OPENROUTER_MODEL_NAME,
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    vlm_options = ApiVlmOptions(
        url=f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        params=params,
        prompt=VLM_SYSTEM_PROMPT,
        response_format=ResponseFormat.MARKDOWN,
        timeout=300,
        concurrency=1
    )
    print(f"✅ VLM 옵션 설정 완료 (모델: {OPENROUTER_MODEL_NAME})")
    return vlm_options

def setup_document_converter(ocr_opts: OcrOptions, vlm_opts: ApiVlmOptions) -> Optional[DocumentConverter]:
    """VLM 파이프라인이 설정된 DocumentConverter를 생성하고 반환합니다."""
    if vlm_opts is None:
        print("❌ VLM 옵션이 설정되지 않아 Converter를 생성할 수 없습니다.")
        return None

    pipeline_options = VlmPipelineOptions(
        ocr_options=ocr_opts,
        vlm_options=vlm_opts,
        enable_remote_services=True
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options
            ),
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options
            )
        }
    )
    print("✅ VLM DocumentConverter 설정 완료")
    return converter
