import logging
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
from docling_core.types.doc.page import TextCell, BoundingRectangle

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page, Size
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions, PaddleOcrOptions
from docling.models.base_ocr_model import BaseOcrModel

_log = logging.getLogger(__name__)


class PaddleOcrModel(BaseOcrModel):
    """
    OCR model using the PaddleOCR engine.
    https://github.com/PaddlePaddle/PaddleOCR
    """

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: PaddleOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PaddleOcrOptions = options

        if not self.enabled:
            self.model = None
            return

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            _log.error(
                "PaddleOCR is not installed. Please install it via: pip install paddlepaddle paddleocr"
            )
            raise

        # Define model directories relative to artifacts_path
        paddle_models_base_path = self.artifacts_path / "paddle_models" if self.artifacts_path else None
        det_model_dir = paddle_models_base_path / "det" if paddle_models_base_path else None
        rec_model_dir = paddle_models_base_path / "rec" if paddle_models_base_path else None
        cls_model_dir = paddle_models_base_path / "cls" if paddle_models_base_path else None

        # Ensure model directories exist if artifacts_path is provided
        # Explicitly set the model directories based on known downloaded paths
        # This is a temporary measure to debug the "No models available" error.
        # In a production setting, a more robust dynamic discovery or configuration
        # mechanism would be preferred.
        final_det_model_dir = paddle_models_base_path / "det" / "PP-OCRv5_server_det" if paddle_models_base_path else None
        final_rec_model_dir = paddle_models_base_path / "rec" / "korean_PP-OCRv5_mobile_rec" if paddle_models_base_path else None
        final_cls_model_dir = paddle_models_base_path / "cls" / "PP-LCNet_x1_0_textline_ori" if paddle_models_base_path else None

        _log.info(f"Initializing PaddleOCR with options: {options}")
        self.model = PaddleOCR(
            use_angle_cls=options.use_angle_cls,
            lang=options.lang,
        )

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled or self.model is None:
            yield from page_batch
            return

        for page in page_batch:
            ocr_rects = self.get_ocr_rects(page)
            if not ocr_rects:
                yield page
                continue

            all_ocr_cells: List[TextCell] = []
            page_image_np = np.array(page.image)

            for ocr_rect in ocr_rects:
                # Crop image to the specific rectangle to perform OCR on
                cropped_image_np = ocr_rect.crop_from_image(page_image_np)

                # Perform OCR
                # The result is a list of [bbox, (text, confidence)]
                ocr_results = self.model.ocr(cropped_image_np, cls=self.options.use_angle_cls)

                if not ocr_results or ocr_results[0] is None:
                    continue

                # Convert results to TextCell objects
                for line in ocr_results[0]:
                    points, (text, confidence) = line
                    
                    # Convert 4-point polygon to a bounding box
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    # Create a docling Rectangle, adjusting for the crop offset
                    rect = BoundingRectangle(
                        l=min(x_coords) + ocr_rect.l,
                        t=min(y_coords) + ocr_rect.t,
                        r=max(x_coords) + ocr_rect.l,
                        b=max(y_coords) + ocr_rect.t,
                    )

                    cell = TextCell(
                        text=text,
                        orig=text,
                        rect=rect,
                        confidence=confidence * 100,
                        from_ocr=True,
                    )
                    all_ocr_cells.append(cell)

            self.post_process_cells(all_ocr_cells, page)

            if self.options.debug_draw_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    def extract_text_from_image_crop(
        self, image_crop: np.ndarray, ocr_rect: BoundingRectangle, page_size: Size
    ) -> List[TextCell]:
        """
        Extracts text from a cropped image region and returns TextCells with page-level coordinates.

        :param image_crop: The cropped numpy array of the image to process.
        :param ocr_rect: The bounding box of the crop on the original page, used for coordinate offset.
        :param page_size: The size of the original page, for context.
        :return: A list of TextCell objects with coordinates relative to the full page.
        """
        if self.model is None:
            return []

        ocr_results = self.model.ocr(image_crop, cls=self.options.use_angle_cls)
        if not ocr_results or ocr_results[0] is None:
            return []

        extracted_cells = []
        for line in ocr_results[0]:
            points, (text, confidence) = line

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            # Create a docling Rectangle, adjusting for the crop offset
            rect = BoundingRectangle(
                l=min(x_coords) + ocr_rect.l,
                t=min(y_coords) + ocr_rect.t,
                r=max(x_coords) + ocr_rect.l,
                b=max(y_coords) + ocr_rect.t,
            )

            cell = TextCell(
                text=text,
                orig=text,
                rect=rect,
                confidence=confidence * 100,
                from_ocr=True,
            )
            extracted_cells.append(cell)

        return extracted_cells

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return PaddleOcrOptions
