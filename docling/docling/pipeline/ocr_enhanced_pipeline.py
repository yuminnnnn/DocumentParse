import json
import logging
from collections.abc import Iterable
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
from docling_core.types.doc import Size
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.page import ImageRef, BoundingRectangle

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import OcrEnhancedPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions
from docling.models.api_vlm_model import ApiVlmModel
from docling.models.code_formula_model import CodeFormulaModel, CodeFormulaModelOptions
from docling.models.factories import get_ocr_factory
from docling.models.paddle_ocr_model import PaddleOcrModel
from docling.models.easyocr_model import EasyOcrModel
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class OcrEnhancedPipeline(PaginatedPipeline):
    """
    A hybrid pipeline that uses a VLM for layout analysis and a dedicated OCR model for text extraction.
    """

    def __init__(self, pipeline_options: OcrEnhancedPipelineOptions):
        super().__init__(pipeline_options)
        self.options = pipeline_options
        self.keep_backend = True

        # 1. Initialize VLM for layout analysis
        # We expect the VLM to return JSON, so we'll set a specific prompt later.
        self.vlm_model = ApiVlmModel(
            enabled=True,
            enable_remote_services=self.options.enable_remote_services,
            vlm_options=self.options.vlm_options,
        )

        # 2. Initialize OCR model using the factory
        ocr_factory = get_ocr_factory(
            allow_external_plugins=self.options.allow_external_plugins
        )
        self.ocr_model = ocr_factory.create_instance(
            options=self.options.ocr_options,
            enabled=True,  # OCR is always enabled in this pipeline
            artifacts_path=self.options.artifacts_path,
            accelerator_options=self.options.accelerator_options,
        )

        # 3. Initialize Formula model for math OCR
        self.formula_model = CodeFormulaModel(
            enabled=True, # Assuming we always want to check for formulas
            artifacts_path=self.options.artifacts_path,
            options=CodeFormulaModelOptions(do_formula_enrichment=True),
            accelerator_options=self.options.accelerator_options,
        )

        _log.info(
            f"Initialized OCR Enhanced Pipeline with VLM, {self.options.ocr_options.kind} OCR, and Formula Model."
        )

    def _process_pages(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """Process each page using the VLM for layout and OCR for text."""
        for page in page_batch:
            _log.info(f"Processing page {page.page_no} with hybrid pipeline...")

            # 1. Get layout structure from VLM in JSON format
            layout_regions = self._get_layout_from_vlm(page)

            # 2. Process each region based on its type
            page_markdown_elements = []
            if page.image and page.size:
                page_image_np = np.array(page.image)
                for region in layout_regions:
                    region_type = region.get("type")
                    if not region_type or "bbox" not in region:
                        continue

                    bbox_from_region = BoundingBox.from_tuple(
                        tuple(region["bbox"]), CoordOrigin.TOPLEFT
                    )
                    region_rect = BoundingRectangle.from_bounding_box(bbox_from_region)
                    cropped_image_np = region_rect.crop_from_image(page_image_np)
                    
                    md_element = ""
                    if region_type == "table":
                        table_data = self._process_table_region(region, page)
                        md_element = self._format_as_markdown(region, table_data)
                    elif region_type == "formula":
                        latex_text = self.formula_model.get_formula_from_image(cropped_image_np)
                        md_element = self._format_as_markdown(region, latex_text)
                    else:  # heading, paragraph, list, etc.
                        text_cells = []
                        if isinstance(self.ocr_model, PaddleOcrModel):
                            text_cells = self.ocr_model.extract_text_from_image_crop(
                                cropped_image_np, region_rect, page.size
                            )
                        elif isinstance(self.ocr_model, EasyOcrModel):
                            # Scale the bounding box to match the high-resolution image
                            scaled_rect = region_rect.scale(self.ocr_model.scale)
                            high_res_image_np = np.array(page.image.resize(
                                (int(page.size.width * self.ocr_model.scale), int(page.size.height * self.ocr_model.scale))
                            ))
                            high_res_crop = scaled_rect.crop_from_image(high_res_image_np)
                            
                            # Pass the original rect for coordinate mapping
                            text_cells = self.ocr_model.extract_text_from_image_crop(
                                high_res_crop, region_rect
                            )

                        region_text = " ".join([cell.text for cell in text_cells])
                        md_element = self._format_as_markdown(region, region_text)

                    if md_element:
                        page_markdown_elements.append(md_element)

            # Store the generated markdown for the page
            page.predictions.vlm_response.text = "\n\n".join(page_markdown_elements)
            yield page

    def _get_layout_from_vlm(self, page: Page) -> List[Dict[str, Any]]:
        """Calls the VLM to get structural layout information as JSON."""
        if not page.image:
            return []

        # Modify the VLM options on-the-fly to request JSON
        original_prompt = self.options.vlm_options.prompt
        vlm_options = self.options.vlm_options.model_copy(deep=True)
        vlm_options.prompt = """Analyze the layout of the given image and return a JSON array of objects.
Each object must have a "type" and "bbox".

- "type": Can be one of:
  - "heading": A title or subtitle.
  - "paragraph": A block of plain text.
  - "list": A bulleted or numbered list.
  - "table": A data table.
  - "chart_bar": A bar chart.
  - "chart_pie": A pie chart.
  - "image": A general image or photograph.
  - "other": Anything else.

- "bbox": An array of four numbers [x1, y1, x2, y2].

- "level" (for "heading" type): A number from 1 to 6, representing the markdown heading level (e.g., 1 for #, 2 for ##).

- "style" (for "paragraph" type): An array of strings, can contain "bold" or "italic" if the entire paragraph has that style.

Do not return any text content, only the layout structure as a JSON object.
"""
        # Temporarily override the model's options for this call
        # This is a simplified approach; a more robust solution might involve a dedicated VLM call method
        vlm_response_text = self.vlm_model.get_response_for_page(page, vlm_options)

        try:
            # Clean up potential markdown code fences
            if vlm_response_text.strip().startswith("```json"):
                vlm_response_text = vlm_response_text.strip()[7:-4]
            
            layout_data = json.loads(vlm_response_text)
            if isinstance(layout_data, list):
                return layout_data
        except (json.JSONDecodeError, TypeError) as e:
            _log.error(f"Failed to parse JSON layout from VLM response: {e}")
            _log.debug(f"VLM Response was: {vlm_response_text}")

        return []

    def _process_table_region(self, region: Dict[str, Any], page: Page) -> List[List[str]]:
        """Process a table region by OCRing each cell."""
        if not page.image or not page.size:
            return []

        page_image_np = np.array(page.image)
        cells_data = region.get("cells", [])
        
        # Find max rows and cols to create the table structure
        max_row = max(cell.get("row", 0) for cell in cells_data)
        max_col = max(cell.get("col", 0) for cell in cells_data)
        table_texts = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

        for cell in cells_data:
            row, col, bbox = cell.get("row"), cell.get("col"), cell.get("bbox")
            if row is None or col is None or bbox is None:
                continue

            cell_rect = BoundingRectangle.from_list(bbox)
            
            text_cells = []
            if isinstance(self.ocr_model, PaddleOcrModel):
                cell_image_crop = cell_rect.crop_from_image(page_image_np)
                text_cells = self.ocr_model.extract_text_from_image_crop(
                    cell_image_crop, cell_rect, page.size
                )
            elif isinstance(self.ocr_model, EasyOcrModel):
                # Scale the bounding box to match the high-resolution image
                scaled_rect = cell_rect.scale(self.ocr_model.scale)
                high_res_image_np = np.array(page.image.resize(
                    (int(page.size.width * self.ocr_model.scale), int(page.size.height * self.ocr_model.scale))
                ))
                high_res_crop = scaled_rect.crop_from_image(high_res_image_np)

                # Pass the original rect for coordinate mapping
                text_cells = self.ocr_model.extract_text_from_image_crop(
                    high_res_crop, cell_rect
                )

            cell_text = " ".join([tc.text for tc in text_cells])
            table_texts[row][col] = cell_text
        
        return table_texts

    def _format_as_markdown(self, region: Dict[str, Any], data: Any) -> str:
        """Formats data into a markdown string based on the structured region info."""
        region_type = region.get("type", "paragraph")

        if region_type == "heading":
            level = region.get("level", 1)
            return f"{'#' * level} {data}"

        if region_type == "list":
            return "\\n".join([f"- {line}" for line in data.split('\\n') if line.strip()])

        if region_type == "table":
            table_data: List[List[str]] = data
            if not table_data:
                return ""
            
            header = "| " + " | ".join(table_data[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
            body = "\n".join(["| " + " | ".join(row) + " |" for row in table_data[1:]])
            
            return f"{header}\n{separator}\n{body}"

        if region_type == "formula":
            return f"$$\n{data}\n$$"

        if region_type in ["chart_bar", "chart_pie", "image"]:
            return f"![{data}]"

        # Default to paragraph
        return str(data)

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Assembles the final document from the generated markdown of each page."""
        all_markdown = "\n\n---\n\n".join(
            [p.predictions.vlm_response.text for p in conv_res.pages if p.predictions.vlm_response]
        )

        response_bytes = BytesIO(all_markdown.encode("utf-8"))
        out_doc = InputDocument(
            path_or_stream=response_bytes,
            filename=conv_res.input.file.name,
            format=InputFormat.MD,
            backend=MarkdownDocumentBackend,
        )
        backend = MarkdownDocumentBackend(
            in_doc=out_doc,
            path_or_stream=response_bytes,
        )
        final_document = backend.convert()

        # Add page info back to the final document
        for pg_idx, page in enumerate(conv_res.pages):
            if page.image:
                final_document.add_page(
                    page_no=pg_idx + 1,
                    size=page.size,
                    image=ImageRef.from_pil(image=page.image, dpi=72),
                )

        conv_res.document = final_document
        return conv_res

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        with TimeRecorder(conv_res, "page_init"):
            page._backend = conv_res.input._backend.load_page(page.page_no)  # type: ignore
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()
                page.parsed_page = page._backend.get_segmented_page()

        return page

    @classmethod
    def get_default_options(cls) -> OcrEnhancedPipelineOptions:
        """Returns the default options for this pipeline."""
        return OcrEnhancedPipelineOptions()
    
    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, PdfDocumentBackend)
