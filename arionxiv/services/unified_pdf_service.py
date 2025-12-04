"""
Unified PDF Processing Service for ArionXiv
Consolidates pdf_processor.py and advanced_pdf_processor.py
Supports basic text extraction, OCR, table extraction, image analysis, and metadata extraction
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile
import base64
from io import BytesIO

import PyPDF2

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

logger = logging.getLogger(__name__)


class UnifiedPDFProcessor:
    """
    Comprehensive PDF processor supporting both basic and advanced operations:
    - Basic text extraction (PyPDF2)
    - Advanced text and layout analysis (PyMuPDF, pdfplumber)
    - OCR for scanned documents (Tesseract)
    - Table extraction (tabula-py)
    - Image extraction and analysis
    - Metadata extraction
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "arionxiv_pdf_processing"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Check available features
        self.features = {
            "basic_extraction": True,  # Always available with PyPDF2
            "advanced_extraction": PYMUPDF_AVAILABLE and PDFPLUMBER_AVAILABLE,
            "ocr": OCR_AVAILABLE,
            "table_extraction": TABULA_AVAILABLE,
            "image_extraction": PYMUPDF_AVAILABLE
        }
        
        logger.info(f"UnifiedPDFProcessor initialized: features={self.features}")
    
    # ====================
    # BASIC PDF PROCESSING (from pdf_processor.py)
    # ====================
    
    async def extract_text_basic(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2 (basic method)"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Basic text extraction failed: path={pdf_path}, error={str(e)}")
            return f"Error extracting text: {str(e)}"
    
    async def extract_metadata_basic(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF using PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata
                return {
                    "title": metadata.get("/Title", "Unknown"),
                    "author": metadata.get("/Author", "Unknown"),
                    "subject": metadata.get("/Subject", "Unknown"),
                    "creator": metadata.get("/Creator", "Unknown"),
                    "producer": metadata.get("/Producer", "Unknown"),
                    "creation_date": str(metadata.get("/CreationDate", "Unknown")),
                    "modification_date": str(metadata.get("/ModDate", "Unknown")),
                    "pages": len(reader.pages)
                }
        except Exception as e:
            logger.error(f"Basic metadata extraction failed: path={pdf_path}, error={str(e)}")
            return {"error": f"Metadata extraction failed: {str(e)}"}
    
    # ====================
    # ADVANCED PDF PROCESSING (from advanced_pdf_processor.py)
    # ====================
    
    async def extract_text_advanced(self, pdf_path: str, ocr_fallback: bool = True) -> Dict[str, Any]:
        """
        Advanced text extraction with multiple fallback methods
        """
        result = {
            "success": False,
            "text": "",
            "method": "",
            "pages": 0,
            "error": None
        }
        
        try:
            # Method 1: Try PyMuPDF first (fastest and most accurate for text PDFs)
            if PYMUPDF_AVAILABLE:
                try:
                    doc = fitz.open(pdf_path)
                    text = ""
                    page_count = len(doc)  # Get page count before closing
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    
                    if text.strip():  # Check if we got meaningful text
                        result.update({
                            "success": True,
                            "text": text,
                            "method": "pymupdf",
                            "pages": page_count
                        })
                        return result
                except Exception as e:
                    logger.warning(f"PyMuPDF extraction failed: {str(e)}")
            
            # Method 2: Try pdfplumber (better for complex layouts)
            if PDFPLUMBER_AVAILABLE:
                try:
                    import pdfplumber
                    with pdfplumber.open(pdf_path) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        
                        if text.strip():
                            result.update({
                                "success": True,
                                "text": text,
                                "method": "pdfplumber",
                                "pages": len(pdf.pages)
                            })
                            return result
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed: {str(e)}")
            
            # Method 3: Fallback to basic PyPDF2
            text = await self.extract_text_basic(pdf_path)
            if text and not text.startswith("Error"):
                result.update({
                    "success": True,
                    "text": text,
                    "method": "pypdf2",
                    "pages": len(PyPDF2.PdfReader(open(pdf_path, 'rb')).pages)
                })
                return result
            
            # Method 4: OCR as last resort (for scanned PDFs)
            if ocr_fallback and OCR_AVAILABLE:
                ocr_result = await self.extract_text_with_ocr(pdf_path)
                if ocr_result["success"]:
                    result.update({
                        "success": True,
                        "text": ocr_result["text"],
                        "method": "ocr",
                        "pages": ocr_result.get("pages", 0)
                    })
                    return result
            
            result["error"] = "All text extraction methods failed"
            return result
            
        except Exception as e:
            logger.error(f"Advanced text extraction failed: path={pdf_path}, error={str(e)}")
            result["error"] = f"Extraction failed: {str(e)}"
            return result
    
    async def extract_text_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text using OCR for scanned PDFs
        """
        if not OCR_AVAILABLE:
            return {
                "success": False,
                "error": "OCR not available (pytesseract not installed)"
            }
        
        try:
            if not PYMUPDF_AVAILABLE:
                return {
                    "success": False,
                    "error": "PyMuPDF required for OCR processing"
                }
            
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # Increase resolution for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(BytesIO(img_data))
                
                # Perform OCR
                page_text = pytesseract.image_to_string(image, lang='eng')
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            
            return {
                "success": True,
                "text": full_text,
                "method": "ocr",
                "pages": len(doc)
            }
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: path={pdf_path}, error={str(e)}")
            return {
                "success": False,
                "error": f"OCR failed: {str(e)}"
            }
    
    async def extract_tables(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract tables from PDF using tabula-py
        """
        if not TABULA_AVAILABLE:
            return {
                "success": False,
                "error": "Table extraction not available (tabula-py not installed)"
            }
        
        try:
            # Extract all tables from all pages
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            table_data = []
            for i, table in enumerate(tables):
                table_dict = {
                    "table_id": i + 1,
                    "columns": table.columns.tolist(),
                    "rows": table.values.tolist(),
                    "shape": table.shape
                }
                table_data.append(table_dict)
            
            return {
                "success": True,
                "tables": table_data,
                "count": len(tables)
            }
            
        except Exception as e:
            logger.error(f"Table extraction failed: path={pdf_path}, error={str(e)}")
            return {
                "success": False,
                "error": f"Table extraction failed: {str(e)}"
            }
    
    async def extract_images(self, pdf_path: str, save_images: bool = False) -> Dict[str, Any]:
        """
        Extract images from PDF
        """
        if not PYMUPDF_AVAILABLE:
            return {
                "success": False,
                "error": "Image extraction not available (PyMuPDF not installed)"
            }
        
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = {
                            "page": page_num + 1,
                            "image_index": img_index,
                            "width": pix.width,
                            "height": pix.height,
                            "colorspace": pix.colorspace.name if pix.colorspace else "Unknown"
                        }
                        
                        if save_images:
                            # Save image to temp directory
                            img_filename = f"page_{page_num + 1}_img_{img_index}.png"
                            img_path = self.temp_dir / img_filename
                            pix.save(str(img_path))
                            img_data["saved_path"] = str(img_path)
                        else:
                            # Convert to base64 for embedding
                            img_bytes = pix.tobytes("png")
                            img_base64 = base64.b64encode(img_bytes).decode()
                            img_data["base64"] = img_base64
                        
                        images.append(img_data)
                    
                    pix = None  # Free memory
            
            doc.close()
            
            return {
                "success": True,
                "images": images,
                "count": len(images)
            }
            
        except Exception as e:
            logger.error(f"Image extraction failed: path={pdf_path}, error={str(e)}")
            return {
                "success": False,
                "error": f"Image extraction failed: {str(e)}"
            }
    
    async def get_document_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze document structure and extract metadata
        """
        try:
            # Get basic metadata first
            basic_metadata = await self.extract_metadata_basic(pdf_path)
            
            structure = {
                "metadata": basic_metadata,
                "structure": {},
                "features": self.features
            }
            
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(pdf_path)
                
                # Get document outline/bookmarks
                outline = doc.get_toc()
                structure["structure"]["outline"] = outline
                
                # Analyze pages
                pages_info = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_info = {
                        "page_number": page_num + 1,
                        "width": page.rect.width,
                        "height": page.rect.height,
                        "rotation": page.rotation,
                        "has_text": bool(page.get_text().strip()),
                        "image_count": len(page.get_images()),
                        "annotation_count": len(page.annots())
                    }
                    pages_info.append(page_info)
                
                structure["structure"]["pages"] = pages_info
                doc.close()
            
            return {
                "success": True,
                "data": structure
            }
            
        except Exception as e:
            logger.error(f"Document structure analysis failed: path={pdf_path}, error={str(e)}")
            return {
                "success": False,
                "error": f"Structure analysis failed: {str(e)}"
            }
    
    # ====================
    # UNIFIED INTERFACE
    # ====================
    
    async def process_pdf(self, pdf_path: str, options: Dict[str, bool] = None) -> Dict[str, Any]:
        """
        Process PDF with all available methods based on options
        """
        if options is None:
            options = {
                "extract_text": True,
                "extract_tables": False,
                "extract_images": False,
                "extract_metadata": True,
                "use_ocr": False,
                "analyze_structure": False
            }
        
        result = {
            "success": True,
            "file": pdf_path,
            "features_used": [],
            "errors": []
        }
        
        try:
            # Extract text
            if options.get("extract_text", True):
                if self.features["advanced_extraction"]:
                    text_result = await self.extract_text_advanced(pdf_path, options.get("use_ocr", False))
                    result["text_extraction"] = text_result
                    result["features_used"].append("advanced_text_extraction")
                else:
                    text = await self.extract_text_basic(pdf_path)
                    result["text_extraction"] = {
                        "success": not text.startswith("Error"),
                        "text": text,
                        "method": "basic"
                    }
                    result["features_used"].append("basic_text_extraction")
            
            # Extract metadata
            if options.get("extract_metadata", True):
                metadata = await self.extract_metadata_basic(pdf_path)
                result["metadata"] = metadata
                result["features_used"].append("metadata_extraction")
            
            # Extract tables
            if options.get("extract_tables", False):
                tables_result = await self.extract_tables(pdf_path)
                result["tables"] = tables_result
                result["features_used"].append("table_extraction")
                if not tables_result["success"]:
                    result["errors"].append(tables_result["error"])
            
            # Extract images
            if options.get("extract_images", False):
                images_result = await self.extract_images(pdf_path)
                result["images"] = images_result
                result["features_used"].append("image_extraction")
                if not images_result["success"]:
                    result["errors"].append(images_result["error"])
            
            # Analyze structure
            if options.get("analyze_structure", False):
                structure_result = await self.get_document_structure(pdf_path)
                result["structure"] = structure_result
                result["features_used"].append("structure_analysis")
                if not structure_result["success"]:
                    result["errors"].append(structure_result["error"])
            
            return result
            
        except Exception as e:
            logger.error(f"PDF processing failed: path={pdf_path}, error={str(e)}")
            result["success"] = False
            result["error"] = f"Processing failed: {str(e)}"
            return result
    
    # ====================
    # BACKWARDS COMPATIBILITY
    # ====================
    
    async def extract_text(self, pdf_path: str) -> str:
        """Backwards compatible text extraction method"""
        if self.features["advanced_extraction"]:
            result = await self.extract_text_advanced(pdf_path)
            return result.get("text", "")
        else:
            return await self.extract_text_basic(pdf_path)
    
    async def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Backwards compatible metadata extraction method"""
        return await self.extract_metadata_basic(pdf_path)


# Global instance
unified_pdf_processor = UnifiedPDFProcessor()

# Backwards compatibility
pdf_processor = unified_pdf_processor
advanced_pdf_processor = unified_pdf_processor

# Export commonly used functions
extract_text = unified_pdf_processor.extract_text
extract_metadata = unified_pdf_processor.extract_metadata
process_pdf = unified_pdf_processor.process_pdf

# Additional aliases for compatibility
pdf_service = unified_pdf_processor

__all__ = [
    'UnifiedPDFProcessor',
    'unified_pdf_processor',
    'pdf_processor',
    'pdf_service',
    'advanced_pdf_processor',
    'extract_text',
    'extract_metadata',
    'process_pdf'
]