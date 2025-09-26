"""
PDF Processing Module for RAG Implementation
Extracts and preprocesses text from PDF documents
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from PyPDF2 import PdfReader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    A class to handle PDF text extraction and preprocessing
    """
    
    def __init__(self):
        self.text_content = ""
        self.metadata = {}
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber (better for complex layouts)
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from the page
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Add page marker for reference
                        text_content.append(f"\\n--- Page {page_num} ---\\n")
                        text_content.append(page_text)
                        logger.debug(f"Extracted text from page {page_num}")
                    else:
                        logger.warning(f"No text found on page {page_num}")
            
            full_text = "\\n".join(text_content)
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {str(e)}")
            raise
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2 (fallback method)
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            text_content = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                logger.info(f"Processing PDF with {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_content.append(f"\\n--- Page {page_num} ---\\n")
                        text_content.append(page_text)
                        logger.debug(f"Extracted text from page {page_num}")
                    else:
                        logger.warning(f"No text found on page {page_num}")
            
            full_text = "\\n".join(text_content)
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned and preprocessed text
        """
        logger.info("Starting text preprocessing")
        
        # Fix OCR artifacts first (before removing page markers)
        text = re.sub(r'\\1 \\2', '', text)  # Remove \1 \2 artifacts
        text = re.sub(r'\\[0-9]+', '', text)  # Remove other numbered artifacts
        text = re.sub(r' 1 1 2', '', text)  # Remove specific OCR artifact pattern
        text = re.sub(r' 1 2 1 1 2', '', text)  # Remove another specific pattern
        text = re.sub(r'1 1 2', '', text)  # Remove pattern without leading space
        text = re.sub(r'1 2 1 1 2', '', text)  # Remove another pattern without leading space
        
        # Remove page markers
        text = re.sub(r'--- Page \\d+ ---', '', text)
        
        # Fix common spacing issues
        text = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', text)  # Add space between camelCase
        text = re.sub(r'([.!?])([A-Z])', r'\\1 \\2', text)  # Add space after punctuation
        text = re.sub(r'([a-z])([0-9])', r'\\1 \\2', text)  # Add space between letter and number
        text = re.sub(r'([0-9])([a-z])', r'\\1 \\2', text)  # Add space between number and letter
        
        # Fix common OCR word concatenations
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\\1 \\2', text)
        text = re.sub(r'([.])([A-Z])', r'\\1 \\2', text)
        
        # Clean up specific patterns found in academic papers
        text = re.sub(r'∗', '*', text)  # Replace special asterisk
        text = re.sub(r'†', '†', text)  # Keep dagger as is
        text = re.sub(r'‡', '‡', text)  # Keep double dagger as is
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\\n+', '\\n', text)
        text = re.sub(r'[ \\t]+', ' ', text)
        
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\\n') if line.strip()]
        text = '\\n'.join(lines)
        
        # Fix some common academic paper artifacts
        text = re.sub(r'\\b([a-z]+)([A-Z][a-z]+)\\b', r'\\1 \\2', text)
        
        logger.info(f"Text preprocessing completed. Final length: {len(text)} characters")
        
        return text
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: PDF metadata
        """
        try:
            metadata = {}
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', ''),
                        'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
                        'modification_date': pdf_reader.metadata.get('/ModDate', ''),
                    })
                
                metadata['num_pages'] = len(pdf_reader.pages)
                metadata['file_size'] = Path(pdf_path).stat().st_size
                
            logger.info(f"Extracted metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}
    
    def process_pdf(self, pdf_path: str, use_pdfplumber: bool = True) -> Dict[str, Any]:
        """
        Main method to process PDF - extract text, preprocess, and get metadata
        
        Args:
            pdf_path (str): Path to the PDF file
            use_pdfplumber (bool): Whether to use pdfplumber (True) or PyPDF2 (False)
            
        Returns:
            Dict[str, Any]: Dictionary containing processed text and metadata
        """
        logger.info(f"Starting PDF processing: {pdf_path}")
        
        # Check if file exists
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text
        if use_pdfplumber:
            try:
                raw_text = self.extract_text_pdfplumber(pdf_path)
            except Exception as e:
                logger.warning(f"pdfplumber failed, falling back to PyPDF2: {str(e)}")
                raw_text = self.extract_text_pypdf2(pdf_path)
        else:
            raw_text = self.extract_text_pypdf2(pdf_path)
        
        # Preprocess text
        processed_text = self.preprocess_text(raw_text)
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path)
        
        # Store results
        self.text_content = processed_text
        self.metadata = metadata
        
        result = {
            'text': processed_text,
            'metadata': metadata,
            'raw_text': raw_text,
            'file_path': pdf_path
        }
        
        logger.info("PDF processing completed successfully")
        return result


def main():
    """
    Test the PDF processor with the Attention paper
    """
    processor = PDFProcessor()
    
    # Process the Attention All You Need paper
    pdf_path = "/Users/sam/Desktop/rag/AttentionAllYouNeed.pdf"
    
    try:
        result = processor.process_pdf(pdf_path)
        
        print(f"\\n=== PDF Processing Results ===")
        print(f"File: {result['file_path']}")
        print(f"Text length: {len(result['text'])} characters")
        print(f"Raw text length: {len(result['raw_text'])} characters")
        print(f"Number of pages: {result['metadata'].get('num_pages', 'Unknown')}")
        print(f"File size: {result['metadata'].get('file_size', 'Unknown')} bytes")
        
        # Show first 500 characters of processed text
        print(f"\\n=== First 500 characters of processed text ===")
        print(result['text'][:500])
        print("...")
        
        # Save processed text to file for inspection
        output_path = "/Users/sam/Desktop/rag/processed_attention_paper.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        print(f"\\nProcessed text saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        raise


if __name__ == "__main__":
    main()
