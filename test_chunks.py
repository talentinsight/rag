#!/usr/bin/env python3
"""
Test script to compare old vs new chunk settings
"""

import sys
import os
sys.path.append('src')

from src.semantic_chunker import SemanticChunker
from src.pdf_processor import PDFProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chunking():
    """Test both old and new chunking settings"""
    
    # Process PDF
    logger.info("Processing PDF...")
    processor = PDFProcessor()
    result = processor.process_pdf('AttentionAllYouNeed.pdf')
    
    if not result:
        logger.error("Failed to process PDF")
        return
    
    text = result['text']
    logger.info(f"PDF processed: {len(text):,} characters")
    
    # Test old settings (current AWS settings)
    logger.info("\n=== Testing OLD settings ===")
    chunker_old = SemanticChunker(
        chunk_size=300, 
        chunk_overlap=30, 
        min_chunk_size=50
    )
    chunks_old = chunker_old.chunk_text(text)
    
    # Test new settings (proposed settings)
    logger.info("\n=== Testing NEW settings ===")
    chunker_new = SemanticChunker(
        chunk_size=800, 
        chunk_overlap=100, 
        min_chunk_size=200
    )
    chunks_new = chunker_new.chunk_text(text)
    
    # Compare results
    print("\n" + "="*50)
    print("CHUNKING COMPARISON RESULTS")
    print("="*50)
    print(f"Text length: {len(text):,} characters")
    print(f"OLD settings (300/30/50):  {len(chunks_old):3d} chunks")
    print(f"NEW settings (800/100/200): {len(chunks_new):3d} chunks")
    print(f"Reduction: {len(chunks_old) - len(chunks_new):3d} chunks ({(1-len(chunks_new)/len(chunks_old))*100:.1f}% less)")
    
    # Show sample chunks
    print(f"\n=== Sample OLD chunk ===")
    if chunks_old:
        sample_old = chunks_old[5]  # 6th chunk
        print(f"ID: {sample_old.chunk_id}")
        print(f"Tokens: {sample_old.token_count}")
        print(f"Content: {sample_old.content[:200]}...")
    
    print(f"\n=== Sample NEW chunk ===")
    if chunks_new:
        sample_new = chunks_new[5]  # 6th chunk
        print(f"ID: {sample_new.chunk_id}")
        print(f"Tokens: {sample_new.token_count}")
        print(f"Content: {sample_new.content[:200]}...")
    
    return len(chunks_old), len(chunks_new)

if __name__ == "__main__":
    test_chunking()
