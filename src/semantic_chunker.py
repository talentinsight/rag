"""
Semantic Text Chunking Module for RAG Implementation
Creates meaningful chunks based on semantic similarity and content structure
"""

import logging
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    Data class to represent a text chunk with metadata
    """
    content: str
    start_char: int
    end_char: int
    chunk_id: str
    section_title: str = ""
    chunk_type: str = "content"  # content, title, abstract, reference, etc.
    token_count: int = 0
    embedding: List[float] = None


class SemanticChunker:
    """
    A class to perform semantic chunking of academic papers
    """
    
    def __init__(self, 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100,
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the semantic chunker
        
        Args:
            chunk_size (int): Target size for chunks in tokens
            chunk_overlap (int): Number of overlapping tokens between chunks
            min_chunk_size (int): Minimum chunk size in tokens
            model_name (str): Model name for tokenization
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.model_name = model_name
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            logger.info(f"Initialized tiktoken tokenizer for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken for {model_name}: {e}")
            # Fallback to a simple tokenizer
            self.tokenizer = None
        
        # Initialize TF-IDF vectorizer for semantic similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.chunks = []
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate token count (1 token ≈ 4 characters)
            return len(text) // 4
    
    def identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify sections in academic paper text
        
        Args:
            text (str): Full paper text
            
        Returns:
            List[Dict]: List of sections with start/end positions and titles
        """
        sections = []
        
        # Common academic paper section patterns
        section_patterns = [
            r'^\\s*(Abstract|ABSTRACT)\\s*$',
            r'^\\s*(Introduction|INTRODUCTION)\\s*$',
            r'^\\s*(Related Work|RELATED WORK|Literature Review)\\s*$',
            r'^\\s*(Method|Methods|METHODS?|Methodology|METHODOLOGY)\\s*$',
            r'^\\s*(Approach|APPROACH)\\s*$',
            r'^\\s*(Model|MODEL|Architecture|ARCHITECTURE)\\s*$',
            r'^\\s*(Experiment|Experiments|EXPERIMENTS?)\\s*$',
            r'^\\s*(Results|RESULTS)\\s*$',
            r'^\\s*(Discussion|DISCUSSION)\\s*$',
            r'^\\s*(Conclusion|Conclusions|CONCLUSIONS?)\\s*$',
            r'^\\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\\s*$',
            r'^\\s*(Appendix|APPENDIX)\\s*$',
            r'^\\s*([0-9]+\\.?\\s+[A-Z][^\\n]{5,50})\\s*$',  # Numbered sections
            r'^\\s*([A-Z][A-Z\\s]{10,50})\\s*$'  # ALL CAPS sections
        ]
        
        lines = text.split('\\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_start = current_pos
            current_pos += len(line) + 1  # +1 for newline
            
            for pattern in section_patterns:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    sections.append({
                        'title': line.strip(),
                        'start_char': line_start,
                        'line_number': i,
                        'pattern': pattern
                    })
                    break
        
        logger.info(f"Identified {len(sections)} sections")
        return sections
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences and paragraphs
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences/paragraphs
        """
        # First split by paragraphs (double newlines or single newlines in academic papers)
        paragraphs = re.split(r'\\n\\s*\\n|\\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        sentences = []
        for paragraph in paragraphs:
            # Split each paragraph into sentences
            para_sentences = re.split(r'(?<=[.!?])\\s+', paragraph)
            para_sentences = [s.strip() for s in para_sentences if s.strip()]
            
            # If paragraph is very long and has no sentence breaks, split by length
            if len(para_sentences) == 1 and len(para_sentences[0]) > 1000:
                # Split long paragraphs by commas or semicolons
                parts = re.split(r'[,;]\\s+', para_sentences[0])
                sentences.extend([part.strip() for part in parts if part.strip()])
            else:
                sentences.extend(para_sentences)
        
        logger.info(f"Split text into {len(sentences)} sentences/segments")
        return sentences
    
    def create_semantic_chunks(self, text: str) -> List[TextChunk]:
        """
        Create semantic chunks from text using a sliding window approach
        
        Args:
            text (str): Input text to chunk
            
        Returns:
            List[TextChunk]: List of semantic chunks
        """
        logger.info("Starting semantic chunking")
        
        chunks = []
        current_section = "Content"
        
        # If sentence splitting fails, use character-based chunking with word boundaries
        sentences = self.split_by_sentences(text)
        
        if len(sentences) <= 1:
            logger.info("Falling back to character-based chunking")
            return self._create_character_based_chunks(text)
        
        logger.info(f"Working with {len(sentences)} sentences")
        
        # Group sentences into chunks
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_start_char = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if (current_chunk_tokens + sentence_tokens > self.chunk_size and 
                current_chunk_sentences and 
                current_chunk_tokens >= self.min_chunk_size):
                
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk_sentences)
                chunk_end_char = chunk_start_char + len(chunk_content)
                
                chunk = TextChunk(
                    content=chunk_content,
                    start_char=chunk_start_char,
                    end_char=chunk_end_char,
                    chunk_id=f"chunk_{len(chunks):04d}",
                    section_title=current_section,
                    token_count=current_chunk_tokens
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk_sentences) > 1:
                    # Keep last sentence for overlap
                    overlap_sentences = [current_chunk_sentences[-1]]
                    overlap_tokens = self.count_tokens(overlap_sentences[0])
                    
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_chunk_tokens = overlap_tokens + sentence_tokens
                else:
                    current_chunk_sentences = [sentence]
                    current_chunk_tokens = sentence_tokens
                
                chunk_start_char = chunk_end_char - len(overlap_sentences[0]) if self.chunk_overlap > 0 else chunk_end_char
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
        
        # Add final chunk if there are remaining sentences
        if current_chunk_sentences and current_chunk_tokens >= self.min_chunk_size:
            chunk_content = ' '.join(current_chunk_sentences)
            chunk_end_char = chunk_start_char + len(chunk_content)
            
            chunk = TextChunk(
                content=chunk_content,
                start_char=chunk_start_char,
                end_char=chunk_end_char,
                chunk_id=f"chunk_{len(chunks):04d}",
                section_title=current_section,
                token_count=current_chunk_tokens
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        
        # Calculate semantic similarity between chunks for quality assessment
        if len(chunks) > 1:
            self._calculate_chunk_similarities(chunks)
        
        self.chunks = chunks
        return chunks
    
    def _create_character_based_chunks(self, text: str) -> List[TextChunk]:
        """
        Create chunks based on character count with word boundaries
        
        Args:
            text (str): Input text
            
        Returns:
            List[TextChunk]: List of chunks
        """
        logger.info("Creating character-based chunks")
        
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        current_chunk_words = []
        current_chunk_chars = 0
        target_chars = self.chunk_size * 4  # Approximate: 1 token ≈ 4 chars
        
        for word in words:
            word_chars = len(word) + 1  # +1 for space
            
            # Check if adding this word would exceed target size
            if (current_chunk_chars + word_chars > target_chars and 
                current_chunk_words and 
                current_chunk_chars >= self.min_chunk_size * 4):
                
                # Create chunk
                chunk_content = ' '.join(current_chunk_words)
                chunk_tokens = self.count_tokens(chunk_content)
                
                chunk = TextChunk(
                    content=chunk_content,
                    start_char=0,  # Simplified for now
                    end_char=len(chunk_content),
                    chunk_id=f"chunk_{len(chunks):04d}",
                    section_title="Content",
                    token_count=chunk_tokens
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk_words) > 10:
                    overlap_words = current_chunk_words[-10:]  # Keep last 10 words
                    current_chunk_words = overlap_words + [word]
                    current_chunk_chars = sum(len(w) + 1 for w in current_chunk_words)
                else:
                    current_chunk_words = [word]
                    current_chunk_chars = word_chars
            else:
                current_chunk_words.append(word)
                current_chunk_chars += word_chars
        
        # Add final chunk
        if current_chunk_words and current_chunk_chars >= self.min_chunk_size * 4:
            chunk_content = ' '.join(current_chunk_words)
            chunk_tokens = self.count_tokens(chunk_content)
            
            chunk = TextChunk(
                content=chunk_content,
                start_char=0,
                end_char=len(chunk_content),
                chunk_id=f"chunk_{len(chunks):04d}",
                section_title="Content",
                token_count=chunk_tokens
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} character-based chunks")
        return chunks
    
    def _calculate_chunk_similarities(self, chunks: List[TextChunk]) -> None:
        """
        Calculate semantic similarities between adjacent chunks
        
        Args:
            chunks (List[TextChunk]): List of chunks to analyze
        """
        try:
            chunk_texts = [chunk.content for chunk in chunks]
            
            # Fit TF-IDF vectorizer and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
            
            # Calculate similarities between adjacent chunks
            similarities = []
            for i in range(len(chunks) - 1):
                similarity = cosine_similarity(
                    tfidf_matrix[i:i+1], 
                    tfidf_matrix[i+1:i+2]
                )[0][0]
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            logger.info(f"Average adjacent chunk similarity: {avg_similarity:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to calculate chunk similarities: {e}")
    
    def optimize_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Optimize chunks by merging small chunks and splitting large ones
        
        Args:
            chunks (List[TextChunk]): Input chunks
            
        Returns:
            List[TextChunk]: Optimized chunks
        """
        logger.info("Optimizing chunks")
        
        optimized_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If chunk is too small, try to merge with next chunk
            if (current_chunk.token_count < self.min_chunk_size and 
                i + 1 < len(chunks) and
                current_chunk.token_count + chunks[i + 1].token_count <= self.chunk_size * 1.2):
                
                next_chunk = chunks[i + 1]
                merged_content = current_chunk.content + " " + next_chunk.content
                merged_tokens = current_chunk.token_count + next_chunk.token_count
                
                merged_chunk = TextChunk(
                    content=merged_content,
                    start_char=current_chunk.start_char,
                    end_char=next_chunk.end_char,
                    chunk_id=f"chunk_{len(optimized_chunks):04d}",
                    section_title=current_chunk.section_title,
                    token_count=merged_tokens
                )
                
                optimized_chunks.append(merged_chunk)
                i += 2  # Skip next chunk as it's been merged
            else:
                # Keep chunk as is, but update ID
                current_chunk.chunk_id = f"chunk_{len(optimized_chunks):04d}"
                optimized_chunks.append(current_chunk)
                i += 1
        
        logger.info(f"Optimized from {len(chunks)} to {len(optimized_chunks)} chunks")
        return optimized_chunks
    
    def chunk_text(self, text: str, optimize: bool = True) -> List[TextChunk]:
        """
        Main method to chunk text semantically
        
        Args:
            text (str): Input text to chunk
            optimize (bool): Whether to optimize chunks after creation
            
        Returns:
            List[TextChunk]: List of semantic chunks
        """
        logger.info(f"Starting semantic chunking of text ({len(text)} characters)")
        
        # Create initial chunks
        chunks = self.create_semantic_chunks(text)
        
        # Optimize chunks if requested
        if optimize:
            chunks = self.optimize_chunks(chunks)
        
        # Store chunks
        self.chunks = chunks
        
        # Log statistics
        if chunks:
            token_counts = [chunk.token_count for chunk in chunks]
            logger.info(f"Chunking complete:")
            logger.info(f"  - Total chunks: {len(chunks)}")
            logger.info(f"  - Average tokens per chunk: {np.mean(token_counts):.1f}")
            logger.info(f"  - Min/Max tokens: {min(token_counts)}/{max(token_counts)}")
        
        return chunks
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the created chunks
        
        Returns:
            Dict[str, Any]: Statistics about chunks
        """
        if not self.chunks:
            return {}
        
        token_counts = [chunk.token_count for chunk in self.chunks]
        char_counts = [len(chunk.content) for chunk in self.chunks]
        
        return {
            'total_chunks': len(self.chunks),
            'total_tokens': sum(token_counts),
            'total_characters': sum(char_counts),
            'avg_tokens_per_chunk': np.mean(token_counts),
            'avg_chars_per_chunk': np.mean(char_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'token_std': np.std(token_counts),
            'sections': list(set(chunk.section_title for chunk in self.chunks))
        }


def main():
    """
    Test the semantic chunker with the processed Attention paper
    """
    # Load processed text
    text_file = "/Users/sam/Desktop/rag/processed_attention_paper.txt"
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"Loaded text: {len(text)} characters")
        
        # Initialize chunker
        chunker = SemanticChunker(
            chunk_size=300,  # Smaller chunks for better semantic coherence
            chunk_overlap=30,
            min_chunk_size=50
        )
        
        # Create chunks
        chunks = chunker.chunk_text(text)
        
        # Print statistics
        stats = chunker.get_chunk_statistics()
        print("\\n=== Semantic Chunking Results ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Show first few chunks
        print("\\n=== First 3 Chunks ===")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\\nChunk {i+1} (ID: {chunk.chunk_id}):")
            print(f"Section: {chunk.section_title}")
            print(f"Tokens: {chunk.token_count}")
            print(f"Content: {chunk.content[:200]}...")
        
        # Save chunks to file
        output_file = "/Users/sam/Desktop/rag/semantic_chunks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(f"=== {chunk.chunk_id} ===\\n")
                f.write(f"Section: {chunk.section_title}\\n")
                f.write(f"Tokens: {chunk.token_count}\\n")
                f.write(f"Content:\\n{chunk.content}\\n\\n")
        
        print(f"\\nChunks saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to process text: {str(e)}")
        raise


if __name__ == "__main__":
    main()
