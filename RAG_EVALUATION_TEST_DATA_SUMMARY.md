# 🔍 RAG Evaluation Test Data - REL (Retrieval Evaluation) Suite

## 📋 **Overview**

I've created comprehensive REL (Retrieval Evaluation) test data for your RAG system based on the "Attention Is All You Need" paper. This test suite follows the standard format for evaluating retrieval-augmented generation systems and includes passages (contexts) and question-answer sets with proper evaluation metadata.

## 📁 **Test Files Created**

### 1. **Passages (Contexts)** - `attention_paper_passages.jsonl`
- **Purpose**: Contains 40 carefully selected passages from the attention paper
- **Structure**: Each passage has ID, text content, and metadata
- **Coverage**: All major sections of the paper
- **Categories**:
  - Architecture (transformer structure)
  - Attention mechanisms (self-attention, multi-head)
  - Mathematical formulas
  - Encoder/decoder components
  - Positional encoding
  - Training and results
  - Model details

### 2. **Basic QA Set** - `attention_paper_qaset.jsonl`
- **Purpose**: 30 fundamental questions covering core concepts
- **Difficulty**: Easy to medium
- **Features**:
  - Required vs. optional questions
  - Robustness testing (paraphrases, synonyms)
  - Prompt robustness (different reasoning modes)
  - Context references to specific passages

### 3. **Challenging QA Set** - `attention_paper_challenging_qaset.jsonl`
- **Purpose**: 15 advanced questions requiring deep understanding
- **Difficulty**: Medium to hard
- **Focus Areas**:
  - Computational complexity analysis
  - Mathematical intuitions
  - Implementation details
  - Comparative analysis
  - Advanced concepts

### 4. **Multi-Hop QA Set** - `attention_paper_multi_hop_qaset.jsonl`
- **Purpose**: 15 complex questions requiring multiple passages
- **Difficulty**: Hard
- **Features**:
  - Cross-section reasoning
  - Information synthesis
  - Process tracing
  - Relationship analysis

## 🎯 **Key Features**

### **Comprehensive Coverage**
```json
{
  "categories": {
    "architecture": "Transformer structure and components",
    "attention_mechanism": "Core attention concepts",
    "multi_head": "Multi-head attention specifics", 
    "encoder": "Encoder architecture and function",
    "decoder": "Decoder architecture and masking",
    "positional_encoding": "Position representation",
    "formulas": "Mathematical expressions",
    "training": "Datasets and training process",
    "results": "Performance metrics and BLEU scores",
    "interpretability": "Model understanding and analysis"
  }
}
```

### **Evaluation Robustness**
- **Paraphrases**: Alternative question formulations
- **Synonyms**: Different terminology for same concepts
- **Prompt Modes**: Simple, Chain-of-Thought (CoT), Scaffolded reasoning
- **Hybrid Requirements**: Multi-modal retrieval testing
- **MMR Lambda**: Diversity parameters for retrieval

### **Difficulty Progression**
1. **Easy (40%)**: Basic concept identification
2. **Medium (35%)**: Concept explanation and relationships  
3. **Hard (25%)**: Deep analysis and multi-hop reasoning

## 📊 **Test Data Statistics**

### **Passages**
- **Total Passages**: 40 contextual chunks
- **Average Length**: ~150 words per passage
- **Section Coverage**: Abstract, Introduction, Architecture, Attention, Experiments
- **Metadata Fields**: Source, category, section, page number

### **Questions**
- **Total Questions**: 60 across all sets
- **Basic QA**: 30 questions (foundation concepts)
- **Challenging QA**: 15 questions (advanced concepts)
- **Multi-Hop QA**: 15 questions (complex reasoning)

### **Evaluation Features**
- **Required Questions**: 45 (75%) - Must pass for system validation
- **Optional Questions**: 15 (25%) - Additional evaluation depth
- **Robustness Tests**: 25 questions with paraphrase variations
- **Prompt Robustness**: 15 questions with multiple reasoning modes

## 🔧 **Passage Examples**

### **Architecture Passage**
```json
{
  "id": "att_p1",
  "text": "The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.",
  "meta": {
    "source": "attention_paper",
    "category": "architecture", 
    "section": "abstract",
    "page": 1
  }
}
```

### **Formula Passage**
```json
{
  "id": "att_p10",
  "text": "Attention(Q, K, V) = softmax(QK^T / √dk)V",
  "meta": {
    "source": "attention_paper",
    "category": "formula",
    "section": "attention",
    "page": 3
  }
}
```

## 🧪 **Question Examples**

### **Basic Question**
```json
{
  "qid": "att_q1",
  "question": "What is the Transformer?",
  "answer": "The Transformer is a model architecture that relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
  "contexts": ["att_p1", "att_p2"],
  "meta": {"category": "architecture", "difficulty": "easy"},
  "task_type": "rag_qa",
  "required": true
}
```

### **Challenging Question with Robustness**
```json
{
  "qid": "att_hard_q2",
  "question": "Why does the paper divide by √dk in the attention formula?",
  "answer": "Dividing by √dk prevents the dot products from growing large in magnitude, which would push the softmax function into regions with extremely small gradients.",
  "contexts": ["att_p8", "att_p7"],
  "meta": {"category": "scaled_dot_product", "difficulty": "hard"},
  "task_type": "rag_qa",
  "required": true,
  "prompt_robustness": {
    "enabled": true,
    "modes": ["cot", "scaffold"],
    "paraphrase_runs": 2
  }
}
```

### **Multi-Hop Question**
```json
{
  "qid": "att_multi_q1",
  "question": "How do the encoder and decoder work together through attention to perform translation?",
  "answer": "The encoder processes the input sequence using self-attention to create representations, then the decoder uses encoder-decoder attention to access these representations while generating the output sequence using masked self-attention.",
  "contexts": ["att_p17", "att_p19", "att_p29", "att_p30"],
  "meta": {"category": "architecture", "difficulty": "hard"},
  "task_type": "rag_qa",
  "required": true,
  "robustness": {
    "paraphrases": ["Explain encoder-decoder interaction in Transformer", "How does Transformer perform translation?"],
    "require_hybrid": true,
    "mmr_lambda": 0.3
  }
}
```

## 📈 **Evaluation Metrics**

### **Retrieval Evaluation**
- **Passage Recall**: How many relevant passages are retrieved
- **Passage Precision**: How many retrieved passages are relevant
- **Context Overlap**: Overlap between retrieved and gold contexts
- **Ranking Quality**: Position of relevant passages in results

### **Generation Evaluation**
- **Answer Accuracy**: Semantic similarity to gold answers
- **Context Utilization**: How well the model uses retrieved passages
- **Factual Correctness**: Accuracy of technical details
- **Completeness**: Coverage of all answer components

### **Robustness Evaluation**
- **Paraphrase Consistency**: Same answers for rephrased questions
- **Prompt Sensitivity**: Performance across different reasoning modes
- **Synonym Handling**: Consistent responses to terminology variations
- **Multi-Hop Reasoning**: Ability to synthesize information across passages

## 🎯 **Test Categories by Difficulty**

### **Easy Questions (Foundation)**
- Basic definitions and concepts
- Simple factual retrieval
- Single-passage answers
- Clear, unambiguous questions

**Examples**:
- "What is the Transformer?"
- "How many attention heads does the Transformer use?"
- "What datasets were used for training?"

### **Medium Questions (Understanding)**
- Concept explanations and relationships
- Process descriptions
- Comparative analysis
- Mathematical understanding

**Examples**:
- "Why is multi-head attention beneficial?"
- "How does self-attention work in the encoder?"
- "What is the formula for Multi-Head Attention?"

### **Hard Questions (Analysis)**
- Deep mathematical intuitions
- Complex process tracing
- Multi-concept integration
- Implementation details

**Examples**:
- "Why does the paper divide by √dk in the attention formula?"
- "How does the masking mechanism preserve auto-regressive properties?"
- "What is the complete mathematical flow from input to attention weights?"

## 🔍 **Retrieval Challenges**

### **Single-Hop Retrieval**
- Direct concept lookup
- Formula retrieval
- Factual information extraction

### **Multi-Hop Retrieval**
- Cross-section reasoning
- Process understanding
- Relationship analysis
- Information synthesis

### **Challenging Retrieval**
- Implicit information
- Mathematical reasoning
- Comparative analysis
- Implementation details

## 🚀 **Usage Instructions**

### **1. Load Test Data**
```bash
# Load passages (contexts)
passages = load_jsonl("attention_paper_passages.jsonl")

# Load question sets
basic_qa = load_jsonl("attention_paper_qaset.jsonl")
challenging_qa = load_jsonl("attention_paper_challenging_qaset.jsonl")
multi_hop_qa = load_jsonl("attention_paper_multi_hop_qaset.jsonl")
```

### **2. Run Retrieval Evaluation**
```python
for question in qa_set:
    # Retrieve relevant passages
    retrieved_passages = rag_system.retrieve(question["question"])
    
    # Evaluate retrieval quality
    recall = calculate_recall(retrieved_passages, question["contexts"])
    precision = calculate_precision(retrieved_passages, question["contexts"])
```

### **3. Run Generation Evaluation**
```python
for question in qa_set:
    # Generate answer
    generated_answer = rag_system.generate(question["question"])
    
    # Evaluate answer quality
    accuracy = evaluate_answer(generated_answer, question["answer"])
    context_usage = evaluate_context_usage(generated_answer, retrieved_passages)
```

### **4. Robustness Testing**
```python
for question in qa_set:
    if "robustness" in question:
        # Test paraphrases
        for paraphrase in question["robustness"]["paraphrases"]:
            test_consistency(original_answer, paraphrase_answer)
        
        # Test prompt modes
        if "prompt_robustness" in question:
            test_prompt_sensitivity(question, modes)
```

## 📊 **Expected Performance Benchmarks**

### **Retrieval Metrics**
- **Passage Recall@5**: >85% for basic, >70% for challenging, >60% for multi-hop
- **Passage Precision@5**: >60% for basic, >45% for challenging, >35% for multi-hop
- **Context Overlap**: >70% for basic, >55% for challenging, >45% for multi-hop

### **Generation Metrics**
- **Answer Accuracy**: >90% for basic, >75% for challenging, >65% for multi-hop
- **Factual Correctness**: >95% for basic, >85% for challenging, >75% for multi-hop
- **Completeness**: >85% for basic, >70% for challenging, >60% for multi-hop

### **Robustness Metrics**
- **Paraphrase Consistency**: >80% semantic similarity
- **Prompt Mode Consistency**: <20% variance across modes
- **Synonym Handling**: >85% consistent responses

## 🎓 **Educational Value**

This test suite serves multiple purposes:

1. **System Evaluation**: Comprehensive assessment of RAG capabilities
2. **Benchmark Creation**: Standard evaluation for attention paper understanding
3. **Research Tool**: Analysis of retrieval and generation quality
4. **Educational Resource**: Learning about transformer architecture through Q&A

The test data is designed to challenge your RAG system across all dimensions of understanding, from basic concept retrieval to complex multi-hop reasoning about the attention mechanism and transformer architecture! 🎯
