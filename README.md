# Embeddings-Based Code Search Engine

A comprehensive code search engine using sentence transformers and vector similarity search, with fine-tuning capabilities and performance analysis.

## Project Overview

This project implements a semantic code search engine that can find relevant code snippets based on natural language queries. The system uses:

- **Sentence Transformers** for generating embeddings
- **USearch** for efficient vector similarity search
- **FastAPI** for the REST API
- **CoSQA dataset** for training and evaluation
- **Fine-tuning** capabilities for domain-specific optimization

## Project Structure

```
embeddings-based-code-search-engine/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── demo_query.py                      # Simple query demonstration
├── report.ipynb                       # Comprehensive analysis report
├── app/                               # Main application code
│   ├── __init__.py
│   ├── main.py                        # FastAPI server
│   ├── engine.py                      # Core search engine
│   ├── models.py                      # Pydantic data models
│   ├── evaluate.py                    # Evaluation script
│   ├── finetune.py                    # Model fine-tuning script
│   ├── compare_models.py              # Model comparison script
│   ├── bonus_analysis.py              # Bonus analysis (function names & hyperparameters)
│   └── configurable_engine.py         # Configurable engine for hyperparameter testing
├── data/                              # Data directory
│   └── documents.json                 # Sample documents (optional)
└── finetuned_model/                   # Fine-tuned model output (created during training)
```

## Features

### Core Functionality
- **Semantic Code Search**: Find relevant code using natural language queries
- **REST API**: FastAPI-based server with `/search`, `/index`, `/health`, and `/reload` endpoints
- **Vector Storage**: Efficient similarity search using USearch
- **Batch Indexing**: Support for indexing large document collections

### Advanced Features
- **Model Fine-tuning**: Domain-specific training on CoSQA dataset
- **Performance Evaluation**: Comprehensive metrics (Recall@k, MRR@k, NDCG@k)
- **Model Comparison**: Side-by-side evaluation of baseline vs fine-tuned models
- **Function Name Analysis**: Comparison of function names vs full code performance
- **Hyperparameter Optimization**: Vector storage configuration analysis

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd embeddings-based-code-search-engine
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Basic Server Setup

Start the search engine server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The server will be available at `http://127.0.0.1:8000`

### 2. Test the Server

Make a simple query:

```bash
python demo_query.py "how to sort a list" 5
```

Check server health:

```bash
curl http://127.0.0.1:8000/health
```

### 3. Model Fine-tuning

Fine-tune the model on CoSQA dataset:

```bash
python -m app.finetune --output_dir ./finetuned_model --num_epochs 3 --batch_size 16
```

This will:
- Load CoSQA training and validation data
- Fine-tune the sentence transformer model
- Save the model to `./finetuned_model`
- Generate training loss visualization

### 4. Model Evaluation

Evaluate the baseline model:

```bash
python -m app.evaluate --api http://127.0.0.1:8000 --split test --k 10
```

Compare baseline vs fine-tuned models:

```bash
python -m app.compare_models --finetuned_path ./finetuned_model --k 10
```

### 5. Use Fine-tuned Model

Set environment variable and restart server:

```bash
export FINETUNED_MODEL_PATH=./finetuned_model
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Advanced Analysis

Run comprehensive analysis of function names and hyperparameters:

```bash
python -m app.bonus_analysis --finetuned_path ./finetuned_model --k 10
```

This analysis explores:
- **Function Names vs Full Code**: Performance comparison using extracted function names
- **Vector Storage Hyperparameters**: Different similarity metrics and data types
- **Memory vs Performance Trade-offs**: FP16 optimization and connectivity tuning

## API Endpoints

### POST `/search`
Search for relevant code snippets.

**Request:**
```json
{
  "query": "how to sort a list",
  "k": 5
}
```

**Response:**
```json
{
  "query": "how to sort a list",
  "results": [
    {
      "id": "doc_1",
      "text": "def sort_list(items): return sorted(items)",
      "score": 0.95,
      "metadata": null
    }
  ]
}
```

### POST `/index`
Index documents for search.

**Request:**
```json
{
  "documents": [
    {
      "id": "doc_1",
      "text": "def sort_list(items): return sorted(items)",
      "metadata": null
    }
  ]
}
```

### GET `/health`
Check server status.

**Response:**
```json
{
  "status": "ok"
}
```

### POST `/reload`
Reload the search engine (useful after model changes).

**Response:**
```json
{
  "status": "reloaded",
  "count": 1000
}
```

## Configuration

### Environment Variables

- `EMBEDDING_MODEL`: Base model name (default: "sentence-transformers/all-MiniLM-L6-v2")
- `FINETUNED_MODEL_PATH`: Path to fine-tuned model
- `DOCS_PATH`: Path to documents file
- `API_URL`: API base URL for evaluation scripts

### Model Configuration

The search engine supports various configurations:

- **Base Models**: Any sentence transformer model
- **Fine-tuned Models**: Custom models trained on code datasets
- **Vector Storage**: Configurable similarity metrics and data types

## Evaluation Metrics

The system uses standard information retrieval metrics:

- **Recall@k**: Fraction of relevant documents retrieved in top-k
- **MRR@k**: Mean Reciprocal Rank of first relevant document
- **NDCG@k**: Normalized Discounted Cumulative Gain

## Fine-tuning Details

### Loss Function: CosineSimilarityLoss

**Why CosineSimilarityLoss?**
- Optimal for semantic similarity tasks
- Works well with normalized embeddings
- Naturally handles positive/negative pairs
- Designed for retrieval and ranking tasks

### Training Process
- **Dataset**: CoSQA training split
- **Epochs**: 3 (sufficient for improvement demonstration)
- **Batch Size**: 16 (balanced memory/performance)
- **Learning Rate**: 2e-5 (standard for transformer fine-tuning)
- **Evaluation**: Regular validation during training

## Advanced Analysis

### Function Name Extraction
The system can extract function names from code using multi-language regex patterns:
- Python: `def function_name(`
- JavaScript: `function functionName(`, `functionName = function`
- Java/C#: `public/private/protected/static returnType functionName(`
- Arrow functions: `functionName = (params) =>`

### Vector Storage Optimization
The analysis tests various hyperparameter configurations:
- **Default**: Cosine similarity, FP32, connectivity=16
- **FP16**: Memory optimization with minimal accuracy loss
- **High Connectivity**: Better recall at cost of search speed
- **Low Connectivity**: Faster search with potential recall loss
- **Alternative Metrics**: L2 distance and inner product comparisons

### Performance Trade-offs
- **Function Names**: Higher precision but lower recall
- **FP16**: ~50% memory reduction with minimal performance impact
- **Connectivity**: Sweet spot between 8-32 depending on dataset size
- **Metrics**: Cosine similarity optimal for normalized embeddings

## Troubleshooting

### Common Issues

1. **ImportError: accelerate>=0.26.0**
   ```bash
   pip install accelerate>=0.26.0
   ```

2. **CUDA out of memory**
   - Reduce batch size in fine-tuning
   - Use CPU-only mode (default)

3. **404 Not Found on /index**
   - Ensure server is running
   - Check API base URL in evaluation scripts

4. **Model loading errors**
   - Verify model path exists
   - Check environment variables

### Performance Tips

- Use FP16 for memory optimization
- Adjust connectivity for speed/accuracy balance
- Fine-tune for domain-specific improvements
- Consider function names for focused search

## Development

### Code Structure
- **Modular Design**: Separate concerns (engine, API, evaluation)
- **Type Hints**: Full type annotation for better IDE support
- **Error Handling**: Comprehensive error handling and validation
- **Documentation**: Detailed docstrings and comments

### Testing
- All scripts include error handling
- Evaluation scripts validate API responses
- Fine-tuning includes progress monitoring