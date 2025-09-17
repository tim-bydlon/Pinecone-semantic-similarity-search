# Smart Q&A Finder

A semantic question similarity search system built with Pinecone's integrated embedding and the Quora Question Pairs dataset. This project demonstrates how to build intelligent Q&A systems using Pinecone's latest integrated inference capabilities for direct text search without manual embedding.

## Overview

The Smart Q&A Finder uses Pinecone's integrated embedding with llama-text-embed-v2 to automatically embed and search user questions against 522,931 Quora questions. Simply type a question and get semantically similar questions instantly - no manual embedding required.

## Features

- **Direct Text Search**: Type questions directly, no embedding needed
- **Integrated Embedding**: Powered by llama-text-embed-v2 (1024 dimensions)
- **Automatic Data Loading**: All 522K+ Quora questions loaded automatically
- **Real Question Text**: Displays actual question content with similarity scores
- **Fast Retrieval**: Sub-second search across entire dataset
- **Pinecone Best Practices**: Uses latest integrated inference capabilities

## Quick Start

### 1. Prerequisites

- Python 3.8+
- Pinecone account with API key

### 2. Installation

```bash
# Clone and setup
git clone <your-repo>
cd pinecone_dev

# Install dependencies
pip install -r requirements.txt

# Set up API key in .env file
echo "PINECONE_API=your_api_key_here" > .env
```

### 3. Run

```bash
python simple_semantic.py
```

That's it! The system will automatically:
- Create an index with integrated embedding
- Load all Quora questions (first run only)
- Start the interactive search interface

## Usage

### Interactive Search

```bash
python simple_semantic.py
```

### Example Session

```
Initializing Simple Semantic Search...
Index 'quora-simple-semantic' already exists.
Index contains 9024 vectors
Simple Semantic Search ready!

Simple Semantic Search
==================================================
Type 'quit' to exit

Your question: How do I learn Python programming?

Searching...

Question: 'How do I learn Python programming?'
============================================================
Found 5 similar questions:

1. Score: 0.5795
   Question: Starting with no programming experience, how long will it take to learn Python 3?

2. Score: 0.5683
   Question: How should I start learning Python?

3. Score: 0.5675
   Question: How should I begin learning Python?

4. Score: 0.5628
   Question: How do I learn Python systematically?

5. Score: 0.4824
   Question: Between Java and Python, which one is better to learn first and why?
```

## How It Works

### Architecture
```
User Question → Pinecone Integrated Embedding → Semantic Search → Similar Questions
```

### Technical Implementation

1. **Index Creation**: Uses `create_index_for_model` with llama-text-embed-v2
2. **Data Loading**: Loads Quora dataset and uses `upsert_records` for text data
3. **Search**: Direct text input via integrated embedding search API
4. **Results**: Returns actual question text with similarity scores

### Key Technical Details

- **Model**: llama-text-embed-v2 (Pinecone hosted)
- **Dimensions**: 1024 (automatically handled)
- **Similarity Metric**: Cosine similarity
- **Index Type**: Pinecone Serverless (AWS us-east-1)
- **Dataset**: 522,931 Quora questions
- **Search Method**: Integrated inference API

## Project Structure

```
pinecone_dev/
├── simple_semantic.py    # 🌟 Main application
├── requirements.txt      # Python dependencies
├── .env                  # API key (create this)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Configuration

### Environment Variables

Create a `.env` file:
```
PINECONE_API=your_pinecone_api_key_here
```

### API Key Setup

1. Sign up at [pinecone.io](https://pinecone.io)
2. Create a project and copy your API key
3. Add it to your `.env` file

## Advanced Usage

### Programmatic Access

```python
from simple_semantic import SimpleSemanticSearch

# Initialize
searcher = SimpleSemanticSearch()

# Search
results = searcher.search_questions("your question here")

# Access results
for hit in results['result']['hits']:
    print(f"Score: {hit['_score']}")
    print(f"Question: {hit['fields']['question_text']}")
```

## Troubleshooting

### Common Issues

1. **"No PINECONE_API environment variable found"**
   - Check your `.env` file has the correct API key
   - Ensure no extra spaces around the `=` sign

2. **"Index is empty, loading Quora data..."**
   - Normal on first run - will load all questions automatically
   - Takes a few minutes but only happens once

3. **SSL Certificate errors**
   - Run: `/Applications/Python\ 3.13/Install\ Certificates.command`
   - Or: `pip install --upgrade certifi`

4. **"Error parsing request: Invalid input: Batch size exceeds 96"**
   - Already handled in code with proper batch sizing

## Production Considerations

This implementation already includes production-ready features:

- ✅ **Integrated Embedding**: No manual model management
- ✅ **Automatic Scaling**: Pinecone handles infrastructure
- ✅ **Full Dataset**: All 522K+ questions loaded
- ✅ **Error Handling**: Proper exception handling and retries
- ✅ **Best Practices**: Following Pinecone's latest API patterns

### Scaling Further

To scale for production use:

1. **Web Interface**: Add Flask/FastAPI REST API
2. **Authentication**: Add user management and API keys
3. **Caching**: Add Redis for frequent queries
4. **Monitoring**: Add logging and metrics
5. **Rate Limiting**: Add request throttling
6. **Answer Integration**: Connect to answer database

## Key Benefits

### vs Manual Embedding
- **No Model Management**: Pinecone handles embedding model
- **Automatic Updates**: Model improvements happen transparently
- **Simplified Code**: Direct text input, no preprocessing needed
- **Better Performance**: Optimized embedding inference

### vs Keyword Search
- **Semantic Understanding**: Finds meaning, not just word matches
- **Query Flexibility**: Works with different phrasings
- **Better Results**: Higher relevance scores
- **Context Aware**: Understands question intent

## Resources

- [Pinecone Integrated Inference](https://www.pinecone.io/blog/integrated-inference/)
- [Pinecone Python SDK](https://docs.pinecone.io/reference/python-sdk)
- [Llama Text Embed v2 Model](https://www.pinecone.io/learn/nvidia-for-pinecone-inference/)
- [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)

## License

This project is for educational and demonstration purposes.