# Smart Q&A Finder

An interactive semantic question similarity search system built with Pinecone vector database, sentence transformers, and the Quora Question Pairs dataset. This project demonstrates how to build intelligent Q&A systems that can find semantically similar questions even when they use different wording.

## Overview

The Smart Q&A Finder embeds user questions in real-time using the all-MiniLM-L6-v2 model and searches against 522,931 pre-embedded Quora questions. Instead of keyword matching, it understands the meaning and context of questions to find the most relevant matches with actual question text and similarity scores.

## Features

- **🔍 Interactive Search**: Type any question and get instant semantic matches
- **🧠 Real-time Embedding**: User questions embedded with all-MiniLM-L6-v2 model
- **📝 Actual Question Text**: Displays real question content from Quora dataset
- **⚡ Fast Retrieval**: Sub-second search across 522K+ pre-embedded questions
- **📊 Similarity Scores**: Precise cosine similarity scores for each match
- **💾 Vector Database**: Powered by Pinecone's scalable infrastructure

## Project Structure

```
pinecone_dev/
├── README.md                 # This file
├── main.py                  # Setup script - creates index and loads data
├── interactive_qa_finder.py # 🌟 Main application - Interactive Q&A Finder
├── simple_qa_finder.py      # Basic demo version
├── .env                     # API key storage (secure)
├── .gitignore              # Git ignore rules
└── requirements.txt        # Python dependencies
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- Pinecone account with API key

### 2. Installation

1. **Clone/download this project**

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv_pc
   source venv_pc/bin/activate  # On Windows: venv_pc\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configuration

1. **Get your Pinecone API key**:
   - Sign up at [pinecone.io](https://pinecone.io)
   - Create a project and copy your API key

2. **Set up your API key**:
   - Open `.env` file
   - Replace `your_api_key_here` with your actual Pinecone API key:
   ```
   PINECONE_API=your_actual_api_key_here
   ```

### 4. Initial Setup (First Time Only)

Run the setup script to create your index and load data:

```bash
python main.py
```

This will:
- Create a Pinecone index called "quora-questions"
- Load and upsert 1,000 sample questions from the Quora dataset
- Display dataset information and confirm successful setup

**Note**: This only needs to be run once. The index will persist in Pinecone.

## Usage

### Running the Interactive Q&A Finder

Execute the main interactive application:

```bash
python interactive_qa_finder.py
```

Choose between:
1. **Interactive mode**: Type your own questions
2. **Demo mode**: See predefined examples

### Example Output

```
🔍 Interactive Smart Q&A Finder
==================================================
Ask any question and find semantically similar questions from Quora's dataset!
Type 'quit' to exit

Your question: How do I learn Python programming?

Searching...

Question: 'How do I learn Python programming?'
============================================================
Found 5 similar questions:

1. Score: 0.5491
   Question: How do I learn a computer language like java?

2. Score: 0.5273
   Question: What's the best way to start learning robotics?

3. Score: 0.5149
   Question: What is Java programming? How To Learn Java Programming Language ?

4. Score: 0.4817
   Question: What math does a complete newbie need to understand algorithms for computer programming?

5. Score: 0.4523
   Question: How can I learn computer security?
```

### Running the Basic Demo

For a simple demonstration without interaction:

```bash
python simple_qa_finder.py
```

## How It Works

### 1. Data Pipeline
- **Dataset**: Uses Quora Question Pairs dataset (`quora_all-MiniLM-L6-bm25`)
- **Embeddings**: Pre-computed using MiniLM sentence transformer (384 dimensions)
- **Storage**: Vectors stored in Pinecone serverless index

### 2. Search Process
1. User types question in interactive interface
2. Question embedded using all-MiniLM-L6-v2 model (384 dimensions)
3. Vector similarity search finds closest matches using cosine distance
4. Results ranked by similarity score (higher = more similar)
5. Returns actual question text from blob field with similarity scores

### 3. Architecture
```
User Question → [Embedding Model] → Query Vector → Pinecone Index → Similar Questions
```

## Technical Details

- **Vector Dimensions**: 384 (MiniLM-L6 model)
- **Similarity Metric**: Cosine similarity
- **Index Type**: Pinecone Serverless (AWS us-east-1)
- **Dataset Size**: 522,931 total questions (1,000 loaded in demo)
- **Response Time**: Sub-second query performance

## Production Considerations

This project already implements core production features:
- ✅ **Real-time Text Embedding**: Integrated all-MiniLM-L6-v2 model
- ✅ **Question Text Access**: Implemented blob field access for actual content
- ✅ **Interactive Interface**: Built command-line interactive demo

To scale further, you could:

1. **Answer Retrieval**: Connect to answer database to return actual responses
2. **Scale Data**: Load the complete 522K dataset instead of 1K sample
3. **Add Metadata**: Include categories, timestamps, vote counts for filtering
4. **Web Interface**: Build Flask/FastAPI web app
5. **Caching**: Add response caching for frequent queries
6. **Authentication**: Add user authentication and query logging

## Files Reference

### `interactive_qa_finder.py` ⭐
- **Purpose**: Main interactive Q&A application
- **Function**: Real-time question embedding and semantic search
- **When to run**: Primary application for asking questions

### `main.py`
- **Purpose**: One-time setup script
- **Function**: Creates index, loads data
- **When to run**: Only when setting up project or recreating index

### `simple_qa_finder.py`
- **Purpose**: Basic demonstration version
- **Function**: Shows working similarity search without embeddings
- **When to run**: Quick demo without interaction

### `.env`
- **Purpose**: Secure API key storage
- **Security**: Already in .gitignore to prevent accidental commits
- **Format**: `PINECONE_API=your_key_here`

## Troubleshooting

### Common Issues

1. **"No PINECONE_API environment variable found"**
   - Check your `.env` file has the correct API key
   - Ensure no extra spaces around the `=` sign

2. **"Unable to prepare type ndarray for serialization"**
   - This is handled in the code with vector conversion
   - If it occurs, ensure vectors are converted to Python lists

3. **SSL Certificate errors**
   - Run: `/Applications/Python\ 3.13/Install\ Certificates.command`
   - Or update certificates: `pip install --upgrade certifi`

4. **Index already exists**
   - Normal behavior - the index persists in Pinecone
   - Delete from Pinecone console if you want to recreate

## Next Steps

Future enhancements for this project:

1. **Answer Integration**: Connect to Quora answer database for complete Q&A
2. **Web Interface**: Build Flask/FastAPI web app with nice UI
3. **Advanced Filtering**: Add metadata-based filtering (categories, dates)
4. **Batch Processing**: Load complete 522K dataset for full coverage
5. **Caching Layer**: Redis cache for frequently asked questions
6. **User Analytics**: Track popular questions and search patterns

## Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Pinecone Python SDK](https://docs.pinecone.io/reference/python-sdk)
- [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)
- [MiniLM Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## License

This project is for educational and demonstration purposes.