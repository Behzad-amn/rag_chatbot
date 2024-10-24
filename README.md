# RagBot: Generative AI Chatbot with Retrieval-Augmented Generation (RAG)

## Overview

This project implements a Generative AI chatbot using Langchain components and OpenAI's GPT models. It enhances the chatbot's capabilities by employing Retrieval-Augmented Generation (RAG), where documents are loaded, split, embedded, and stored for retrieval to answer questions based on context from the documents.

The bot is designed to load documents, break them into smaller chunks, save embeddings in Chroma, and retrieve the most relevant chunks to answer questions.

## Features

- **Document Loading and Splitting**: Load documents from a specified folder, split them into chunks, and store them for future queries.
- **Embeddings Storage**: Save document embeddings in the Chroma vector database for quick retrieval based on similarity search.
- **Context-Aware Responses**: The bot retrieves relevant document sections to answer user queries accurately using GPT-4 models.
- **Document Updating**: The bot can detect and add new documents into the database when updates occur.
- **Stream Output**: Option to stream chatbot responses as they are generated.

## Files

### 1. `Chatbot.py`

This script initializes and invokes the `RagBot` class. You need to replace the OpenAI API key and the prompt to start querying the bot.

**Usage**:
```python
from RagBot import RagBot

API_KEY = "Replace with own OpenAI API_KEY"
prompt = "Replace with own prompt"
agent = RagBot(api_key=API_KEY)

Response = agent.invoke(prompt) 
print(Response)
```


## 2. RagBot.py

This is the core component of the chatbot. It handles:

- Document loading from a directory.
- Splitting documents into manageable chunks.
- Storing and retrieving document embeddings using Chroma.
- Generating responses by leveraging OpenAI's language model with the context provided by the document retrieval.

### Methods:

- `create_database()`: Loads and stores the document embeddings into a Chroma vector database.
- `update_database()`: Checks for new documents and updates the database if necessary.
- `invoke(prompt: str)`: Main method for querying the bot, where the prompt is processed, and relevant document chunks are retrieved and used to generate a response.

### Getting Started

#### Prerequisites:

- Python 3.8+
- OpenAI API Key

#### Required Python libraries:

- `langchain_community`
- `langchain_openai`
- `langchain_chroma`

You can install the required libraries with:

```bash
pip install langchain_community langchain_openai langchain_chroma
```

## Installation

### Clone the repository:

```bash
git clone https://github.com/your-repo/ragbot.git
cd ragbot
```

### Set up your OpenAI API key:
Replace the `API_KEY` in `Chatbot.py` with your OpenAI API key.

### Prepare the document folder:
Place your documents in the `documents` directory. The bot will automatically load and process these files.

### Run the chatbot:

```bash
python Chatbot.py

```

## Example Usage

To generate a response based on a user prompt:

```python
prompt = "What is the process for AI model training?"
response = agent.invoke(prompt)
print(response)
```

## Customization

You can adjust the following parameters in `RagBot.py` for your use case:

- **Document Path**: Change the `document_path` in `RagBot.py` to point to your specific document directory.
- **Chunk Size and Overlap**: Modify `chunk_size` and `chunk_overlap` to change how documents are split into chunks.
- **OpenAI Model**: Adjust `openai_model` to select different GPT models (e.g., GPT-4).

## Contributing

Feel free to fork this project and contribute improvements. Pull requests are welcome!

## License

This project is licensed under the MIT License.

