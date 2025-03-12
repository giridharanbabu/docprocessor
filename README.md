# Document Processing Application

## Installation & Setup

### Clone the Repository
```sh
git clone <repo-url>
cd document-processor
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Start the Ollama Model
```sh
ollama serve --model llama3
```

### Run the Streamlit Application
```sh
streamlit run app.py
```

## Features
- 📄 Upload PDF, TXT, CSV files
- 📝 Summarize documents
- 🔍 Ask questions based on uploaded documents
- 📊 Extract structured data
- 🎤 Search using voice commands

## Usage
1. **Upload** one or more documents.
2. **Select** a processing option:
   - Generate a **summary**
   - Ask a **question**
   - Extract **structured data**
   - Perform **voice-based search**
3. **View results** in the UI.

## Application Flow

### File Upload & Processing
- Users upload multiple files (PDF, TXT, CSV).
- Extracted text is stored in `st.session_state.raw_documents`.

### Text Summarization
- Uses **Ollama LLM** for summarization.
- Summary is displayed in the UI.

### Conversational Q&A
- **FAISS Vector Store** stores embeddings for document retrieval.
- Retrieval-based Q&A is handled by `RetrievalQA`.
- Displays responses along with conversation history.

### Structured Data Extraction
- User provides keys for structured extraction.
- The model extracts and displays structured data in JSON format.

### Audio-Based Search
- Speech-to-text conversion using an external `speechtotext` module.
- The converted text is used for document search.

## Architecture Design

### Components
1. **Frontend (Streamlit)** - Handles UI interactions.
2. **LLM Engine (Ollama)** - Summarization, QA, and structured extraction.
3. **Vector Store (FAISS)** - Stores document embeddings.
4. **Speech-to-Text (External Module)** - Converts audio to text.
5. **Session Storage** - Stores extracted text, summaries, and history.

### High-Level Architecture Diagram
```
+------------------+
| Streamlit UI    |
+------------------+
        |
        v
+--------------------+
| File Processing   |
| (PDF, TXT, CSV)  |
+--------------------+
        |
        v
+--------------------+
| Vector Store (FAISS) |
+--------------------+
        |
        v
+------------------------+
| Ollama LLM (QA, Summarization, Extraction) |
+------------------------+
        |
        v
+----------------------+
| Response Generation |
+----------------------+
```

---

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
