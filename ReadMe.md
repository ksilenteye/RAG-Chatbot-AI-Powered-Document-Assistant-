# 🧠 RAG Chatbot (AI-Powered Document Assistant)

This project implements a Retrieval-Augmented Generation (RAG) pipeline to create an intelligent chatbot that can answer questions based on a given document. The chatbot uses a powerful combination of sentence transformers for semantic search and a large language model (LLM) for generating human-like responses.

The user interface is built with Streamlit, providing an interactive and user-friendly way to interact with the chatbot.

## 🚀 Features

* **Document Processing**: Extracts text from PDF documents and splits it into manageable chunks.
* **Vector Embeddings**: Converts text chunks into high-dimensional vectors using `sentence-transformers`.
* **Efficient Retrieval**: Uses FAISS (Facebook AI Similarity Search) for fast and efficient retrieval of relevant document chunks.
* **Generative Answering**: Leverages a pre-trained LLM (`microsoft/DialoGPT-medium`) to generate answers based on the retrieved context.
* **Confidence Scoring**: Provides a confidence score for each generated answer.
* **Hallucination Detection**: Includes a mechanism to detect potential hallucinations in the generated responses.
* **Interactive UI**: A Streamlit-based web interface for easy interaction with the chatbot.
* **Customizable Parameters**: Allows users to adjust parameters like the number of chunks to retrieve, similarity threshold, and max tokens for the generated response.

## 📂 Project Structure

├── data/│   └── AI Training Document.pdf├── chunks/│   └── chunks.txt├── vectordb/│   ├── chunks.pkl│   └── index.faiss├── src/│   └── rag_pipeline.py├── app.py├── Preprocess.ipynb├── embed.ipynb└── requirements.txt
* `data/`: Contains the source PDF document.
* `chunks/`: Stores the processed text chunks from the PDF.
* `vectordb/`:  Stores the FAISS index and the pickled chunks.
* `src/rag_pipeline.py`: The core RAG pipeline implementation.
* `app.py`: The Streamlit web application.
* `Preprocess.ipynb`: Jupyter notebook for text extraction and chunking.
* `embed.ipynb`: Jupyter notebook for creating and storing vector embeddings.
* `requirements.txt`: A list of the Python dependencies for the project.

## ⚙️ How It Works

The RAG pipeline works in the following steps:

1.  **Preprocessing**:
    * The text is extracted from the `AI Training Document.pdf`.
    * The extracted text is split into smaller, overlapping chunks to ensure semantic continuity.
    * These chunks are saved to `chunks/chunks.txt`.

2.  **Embedding and Indexing**:
    * The text chunks are loaded.
    * Each chunk is converted into a numerical vector (embedding) using the `all-MiniLM-L6-v2` model.
    * These embeddings are stored in a FAISS index for efficient similarity search. The index and chunks are saved to the `vectordb/` directory.

3.  **Retrieval and Generation (RAG Pipeline)**:
    * When a user asks a question, the query is converted into an embedding.
    * The FAISS index is used to find the most relevant chunks from the document based on semantic similarity.
    * The retrieved chunks and the user's question are fed into the `microsoft/DialoGPT-medium` model.
    * The LLM generates an answer based on the provided context.
    * The answer, along with metadata like confidence scores and retrieved chunks, is displayed to the user.

## 🛠️ Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ Running the Project

1.  **Run the preprocessing and embedding notebooks**:
    * Open and run the `Preprocess.ipynb` notebook to create the text chunks.
    * Open and run the `embed.ipynb` notebook to create the vector database.

2.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

    This will start the web application, and you can interact with the RAG chatbot in your browser.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
