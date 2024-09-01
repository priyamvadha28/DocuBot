# DocuBot
This project, DocuBot is an advanced document-based Q&A system developed by Priyamvadha Pradeep. It enables users to upload various types of documents and interact with their content through natural language questions. Leveraging the power of Azure OpenAI and LangChain, this tool provides accurate and context-aware responses, making it an essential tool for document analysis and information retrieval.

Features:
- Multi-format Document Support: DocuBot supports multiple document types including PDF, Word (.docx), PowerPoint (.pptx), and Excel (.xlsx), making it versatile for different use cases.
- Streamlit Interface: A user-friendly interface built with Streamlit that allows seamless interaction with uploaded documents.
- Azure OpenAI Integration: Utilizes Azure OpenAI’s language models to understand and generate human-like responses based on the content of the uploaded documents.
- LangChain for Text Processing: Implements LangChain to split, process, and retrieve relevant information from the documents. This ensures that responses are contextually accurate and informative.
- Session-based Chat History: The application stores the chat history for each session, allowing users to track their questions and the responses provided by DocuBot.

How It Works: 
- Upload Document: Users can upload a document of their choice (PDF, Word, PowerPoint, or Excel) via the Streamlit sidebar.
- Text Extraction: The application extracts text from the uploaded document using the appropriate Python libraries (PyPDF2, docx, pptx, pandas).
- Text Chunking: The extracted text is split into manageable chunks using LangChain’s RecursiveCharacterTextSplitter. This allows for efficient processing and retrieval of relevant content.
- Embeddings and Vector Store: The text chunks are then converted into embeddings using Azure OpenAI Embeddings and stored in a FAISS vector store. This enables fast similarity searches when answering user queries.
- Ask Questions: Users can input their questions, and DocuBot will search through the vector store for relevant information, generate a response, and display it in the chat interface.
- Chat History: All questions and responses are saved in the session state, allowing users to review their interactions.

Getting Started: 
Prerequisites:
- Python 3.8 or higher
- Streamlit
- Azure OpenAI API Key
- Python libraries: PyPDF2, docx, pptx, pandas, langchain, faiss, langchain_community

Installation
1. Clone the repository:
git clone https://github.com/yourusername/docubot.git
cd docubot

2. Install the required Python libraries:
pip install -r requirements.txt

3. Set up your Azure OpenAI credentials and configure them in the Chat_Model dictionary in the code.

4. Run the application:
streamlit run app.py

5. Upload a document, type your question, and start interacting with DocuBot!

Usage:
Once the application is running, you can:

- Upload Documents: Use the sidebar to upload your document.
- Ask Questions: Type your questions in the provided input field and get responses based on the document content.
- Review History: Expand the chat history to review previous interactions.

Author:
This project is developed and maintained by Priyamvadha Pradeep.
