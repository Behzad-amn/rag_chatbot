from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
import os


class RagBot:
    def __init__(
        self,
        api_key: str,
        document_path: str = "documents",
        stored_embedding_path: str = "stored_embedding",
        openai_model: str = "gpt-4o-mini",
        chunk_size: int = 300,
        chunk_overlap: int = 100
    ):
        """
        Initialize the CoreAIModel with API key and paths for documents and embeddings.
        """
        self.api_key = api_key
        self.document_path = document_path
        self.stored_embedding_path = stored_embedding_path
        self.openai_model = openai_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_database(self) -> None:
        """
        Load, split, and save documents into Chroma database.
        """
        documents = self._load_documents()
        split_documents = self._split_documents(documents)
        self._save_to_database(split_documents)

    def update_database(self) -> None:
        """
        Update the Chroma database if new documents are detected.
        """
        document_filenames = self._document_filenames()
        embedded_filenames = self._embedded_filenames()

        if document_filenames != embedded_filenames:
            print("New files detected. Updating database.")
            for file in document_filenames:
                if file not in embedded_filenames:
                    documents = self._load_documents(file)
                    split_documents = self._split_documents(documents)
                    self._save_to_database(split_documents)
        else:
            print("No update required.")

    def invoke(self, prompt: str, stream_output: bool = False) -> str:
        """
        Generate a response based on the provided prompt.
        """
        rag_chain = (
            {"context": self._retriever() | self._format_documents,
             "question": RunnablePassthrough()}
            | self._get_prompt_template()
            | self._initialize_llm()
            | StrOutputParser()
        )

        if stream_output:
            response = "\n"
            print("\n---------\nAnswer:\n")
            for chunk in rag_chain.stream(prompt):
                response += chunk
                print(chunk, end="", flush=True)
            print("\n---------")
        else:
            response = "\n" + rag_chain.invoke(prompt)

        return response

    # Private helper methods

    def _load_documents(self, glob: str = "**/[!.]*") -> List[Document]:
        """
        Load documents from the specified directory using a glob pattern.
        """
        loader = DirectoryLoader(
            path=self.document_path, show_progress=True, glob=glob)
        return loader.lazy_load()

    def _document_filenames(self) -> List[str]:
        """
        Get a sorted list of document filenames in the specified directory.
        """
        return sorted(f for f in os.listdir(self.document_path) if os.path.isfile(os.path.join(self.document_path, f)))

    def _embedded_filenames(self) -> List[str]:
        """
        Get a list of filenames already embedded in Chroma.
        """
        try:
            vectorstore = Chroma(
                persist_directory=self.stored_embedding_path,
                embedding_function=OpenAIEmbeddings(api_key=self.api_key)
            )
            return sorted({item['source'].replace(self.document_path + "\\", '') for item in vectorstore.get()['metadatas']})
        except Exception:
            return []

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        return splitter.split_documents(documents)

    def _save_to_database(self, documents: List[Document]) -> None:
        """
        Save document chunks to Chroma with OpenAI embeddings.
        """
        Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(api_key=self.api_key),
            persist_directory=self.stored_embedding_path
        )

    def _retriever(self):
        """
        Retrieve documents from the Chroma database.
        """
        if not os.path.exists(self.stored_embedding_path):
            print("Embedding database not found. Creating database...")
            self.create_database()

        try:
            vectorstore = Chroma(
                persist_directory=self.stored_embedding_path,
                embedding_function=OpenAIEmbeddings(api_key=self.api_key)
            )
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"

    def _initialize_llm(self):
        """
        Initialize the language model using OpenAI.
        """
        return ChatOpenAI(model=self.openai_model, api_key=self.api_key)

    def _get_prompt_template(self) -> ChatPromptTemplate:
        """
        Generate a prompt template for the language model.
        """
        system_prompt = """
            You are an assistant for answering questions. Use the retrieved context to answer the question.
            If you don't know the answer, say that. Limit your response to three sentences.
            Include the source document for your answer.
            Question: {question}
            Context: {context}
        """
        return ChatPromptTemplate.from_messages([("system", system_prompt)])

    @staticmethod
    def _format_documents(docs: List[Document]) -> str:
        """
        Format documents for output display.
        """
        return "\n\n".join(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n" for doc in docs)


