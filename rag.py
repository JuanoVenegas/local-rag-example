import os
import sys
import shutil
from pprint import pprint
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import (
    FastEmbedEmbeddings,
    FakeEmbeddings,
    OllamaEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL = [
    "mistral-nemo",
    "llama3.1",
][1]

SYSTEM_PROMPT = {
    "en": (
        "<s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context"
        " to answer the question. If you don't know the answer, just say that you don't know. Use three sentences"
        " maximum and keep the answer concise. [/INST] </s>"
        "\n[INST] Question: {question}"
        "\nContext: {context}"
        "\nAnswer: [/INST]"
    ),
    "es": (
        "<s> [INST] Eres un asistente para tareas de pregunta-respuesta. Usa los siguientes fragmentos de contexto"
        " recuperados para responder la pregunta. Si no sabes la respuesta, simplemente di que no lo sabes. Usa tres"
        " oraciones como máximo y mantén la respuesta concisa. [/INST] </s>"
        "\n[INST] Pregunta: {question}"
        "\nContexto: {context}"
        "\nRespuesta: [/INST]"
    ),
}["es"]


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model=MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = PromptTemplate.from_template(
            SYSTEM_PROMPT,
        )

    def ingest(self, file_path: str, doc_type: str):
        print("Ingesting file:", file_path)
        # Check if it's a pdf file or a text file
        if doc_type.lower() == "pdf":
            docs = PyPDFLoader(file_path=file_path).load()
        else:
            docs = TextLoader(file_path=file_path).load()
        print("  * Documents loaded:", len(docs))

        print("* Splitting documents")
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        pprint(
            [
                f"{i}: {sz} chars"
                for i, sz in enumerate(map(len, [doc.page_content for doc in chunks]))
            ]
        )
        print("  * Chunks created:", len(chunks))

        print("* Creating vector store")
        # Delete previous vector store if it exists (otherwise it fails)
        persist_dir = "./db"
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            print("  * Removed previous vector store")
        ##
        vector_store = Chroma.from_documents(
            documents=chunks,
            persist_directory=persist_dir,  # It fails without this !
            embedding=FastEmbedEmbeddings(),
            # embedding=FakeEmbeddings(size=512)
            # embedding=OllamaEmbeddings(model="nomic-embed-text"),
        )
        print("  * Vector store created")

        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        print("* Retriever created")

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        print("* Chain created")

    def ask(self, query: str):
        if not self.chain:
            return "Por favor, agrega un documento PDF primero."
        print("Query:", query)
        ans = self.chain.invoke(query)
        print("Answer:", ans)
        return ans

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
