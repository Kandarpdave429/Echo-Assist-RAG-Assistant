from qdrant_client import QdrantClient
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import QueryBundle
import os
from dotenv import load_dotenv
load_dotenv(override=True)

import warnings
warnings.filterwarnings("ignore")


class AIVoiceAssistant:
    """
    RAG assistant: Qdrant + HuggingFace embeddings + Groq LLM via llama-index.
    interact_with_llm(...) ALWAYS returns {"answer": str, "retrieved_chunks": [str,...]}
    """

    def __init__(self, qdrant_url=None, collection_name="manual_db"):
        self._qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)

        # Groq LLM (ensure GROQ_API_KEY in env)
        self._llm = Groq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "1024"))
        )

        # Embedding model
        self._embed_model = HuggingFaceEmbedding(
            model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )

        # Apply to global settings
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model

        # Index and chat engine
        self._index = None
        self._collection_name = collection_name
        self._create_kb()
        if self._index is not None:
            self._create_chat_engine()
        else:
            print("❌ Failed to create knowledge base. Chat engine not initialized.")

    def _kb_path(self):
        base_dir = os.path.dirname(__file__)
        return os.path.join(base_dir, "pg_manual.txt")

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(input_files=[self._kb_path()])
            documents = reader.load_data()

            vector_store = QdrantVectorStore(client=self._client, collection_name=self._collection_name)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Build index and upload embeddings into Qdrant collection
            self._index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            print("✅ Knowledge base created successfully and stored in Qdrant.")
        except Exception as e:
            print(f"❌ Error while creating knowledge base: {e}")

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def interact_with_llm(self, customer_query, top_k=5):
        """
        Always returns:
            {"answer": <str>, "retrieved_chunks": [<str>, ...]}
        """
        try:
            if not customer_query or not customer_query.strip():
                return {"answer": "", "retrieved_chunks": []}

            # Generate embedding for retrieval
            query_embedding = self._embed_model.get_text_embedding(customer_query)
            if query_embedding is None:
                return {"answer": "❌ Could not generate embedding for your query.", "retrieved_chunks": []}

            # Retrieve nodes from index
            retriever = self._index.as_retriever()
            bundle = QueryBundle(query_str=customer_query, embedding=query_embedding)
            nodes = retriever.retrieve(bundle)
            print(f"✅ Retrieved {len(nodes)} relevant chunks.")

            # Extract texts from nodes robustly
            retrieved_texts = []
            for n in nodes:
                text = None
                try:
                    text = n.node.get_text()
                except Exception:
                    text = getattr(n, "text", None) or getattr(n, "content", None) or getattr(n, "page_content", None)
                if not text:
                    text = str(n)
                retrieved_texts.append(text)

            # Ask chat engine (llama-index chat wrapper using Groq)
            response = self._chat_engine.chat(customer_query)
            answer_text = getattr(response, "response", None) or str(response)

            return {"answer": answer_text, "retrieved_chunks": retrieved_texts}
        except Exception as e:
            return {"answer": f"❌ Error in interaction: {e}", "retrieved_chunks": []}



    @property
    def _prompt(self):
        return """
        You are an AI assistant helping prospective students with queries about the IIT Kanpur PG Admission process.
        Answer questions clearly and concisely using the information from the admission manual.

        If you don't know the answer, say you don't know. Keep answers short and to the point.
        """
