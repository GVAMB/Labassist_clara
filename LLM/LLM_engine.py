"""
labassist_rag.py

Class-based version of your internet-integrated LabAssist RAG.

- Builds / loads a FAISS index over local SOP PDFs
- Uses LangChain retriever to get relevant chunks
- Uses `internet_research.integrate_with_rag` to optionally add web snippets
- Calls OpenAI GPT-4o-mini with a clean SYSTEM prompt
"""

import os
from typing import List, Dict

from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA

from LLM.internet_research import integrate_with_rag
from openai import OpenAI


class LabAssistRAG:
    """
    Internet-integrated RAG helper.

    Typical usage:

        from labassist_rag import LabAssistRAG

        rag = LabAssistRAG(
            pdf_dir="sample_pdfs",
            index_dir="labassist_index",
        )

        answer = rag.answer_query("How to store sample X?")
        print(answer)
    """

    def __init__(
        self,
        pdf_dir: str = "LLM/Sample_Pdfs",
        index_dir: str = "LLM/labassist_index",
        model_name: str = "gpt-4o-mini",
    ):
        # 1. Load your OpenAI API key from .env
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

        # For langchain-openai
        os.environ["OPENAI_API_KEY"] = api_key

        # 2. Paths
        self.PDF_DIR = pdf_dir
        self.INDEX_DIR = index_dir

        # 3. Core components
        self.embeddings = OpenAIEmbeddings()
        self.db = None
        self.retriever = None
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.qa = None  # we keep this in case you still want pure RetrievalQA
        self.openai_client = OpenAI()  # for direct chat.completions
        self.model_name = model_name

        # Build / load index immediately
        self._build_or_load_index()

        # System prompt (same semantics as script)
        self.SYSTEM_PROMPT = (
            "You are LabAssist, a laboratory assistant. Always: "
            "1) Prefer internal SOP documents first when answering lab procedures. "
            "2) Use web sources only for background / standards. "
            "3) Never provide hazardous step-by-step procedures. "
            "4)Do not include source tags or external references. Focus solely on the relevant content to provide accurate and helpful responses."
            "Answer concisely."
        )

    # ------------------------------------------------------------------
    # Index build/load
    # ------------------------------------------------------------------
    def _build_or_load_index(self, chunk_size: int = 700, chunk_overlap: int = 100):
        """Create or load the FAISS index for local PDFs."""
        if not os.path.exists(self.INDEX_DIR):
            print("🔧 Creating vector index from PDFs...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            texts: List[str] = []
            metas: List[Dict] = []

            if not os.path.isdir(self.PDF_DIR):
                raise RuntimeError(f"PDF_DIR does not exist: {self.PDF_DIR}")

            for file in os.listdir(self.PDF_DIR):
                if not file.lower().endswith(".pdf"):
                    continue

                pdf_path = os.path.join(self.PDF_DIR, file)
                reader = PdfReader(pdf_path)
                text = "\n".join(
                    p.extract_text() for p in reader.pages if p.extract_text()
                )
                if not text.strip():
                    continue

                for chunk in splitter.split_text(text):
                    texts.append(chunk)
                    metas.append({"source": file})

            if not texts:
                raise RuntimeError(
                    f"No text extracted from any PDFs in {self.PDF_DIR}. "
                    "Check that the files are readable."
                )

            self.db = FAISS.from_texts(texts, embedding=self.embeddings, metadatas=metas)
            self.db.save_local(self.INDEX_DIR)
        else:
            print("✅ Loading existing index...")
            self.db = FAISS.load_local(
                self.INDEX_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        # Set up retriever and (optional) RetrievalQA chain
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)

    # ------------------------------------------------------------------
    # Core query method
    # ------------------------------------------------------------------
    def answer_query(self, question: str, max_web: int = 2) -> str:
        """
        Answer a user question using:
        - local PDFs via FAISS retriever
        - plus optional web snippets via `integrate_with_rag`.
        """

        if self.retriever is None:
            # Safety fallback – try to rebuild index
            self._build_or_load_index()

        # 1) Local retrieval (SOPs)
        try:
            docs = self.retriever._get_relevant_documents(question, run_manager=None)
        except TypeError:
            # older LC versions may not accept run_manager
            docs = self.retriever._get_relevant_documents(question)
        sop_chunks = []
        for d in docs:
            text = getattr(d, "page_content", "") or ""
            source = (
                d.metadata.get("source")
                if hasattr(d, "metadata") and d.metadata
                else "unknown"
            )
            sop_chunks.append({"text": text, "source": source, "score": None})

        # 2) Internet research (safe, optional)
        combined = integrate_with_rag(sop_chunks, question, max_web=max_web)

        # Build combined context
        context_pieces = []
        for c in combined:
            src = c.get("source", "unknown")
            txt = c.get("text", "") or ""
            context_pieces.append(f"[{src}] {txt[:1200].strip()}")
        combined_context = "\n\n".join(context_pieces)

        # 3) Call OpenAI LLM with RAG + web context
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{combined_context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer concisely."
                ),
            },
        ]

        try:
            resp = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=800,
            )
        except Exception as e:
            return f"[ERROR contacting OpenAI API] {e}"

        # Robust extraction – same logic as your script
        content = None
        try:
            content = resp.choices[0].message.content
        except Exception:
            try:
                content = resp["choices"][0]["message"]["content"]
            except Exception:
                try:
                    content = resp.choices[0].text
                except Exception:
                    try:
                        s = repr(resp)
                        content = s[:1200] + ("..." if len(s) > 1200 else "")
                    except Exception:
                        content = "[ERROR: failed to extract assistant content]"

        return content or "[Empty response from model]"

    # ------------------------------------------------------------------
    # Optional REPL (for debugging)
    # ------------------------------------------------------------------
    def run_repl(self):
        print("\n🤖 LabAssist RAG Prototype Ready! (class, with internet research)")
        print("Ask a question (type 'exit' to quit)\n")

        while True:
            q = input("You: ")
            if q.lower() in ("exit", "quit"):
                break
            ans = self.answer_query(q)
            print(f"LabAssist: {ans}\n")


if __name__ == "__main__":
    # Standalone test
    rag = LabAssistRAG(
        pdf_dir="sample_pdfs",
        index_dir="labassist_index",
    )
    rag.run_repl()
