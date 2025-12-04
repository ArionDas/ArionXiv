import warnings
import pytest
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

warnings.filterwarnings("ignore", message="PyPDF2 is deprecated", category=DeprecationWarning)

from arionxiv.rag_techniques.basic_rag import BasicRAG, EmbeddingProvider

pytestmark = pytest.mark.filterwarnings("ignore:PyPDF2 is deprecated:DeprecationWarning")


class FakeConfigService:
    def get_rag_config(self):
        return {
            "chunk_size": 120,
            "chunk_overlap": 20,
            "top_k_results": 3,
            "ttl_hours": 1,
            "vector_collection": "paper_embeddings",
            "chat_collection": "chat_sessions",
        }

    def get_embedding_config(self):
        return {
            "primary_model": "fake-primary",
            "fallback_1": "fake-fallback-1",
            "fallback_2": "fake-fallback-2",
            "dimension_default": 4,
            "enable_gemini": False,
            "enable_huggingface": False,
            "batch_size": 2,
            "cache_enabled": False,
        }


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._dimension = 4

    async def get_embeddings(self, texts):
        return [[float(len(text) % 5 + i) for i in range(self._dimension)] for text in texts]

    def get_dimension(self):
        return self._dimension

    def get_name(self):
        return "fake-embedding"


class FakeLLMClient:
    def __init__(self):
        self.prompts = []

    async def get_completion(self, prompt):
        self.prompts.append(prompt)
        return {"success": True, "content": "Assistant response"}


class InMemoryDBService:
    def __init__(self):
        self.collections = defaultdict(list)

    async def insert_one(self, collection, document):
        self.collections[collection].append(document)
        return SimpleNamespace(inserted_id=len(self.collections[collection]))

    async def find_one(self, collection, filter_dict):
        for doc in self.collections.get(collection, []):
            if all(doc.get(k) == v for k, v in filter_dict.items()):
                return doc
        return None

    async def update_one(self, collection, filter_dict, update_dict):
        doc = await self.find_one(collection, filter_dict)
        if not doc:
            return SimpleNamespace(modified_count=0)

        for field, value in update_dict.get("$set", {}).items():
            doc[field] = value

        push_ops = update_dict.get("$push", {})
        for field, payload in push_ops.items():
            if isinstance(payload, dict) and "$each" in payload:
                doc.setdefault(field, [])
                doc[field].extend(payload["$each"])
            else:
                doc.setdefault(field, [])
                doc[field].append(payload)

        return SimpleNamespace(modified_count=1)


class _TestableBasicRAG(BasicRAG):
    def _setup_embedding_providers(self, embedding_config):
        self.embedding_providers = [FakeEmbeddingProvider()]
        self.current_embedding_provider = self.embedding_providers[0]


@pytest.fixture(autouse=True)
def stub_prompt_formatter(monkeypatch):
    def fake_prompt(name, **kwargs):
        return (
            f"[CTX]{kwargs.get('context', '')}\n"
            f"[HIST]{kwargs.get('history', '')}\n"
            f"[MSG]{kwargs.get('message', '')}"
        )

    monkeypatch.setattr("arionxiv.prompts.format_prompt", fake_prompt)
    monkeypatch.setattr("arionxiv.prompts.prompts.format_prompt", fake_prompt)


@pytest.fixture
def rag_setup():
    db_service = InMemoryDBService()
    config_service = FakeConfigService()
    llm_client = FakeLLMClient()
    rag = _TestableBasicRAG(db_service, config_service, llm_client)
    return rag, db_service, llm_client


@pytest.mark.asyncio
async def test_add_document_to_index_creates_multiple_chunks(rag_setup):
    rag, db_service, _ = rag_setup
    text = " ".join(f"Sentence {i}." for i in range(40))

    await rag.add_document_to_index("doc-1", text, {"topic": "ml"})

    stored_docs = db_service.collections[rag.vector_collection]
    assert len(stored_docs) >= 3
    assert all(doc["doc_id"] == "doc-1" for doc in stored_docs)
    assert all(doc["metadata"]["topic"] == "ml" for doc in stored_docs)


def test_build_chat_prompt_includes_context_and_history(rag_setup):
    rag, _, _ = rag_setup
    session = {
        "messages": [
            {"type": "user", "content": "Explain dataset"},
            {"type": "assistant", "content": "Dataset described"},
        ]
    }

    prompt = rag._build_chat_prompt(session, "What are the key results?", "Important context")

    assert "Important context" in prompt
    assert "User: Explain dataset" in prompt
    assert "Assistant: Dataset described" in prompt
    assert "key results" in prompt


@pytest.mark.asyncio
async def test_chat_with_session_records_messages_and_returns_llm_result(monkeypatch, rag_setup):
    rag, db_service, llm_client = rag_setup
    session_doc = {
        "session_id": "session-1",
        "messages": [],
        "user_id": "tester",
        "paper_ids": ["doc-1"],
        "created_at": datetime.utcnow(),
        "last_activity": datetime.utcnow(),
    }
    await db_service.insert_one(rag.chat_collection, session_doc)

    async_mock = AsyncMock(return_value=[{"doc_id": "doc-1", "text": "Chunk text", "metadata": {}}])
    monkeypatch.setattr(rag, "search_similar_documents", async_mock)

    result = await rag._chat_with_session("session-1", "Summarize contributions")

    assert result["success"] is True
    assert result["relevant_chunks"] == 1
    assert len(session_doc["messages"]) == 2
    assert session_doc["messages"][0]["type"] == "user"
    assert session_doc["messages"][1]["type"] == "assistant"
    assert "Summarize contributions" in llm_client.prompts[-1]
