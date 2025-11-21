"""
Hybrid QA endpoint that fuses BM25/Vector retrieval with GraphRAG index-time graph lookup,
and attaches KG triples as evidence. Keeps KG as ground truth while using graph for fast recall.
"""
import time
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Body
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import settings
from ..rag.graph_retriever import GraphRetriever
from ..rag.bm25_hybrid_retriever import BM25HybridRetriever
from ..rag.vectorstore import YonEarthVectorStore
from ..character.gaia import GaiaCharacter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["Hybrid QA"])
limiter = Limiter(key_func=get_remote_address)


class HybridQARequest(BaseModel):
    query: str = Field(..., description="User question")
    k: int = Field(10, description="Number of results to return")
    category_threshold: float = Field(0.7, description="Semantic category threshold for BM25 chain")


class HybridSource(BaseModel):
    chunk_id: Optional[str]
    title: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_preview: Optional[str] = None
    source_type: str = Field("graph|bm25", description="Origin of this source")


class HybridQAResponse(BaseModel):
    query: str
    answer: Optional[str]
    sources: List[HybridSource]
    graph_evidence: List[Dict[str, Any]]
    matched_entities: List[str]
    used_bm25: bool
    processing_time: float
    note: Optional[str] = None


def _safe_init_bm25() -> Optional[BM25HybridRetriever]:
    try:
        vs = YonEarthVectorStore()
        retriever = BM25HybridRetriever(vs, use_reranker=False)
        return retriever
    except Exception as e:
        logger.warning(f"BM25/Vector not available: {e}")
        return None


@router.post("/hybrid", response_model=HybridQAResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def qa_hybrid(request: Request, body: HybridQARequest = Body(...)) -> HybridQAResponse:
    start = time.time()
    try:
        # Graph retrieval (fast, offline). If index missing, continue with BM25 only.
        g_res = None
        try:
            graph = GraphRetriever()
            g_res = graph.retrieve(body.query, k=body.k)
        except Exception as ge:
            logger.warning(f"Graph index not available: {ge}")
            g_res = None

        sources: List[HybridSource] = []
        if g_res:
            for d in g_res.chunks:
                sources.append(HybridSource(
                    chunk_id=d.metadata.get("chunk_id"),
                    title=d.metadata.get("chapter_title"),
                    metadata=dict(d.metadata or {}),
                    content_preview=d.page_content,
                    source_type="graph"
                ))

        used_bm25 = False
        # Try BM25+vector fusion (if configured)
        bm25 = _safe_init_bm25()
        bm_docs = []
        if bm25:
            used_bm25 = True
            bm_docs = bm25.hybrid_search(body.query, k=body.k)
            for d in bm_docs:
                sources.append(HybridSource(
                    chunk_id=d.metadata.get("chunk_id"),
                    title=d.metadata.get("title") or d.metadata.get("chapter_title"),
                    metadata=dict(d.metadata or {}),
                    content_preview=d.page_content[:200] if d.page_content else None,
                    source_type="bm25"
                ))

        # Deduplicate by chunk_id + preview
        seen = set()
        deduped: List[HybridSource] = []
        for s in sources:
            key = (s.chunk_id, s.content_preview)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)

        # Synthesize answer over fused results (BM25 + Graph + KG triples)
        answer = None
        note = None
        try:
            retrieved_docs = []
            retrieved_docs.extend(bm_docs)
            if g_res:
                retrieved_docs.extend(g_res.chunks)
                if g_res.triples:
                    from langchain_core.documents import Document
                    triples_text = "\n".join(
                        f"- ({t.get('p_true',1.0):.2f}) {t['source']} --{t['predicate']}--> {t['target']} | {t.get('context','')}"
                        for t in g_res.triples[: min(20, len(g_res.triples))]
                    )
                    retrieved_docs.append(Document(page_content=f"KG Evidence:\n{triples_text}", metadata={"content_type":"kg_evidence"}))
            try:
                gaia = GaiaCharacter()
                gen = gaia.generate_response(user_input=body.query, retrieved_docs=retrieved_docs)
                answer = gen.get("response")
            except Exception as llme:
                logger.warning(f"LLM synthesis failed: {llme}")
        except Exception as se:
            logger.warning(f"Synthesis unavailable: {se}")

        elapsed = time.time() - start
        if not used_bm25:
            if g_res:
                note = "BM25/Vector unavailable; returning graph sources + metadata only."
            else:
                note = "Neither graph index nor BM25/Vector available; no retrieval sources attached."

        return HybridQAResponse(
            query=body.query,
            answer=answer,
            sources=deduped[: body.k],
            graph_evidence=g_res.triples if g_res else [],
            matched_entities=g_res.matched_entities if g_res else [],
            used_bm25=used_bm25,
            processing_time=elapsed,
            note=note,
        )
    except Exception as e:
        logger.error(f"Hybrid QA error: {e}")
        raise HTTPException(status_code=500, detail="Hybrid QA failed")
