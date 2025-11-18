"""
LangChain compatibility layer for imports across versions.

Provides unified imports for:
 - Document
 - BaseMessage, HumanMessage, AIMessage
 - ChatPromptTemplate, MessagesPlaceholder
 - ConversationSummaryBufferMemory

Prefers langchain_core / langchain_community when available,
falls back to legacy langchain.* modules for older versions.
"""
from __future__ import annotations

# Document
try:
    from langchain_core.documents import Document  # type: ignore
except Exception:  # pragma: no cover - fallback path
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        from langchain.docstore.document import Document  # type: ignore

# Messages
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage  # type: ignore
except Exception:  # pragma: no cover
    from langchain.schema import BaseMessage, HumanMessage, AIMessage  # type: ignore

# Prompts
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore
except Exception:  # pragma: no cover
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # type: ignore

# Memory
try:
    from langchain_community.memory import ConversationSummaryBufferMemory  # type: ignore
except Exception:  # pragma: no cover
    from langchain.memory import ConversationSummaryBufferMemory  # type: ignore

