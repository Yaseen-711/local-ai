"""Local AI Foundation - Standalone RAG Subsystem."""

from rag.offline import ensure_offline_environment

# Enforce zero-network runtime safety by default upon package import
ensure_offline_environment()
