"""Structural document chunker for the RAG subsystem.

Transforms a NormalizedDocument into a sequence of structural rag.domain.models.Chunk
objects. Respects document semantic boundaries (sections, headings, tables, lists,
paragraphs, code blocks) while maintaining configurable chunk sizes, genuine overlap,
heading hierarchy preservation, and fallback splitting for oversized elements.
Strictly guarantees that every final Chunk satisfies len(chunk.content) <= max_chunk_size.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from rag.chunking.interfaces import DocumentChunker
from rag.chunking.options import ChunkingOptions, FallbackSplitStrategy
from rag.domain.models import Chunk
from rag.metadata.models import ChunkMetadata
from rag.normalization.models import (
    NormalizedDocument,
    NormalizedElement,
    NormalizedElementType,
)


def _split_by_character(text: str, max_size: int, overlap: int) -> List[str]:
    """Slice text by fixed character windows with overlap."""
    if len(text) <= max_size:
        return [text]

    safe_overlap = max(0, min(overlap, max_size - 1))
    chunks: List[str] = []
    start = 0
    step = max(1, max_size - safe_overlap)

    while start < len(text):
        end = min(start + max_size, len(text))
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start += step

    return chunks


def _split_oversized_element_content(
    content: str,
    max_size: int,
    overlap: int,
    strategy: FallbackSplitStrategy,
) -> List[str]:
    """Apply the chosen FallbackSplitStrategy to an oversized text block with genuine overlap."""
    content = content.strip()
    if not content:
        return []
    if len(content) <= max_size:
        return [content]

    safe_overlap = max(0, min(overlap, max_size - 1))

    if strategy == FallbackSplitStrategy.CHARACTER:
        return _split_by_character(content, max_size, safe_overlap)

    # Hierarchically break down text into atomic semantic pieces <= max_size
    def _break_down(text_block: str, level: int) -> Tuple[List[str], str]:
        t = text_block.strip()
        if not t:
            return [], " "
        if len(t) <= max_size:
            return [t], " "

        if strategy == FallbackSplitStrategy.LINE:
            delims = ["\n", " "]
        else:
            delims = ["\n\n", "\n", "sentence", " "]

        if level >= len(delims):
            # Atomic fallback for unbreakable token (e.g. huge URL or unbroken code)
            step = max(1, max_size - safe_overlap)
            sliced = []
            for i in range(0, len(t), step):
                part = t[i : i + max_size].strip()
                if part:
                    sliced.append(part)
                if i + max_size >= len(t):
                    break
            return sliced, ""

        delim = delims[level]
        if delim == "sentence":
            raw_parts = re.split(r"(?<=[.?!])\s+", t)
            sep = " "
        else:
            raw_parts = t.split(delim)
            sep = delim

        parts = [p.strip() for p in raw_parts if p.strip()]
        if len(parts) <= 1:
            # Delimiter was not found or did not split, advance to finer delimiter
            return _break_down(t, level + 1)

        atomic_units: List[str] = []
        for p in parts:
            if len(p) <= max_size:
                atomic_units.append(p)
            else:
                sub_units, _ = _break_down(p, level + 1)
                atomic_units.extend(sub_units)

        return atomic_units, sep

    units, sep = _break_down(content, 0)
    if not units:
        return []
    if not sep:
        sep = " "

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for u in units:
        u_len = len(u)
        add_len = u_len if not current_chunk else u_len + len(sep)

        if current_len + add_len <= max_size:
            current_chunk.append(u)
            current_len += add_len
        else:
            if current_chunk:
                chunks.append(sep.join(current_chunk))

            overlap_parts: List[str] = []
            overlap_len = 0

            if safe_overlap > 0:
                # Accumulate tail units from previous chunk up to safe_overlap
                for part in reversed(current_chunk):
                    p_len = len(part) + (len(sep) if overlap_parts else 0)
                    if overlap_len + p_len <= safe_overlap:
                        overlap_parts.insert(0, part)
                        overlap_len += p_len
                    else:
                        if not overlap_parts and len(part) <= safe_overlap:
                            overlap_parts.insert(0, part)
                            overlap_len += len(part)
                        break

                # Ensure strict forward progress
                if len(overlap_parts) >= len(current_chunk) and overlap_parts:
                    overlap_parts.pop(0)
                    overlap_len = sum(len(p) for p in overlap_parts) + max(0, len(overlap_parts) - 1) * len(sep)

                # Ensure overlap + next unit fits within max_size
                while overlap_parts and (overlap_len + len(sep) + u_len > max_size):
                    overlap_parts.pop(0)
                    overlap_len = sum(len(p) for p in overlap_parts) + max(0, len(overlap_parts) - 1) * len(sep)

            current_chunk = overlap_parts + [u]
            current_len = sum(len(p) for p in current_chunk) + max(0, len(current_chunk) - 1) * len(sep)

    if current_chunk:
        chunks.append(sep.join(current_chunk))

    return chunks


def _infer_heading_level(elem: NormalizedElement, current_stack: List[Tuple[int, str]]) -> int:
    """Infer hierarchical heading level (1 for title/H1, 2 for H2, etc.)."""
    if elem.heading_level is not None and elem.heading_level > 0:
        return elem.heading_level

    content = elem.content.strip()

    match_hash = re.match(r"^(#{1,6})\s+", content)
    if match_hash:
        return len(match_hash.group(1))

    match_num = re.match(r"^(\d+(?:\.\d+)*)\b", content)
    if match_num:
        dots = match_num.group(1).count(".")
        return 1 + dots

    if elem.element_type == NormalizedElementType.TITLE:
        return 1

    if current_stack:
        if current_stack[-1][0] == 1:
            return 2
        return current_stack[-1][0]

    return 2


class StructuralChunker(DocumentChunker):
    """Document chunker that respects semantic document hierarchy.

    Groups elements into chunks based on headings, sections, tables, lists, and paragraphs.
    Avoids arbitrary character splits while enforcing configurable bounds (max_chunk_size,
    min_chunk_size, overlap_size), preserving heading hierarchy paths, and generating
    deterministic chunk IDs and metadata.
    Strictly guarantees that len(chunk.content) <= max_chunk_size for all chunks.
    """

    def __init__(self, options: Optional[ChunkingOptions] = None) -> None:
        """Initialize the chunker with options.

        Args:
            options: Configuration options. Defaults to baseline ChunkingOptions().
        """
        self.options = options or ChunkingOptions()

    def chunk(self, document: NormalizedDocument) -> List[Chunk]:
        """Convert a normalized document into a list of structural domain Chunks.

        Args:
            document: Clean, normalized document containing ordered structural elements.

        Returns:
            List of domain Chunk objects with deterministic IDs and complete metadata.
        """
        elements = document.elements
        if not elements:
            if not document.text or not document.text.strip():
                return []
            elements = [
                NormalizedElement(
                    index=0,
                    element_type=NormalizedElementType.PARAGRAPH,
                    content=document.text.strip(),
                )
            ]

        valid_elements = [
            elem for elem in elements if elem.content and elem.content.strip()
        ]
        if not valid_elements:
            return []

        chunks: List[Chunk] = []
        heading_stack: List[Tuple[int, str]] = []
        unemitted_heading_elements: List[NormalizedElement] = []
        pending_elements: List[NormalizedElement] = []
        pending_len: int = 0
        pending_is_list: bool = False

        def _get_heading_context(elements_to_emit: List[NormalizedElement]) -> Tuple[Optional[str], List[str]]:
            """Return (active_heading_str, heading_path_list)."""
            if heading_stack:
                path = [h for _, h in heading_stack]
                return " > ".join(path), list(path)

            if elements_to_emit:
                for e in elements_to_emit:
                    if e.parent_heading:
                        return e.parent_heading, [e.parent_heading]

            return None, []

        def _resolve_prefix_and_budget(
            heading_str: Optional[str],
            heading_path: List[str],
        ) -> Tuple[str, int]:
            """Determine heading prefix to prepend and the remaining budget for body text.

            Guarantees len(prefix) + body_budget <= max_chunk_size.
            Leaves at least min_body_room for body text; if prefix is too long,
            falls back to immediate heading, or empty prefix if even immediate heading is too large.
            """
            if not self.options.include_heading_context or not heading_str or not heading_path:
                return "", self.options.max_chunk_size

            min_body_room = min(30, max(10, self.options.max_chunk_size // 4))

            # Candidate 1: Full hierarchy breadcrumb
            full_prefix = f"{heading_str}\n\n"
            if len(full_prefix) + min_body_room <= self.options.max_chunk_size:
                return full_prefix, self.options.max_chunk_size - len(full_prefix)

            # Candidate 2: Immediate section heading only
            immediate = heading_path[-1]
            immediate_prefix = f"{immediate}\n\n"
            if len(immediate_prefix) + min_body_room <= self.options.max_chunk_size:
                return immediate_prefix, self.options.max_chunk_size - len(immediate_prefix)

            # Candidate 3: Do not prepend in content (heading remains in metadata)
            return "", self.options.max_chunk_size

        def _emit_chunk(
            elements_to_emit: List[NormalizedElement],
            forced_content: Optional[str] = None,
            extra_meta: Optional[Dict[str, Any]] = None,
        ) -> None:
            """Internal helper to construct and append a single Chunk."""
            if not elements_to_emit and not forced_content:
                return

            if forced_content is not None:
                body_text = forced_content.strip()
            else:
                is_all_list = all(
                    e.element_type == NormalizedElementType.LIST_ITEM
                    for e in elements_to_emit
                )
                sep = "\n" if is_all_list else "\n\n"
                body_text = sep.join(e.content.strip() for e in elements_to_emit)

            if not body_text:
                return

            heading_for_chunk, heading_path = _get_heading_context(elements_to_emit)
            prefix, _ = _resolve_prefix_and_budget(heading_for_chunk, heading_path)

            final_content = body_text
            if prefix and not body_text.startswith(prefix.strip()):
                final_content = f"{prefix}{body_text}"

            # Invariant safety: if final_content exceeds max_chunk_size, drop prefix if body alone fits
            if len(final_content) > self.options.max_chunk_size:
                if prefix and len(body_text) <= self.options.max_chunk_size:
                    final_content = body_text
                else:
                    final_content = final_content[: self.options.max_chunk_size].strip()

            chunk_idx = len(chunks)
            chunk_id = f"{document.document_id}_chk_{chunk_idx:04d}"

            doc_title = getattr(document, "title", None) or document.metadata.get("title")

            chunk_meta = ChunkMetadata.build(
                chunk_id=chunk_id,
                document_id=document.document_id,
                chunk_index=chunk_idx,
                elements=elements_to_emit,
                heading=heading_for_chunk,
                heading_path=heading_path,
                source_path=str(document.file_path) if document.file_path else document.metadata.get("source_path"),
                format=document.format or document.metadata.get("format", ""),
                document_title=doc_title,
                is_split=bool(extra_meta and extra_meta.get("is_split")),
                split_part=extra_meta.get("split_part") if extra_meta else None,
                total_parts=extra_meta.get("total_parts") if extra_meta else None,
                extra_meta=extra_meta,
            )

            is_table = chunk_meta.is_table
            is_split = chunk_meta.is_split

            # Check min_chunk_size merge with previous chunk under exact same heading
            if (
                chunks
                and not is_table
                and not is_split
                and len(final_content) < self.options.min_chunk_size
                and not chunks[-1].metadata.get("is_table", False)
                and not chunks[-1].metadata.get("is_split", False)
                and chunks[-1].metadata.get("heading") == heading_for_chunk
                and (len(chunks[-1].content) + len("\n\n") + len(body_text) <= self.options.max_chunk_size)
            ):
                prev = chunks[-1]
                prev_meta = ChunkMetadata.from_dict(prev.metadata)
                merged_meta = prev_meta.merge_elements(elements_to_emit, extra_meta=extra_meta)
                merged_content = prev.content + "\n\n" + body_text
                chunks[-1] = Chunk(
                    id=prev.id,
                    document_id=prev.document_id,
                    content=merged_content,
                    metadata=merged_meta.to_dict(),
                )
                return

            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.document_id,
                    content=final_content,
                    metadata=chunk_meta.to_dict(),
                )
            )

        def _flush_pending() -> None:
            nonlocal pending_elements, pending_len, pending_is_list
            if not pending_elements:
                return

            heading_for_chunk, heading_path = _get_heading_context(pending_elements)
            _, body_budget = _resolve_prefix_and_budget(heading_for_chunk, heading_path)

            sep = "\n" if pending_is_list else "\n\n"
            combined_text = sep.join(e.content.strip() for e in pending_elements)

            if len(combined_text) <= body_budget:
                _emit_chunk(pending_elements)
            else:
                sub_buf: List[NormalizedElement] = []
                sub_len = 0
                for elem in pending_elements:
                    elem_len = len(elem.content.strip())
                    if elem_len > body_budget:
                        if sub_buf:
                            _emit_chunk(sub_buf)
                            sub_buf = []
                            sub_len = 0
                        safe_ovlp = min(self.options.overlap_size, max(0, body_budget - 1))
                        splits = _split_oversized_element_content(
                            elem.content,
                            body_budget,
                            safe_ovlp,
                            self.options.fallback_strategy,
                        )
                        for part_idx, part in enumerate(splits):
                            _emit_chunk(
                                [elem],
                                forced_content=part,
                                extra_meta={
                                    "is_split": True,
                                    "split_part": part_idx + 1,
                                    "total_parts": len(splits),
                                },
                            )
                    elif sub_len + elem_len + len(sep) <= body_budget:
                        sub_buf.append(elem)
                        sub_len += elem_len + len(sep)
                    else:
                        if sub_buf:
                            _emit_chunk(sub_buf)
                        sub_buf = [elem]
                        sub_len = elem_len

                if sub_buf:
                    _emit_chunk(sub_buf)

            pending_elements = []
            pending_len = 0
            pending_is_list = False

        # Walk through normalized elements
        for elem in valid_elements:
            etype = elem.element_type
            content = elem.content.strip()

            # Case 1: Headings (TITLE, SECTION_HEADER)
            if etype in (NormalizedElementType.TITLE, NormalizedElementType.SECTION_HEADER):
                _flush_pending()
                lvl = _infer_heading_level(elem, heading_stack)
                while heading_stack and heading_stack[-1][0] >= lvl:
                    heading_stack.pop()
                heading_stack.append((lvl, content))
                unemitted_heading_elements.append(elem)
                continue

            if unemitted_heading_elements:
                unemitted_heading_elements.clear()

            heading_for_chunk, heading_path = _get_heading_context([elem])
            _, body_budget = _resolve_prefix_and_budget(heading_for_chunk, heading_path)

            # Case 2: Tables
            if etype == NormalizedElementType.TABLE and self.options.preserve_tables:
                _flush_pending()
                table_len = len(content)
                extra_table_meta: Dict[str, Any] = {"is_table": True}
                if "num_rows" in elem.metadata:
                    extra_table_meta["num_rows"] = elem.metadata["num_rows"]
                if "num_cols" in elem.metadata:
                    extra_table_meta["num_cols"] = elem.metadata["num_cols"]

                if table_len > body_budget:
                    safe_ovlp = min(self.options.overlap_size, max(0, body_budget - 1))
                    splits = _split_oversized_element_content(
                        content,
                        body_budget,
                        safe_ovlp,
                        self.options.fallback_strategy,
                    )
                    for part_idx, part in enumerate(splits):
                        split_meta = {
                            **extra_table_meta,
                            "is_split": True,
                            "split_part": part_idx + 1,
                            "total_parts": len(splits),
                        }
                        _emit_chunk([elem], forced_content=part, extra_meta=split_meta)
                else:
                    _emit_chunk([elem], extra_meta=extra_table_meta)
                continue

            # Case 3: List Items
            if etype == NormalizedElementType.LIST_ITEM and self.options.preserve_lists:
                if pending_elements and not pending_is_list:
                    _flush_pending()

                pending_is_list = True
                elem_len = len(content)

                if pending_len + elem_len + 1 > body_budget:
                    _flush_pending()
                    pending_is_list = True

                pending_elements.append(elem)
                pending_len += elem_len + 1
                continue

            # Case 4: Paragraphs, Code, and Other Text Elements
            if pending_is_list:
                _flush_pending()

            elem_len = len(content)
            if elem_len > body_budget:
                _flush_pending()
                safe_ovlp = min(self.options.overlap_size, max(0, body_budget - 1))
                splits = _split_oversized_element_content(
                    content,
                    body_budget,
                    safe_ovlp,
                    self.options.fallback_strategy,
                )
                for part_idx, part in enumerate(splits):
                    _emit_chunk(
                        [elem],
                        forced_content=part,
                        extra_meta={
                            "is_split": True,
                            "split_part": part_idx + 1,
                            "total_parts": len(splits),
                        },
                    )
                continue

            if pending_elements and (pending_len + elem_len + 2 > body_budget):
                _flush_pending()

            pending_elements.append(elem)
            pending_len += elem_len + 2

        _flush_pending()

        # Outline edge-case (document with only headings)
        if not chunks and unemitted_heading_elements:
            for h_elem in unemitted_heading_elements:
                _emit_chunk([h_elem], forced_content=h_elem.content.strip())

        return chunks
