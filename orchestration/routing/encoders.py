"""Air-gapped, offline embedding encoders for semantic routing."""

import hashlib
import math
import re
from typing import List

from orchestration.routing.base import SemanticRouterEncoder


class DeterministicHashEncoder(SemanticRouterEncoder):
    """Zero-dependency deterministic n-gram hash encoder for offline semantic routing.

    Projects text into a fixed-dimension unit vector using sub-word and token
    hashes. Completely air-gapped, requires no model downloads, and guarantees
    stable, reproducible vector similarity.
    """

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode batch of texts into normalized embedding vectors."""
        results = []
        for text in texts:
            results.append(self._encode_single(text))
        return results

    def _encode_single(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        cleaned = text.lower().strip()
        if not cleaned:
            return vec

        # Token and character 3-gram extraction
        tokens = re.findall(r"\w+", cleaned)
        features = list(tokens)
        for i in range(len(cleaned) - 2):
            features.append(cleaned[i : i + 3])

        for feat in features:
            h = int(hashlib.sha256(feat.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) == 1 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec
