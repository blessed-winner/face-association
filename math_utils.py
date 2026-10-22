"""
Pure-Python mathematical and geometric primitives.
Provides zero-dependency vector math, matrix operations, bounding box operations,
and Hungarian (Kuhn-Munkres) algorithm for linear sum assignment.
Runs 100% reliably in pure CPU Python without requiring binary C-extensions,
while seamlessly leveraging numpy if present.
"""
from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any, Union

class BBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float = 1.0):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = float(confidence)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def clip(self, max_w: float, max_h: float) -> BBox:
        return BBox(
            x1=max(0.0, min(self.x1, max_w)),
            y1=max(0.0, min(self.y1, max_h)),
            x2=max(0.0, min(self.x2, max_w)),
            y2=max(0.0, min(self.y2, max_h)),
            confidence=self.confidence
        )

    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2, "confidence": self.confidence}

    def __repr__(self) -> str:
        return f"BBox([{self.x1:.1f}, {self.y1:.1f}, {self.x2:.1f}, {self.y2:.1f}], conf={self.confidence:.2f})"

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 <= 1e-9 or norm2 <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm1 * norm2)))

def hungarian_algorithm(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Standard Kuhn-Munkres (Hungarian) algorithm implementation for optimal bipartite matching.
    Takes an N x M cost matrix and returns a list of (row_idx, col_idx) optimal matches.
    """
    if not cost_matrix or not cost_matrix[0]:
        return []
    
    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0])
    
    # Square matrix padding for Hungarian
    dim = max(n_rows, n_cols)
    max_val = 0.0
    for r in range(n_rows):
        for c in range(n_cols):
            if cost_matrix[r][c] > max_val:
                max_val = cost_matrix[r][c]
    pad_val = max_val + 1000.0

    mat = [[pad_val for _ in range(dim)] for _ in range(dim)]
    for r in range(n_rows):
        for c in range(n_cols):
            mat[r][c] = cost_matrix[r][c]

    # Row reduction
    for r in range(dim):
        min_r = min(mat[r])
        for c in range(dim):
            mat[r][c] -= min_r

    # Col reduction
    for c in range(dim):
        min_c = min(mat[r][c] for r in range(dim))
        for r in range(dim):
            mat[r][c] -= min_c

    # Augmenting path implementation
    u = [0.0] * (dim + 1)
    v = [0.0] * (dim + 1)
    p = [0] * (dim + 1)
    way = [0] * (dim + 1)

    for i in range(1, dim + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (dim + 1)
        used = [False] * (dim + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, dim + 1):
                if not used[j]:
                    cur = mat[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(dim + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    matches = []
    for j in range(1, dim + 1):
        if p[j] > 0:
            r = p[j] - 1
            c = j - 1
            if r < n_rows and c < n_cols:
                matches.append((r, c))
    return matches
