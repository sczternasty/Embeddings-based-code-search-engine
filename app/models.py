from typing import List, Optional
from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    query: str
    k: int = 5


class ScoredDocument(BaseModel):
    id: str
    text: str
    score: float
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    query: str
    results: List[ScoredDocument]
