"""
BM25 retriever using Elasticsearch with fallback to pure Python BM25
"""

from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class SimpleBM25:
    """Simple Python-based BM25 implementation for fallback (when Elasticsearch unavailable)"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_index = []
        self.idf = {}
        self.avgdl = 0
        self.corpus_size = 0
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 retrieval"""
        import math
        from collections import Counter
        
        self.doc_index = []
        term_doc_freq = {}
        
        for doc in documents:
            doc_id = str(doc.get("doc_id"))
            text = (doc.get("title", "") + " " + doc.get("abstract", "")).lower().split()
            self.doc_index.append({
                "doc_id": doc_id,
                "terms": text,
                "length": len(text),
                "doc": doc
            })
            
            for term in set(text):
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1
        
        self.corpus_size = len(self.doc_index)
        self.avgdl = sum(d["length"] for d in self.doc_index) / max(1, self.corpus_size)
        
        # Calculate IDF
        for term, df in term_doc_freq.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25 algorithm"""
        from collections import Counter
        
        query_terms = query.lower().split()
        scores = {}
        
        for doc in self.doc_index:
            doc_id = doc["doc_id"]
            score = 0
            term_freq = Counter(doc["terms"])
            
            for term in query_terms:
                if term in term_freq:
                    idf = self.idf.get(term, 0)
                    tf = term_freq[term]
                    numerator = idf * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc["length"] / self.avgdl))
                    score += numerator / denominator
            
            if score > 0:
                scores[doc_id] = {"score": score, "doc": doc["doc"]}
        
        # Return top-k results
        results = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]
        return [
            {"doc_id": doc_id, "score": data["score"], "source": data["doc"]}
            for doc_id, data in results
        ]


class BM25Retriever:
    """BM25 lexical retriever using Elasticsearch with Python fallback"""
    
    def __init__(self, host: str = "localhost", port: int = 9200, index_name: str = "medical_docs", k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            index_name: Name of the index
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        self.k1 = k1
        self.b = b
        self.es = None
        self.fallback_bm25 = SimpleBM25(k1=k1, b=b)
        self._connect()
    
    def _connect(self):
        """Connect to Elasticsearch with a short timeout so unavailability fails fast."""
        try:
            self.es = Elasticsearch(
                [f"http://{self.host}:{self.port}"],
                request_timeout=60,
                max_retries=1,
                retry_on_timeout=False,
            )
            if not self.es.ping(request_timeout=5):
                print(f"Warning: Could not connect to Elasticsearch at {self.host}:{self.port}")
                self.es = None
        except Exception as e:
            print(f"Warning: Elasticsearch connection failed: {e}")
            self.es = None
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        entities: Optional[List[Dict[str, Any]]] = None,
        entity_boost: float = 2.0,
        max_entities: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of search results with doc_id and score
        """
        # Use Elasticsearch if available, otherwise fallback
        if self.es is not None:
            return self._search_elasticsearch(query, top_k, entities, entity_boost, max_entities)
        else:
            return self._search_fallback(query, top_k)
    
    def _search_elasticsearch(
        self,
        query: str,
        top_k: int,
        entities: Optional[List[Dict[str, Any]]] = None,
        entity_boost: float = 2.0,
        max_entities: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search using Elasticsearch BM25"""
        if self.es is None:
            return []
        
        # Build must clauses for main query
        must_clauses = [
            {
                "match": {
                    "title": {
                        "query": query,
                        "boost": 3.0,
                        "operator": "or"
                    }
                }
            },
            {
                "match": {
                    "abstract": {
                        "query": query,
                        "boost": 2.0,
                        "operator": "or"
                    }
                }
            }
        ]
        
        # Build should clauses for entity boosts
        should_clauses: List[Dict[str, Any]] = []
        if entities:
            for ent in entities[: max(0, max_entities)]:
                ent_text = (ent.get("text") if isinstance(ent, dict) else str(ent)) or ""
                ent_text = ent_text.strip()
                if not ent_text:
                    continue
                should_clauses.append(
                    {
                        "match": {
                            "title": {
                                "query": ent_text,
                                "boost": float(entity_boost) * 3.0,
                                "operator": "and"
                            }
                        }
                    }
                )
                should_clauses.append(
                    {
                        "match": {
                            "abstract": {
                                "query": ent_text,
                                "boost": float(entity_boost) * 2.0,
                                "operator": "and"
                            }
                        }
                    }
                )

        search_body = {
            "query": {
                "bool": {
                    "should": must_clauses + should_clauses,
                    "minimum_should_match": 1,
                }
            },
            "size": top_k,
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response["hits"]["hits"]:
                src = hit.get("_source", {}) or {}
                # Prefer external PMID from source metadata if available; fallback to ES _id
                try:
                    pmid = src.get("metadata", {}).get("pmid")
                except Exception:
                    pmid = None
                doc_id_val = pmid if pmid else hit.get("_id")
                score = hit.get("_score", 0.0)
                item = {
                    "doc_id": str(doc_id_val),
                    "score": float(score),
                    "source": src
                }
                results.append(item)
            
            return results
        
        except Exception as e:
            print(f"Elasticsearch search error: {e}")
            return []
    
    def _search_fallback(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback search using Python BM25 implementation"""
        return self.fallback_bm25.search(query, top_k)
    
    def index_exists(self) -> bool:
        """Check if the index exists"""
        if self.es is None:
            return False
        try:
            return self.es.indices.exists(index=self.index_name)
        except:
            return False

    def create_index(self):
        """Create index with basic BM25-friendly mappings if it doesn't exist"""
        if self.es is None:
            return False
        if self.index_exists():
            return True
        settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "english_custom": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text", "analyzer": "english"},
                    "abstract": {"type": "text", "analyzer": "english"},
                    "pub_date": {"type": "date", "ignore_malformed": True},
                    "metadata": {"type": "object"}
                }
            }
        }
        try:
            self.es.indices.create(index=self.index_name, body=settings)
            return True
        except Exception:
            # If race condition or already exists
            return self.index_exists()

    def reset_index(self) -> bool:
        """Delete the existing index (if any) and recreate it fresh for this run."""
        if self.es is None:
            return False
        try:
            if self.index_exists():
                self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        except Exception:
            # Best-effort delete; proceed to create
            pass
        return self.create_index()

    def index_documents(self, documents: List[Dict[str, Any]], index_fallback: bool = True):
        """Bulk index documents into Elasticsearch (or fallback)"""
        if not documents:
            return False
        
        # Optionally index in fallback BM25 for hybrid retrieval
        if index_fallback:
            self.fallback_bm25.index_documents(documents)
        
        # Try to index in Elasticsearch if available
        if self.es is None:
            if not index_fallback:
                print("Warning: Elasticsearch not available and fallback BM25 disabled")
                return False
            print("Note: Using Python-based BM25 (Elasticsearch not available)")
            return True
        
        self.create_index()
        actions = []
        for doc in documents:
            doc_id = str(doc.get("doc_id"))
            source = {
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", ""),
                "pub_date": doc.get("pub_date"),
                "metadata": doc.get("metadata", {})
            }
            actions.append({
                "_index": self.index_name,
                "_id": doc_id,
                "_source": source
            })
        try:
            bulk(self.es, actions)
            return True
        except Exception as e:
            print(f"Elasticsearch bulk index error: {e}")
            return False
