from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和重排序"""
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.vector_retriever = None
        self.bm25_retriever = None
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5
        )
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """混合检索 - 结合向量检索和BM25检索，使用RRF重排"""
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def _rrf_rerank(self, vector_results: List[Document], bm25_results: List[Document]) -> List[Document]:
        """RRF倒数排名融合算法"""
        rrf_scores = {}
        k = 60
        for rank, doc in enumerate(vector_results):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(bm25_results):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        all_docs = {id(doc): doc for doc in vector_results + bm25_results}
        sorted_docs = sorted(
            all_docs.items(),
                             key=lambda x: rrf_scores.get(x[0], 0),
            reverse=True
        )

        return [doc for _, doc in sorted_docs]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any],
                                 top_k: int = 5) -> List[Document]:
        """基于元数据过滤的检索"""
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k * 3, "filter": filters}
        )

        results = vector_retriever.invoke(query)
        return results[:top_k]
