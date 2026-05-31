"""
向量检索的优势：

理解语义相似性，如"简单易做的菜"能匹配到标记为"简单"的菜谱
处理同义词和近义词，如"制作方法"和"做法"、"烹饪步骤"
理解用户意图，如"适合新手"能找到难度较低的菜谱
BM25检索的优势：

精确匹配菜名，如"宫保鸡丁"能准确找到对应菜谱
匹配具体食材，如"土豆丝"、"西红柿"等关键词
处理专业术语，如"爆炒"、"红烧"等烹饪手法
RRF算法能综合两种检索方式的排名信息，既保证了语义理解的准确性，又确保了关键词匹配的精确性。
当然还可以用路由的方式，根据查询类型智能选择使用向量检索还是BM25检索。
这种方法针对性强，能为不同类型的查询选择最优的检索方式；
不足是路由规则的设计和维护比较复杂，边界情况难以处理，而且通常需要调用LLM来判断查询类型，
会增加延迟和成本。
"""



from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document




class RetrievalOptimizationModule:
    """检索优化"""
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    #检索器设置
    def setup_retriever(self):
        """设置向量检索器和BM25检索器"""
        self.vector_retriever=self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":5}
        )
        self.bm25_retriever=BM25Retriever.from_documents(
            self.chunks,
            k=5
        )
    def hybrid_search(self,query:str,top_k:int=3)->List[Document]:
        """混合检索 - 结合向量检索和BM25检索，使用RRF重排"""
        #两种检索结果
        vector_docs=self.vector_retriever._get_relevant_documents(query)
        bm25_docs=self.bm25_retriever._get_relevant_documents(query)

        #RRF重排
        reranked_docs=self._rrf_rerank(vector_docs,bm25_docs)
        return reranked_docs[:top_k]

    def _rrf_rerank(self, vector_results: List[Document], bm25_results: List[Document]):
        """RRF"""
        rrf_scores={}
        k=60 #rrf参数
        #计算向量检索rrf分数
        for rank, doc in enumerate(vector_results):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        # 计算BM25检索的RRF分数
        for rank, doc in enumerate(bm25_results):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
        # 合并所有文档并按RRF分数排序
        all_docs = {id(doc): doc for doc in vector_results + bm25_results}
        sorted_docs = sorted(all_docs.items(),
                             key=lambda x: rrf_scores.get(x[0], 0),
                             reverse=True)

        return [doc for _, doc in sorted_docs]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any],
                                 top_k: int = 5) -> List[Document]:
        """基于元数据过滤的检索"""
        # 先进行向量检索
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k * 3, "filter": filters}  # 扩大检索范围
        )

        results = vector_retriever.invoke(query)
        return results[:top_k]
