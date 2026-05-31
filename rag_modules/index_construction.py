from pathlib import Path
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "text-embedding-v2",
                 index_save_path: str = "./vector_index"):
        self.index_save_path = index_save_path
        self.model_name = model_name
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型"""
        self.embeddings = DashScopeEmbeddings(model=self.model_name)

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """构建FAISS向量索引"""
        if not chunks:
            raise ValueError("文档不能为空")

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        return self.vectorstore

    def save_index(self):
        """保存向量索引到配置的路径"""
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)

    def load_index(self):
        """从配置的路径加载向量索引"""
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            return None

        self.vectorstore = FAISS.load_local(
            self.index_save_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore


