import uuid
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射

    def load_documents(self) -> List[Document]:
        """加载文档数据"""
        documents = []
        data_path_obj = Path(self.data_path)

        if not data_path_obj.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")

        for md_file in data_path_obj.rglob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            parent_id = str(uuid.uuid4())
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "parent_id": parent_id,
                    "doc_type": "parent"
                }
            )
            documents.append(doc)
        for doc in documents:
            self._enhance_metadata(doc)
        self.documents = documents
        return documents

    def _enhance_metadata(self, doc: Document):
        """增强文档元数据"""
        file_path = Path(doc.metadata.get("source", " "))
        category_mapping = {
            'meat_dish': '荤菜', 'vegetable_dish': '素菜', 'soup': '汤品',
            'dessert': '甜品', 'breakfast': '早餐', 'staple': '主食',
            'aquatic': '水产', 'condiment': '调料', 'drink': '饮品'
        }
        doc.metadata['category'] = '其他'
        for key, value in category_mapping.items():
            if key in file_path.parts:
                doc.metadata['category'] = value
                break
        doc.metadata['dish_name'] = file_path.stem

    def chunk_documents(self) -> List[Document]:
        """将父文档分割成子块（简单按行分割）"""
        self.chunks = []
        for doc in self.documents:
            lines = doc.page_content.split('\n')
            chunk_text = ""
            chunk_id = str(uuid.uuid4())
            for line in lines:
                if line.startswith('## ') and chunk_text:
                    sub_doc = Document(
                        page_content=chunk_text.strip(),
                        metadata={
                            "source": doc.metadata["source"],
                            "parent_id": doc.metadata["parent_id"],
                            "doc_type": "chunk",
                            "chunk_id": chunk_id
                        }
                    )
                    self.chunks.append(sub_doc)
                    chunk_text = line + '\n'
                    chunk_id = str(uuid.uuid4())
                else:
                    chunk_text += line + '\n'
            if chunk_text.strip():
                sub_doc = Document(
                    page_content=chunk_text.strip(),
                    metadata={
                        "source": doc.metadata["source"],
                        "parent_id": doc.metadata["parent_id"],
                        "doc_type": "chunk",
                        "chunk_id": chunk_id
                    }
                )
                self.chunks.append(sub_doc)
        return self.chunks

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """根据子块获取对应的父文档（智能去重）"""
        parent_relevance = {}
        parent_docs_map = {}
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        sorted_parent_ids = sorted(
            parent_relevance.keys(),
            key=lambda x: parent_relevance[x],
            reverse=True
        )

        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        return parent_docs
