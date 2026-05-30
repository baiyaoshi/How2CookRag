import uuid
from importlib import metadata
from importlib.resources import contents
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射



    """
    rglob("*.md"): 递归查找所有Markdown文件
    parent_id: 为每个父文档分配唯一ID，建立父子关系的关键
    doc_type: 标记为"parent"，便于区分父子文档
    """

    def load_documents(self) -> List[Document]:
        """加载文档数据"""
        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"):
            # 读取文件内容，保持Markdown格式
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # 为每个父文档分配唯一ID
            parent_id = str(uuid.uuid4())
            # 创建Document对象
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(md_file),
                    "parent_id": parent_id,
                    "doc_type": "parent"  # 标记为父文档
                }
            )
            documents.append(doc)
        # 增强文档元数据
        for doc in documents:
            self._enhance_metadata(doc)
        self.documents = documents
        return documents

    #增强元数组
    def _enhance_metadata(self,doc:Document):
        """增强文件元数组"""
        file_path=Path(doc.metadata.get("source"," "))
        # .parts 把路径拆成元组
        #"data/素菜/番茄炒蛋.md"
        #('data', '素菜', '番茄炒蛋.md')
        path_parts = file_path.parts
        # 提取菜品分类
        category_mapping = {
            'meat_dish': '荤菜', 'vegetable_dish': '素菜', 'soup': '汤品',
            'dessert': '甜品', 'breakfast': '早餐', 'staple': '主食',
            'aquatic': '水产', 'condiment': '调料', 'drink': '饮品'
        }
        # 从文件路径推断分类
        doc.metadata['category'] = '其他'
        for key, value in category_mapping.items():
            if key in file_path.parts:
                doc.metadata['category'] = value
                break

        # 提取菜品名称
        doc.metadata['dish_name'] = file_path.stem
        # 分析难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    #分块策略
    def chunk_documents(self)->List[Document]:
        """Markdown结构感知分块"""
        if not self.documents:
            raise ValueError("请先加载文档")
        #使用Markdown标题分割器
        chunks = self._markdown_header_split()
        #为chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 如果没有chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['batch_index'] = i  # 在当前批次中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        return chunks

    #标题分割器
    """
    三级标题分割: 按照#、##、###进行层级分割
    保留标题: 设置strip_headers=False，保留标题信息便于理解上下文
    父子关系: 每个子块都记录其父文档的parent_id
    唯一标识: 每个子块都有独立的child_id
    """

    """
    原文档：西红柿炒鸡蛋的做法.md (父文档)
    ├── 子块1：# 西红柿炒鸡蛋的做法 + 简介 + 难度评级
    ├── 子块2：## 必备原料和工具 + 食材清单
    ├── 子块3：## 计算 + 用量配比公式
    ├── 子块4：## 操作 + 详细制作步骤
    └── 子块5：## 附加内容
    """

    def _markdown_header_split(self) -> List[Document]:
        headers_to_split_on = [
            ("#", "主标题"),  # 菜品名称
            ("##", "二级标题"),  # 必备原料、计算、操作等
            ("###", "三级标题")  # 简易版本、复杂版本等
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )
        all_chunks=[]
        for doc in self.documents:
            md_chunks=markdown_splitter.split_text(doc.page_content)
            # 为每个子块建立与父文档的关系
            parent_id = doc.metadata["parent_id"]
            for i,chunk in enumerate(md_chunks):
                child_id=str(uuid.uuid4())
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "parent_id":parent_id,
                    "doc_type": "child",  # 标记为子文档
                    "chunk_index": i  # 在父文档中的位置
                })
                # 建立父子映射关系
                self.parent_child_map[child_id] = parent_id
            all_chunks.extend(md_chunks)
        return all_chunks

    #智能去重
    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """根据子块获取对应的父文档（智能去重）"""
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}
        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序并构建去重后的父文档列表
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                   key=lambda x: parent_relevance[x], reverse=True)

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        return parent_docs
