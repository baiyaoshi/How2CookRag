import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

from config import RAGConfig, DEFAULT_CONFIG
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.index_construction import IndexConstructionModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径和API密钥
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        # 1. 初始化数据准备模块
        logger.info("正在初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        logger.info("正在初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        logger.info("正在初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

    def build_knowledge_base(self):
        """构建知识库"""
        # 1. 加载文档并分割
        logger.info("正在加载文档...")
        self.data_module.load_documents()
        chunks = self.data_module.chunk_documents()
        logger.info(f"文档分割完成，共 {len(chunks)} 个子块")

        # 2. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            logger.info("已加载已有向量索引")
        else:
            # 构建新索引
            logger.info("正在构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            self.index_module.save_index()
            logger.info("向量索引构建并保存完成")

        # 3. 初始化检索优化模块
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
        logger.info("检索优化模块初始化完成")

    def ask_question(self, question: str, stream: bool = False) -> str:
        """回答用户问题"""
        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        logger.info(f"查询路由结果: {route_type}")

        # 2. 智能查询重写
        if route_type == 'list':
            rewritten_query = question
        else:
            rewritten_query = self.generation_module.query_rewrite(question)

        # 3. 检索相关子块
        relevant_chunks = self.retrieval_module.hybrid_search(
            rewritten_query, top_k=self.config.top_k
        )

        # 4. 根据路由类型选择回答方式
        if route_type == 'list':
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            if route_type == "detail":
                return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                return self.generation_module.generate_basic_answer(question, relevant_docs)

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        self.initialize_system()
        self.build_knowledge_base()

        while True:
            user_input = input("\n您的问题: ").strip()
            if user_input.lower() in ['退出', 'quit', 'exit']:
                break

            answer = self.ask_question(user_input, stream=False)
            print(f"\n回答:\n{answer}\n")


if __name__ == "__main__":
    system = RecipeRAGSystem()
    system.run_interactive()
