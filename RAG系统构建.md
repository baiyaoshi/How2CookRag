# 食谱 RAG 系统构建详解

> 本文档面向 RAG 新手，带你一步步理解这个"食谱问答系统"是如何构建出来的。

---

## 目录

1. [什么是 RAG？](#1-什么是rag)
2. [项目结构总览](#2-项目结构总览)
3. [从零搭建的每一步](#3-从零搭建的每一步)
4. [核心概念通俗解释](#4-核心概念通俗解释)
5. [数据如何流动](#5-数据如何流动)
6. [常见问题 FAQ](#6-常见问题-faq)

---

## 1. 什么是 RAG？

### 一句话解释

**RAG = 检索 + 生成**

你把一堆菜谱文档给系统，问"番茄炒蛋怎么做"，系统会：
1. **检索**：从菜谱中找出跟"番茄炒蛋"最相关的内容
2. **生成**：把找到的内容喂给大语言模型（如通义千问），让它整理成完整的回答

### 没有 RAG 会怎样？

| 场景 | 没有 RAG | 有 RAG |
|------|---------|--------|
| 问"番茄炒蛋怎么做" | 大模型凭记忆回答，可能不准确 | 先从你的菜谱里找到原文，再生成回答 |
| 问"推荐几个素菜" | 大模型不知道你有什么菜 | 根据你的菜谱数据推荐，更靠谱 |
| 自定义数据 | 大模型没学过你的私有数据 | 可以把任何文档变成知识库 |

---

## 2. 项目结构总览

```
How2CookRag/
├── main.py                          # 入口文件（启动系统）
├── config.py                        # 配置文件
├── pyproject.toml                   # 项目依赖
├── .env                             # API密钥等环境变量
├── data/recipes/                    # 菜谱数据（Markdown文件）
│   ├── vegetable_dish/番茄炒蛋.md
│   ├── meat_dish/红烧肉.md
│   ├── soup/番茄蛋汤.md
│   ├── dessert/提拉米苏.md
│   └── breakfast/蛋炒饭.md
├── rag_modules/                     # 核心模块
│   ├── __init__.py                  # 模块导出
│   ├── data_preparation.py          # 数据准备模块
│   ├── index_construction.py        # 索引构建模块
│   ├── retrieval_optimization.py    # 检索优化模块
│   └── generation_integration.py    # 生成集成模块
└── vector_index/                    # 保存到本地的向量索引
```

### 四个核心模块的关系

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────┐
│  RecipeRAGSystem (main.py)                   │
│  负责串联所有模块                             │
└────┬────┬────┬────┬──────────────────────────┘
     │    │    │    │
     ▼    ▼    ▼    ▼
 数据准备  索引构建  检索优化  生成集成
```

---

## 3. 从零搭建的每一步

### 第1步：准备数据（Markdown 文件）

在 `data/recipes/` 目录下放菜谱，每个菜一个 `.md` 文件，按分类放在子目录中：

```
data/recipes/
├── vegetable_dish/      ← 目录名 = 分类英文名
│   └── 番茄炒蛋.md      ← 文件名 = 菜名.md
├── meat_dish/
│   └── 红烧肉.md
├── soup/
│   └── 番茄蛋汤.md
└── ...
```

每个 `.md` 文件的结构：

```markdown
# 番茄炒蛋

## 菜品介绍
...

## 所需食材
...

## 制作步骤

### 步骤一：准备
...

### 步骤二：炒鸡蛋
...

## 制作技巧
...
```

> **为什么这么组织？** 因为后面代码会从目录名自动推断菜品分类（如 `vegetable_dish`→素菜），从文件名提取菜名。

---

### 第2步：配置文件（config.py）

```python
@dataclass
class RAGConfig:
    data_path: str = "./data/recipes"       # 菜谱路径
    index_save_path: str = "./vector_index" # 索引保存路径
    embedding_model: str = "text-embedding-v2"  # 向量模型
    llm_model: str = "qwen-plus"              # 大模型
    top_k: int = 3                            # 返回几个结果
    temperature: float = 0.1                  # 生成随机性
    max_tokens: int = 2048                    # 最大输出长度
```

> **`@dataclass`** 是 Python 的"数据类"装饰器，自动帮你生成 `__init__` 方法，省去写构造函数的麻烦。

---

### 第3步：数据准备模块（data_preparation.py）

这个模块做三件事：**加载 → 增强 → 分割**

#### 3.1 加载文档（load_documents）

```python
for md_file in data_path_obj.rglob("*.md"):   # 递归找所有 .md 文件
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()                      # 读取文件内容
    parent_id = str(uuid.uuid4())              # 生成唯一ID
    doc = Document(                             # 创建LangChain的文档对象
        page_content=content,                   # 正文就是文件内容
        metadata={
            "source": str(md_file),             # 来源路径
            "parent_id": parent_id,             # 唯一ID（后续关联用）
            "doc_type": "parent"                # 标记为父文档
        }
    )
    documents.append(doc)
```

> **`rglob("*.md")`** 中的 `r` 表示 recursive（递归），会搜索所有子目录。

> **`uuid.uuid4()`** 生成一个全球唯一的 ID，用来标识每个菜谱。

#### 3.2 增强元数据（_enhance_metadata）

```python
def _enhance_metadata(self, doc: Document):
    file_path = Path(doc.metadata.get("source"))
    # 从文件路径推断分类
    category_mapping = {
        'meat_dish': '荤菜', 'vegetable_dish': '素菜', ...
    }
    # 检查路径的每一级目录名是否匹配分类
    for key, value in category_mapping.items():
        if key in file_path.parts:              # parts = ('data', 'vegetable_dish', '番茄炒蛋.md')
            doc.metadata['category'] = value
            break
    doc.metadata['dish_name'] = file_path.stem   # stem = '番茄炒蛋'（不带后缀）
```

> **`file_path.parts`** 把路径拆成元组：`('data', 'vegetable_dish', '番茄炒蛋.md')`

> **`file_path.stem`** 获取不带后缀的文件名：`番茄炒蛋`

#### 3.3 分割文档（chunk_documents）

把一个大文档按 `##` 标题切分成多个小块：

```python
# 原始文档内容（一个文件）
番茄炒蛋.md
├── # 番茄炒蛋
├── ## 所需食材        → 切成一个小块
├── ## 制作步骤
│   ├── ### 步骤一     → 切成一个小块
│   ├── ### 步骤二     → 切成一个小块
└── ## 制作技巧        → 切成一个小块
```

```python
for line in lines:
    if line.startswith('## ') and chunk_text:  # 遇到 ## 标题就切分
        sub_doc = Document(
            page_content=chunk_text.strip(),
            metadata={
                "source": doc.metadata["source"],
                "parent_id": doc.metadata["parent_id"],  # 关联到父文档
                "doc_type": "chunk"
            }
        )
        self.chunks.append(sub_doc)
```

> **为什么切分？** 一个大文档直接传给大模型会超长（token限制），而且搜索时按小块匹配更精准。

#### 3.4 获取父文档（get_parent_documents）

搜索时匹配到的是小块（chunks），但最终要给用户看完整菜谱。这个方法把小块还原成完整菜谱，并按匹配数量排序去重：

```python
def get_parent_documents(self, child_chunks):
    parent_relevance = {}  # 统计每个菜谱被匹配了多少次
    parent_docs_map = {}   # 缓存菜谱对象
    
    for chunk in child_chunks:
        parent_id = chunk.metadata.get("parent_id")
        parent_relevance[parent_id] += 1  # 计数+1
        # 缓存菜谱，避免下次重复查找
        if parent_id not in parent_docs_map:
            parent_docs_map[parent_id] = 找到的完整菜谱
    
    # 按匹配次数从多到少排序
    sorted_ids = sorted(parent_relevance.keys(), 
                        key=lambda x: parent_relevance[x], reverse=True)
    
    return [parent_docs_map[id] for id in sorted_ids]
```

> **为什么去重？** 搜索"番茄炒蛋怎么做"可能匹配到3个小块（食材、步骤、技巧），但它们都属于同一个菜谱，只需返回一个完整的菜谱即可。

---

### 第4步：索引构建模块（index_construction.py）

把文本转换成向量（数学上的"坐标"），存入 FAISS 数据库。

#### 4.1 初始化嵌入模型

```python
def setup_embeddings(self):
    self.embeddings = DashScopeEmbeddings(model=self.model_name)
```

> **嵌入模型是什么？** 它把"番茄炒蛋"这样的文字转换成一串数字（如 `[0.12, -0.34, 0.56, ...]`），
> 让计算机能"理解"语义。语义相近的文本，向量距离也近。

#### 4.2 构建向量索引

```python
def build_vector_index(self, chunks):
    texts = [chunk.page_content for chunk in chunks]     # 取出所有文本
    metadatas = [chunk.metadata for chunk in chunks]     # 取出所有元数据
    
    self.vectorstore = FAISS.from_texts(
        texts=texts,           # 文本列表
        embedding=self.embeddings,  # 嵌入模型
        metadatas=metadatas    # 元数据（存着parent_id等）
    )
```

> **FAISS** 是 Facebook 开源的向量搜索引擎，用来快速找"最相似的向量"。

> 索引构建的完整过程：
> ```
> "番茄炒蛋" → [0.12, -0.34, ...]  ← 向量1
> "所需食材" → [0.45, 0.12, ...]   ← 向量2
> "制作步骤" → [-0.21, 0.67, ...]  ← 向量3
>                          ↓
>                  全部存入 FAISS
> ```

#### 4.3 保存/加载索引

```python
def save_index(self):
    Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
    self.vectorstore.save_local(self.index_save_path)  # 存到磁盘

def load_index(self):
    self.vectorstore = FAISS.load_local(
        self.index_save_path,
        self.embeddings,
        allow_dangerous_deserialization=True  # 允许反序列化
    )
```

> **为什么保存索引？** 构建索引很慢（需要调用 API 把每个文本转成向量），第一次构建后保存到磁盘，下次启动直接加载，节省时间和费用。

---

### 第5步：检索优化模块（retrieval_optimization.py）

用两种方式检索，然后融合结果。

#### 5.1 混合检索（向量检索 + BM25）

```python
def setup_retrievers(self):
    # 检索器1：向量检索（理解语义）
    self.vector_retriever = self.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # 返回前5个
    )
    # 检索器2：BM25（关键词匹配）
    self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)

def hybrid_search(self, query, top_k=3):
    vector_docs = self.vector_retriever.invoke(query)  # 向量检索结果
    bm25_docs = self.bm25_retriever.invoke(query)       # BM25检索结果
    
    reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
    return reranked_docs[:top_k]
```

| 检索方式 | 优点 | 例子 |
|---------|------|------|
| **向量检索** | 理解语义，能匹配同义词 | "简单易做的菜" → 能找到标记为"简单"的菜谱 |
| **BM25** | 精准匹配关键词 | "宫保鸡丁" → 精确找到宫保鸡丁的菜谱 |

#### 5.2 RRF 融合排序

```python
def _rrf_rerank(self, vector_results, bm25_results):
    rrf_scores = {}
    k = 60
    
    # 给向量检索的结果打分：排名越靠前分数越高
    for rank, doc in enumerate(vector_results):
        rrf_scores[id(doc)] += 1 / (k + rank + 1)
    
    # 给BM25的结果打分，同一个文档分数累加
    for rank, doc in enumerate(bm25_results):
        rrf_scores[id(doc)] += 1 / (k + rank + 1)
    
    # 按总分从高到低排序
    ...
```

> **RRF 核心思想**：一个文档在两个检索结果中都排名靠前，它的总排名就应该更高。
> 比如"番茄炒蛋"在向量检索排第2（得1/61分），在BM25排第1（得1/61分），总分为 2/61，
> 而"番茄牛腩"只在向量检索排第3（得1/63分），总分只有 1/63，所以"番茄炒蛋"排前面。

> **为什么用 `id(doc)` 而不是 `doc.metadata.get('parent_id')`？**
> `id(doc)` 是 Python 对象的内存地址，每个 Document 对象唯一，可以区分不同的 chunks。
> 而 `parent_id` 是菜谱级别的ID，多个 chunks 共享同一个 parent_id。

#### 5.3 元数据过滤搜索

```python
def metadata_filtered_search(self, query, filters, top_k=5):
    # 只搜索符合过滤条件的文档
    vector_retriever = self.vectorstore.as_retriever(
        search_kwargs={"k": top_k * 3, "filter": filters}
    )
    results = vector_retriever.invoke(query)
    return results[:top_k]
```

> **使用场景**：用户说"推荐几个素菜"，可以设置 `filters={"category": "素菜"}`，只搜素菜分类。

---

### 第6步：生成集成模块（generation_integration.py）

负责与大模型交互，生成最终回答。

#### 6.1 初始化大模型

```python
def setup_llm(self):
    self.llm = ChatOpenAI(
        model=self.model_name,
        api_key=os.getenv("DASHSCOPE_API_KEY"),   # 从环境变量读取
        base_url=os.getenv("DASHSCOPE_BASE_URL"),  # 通义千问的API地址
        temperature=self.temperature,  # 0.1 = 低随机性，回答更稳定
        max_tokens=self.max_tokens     # 最大输出长度
    )
```

> **ChatOpenAI** 是 LangChain 的通用 OpenAI 接口封装。虽然叫 OpenAI，但可以用来调用任何兼容 OpenAI API 的服务（包括通义千问）。

> **为什么用环境变量存 API Key？** 安全！API Key 不能写在代码里，否则上传到 GitHub 就泄露了。

#### 6.2 查询路由（query_router）

判断用户问题属于哪种类型：

```python
def query_router(self, query):
    prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐
   例如：推荐几个素菜、有什么川菜

2. 'detail' - 用户想要具体的制作方法
   例如：宫保鸡丁怎么做、制作步骤

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧

请只返回分类结果：list、detail 或 general

用户问题: {query}
分类结果:""")

    chain = (
        {"query": RunnablePassthrough()}  # 输入透传
        | prompt                          # 填入提示词模板
        | self.llm                        # 调用大模型
        | StrOutputParser()              # 转成字符串
    )
    return chain.invoke(query).strip()
```

> **LCEL（LangChain Expression Language）** 的 `|` 语法，数据从左到右流动：
> ```
> 用户问题 → 包装成字典 → 填入提示词 → 发给大模型 → 提取文本
> ```

#### 6.3 查询重写（query_rewrite）

把模糊的问题改得更清晰：

| 原始问题 | 重写后 |
|---------|--------|
| "做菜" | "简单易做的家常菜谱" |
| "推荐个菜" | "简单家常菜推荐" |
| "川菜" | "经典川菜菜谱" |
| "宫保鸡丁怎么做" | ❌ 不重写（已经明确了） |

#### 6.4 三种回答模式

| 模式 | 触发条件 | 返回内容 |
|------|---------|---------|
| **list** | "推荐几个菜" | 简单的菜品名称列表 |
| **detail** | "番茄炒蛋怎么做" | 分步骤的详细指导（带食材、步骤、技巧） |
| **general** | "什么是川菜" | 常规回答 |

#### 6.5 构建上下文

```python
def _build_context(self, docs, max_length=2000):
    # 把检索到的文档拼接成一段文本
    # 加上元数据信息如菜名、分类等
    # 限制总长度不超过 max_length，防止超出大模型上下文窗口
```

---

### 第7步：主入口（main.py）

把所有模块串联起来。

#### 7.1 初始化流程

```python
class RecipeRAGSystem:
    def __init__(self, config):
        # 1. 检查数据路径是否存在
        # 2. 检查API Key是否设置
        
    def initialize_system(self):
        # 1. 创建数据准备模块
        self.data_module = DataPreparationModule(config.data_path)
        # 2. 创建索引构建模块
        self.index_module = IndexConstructionModule(...)
        # 3. 创建生成集成模块
        self.generation_module = GenerationIntegrationModule(...)
```

#### 7.2 构建知识库

```python
def build_knowledge_base(self):
    # 1. 加载文档并分割成小块
    self.data_module.load_documents()
    chunks = self.data_module.chunk_documents()
    
    # 2. 尝试加载已有索引（如果之前构建过）
    vectorstore = self.index_module.load_index()
    if vectorstore is None:
        # 3. 没有就新建索引
        vectorstore = self.index_module.build_vector_index(chunks)
        self.index_module.save_index()  # 保存到磁盘
    
    # 4. 初始化检索优化模块
    self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
```

#### 7.3 回答问题的完整流程

```python
def ask_question(self, question):
    # 第1步：判断问题类型（列表/详细/一般）
    route_type = self.generation_module.query_router(question)
    
    # 第2步：模糊问题重写（列表类型不重写）
    if route_type != 'list':
        rewritten_query = self.generation_module.query_rewrite(question)
    
    # 第3步：混合检索（向量+BM25）
    relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=3)
    
    # 第4步：还原成完整菜谱，按匹配次数排序
    relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
    
    # 第5步：根据问题类型选择回答方式
    if route_type == 'list':
        return self.generation_module.generate_list_answer(question, relevant_docs)
    elif route_type == 'detail':
        return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
    else:
        return self.generation_module.generate_basic_answer(question, relevant_docs)
```

---

## 4. 核心概念通俗解释

### Document（文档对象）

可以把 `Document` 想象成一个**带标签的文件袋**：

```
Document {
    page_content: "番茄炒蛋的做法...",  ← 袋子里的内容
    metadata: {                          ← 袋子外面的标签
        "source": "data/.../番茄炒蛋.md",
        "parent_id": "abc-123-...",
        "category": "素菜",
        "dish_name": "番茄炒蛋"
    }
}
```

### 父文档 vs 子文档（Parent vs Chunk）

```
父文档（完整菜谱）               子文档（小块）
┌──────────────────────┐     ┌──────────────┐
│ # 番茄炒蛋           │     │ ## 所需食材   │ ← chunk1
│ ## 所需食材          │  →  ├──────────────┤
│ 番茄2个、鸡蛋3个...  │     │ ## 制作步骤   │ ← chunk2
│ ## 制作步骤...       │     ├──────────────┤
│ ## 制作技巧...       │     │ ## 制作技巧   │ ← chunk3
└──────────────────────┘     └──────────────┘
     parent_id: A                 parent_id: A（关联到同一个父文档）
```

### 嵌入向量（Embedding）

```
"番茄炒蛋"  →  嵌入模型  →  [0.12, -0.34, 0.56, 0.78, -0.23, ...]
"红烧肉"    →  嵌入模型  →  [-0.45, 0.67, 0.12, -0.89, 0.34, ...]
                               ↑ 两串数字越接近，语义越相似
```

### RRF 分数计算

```
向量检索排名： [番茄炒蛋(第1), 红烧肉(第2), 番茄蛋汤(第3)]
BM25排名：     [番茄蛋汤(第1), 番茄炒蛋(第2), 蛋炒饭(第3)]

分数计算（k=60）：
番茄炒蛋：1/61(向量第1) + 1/62(BM25第2) = 0.0325  ← 总分最高
番茄蛋汤：1/63(向量第3) + 1/61(BM25第1) = 0.0322
红烧肉：  1/62(向量第2) + 0(不在BM25)    = 0.0161
蛋炒饭：  0(不在向量)   + 1/63(BM25第3)  = 0.0159
```

---

## 5. 数据如何流动

### 构建阶段

```
硬盘上的.md文件
    │
    ▼
DataPreparationModule.load_documents()
读取所有 .md 文件，创建 Document 对象
    │
    ▼
DataPreparationModule._enhance_metadata()
从路径推断分类、菜名
    │
    ▼
DataPreparationModule.chunk_documents()
按 ## 标题切分成小块
    │
    ▼
IndexConstructionModule.build_vector_index()
用嵌入模型转成向量，存入 FAISS
    │
    ▼
IndexConstructionModule.save_index()
保存到磁盘 vector_index/ 目录
```

### 问答阶段

```
用户输入："番茄炒蛋怎么做"
    │
    ▼
generation_module.query_router("番茄炒蛋怎么做")
    │ 判断为 'detail'（详细模式）
    ▼
generation_module.query_rewrite("番茄炒蛋怎么做")
    │ 已经是具体问题，不重写
    ▼
retrieval_module.hybrid_search("番茄炒蛋怎么做", top_k=3)
    │ 向量检索 + BM25 + RRF融合
    ▼
获取到 3 个相关的 chunks（小块）
    │
    ▼
data_module.get_parent_documents(chunks)
    │ 还原成完整菜谱，按匹配次数排序
    ▼
generation_module.generate_step_by_step_answer(
    "番茄炒蛋怎么做", 
    [番茄炒蛋完整菜谱]
)
    │ 大模型整理成步骤清晰的回答
    ▼
输出："番茄炒蛋的做法如下：\n\n🥘 菜品介绍...\n🛒 所需食材...\n👨‍🍳 制作步骤..."
```

---

## 6. 常见问题 FAQ

### Q: 为什么需要切分成小块（chunking）？

**A:** 三个原因：
1. **搜索更精准**：搜索"番茄"时，匹配到"食材"小块比匹配整个文档更准确
2. **节省费用**：大模型按 token 收费，传整个文档比传小块贵得多
3. **突破长度限制**：大模型有上下文窗口限制（如 8k tokens），长文档可能超限

### Q: 为什么要用两种检索方式（向量+BM25）？

**A:** 两种方式互补：

| 场景 | 向量检索 | BM25 |
|------|---------|------|
| "简单易做的菜"（语义匹配） | ✅ 能匹配"简单"菜谱 | ❌ 找不到关键词"简单易做" |
| "宫保鸡丁"（精确匹配） | ✅ 也能匹配 | ✅ 更精准 |
| "适合新手的菜" | ✅ 理解"新手"=简单 | ❌ 不命中 |

### Q: `向量索引` 和 `向量数据库` 有什么区别？

**A:** 在这个项目中，FAISS 是**向量索引**（存内存里），`save_index()` 存到磁盘，`load_index()` 加载回来。向量数据库（如 Milvus、Weaviate）是更完整的服务，支持分布式、实时增删改等。FAISS 对小型项目够用。

### Q: 第一次运行和后续运行有什么区别？

**第一次运行：**
1. 没有已有索引 → 调用 `build_vector_index()` 构建
2. 调用 `save_index()` 保存到磁盘
3. 耗时较长（需要调用嵌入 API）

**后续运行：**
1. `load_index()` 从磁盘加载已有索引
2. 跳过构建步骤，直接使用
3. 启动快，节省 API 费用

### Q: metadata 具体有哪些字段，有什么用？

```python
metadata = {
    "source": "data/recipes/vegetable_dish/番茄炒蛋.md",  # 文件来源
    "parent_id": "a1b2c3d4-...",    # 唯一ID，父子文档关联
    "doc_type": "parent",           # parent(父文档) 或 chunk(子块)
    "chunk_id": "e5f6g7h8-...",    # 子块自己的ID（仅子块有）
    "category": "素菜",             # 菜品分类
    "dish_name": "番茄炒蛋"         # 菜品名称
}
```

---

## 总结：RAG 系统的核心流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   数据准备    │ ──→ │   索引构建    │ ──→ │   检索优化    │
│  加载.md文件  │     │ 文本→向量→FAISS│     │ 向量+BM25+RRF │
│  增强元数据   │     │ 保存/加载索引  │     │              │
│  分割成小块   │     │              │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   最终回答    │ ←── │   生成集成    │ ←── │   相关文档    │
│  显示给用户   │     │ 查询路由+重写  │     │  get_parent  │
│              │     │ 三种回答模式   │     │  去重排序    │
└──────────────┘     └──────────────┘     └──────────────┘
```

如果你看懂了这篇文档，恭喜你！你已经理解了 RAG 的完整工作流程。
