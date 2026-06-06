# UTC-TA — Mô tả Hệ thống

## Tổng quan

**UTC-TA (University of Transport and Communications Teaching Assistant)** là một chatbot RAG (Retrieval-Augmented Generation) được xây dựng bằng **Streamlit + LangGraph**, phục vụ dạy kèm toán học cho sinh viên Đại học Giao thông Vận tải (UTC). Toàn bộ UI và prompt đều bằng **tiếng Việt**.

---

## Cấu trúc thư mục

```
UTC-TA/
├── main.py                          # Entry point
├── .env                             # API keys (gitignored)
├── AGENTS.md                        # Mô tả kiến trúc
├── README.md
├── test.json                        # Log mẫu
├── config/                          # (trống)
├── data/                            # (trống)
├── logs/
│   ├── logging.py                   # Module ghi log JSON
│   ├── chat_logs.json               # Log hội thoại
│   └── result_logs.json             # Log kết quả
├── test/
│   ├── converted_result.md          # Kết quả convert file
│   ├── data_mining.md               # File test mẫu
│   ├── football_law.md              # File test mẫu
│   ├── test_api.py                  # Script test Gemini (ad-hoc)
│   └── test_md.md                   # File test mẫu
└── src/
    ├── frontend/
    │   └── gui.py                   # Giao diện Streamlit
    └── backend/
        ├── graph.py                 # LangGraph state machine
        ├── processing.py            # Convert file → Markdown
        ├── embedding.py             # Embedding backends
        ├── agents/
        │   ├── llm.py               # LLM backends
        │   ├── retriever.py         # SemanticSearch wrapper
        │   ├── chunker.py           # 4 chiến lược chunking
        │   └── prompt.py            # System + Router prompt
        └── store/
            └── embedding_store.py   # Vector store
```

---

## Luồng xử lý chi tiết

### 1. Khởi động (`main.py`)

```python
from src.frontend.gui import Interface
Interface()
```

### 2. Giao diện (`src/frontend/gui.py` — class `Interface`)

**Sidebar:**
- Upload file (PDF, TXT, DOCX), chấp nhận nhiều file
- Nút **"Nạp tài liệu"** → pipeline xử lý:
  1. `file_converter()` → Markdown
  2. `chunk_structure_aware()` → chia nhỏ theo header
  3. `get_embedder()` → embedding vectors
  4. `EmbeddingStore.add_documents()` → lưu vào store
  5. `SemanticSearch(store)` → khởi tạo retriever
- Selectbox chọn model: "Open AI" | "Gemma" | "DeepSeek"

**Chat:**
- Lịch sử chat HTML bubble (iframe + MathJax cho LaTeX)
- Input → LangGraph → fake streaming (split words + `time.sleep(0.05)`)
- Ghi log sau mỗi lượt chat

**Xử lý LaTeX:**
- `_protect_math()` → thay placeholder → `markdown.markdown()` → `_restore_math()`
- MathJax CDN cho render toán học

### 3. LangGraph (`src/backend/graph.py`)

**GraphState:**
- `chat_history`, `context`, `route`, `response`

**3 nodes:**
- **`router`**: nếu chưa upload → `"direct"`; nếu có → gọi LLM router quyết định `"retrieve"` hay `"direct"`
- **`retriever`**: gọi `SemanticSearch.search()` với lịch sử chat → context
- **`generator`**: gọi LLM với SYSTEM_PROMPT + chat_history + context → response

```
START → router ──(direct)──→ generator → END
                └─(retrieve)─→ retriever → generator → END
```

Graph cached bằng `@st.cache_resource`.

### 4. Chunking (`src/backend/agents/chunker.py`)

| Strategy | Mô tả |
|---|---|
| `chunk_basic` | Split theo paragraph, ~500 ký tự/chunk (baseline) |
| `chunk_semantic` | SentenceTransformer → cosine similarity → gộp câu cùng chủ đề |
| `chunk_hierarchical` | Parent-child: child (precision) → parent (context) |
| **`chunk_structure_aware`** | **Production**: parse Markdown headers, mỗi section = 1 chunk (tối đa 1000 ký tự) |

### 5. LLM Backends (`src/backend/agents/llm.py`)

- **`BaseAssistant`** (ABC): `get_response()`, `get_router_response()`
- **`OpenAIAssistant`**: dùng OpenAI API (`OPENAI_MODEL`, `OPENAI_API_KEY`)
- **`GemmaAssistant`**: dùng Google GenAI (`GEMMA_MODEL`, `GOOGLE_API_KEY`), model hardcode `"gemma-4-31b-it"` với `thinking_level="high"`
- **`get_assistant(model)`** (cached): match case → trả về backend tương ứng

### 6. Prompt (`src/backend/agents/prompt.py`)

**`SYSTEM_PROMPT`** (tiếng Việt):
- Role: Tutor UTC — hướng dẫn, không giải đáp án
- Context từ `<context_data>`
- Scaffolding 3 mức: tổng quan → gợi mở → chi tiết
- LaTeX cho mọi công thức

**`ROUTER_PROMPT`**:
- `"retrieve"` — chuyên môn, bài tập, giáo trình
- `"direct"` — chào hỏi, tán gẫu, kiến thức phổ thông

### 7. Embedding (`src/backend/embedding.py`)

| Backend | Mô tả |
|---|---|
| `MockEmbedder` | MD5 → 64 chiều (test) |
| **`LocalEmbedder`** (default) | SentenceTransformers `all-MiniLM-L6-v2` (384 chiều, batch support) |
| `OpenAIEmbedder` | OpenAI `text-embedding-3-small` API |

### 8. Vector Store (`src/backend/store/embedding_store.py`)

- **ChromaDB** (nếu import được) — `collection.query()`
- **In-memory** (fallback) — list of dicts + dot product search

Phương thức: `add_documents()`, `search()`, `search_with_filter()`, `delete_document()`, `get_collection_size()`.

### 9. Processing (`src/backend/processing.py`)

| Loại | Công cụ |
|---|---|
| PDF | `pymupdf4llm` (từng trang, table_strategy="lines") |
| TXT | Đọc UTF-8 trực tiếp |
| DOCX | `markitdown` (không LLM) |

### 10. Logging (`logs/logging.py`)

- `chat_logging()` — append JSON vào `chat_logs.json` (timestamp + route + context + conversation)
- `result_logging()` — append vào `result_logs.json`
- `context_logging()` — ghi đè vào `context_logs.json`

---

## Flow End-to-End

```
1. streamlit run main.py
2. Upload file → bấm "Nạp tài liệu"
3. file_converter() → Markdown → chunk_structure_aware() → EmbeddingStore.add_documents()
4. Nhập câu hỏi
5. LangGraph:
   a. Router → direct / retrieve
   b. Nếu retrieve → SemanticSearch.search() → context
   c. Generator → SYSTEM_PROMPT + context + chat_history → LLM → response
6. Fake streaming (word-by-word, 0.05s/word)
7. Log → chat_logs.json
8. st.rerun()
```

---

## Hạn chế

1. **Fake streaming**: không phải real LLM streaming
2. **Không có test suite**: `test/` chỉ là file mẫu + script ad-hoc
3. **Không có requirements.txt**: dependencies phải tự cài
4. **Graph cached cứng**: cần restart Streamlit để reset
5. **DeepSeek không có implementation**: fallback về OpenAI
6. **Gemma hardcode model name**: `"gemma-4-31b-it"` thay vì dùng biến môi trường
