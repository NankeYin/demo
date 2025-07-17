from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_handler import RAGHandler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bank assistant RAG API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 初始化RAG处理器
rag_handler = RAGHandler()

class QueryRequest(BaseModel):
    question: str
    max_results: int = 3

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        logger.info(f"Processing question: {request.question}")
        result = rag_handler.get_answer(
            request.question,
            max_results=request.max_results
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "context": result["context"]
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your question: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "qwen:7b",
        "service": "Bank Assistant RAG API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")