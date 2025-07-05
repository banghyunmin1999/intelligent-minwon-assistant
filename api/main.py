from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os

# AI 엔진 모듈 import
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from ..ai_engine import init_ai_engine

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="지능형 농지민원 답변 API",
    description="농지민원 상담을 위한 AI 답변 시스템 API",
    version="1.0.0"
)

# 요청/응답 모델 정의
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.get("/healthcheck")
async def health_check():
    return {"message": "지능형 농지민원 답변 API 서버가 정상적으로 실행 중입니다."}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # AI 엔진 초기화
        ai_engine = init_ai_engine()
        
        # 질문 처리
        answer = await ai_engine.process_question(request.question)
        
        return AnswerResponse(answer=answer)
        
    except Exception as e:
        import traceback
        print("\n=== 오류 발생 ===")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 메시지: {str(e)}")
        print("\n=== 상세 오류 추적 ===")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
