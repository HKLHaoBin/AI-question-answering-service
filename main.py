import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from openai import OpenAI
import uvicorn

# Load variables defined in .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIMPLE_TOKEN = os.getenv("AI_SIMPLE_TOKEN", "change-this-token")

app = FastAPI(title="AI Question Helper", version="0.1.0")


class QuestionRequest(BaseModel):
    title: str = Field(..., description="题目标题")
    options: Optional[List[str]] = Field(
        default=None,
        description="选项列表，例如 ['A. xxx', 'B. xxx']",
    )
    type: Optional[str] = Field(
        default="single",
        description="题目类型：single/multiple/judgement/completion 等",
    )
    token: Optional[str] = Field(
        default=None,
        description="简单鉴权用，与服务端 AI_SIMPLE_TOKEN 一致即可",
    )


class AiAnswerData(BaseModel):
    question: str
    answer: str
    analysis: str
    ai: bool = True


class AiAnswerResponse(BaseModel):
    code: int
    msg: str = "ok"
    data: Optional[AiAnswerData] = None


def build_prompt(title: str, options: Optional[List[str]], q_type: Optional[str]) -> str:
    type_map = {
        "single": "单选题",
        "multiple": "多选题",
        "judgement": "判断题",
        "completion": "填空题",
    }
    type_text = type_map.get(q_type or "", "未知类型题目")

    if options:
        option_text = "\n".join(o.strip() for o in options)
    else:
        option_text = "（本题无选项，可能是简答或填空题）"

    prompt = f"""
请根据给出的题目和选项，推理出最合理的答案，并给出详细解析。
题目类型：{type_text}

题目：
{title}

选项：
{option_text}

请严格按下面 JSON 格式返回（不要输出多余文字，不要加注释）：
{{
  "answer": "例如 A 或 A#C（多选用#分隔，主观题可以用完整文字答案）",
  "analysis": "详细解析，说明思路和知识点，仅用于学习和复习"
}}
    """.strip()

    return prompt


def parse_model_json(content: str) -> dict:
    """
    模型有时会加 ```json ``` 代码块，这里做个简单清洗再 json.loads
    """
    text = content.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "answer": "",
            "analysis": text,
        }


def parse_options_query(raw: Optional[str]) -> Optional[List[str]]:
    """
    将查询参数中的选项字符串解析为列表。
    支持 JSON 数组、单个字符串以及按常见分隔符拆分。
    """
    if raw is None:
        return None

    text = raw.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        cleaned = [str(item).strip() for item in parsed if str(item).strip()]
        return cleaned or None
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            return None

    separators = ["\n", "|||", "||", "|", "#", ";", "；", ",", "，"]
    for sep in separators:
        if sep in text:
            items = [part.strip() for part in text.split(sep)]
            options = [item for item in items if item]
            if options:
                return options

    return [text]


def process_ai_answer(req: QuestionRequest) -> AiAnswerResponse:
    if SIMPLE_TOKEN and req.token != SIMPLE_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")

    if not req.title:
        raise HTTPException(status_code=400, detail="title is required")

    prompt = build_prompt(req.title, req.options, req.type)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的 AI 助教。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    content = completion.choices[0].message.content or ""
    parsed = parse_model_json(content)

    answer = str(parsed.get("answer", "")).strip()
    analysis = str(parsed.get("analysis", "")).strip()

    data = AiAnswerData(
        question=req.title,
        answer=answer,
        analysis=analysis,
        ai=True,
    )

    return AiAnswerResponse(code=1, msg="ok", data=data)


@app.post("/ai-answer", response_model=AiAnswerResponse)
async def ai_answer(req: QuestionRequest) -> AiAnswerResponse:
    return process_ai_answer(req)


@app.get("/ai-answer", response_model=AiAnswerResponse)
async def ai_answer_by_query(
    title: str = Query(..., description="题目标题"),
    options: Optional[str] = Query(
        default=None,
        description="选项，支持 JSON 数组或使用换行/分隔符分割的字符串",
    ),
    question_type: Optional[str] = Query(
        default="single",
        alias="type",
        description="题目类型：single/multiple/judgement/completion 等",
    ),
    token: Optional[str] = Query(
        default=None,
        description="简单鉴权 token，与环境变量 AI_SIMPLE_TOKEN 保持一致",
    ),
) -> AiAnswerResponse:
    parsed_options = parse_options_query(options)
    req = QuestionRequest(
        title=title,
        options=parsed_options,
        type=question_type,
        token=token,
    )
    return process_ai_answer(req)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_RELOAD", "true").lower() == "true",
    )
