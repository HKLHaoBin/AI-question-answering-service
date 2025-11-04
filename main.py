import json
import os
import re
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
        "completion": "填空题",
    }

    normalized_type = (q_type or "").lower()
    type_text = type_map.get(normalized_type, "单选题")
    if normalized_type == "judgement":
        type_text = "单选题"

    if options:
        option_text = "\n".join(o.strip() for o in options)
    else:
        option_text = "（本题无选项，可能是简答或填空题）"

    is_completion = normalized_type == "completion" or not options

    if is_completion:
        prompt_lines = [
            "请根据给出的题目内容，写出最合理的答案，并提供必要的解析。",
            f"题目类型：{type_text}",
            "",
            "题目：",
            title,
            "",
            "选项：",
            option_text,
            "",
            "请严格按下面 JSON 格式返回（不要输出多余文字，不要加注释）：",
            "{",
            '  "answer": "完整文字答案",',
            '  "analysis": "详细解析，说明思路和知识点，仅用于学习和复习"',
            "}",
            "附加要求：",
            "- 这是填空题或无选项题，请输出完整文字答案。",
            "- analysis 字段必须包含详细解析，说明解题思路和知识点。",
            "- 答案和解析仅用于学习、复习，不得包含无关内容。",
        ]
    else:
        prompt_lines = [
            "请根据给出的题目和选项，推理出最合理的答案。",
            f"题目类型：{type_text}",
            "",
            "题目：",
            title,
            "",
            "选项：",
            option_text,
            "",
            "请严格按下面 JSON 格式返回（不要输出多余文字，不要加注释）：",
            "{",
            '  "answer": "例如 A 或 AC 或 ABCD",',
            '  "analysis": "与 answer 完全一致的字符串"',
            "}",
            "附加要求：",
            "- 这是选择题（包括单选、多选、判断），请仅返回选项字母且使用大写字母。",
            "- 单选题返回一个字母，多选题按字母顺序拼写（例如 AC），如果所有选项都正确，请返回 ABCD（或按实际选项数量的全量字母）。",
            "- 判断题也仅返回 A 或 B。",
            "- analysis 字段必须与 answer 完全一致，不允许包含任何其他字符或说明。",
            "- 不要输出任何解题过程、推理步骤或额外文字。",
            "- answer 字段不要出现空格、# 号或无关内容。",
        ]

    return "\n".join(prompt_lines).strip()


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


OPTION_LABEL_PATTERN = re.compile(r"^\s*([A-Za-z])[\s\.\、,，:：\)\（\(\）]*(.*)$")


def build_option_lookup(options: Optional[List[str]]) -> dict:
    """
    构建选项字母到选项内容的映射。
    """
    lookup = {}
    if not options:
        return lookup

    for raw in options:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        match = OPTION_LABEL_PATTERN.match(text)
        if match:
            letter = match.group(1).upper()
            remainder = match.group(2).strip()
            lookup[letter] = remainder or text
        else:
            letter = text[:1].upper()
            lookup[letter] = text
    return lookup


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

    normalized_type = (req.type or "").lower()
    is_completion = normalized_type == "completion" or not req.options

    if not answer:
        raise HTTPException(status_code=500, detail="model returned empty answer")

    if not is_completion:
        answer_letters = re.findall(r"[A-Za-z]", answer.upper())
        if not answer_letters:
            raise HTTPException(status_code=500, detail="model returned invalid answer")

        unique_letters = []
        for letter in answer_letters:
            if letter not in unique_letters:
                unique_letters.append(letter)

        answer = "".join(unique_letters)

        option_lookup = build_option_lookup(req.options)
        matched_texts = []
        for letter in unique_letters:
            option_text = option_lookup.get(letter)
            if option_text:
                matched_texts.append(f"{letter}. {option_text}")
            else:
                matched_texts.append(letter)

        analysis = "\n".join(matched_texts) if matched_texts else answer
    else:
        if not analysis:
            analysis = "解析略"

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
