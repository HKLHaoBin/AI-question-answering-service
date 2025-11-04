# AI 题目解析服务

使用 FastAPI + OpenAI SDK 实现的后端服务，提供题目解析与答案生成能力，可供任意 HTTP 客户端调用。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 配置环境变量：复制 `.env` 并填写自己的 `OPENAI_API_KEY`，按需调整 `OPENAI_MODEL`、`AI_SIMPLE_TOKEN` 等。
3. 启动服务：
   ```bash
   python main.py
   ```
   默认在 `http://127.0.0.1:8000` 提供 Swagger UI 文档。

## 题库/前端调用示例

支持普通 HTTP POST 调用（示例）：

```bash
curl -X POST "http://127.0.0.1:8000/ai-answer" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "下列关于 HTTP 的描述正确的是？",
    "options": [
      "A. HTTP 是有状态协议",
      "B. HTTP 默认端口是 80",
      "C. HTTP 只支持 GET 和 POST",
      "D. HTTP 必须加密传输"
    ],
    "type": "single",
    "token": "change-this-token"
  }'
```

## OCS 配置示例

若在 OCS 等平台集成，直接粘贴下面的 JSON 字符串（将 `token` 改成 `.env` 中的 `AI_SIMPLE_TOKEN`，把 `url` 换成实际部署地址即可）。该配置默认使用 GET 请求，`options` 可为 JSON 数组或换行分隔的字符串：

```json
{"contentType":"json","data":{"options":"${options}","title":"${title}","token":"change-this-token","type":"${type}"},"handler":"return (res)=>res.code === 1 && res.data ? [res.data.answer,res.data.analysis,{ai: res.data.ai}] : [res.msg || '调用失败', undefined, { ai: false }];","homepage":"http://127.0.0.1:8000/docs","method":"get","name":"本地AI题库","type":"GM_xmlhttpRequest","url":"http://127.0.0.1:8000/ai-answer"}
```

粘贴时请保留为单行 JSON，以满足“格式为 JSON 字符串”要求。
