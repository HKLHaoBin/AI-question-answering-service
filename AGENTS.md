# Repository Guidelines

## Project Structure & Module Organization
- `main.py` hosts the FastAPI application plus OpenAI integration. Treat it as the single entry point for business logic, request validation, and prompt orchestration.
- `requirements.txt` captures runtime and dev dependencies. Update it whenever you add imports.
- `README.md` documents deployment basics; keep it aligned with code changes.
- Use `__pycache__/` only for interpreter artifacts; never commit files inside it.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs pinned dependencies for local dev and CI.
- `python main.py` launches the API with Uvicorn, enabling reload when `APP_RELOAD=true`.
- `uvicorn main:app --host 0.0.0.0 --port 8000` is a manual alternative when you need custom flags.
- If you add automated tests, prefer `pytest` and document any fixtures in this guide.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indents, clear docstrings, and snake_case for variables/functions.
- Keep prompts and validation helpers (e.g., `build_prompt`, `parse_model_json`) pure and composable.
- Add short, meaningful comments only when logic is non-obvious—avoid restating the code.
- For environment variables, mirror the existing uppercase underscore style (`OPENAI_MODEL`, `AI_SIMPLE_TOKEN`).

## Testing Guidelines
- Add tests under a future `tests/` directory, mirroring module paths (e.g., `tests/test_main.py`).
- Name tests by behavior (`test_ai_answer_requires_token`) and use fixtures for repeated request payloads.
- Run `pytest -q` locally before pushing; aim to cover prompt building, option parsing, and token validation.

## Commit & Pull Request Guidelines
- Write commits in the imperative mood (`Add completion prompt guard`). Group related edits together.
- Reference issue IDs in the summary when applicable and describe visible changes plus testing evidence in the body.
- Pull requests should include: scope description, screenshots or curl output for API changes, and notes on config or env var updates.

## Security & Configuration Tips
- Never commit `.env` or API keys. Use `.env.example` if you need to illustrate required variables.
- Validate the `token` field before calling the model to avoid unauthenticated usage; keep the guard intact when refactoring.
