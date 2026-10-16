# FastAPI and Pydantic: key concepts

Short notes on the ideas the service is built on.

## ASGI, FastAPI, Uvicorn

- ASGI is the async successor of WSGI: an interface between a Python web app
  and a server that supports `async` handlers, WebSockets and long-lived
  connections.
- FastAPI is an ASGI framework built on Starlette (routing, requests,
  responses) and Pydantic (validation, serialization).
- Uvicorn is the ASGI server that runs the app: `uvicorn package.module:app`.
  `--reload` restarts it on code changes and is for development only; in
  production run several workers (`--workers N` or Gunicorn with Uvicorn workers).

## Path operations

- An endpoint is a function decorated with `@app.get`, `@router.post`, etc.
- FastAPI reads the function signature: path params, query params, and a
  Pydantic model parameter becomes the JSON request body.
- The return annotation (or `response_model`) defines the response schema:
  the output is validated and filtered, so internal fields do not leak.
- `def` endpoints run in a thread pool, `async def` endpoints run on the event
  loop. Blocking code (CPU-heavy model inference, sync DB drivers) must not go
  into `async def`, or it freezes every request.
- `APIRouter` groups endpoints by feature; `app.include_router` mounts them.

## Pydantic models

- A `BaseModel` subclass declares fields with type hints. Data is parsed and
  validated on creation; an invalid payload raises `ValidationError`, which
  FastAPI turns into HTTP 422 with the location of each error.
- Lax mode by default: `"3"` is coerced to `3`, but `1.5` is rejected for
  an `int` field. `strict=True` disables coercion.
- `Field(...)` adds constraints (`ge`, `le`, `min_length`, `pattern`) and
  documentation (`description`, `examples`).
- `Literal[...]` restricts a field to a fixed set of values - a cheap way to
  validate categorical features before they reach the model.
- Inheritance reuses fields: a labeled dataset row is the feature vector plus
  the target column.
- `model_dump()` / `model_dump_json()` serialize, `model_validate()` parses
  a dict, `model_json_schema()` produces the JSON Schema.

## OpenAPI and /docs

- FastAPI generates an OpenAPI schema from routes and models at
  `/openapi.json`.
- `/docs` (Swagger UI) and `/redoc` render it, and Swagger UI can send test
  requests. The docs always match the code because both come from the same
  type hints.

## Why validate input in an ML service

- A model silently produces garbage for out-of-range or unknown categorical
  values; rejecting them at the API boundary with a clear 422 is cheaper than
  debugging wrong predictions.
- The schema is a contract between clients and the model: the same feature
  set is used for training rows and for prediction requests.

## Testing a FastAPI app

- `httpx.AsyncClient(transport=ASGITransport(app=app))` calls the app
  in-process, without a network socket or a running server. Tests are
  `async def` and run under `pytest-asyncio`.
- `ASGITransport` does not send lifespan events: whatever `lifespan` creates
  (DB pool, loaded model) is missing. Either wrap the client in
  `asgi_lifespan.LifespanManager(app)` or replace the dependency through
  `app.dependency_overrides`.
- `dependency_overrides` is keyed by the same function object used in
  `Depends(...)`, and it lives on the app, so clear it in fixture teardown.
- The sync `TestClient` runs its own event loop, so it does not mix with async
  fixtures; keep it for quick smoke checks only.
