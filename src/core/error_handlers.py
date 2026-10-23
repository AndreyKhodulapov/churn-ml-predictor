from fastapi import Request
from fastapi.responses import JSONResponse


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Starlette re-raises the exception after sending this response, so the traceback still reaches the server log.
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
