import json

from fastapi import HTTPException
from starlette.requests import Request

from src.core.error_handlers import http_error_handler

REQUEST = Request({"type": "http", "method": "GET", "path": "/"})


async def test_http_error_handler_keeps_a_string_detail_as_the_message():
    response = await http_error_handler(REQUEST, HTTPException(403, detail="Training is disabled"))

    assert response.status_code == 403
    assert json.loads(response.body) == {"code": "forbidden", "message": "Training is disabled", "details": None}


async def test_http_error_handler_moves_a_structured_detail_to_details():
    response = await http_error_handler(REQUEST, HTTPException(400, detail={"field": "n"}))

    assert json.loads(response.body) == {"code": "bad_request", "message": "Bad Request", "details": {"field": "n"}}


async def test_http_error_handler_keeps_the_headers():
    response = await http_error_handler(REQUEST, HTTPException(401, headers={"WWW-Authenticate": "Bearer"}))

    assert response.headers["www-authenticate"] == "Bearer"


async def test_http_error_handler_names_a_status_http_does_not_know():
    response = await http_error_handler(REQUEST, HTTPException(499, detail={"reason": "closed"}))

    assert response.status_code == 499
    assert json.loads(response.body) == {"code": "http_error", "message": "HTTP error", "details": {"reason": "closed"}}
