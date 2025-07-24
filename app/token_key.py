from fastapi import Header, HTTPException
import os


LLM_GATEWAY_SECRET_KEY = "llm-container"

def validate_token(x_gateway_token: str = Header(...)):
    if x_gateway_token != LLM_GATEWAY_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
