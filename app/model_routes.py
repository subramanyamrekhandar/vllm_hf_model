from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from vllm import SamplingParams
from app.token_key import validate_token
import asyncio
from app.inference_queue import inference_queue

router = APIRouter(prefix="/v1", tags=["LLM Generate Endpoint"])

class InferenceRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    images: Optional[List[str]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    max_tokens: Optional[int] = 1024
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    stop: Optional[List[str]] = None

@router.post("/generate")
async def generate(req: InferenceRequest, request: Request, _=Depends(validate_token)):
    # llm = request.app.state.llm
    sampling = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k if req.top_k is not None else -1,  # fallback,
        max_tokens=req.max_tokens,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        repetition_penalty=req.repetition_penalty,
        stop=req.stop
    )
    future = asyncio.Future()
    # future = asyncio.get_event_loop().create_future()
    await inference_queue.put((req.prompt, sampling, future))
    outputs = await future
    if outputs and outputs[0].outputs:
        return {"output": outputs[0].outputs[0].text}
    return {"error": "No output generated"}
