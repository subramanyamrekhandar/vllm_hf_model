# main.py
import os
import asyncio
import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from contextlib import asynccontextmanager

from app.model_routes import router as model_routes
from app.inference_queue import start_batch_workers

# Load env variables
load_dotenv()

HF_MODEL_ID = os.getenv("HF_MODEL_ID")  # e.g., "mistralai/Mistral-7B-Instruct-v0.1"
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "")

if not HF_MODEL_ID:
    raise ValueError("Environment variable HF_MODEL_ID is not set.")

# Authenticate Hugging Face download
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

"""
Warmup load
"""
async def warmup_model(llm, retries=3):
    for attempt in range(1, retries + 1):
        try:
            llm.generate("Hello", sampling_params=SamplingParams(max_tokens=8))
            print("Model warm-up complete")
            return
        except Exception as e:
            print(f"Warm-up failed (attempt {attempt}): {str(e)}")
            traceback.print_exc()
            await asyncio.sleep(2)
    raise RuntimeError("Warm-up failed after multiple attempts.")

"""
Lifespan (Startup + Cleanup) 
""" 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading HF model:", HF_MODEL_ID)
    llm = LLM(
        model=HF_MODEL_ID,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.85,  # Reduce to avoid OOM
        max_num_seqs=64,
    )
    app.state.llm = llm

    await warmup_model(llm)
    asyncio.create_task(start_batch_workers(app))
    yield

"""
FastAPI App and Health Check
"""
app = FastAPI(title="Unified LLM API", lifespan=lifespan)
app.include_router(model_routes)

@app.get("/healthz", tags=["Health"])
async def health_check():
    try:
        llm = app.state.llm
        llm.generate("ping", sampling_params=SamplingParams(max_tokens=1))
        return JSONResponse(content={"status": "ok", "model": HF_MODEL_ID}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)}, status_code=500)
