
import asyncio
from vllm import SamplingParams

# Create queue for requests
inference_queue = asyncio.Queue()
NUM_WORKERS = 4

async def inference_worker(app):
    while True:
        prompt, sampling_params, future = await inference_queue.get()
        try:
            llm = app.state.llm
            outputs = llm.generate(prompt, sampling_params=sampling_params)
            future.set_result(outputs)
        except Exception as e:
            future.set_exception(e)

async def start_batch_workers(app):
    for _ in range(NUM_WORKERS):
        asyncio.create_task(inference_worker(app))
