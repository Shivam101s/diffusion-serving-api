import json
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from taskqueue import TaskQueue, Worker, task

from app import config
from app.generator import build_generator

_queue: Optional[TaskQueue] = None
_worker: Optional[Worker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queue, _worker

    _queue = TaskQueue(config.QUEUE_DB_PATH)
    generator = build_generator(config.GENERATOR, config.CHECKPOINT_PATH, config.DDIM_STEPS)

    @task(name="generate")
    def generate(num_samples: int, seed: Optional[int] = None):
        return generator.generate(num_samples=num_samples, seed=seed)

    _worker = Worker(_queue, concurrency=config.WORKER_CONCURRENCY, poll_interval=0.1)
    threading.Thread(target=_worker.run, daemon=True).start()

    yield

    _worker.stop()


app = FastAPI(title="diffusion-serving-api", lifespan=lifespan)


class GenerateRequest(BaseModel):
    num_samples: int = Field(default=1, ge=1, le=config.MAX_SAMPLES_PER_JOB)
    seed: Optional[int] = None


class JobCreatedResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    images: Optional[list] = None
    error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "generator": config.GENERATOR}


@app.post("/jobs", response_model=JobCreatedResponse, status_code=201)
def create_job(request: GenerateRequest):
    job_id = _queue.enqueue(
        "generate",
        args=[request.num_samples],
        kwargs={"seed": request.seed},
    )
    return JobCreatedResponse(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    task_obj = _queue.get(job_id)
    if task_obj is None:
        raise HTTPException(status_code=404, detail="job not found")

    images = json.loads(task_obj.result) if task_obj.result else None
    return JobStatusResponse(
        job_id=task_obj.id,
        status=task_obj.status,
        images=images,
        error=task_obj.error,
    )
