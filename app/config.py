import os

# "mock" returns random noise instantly (used in tests/CI, no GPU or
# checkpoint required); "diffusion" loads a real trained checkpoint.
GENERATOR = os.environ.get("GENERATOR", "mock")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/model_final.pt")
DDIM_STEPS = int(os.environ.get("DDIM_STEPS", "50"))
QUEUE_DB_PATH = os.environ.get("QUEUE_DB_PATH", "queue.db")
MAX_SAMPLES_PER_JOB = int(os.environ.get("MAX_SAMPLES_PER_JOB", "8"))
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "2"))
