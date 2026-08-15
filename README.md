# diffusion-serving-api

![tests](https://github.com/Shivam101s/diffusion-serving-api/actions/workflows/tests.yml/badge.svg)

A small FastAPI service that turns a slow generative model into an async
job API: `POST /jobs` returns immediately with a job id, a worker pool picks
the job up and runs it, and `GET /jobs/{id}` lets the caller poll for the
result. This is the pattern any real image/video generation product needs —
you can't hold an HTTP connection open for the seconds-to-minutes a
diffusion sampler takes — scoped down to something readable end to end.

It wraps the model from [mini-diffusion](https://github.com/Shivam101s/mini-diffusion)
(a from-scratch DDPM trained on CIFAR-10) and uses
[mini-taskqueue](https://github.com/Shivam101s/mini-taskqueue) — also mine —
as the actual job queue, rather than reaching for Celery+Redis for what's a
single-machine service. All three repos are meant to be read as one system.

## Architecture

```
POST /jobs  ──►  enqueue("generate", num_samples, seed)  ──►  SQLite queue
                                                                    │
GET /jobs/{id} ◄── poll for status/result ◄── worker pool picks it up, runs
                                               the model, writes the result
```

The generator is pluggable behind a small interface
([`app/generator.py`](app/generator.py)):

- **`MockGenerator`** — returns random noise instantly, no model or GPU
  needed. This is what CI and the test suite run against, so the API
  contract (validation, job lifecycle, error handling) is verified on every
  push without needing a trained checkpoint or an accelerator.
- **`DiffusionGenerator`** — loads a real mini-diffusion checkpoint and
  samples from it with DDIM.

Switch between them with the `GENERATOR` env var.

## Usage

```bash
pip install -r requirements.txt

# random-noise mode, no checkpoint needed
GENERATOR=mock uvicorn app.main:app --reload

# real model
GENERATOR=diffusion CHECKPOINT_PATH=/path/to/model_final.pt uvicorn app.main:app
```

```bash
curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
  -d '{"num_samples": 4, "seed": 42}'
# {"job_id": "..."}

curl localhost:8000/jobs/<job_id>
# {"job_id": "...", "status": "done", "images": ["<base64 png>", ...], "error": null}
```

Or with Docker:

```bash
docker build -t diffusion-serving-api .
docker run -p 8000:8000 -v $(pwd)/data:/data diffusion-serving-api
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

11 tests: the generator interface (shape, valid PNG output, seeded
determinism), and the API layer end to end — job creation, validation
(`num_samples` bounds), polling through to completion, and 404s for unknown
jobs — all against `MockGenerator`, so the suite runs in under two seconds
with no GPU.

## License

MIT
