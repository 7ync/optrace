# optrace

A Python engine for matrix multiplication, matrix-vector multiplication, vector addition and dot product. It computes the result and, as it runs, yields a step-by-step trace of the multiplications, additions, scalars indexed and output writes it performs. Written to understand how these operations work in code rather than for speed or completeness.

A FastAPI backend exposes the engine over HTTP, and a plain HTML/JS frontend replays the trace as a step-through visualisation.

[Try the live demo](https://baileyward.ai/demo/) or [read about what I learned building it](https://baileyward.ai/optrace.html).

## Running it

Backend:

```bash
git clone https://github.com/7ync/optrace
cd optrace
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The API listens on `http://localhost:8000`.

Frontend (plain HTML/CSS/JS). The backend's CORS config allows `http://localhost:3000` and `http://127.0.0.1:3000` for local development, so serve `frontend/` on port 3000:

```bash
cd frontend
python3 -m http.server 3000
```

Then open `http://localhost:3000`.

Tests (from the repository root):

```bash
pytest
```

## Using the engine

`Trace.calculate(A, B, op)` is a generator. Exhaust it to run the calculation, then read the result and the counts:

```python
from backend.engine import Trace

trace = Trace()
for event in trace.calculate([[1, 2], [3, 4]], [[5, 6], [7, 8]], "matmul"):
    ...

trace.C()                   # [[19, 22], [43, 50]]
trace.get_expected_cost()   # {"muls": 8, "adds": 4, "scalars_indexed": 16, "writes": 4}
trace.report()              # prints the result and counts
```

A `Trace` moves through three states: `inactive`, `in_progress` and `complete`. It refuses to start a second calculation while one is in progress, refuses to hand back a result before one has finished, and resets itself if a generator is abandoned partway through.

To use the engine on its own without either server running:

```bash
python -m backend.run
```

It runs a 12×12 matrix multiplication and prints the report. You can change the inputs to use any operation or matrix/vector pair.

## What it counts

`Trace` tracks four counters as it runs:

- `muls` — scalar multiplications performed
- `adds` — scalar additions performed
- `scalars_indexed` — scalars indexed from the operands
- `writes` — values written to the output

For matrix multiplication of an m×n matrix by an n×p matrix, `Trace.get_expected_cost()` gives:

- muls = m·n·p
- adds = m·p·(n−1)
- scalars_indexed = 2·m·n·p
- writes = m·p

`Trace.report()` recomputes these formulas independently and compares them against the counters accumulated during the run, raising if they disagree. This check runs on demand when `report()` is called, not automatically after every calculation.

This is a logical operation count for the specific triple-loop implementation, not a runtime cost model. It says nothing about memory or how the operation would cost out on real hardware.

## Events

`calculate()` yields dicts with an `"event"` key: `"init"` once at the start with the shapes, `"compute"` after each scalar multiply/add, and `"write"` after each value is committed to the output. Compute events carry `muls`/`adds`/`scalars_indexed`; write events carry `writes`.

Example sequence for a 2×2 matmul (from backend/tests/test_engine.py):

```python
{"event": "init", "A_rows": 2, "A_cols": 2, "B_rows": 2, "B_cols": 2, "C_rows": 2, "C_cols": 2}
{"event": "compute", "muls": 1, "adds": 0, "scalars_indexed": 2, "i": 0, "j": 0, "k": 0}
{"event": "compute", "muls": 2, "adds": 1, "scalars_indexed": 4, "i": 0, "j": 0, "k": 1}
{"event": "write", "writes": 1, "i": 0, "j": 0}
```

Position keys differ per operation. See backend/engine.py for the exact keys each yields.

## HTTP API

Four routes, one per operation: `POST /matmul`, `/matvec`, `/addvec`, `/dot`. Each takes the two operands and returns the full event list.

```bash
curl -X POST http://localhost:8000/matmul \
  -H "Content-Type: application/json" \
  -d '{"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}'
```

```json
{"status": "success", "events": [...]}
```

Invalid input returns `{"status": "fail", "reason": "..."}` rather than an error status code.

Operands are capped at 6×6 for matrices and 6 elements for vectors (backend/schemas.py). The engine itself has no limit. The cap exists for the frontend, and calling `Trace.calculate()` directly bypasses it.

## Layout

```text
backend/
  engine.py        Trace (the engine), Validators, Visualiser
  main.py          FastAPI app and the four POST routes
  schemas.py       pydantic request models, capped at 6x6 / 6 elements
  run.py           standalone script that runs a matmul trace and prints a report
  tests/
    test_engine.py
frontend/
  index.html
  script.js        fetches from the backend and drives the step visualiser
  style.css
```