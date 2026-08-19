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

Frontend:

```bash
cd frontend
python3 -m http.server 3000
```

Then open `http://localhost:3000`.

The backend's CORS config allows `http://localhost:3000` and `http://127.0.0.1:3000` for local development.

Tests (from the repository root):

```bash
pytest
```

## Using the engine

`Trace.calculate(A, B, op)` is a generator. Exhaust it to run the calculation, then read the result and print the report:

```python
from backend.engine import Trace

trace = Trace()
for event in trace.calculate([[1, 2], [3, 4]], [[5, 6], [7, 8]], "matmul"):
    ...

trace.C()                   # [[19, 22], [43, 50]]
trace.report()              # prints the result and counts
```

A `Trace` moves through three states. It starts `inactive`, holding no operands or result. Only `calculate()` can be called. While a generator is being consumed it is `in_progress`. A second calculation cannot be started, and an error or an abandoned generator resets it to `inactive`. Once the generator is exhausted it is `complete`, and `C()` and `report()` become available.

To use the engine on its own without either server running:

```bash
python -m backend.run
```

It runs a 12×12 matrix multiplication and prints the report. You can change the inputs to use any valid operation or matrix/vector pair.

## What it counts

`Trace` tracks four counters as it runs:

- `muls` — scalar multiplications performed
- `adds` — scalar additions performed
- `scalars_indexed` — scalars indexed from the operands
- `writes` — values written to the output

For matrix multiplication of an m×n matrix by an n×p matrix:

- muls = m·n·p
- adds = m·p·(n−1)
- scalars_indexed = 2·m·n·p
- writes = m·p

`Trace.report()` compares these formulas, derived by `Trace.get_expected_cost()`, against the counters accumulated during the run. A RuntimeError is raised if they disagree. This check runs on demand when `report()` is called.

This is a logical operation count for the specific triple-loop implementation. It says nothing about runtime, memory, or how the operation actually behaves on hardware.

## Events

`calculate()` yields dicts with an `"event"` key. There is one `"init"` at the start carrying the shapes, a `"compute"` after each scalar multiply/add, and a `"write"` after each value is committed to the output.

Compute events carry `muls`/`adds`/`scalars_indexed`, and write events carry `writes`.

A `Visualiser` class takes an engine and collects its events into a list. Nothing in the engine knows about it, so another consumer could handle the same events differently.

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

Invalid input returns `{"status": "fail", "reason": "..."}`.

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