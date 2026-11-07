# optrace

A Python project exploring numerical computation, execution tracing and software testing.

Implements matrix multiplication, matrix-vector multiplication, vector addition and dot products in pure Python. The engine uses generators to produce step-by-step execution events, tracking arithmetic operations, operand indexing and output writes.

**[Try the interactive demo](https://baileyward.ai/demo/)**

The interactive frontend is hosted separately. This repository contains the Python engine, automated tests and FastAPI interface.

## Using the engine

```python
from backend.engine import Trace

trace = Trace()

for event in trace.calculate(
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    "matmul"
):
    print(event)

print(trace.C())
trace.report()
```

Alternatively, run the standalone example from the repository root:

```bash
python -m backend.run
```

## Testing

The project uses pytest to verify numerical results, execution events, operation counts, input validation and calculation states.

```bash
pip install -r requirements.txt
pytest
```

## Implementation

The engine is implemented in `backend/engine.py` and operates independently of the web interface.

`Trace.calculate()` yields execution events while calculating, and `Trace.report()` compares recorded operation counts against expected counts.

These counts describe the operations performed by the implementation, rather than actual runtime or hardware performance.
