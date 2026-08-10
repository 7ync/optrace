from fastapi import FastAPI
from engine import Trace, Visualiser
from schemas import Matmul, Matvec, Addvec, Dot

app = FastAPI()
trace = Trace()
visualiser = Visualiser(trace)

def execute(op: str, payload1, payload2):
    try:
        events = list(visualiser.get_events(payload1, payload2, op))
        return {"status": "success", "events": events}
    except ValueError as e:
        return {"status": "fail", "reason": str(e)}

@app.post("/matmul")
def handle_matmul(payload: Matmul):
    return execute("matmul", payload.A, payload.B)

@app.post("/matvec")
def handle_matvec(payload: Matvec):
    return execute("matvec", payload.A, payload.B)

@app.post("/addvec")
def handle_addvec(payload: Addvec):
    return execute("addvec", payload.A, payload.B)

@app.post("/dot")
def handle_dot(payload: Dot):
    return execute("dot", payload.A, payload.B)