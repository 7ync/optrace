from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.engine import Trace, Visualiser
from backend.schemas import Matmul, Matvec, Addvec, Dot

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def execute(op: str, payload1, payload2):
    visualiser = Visualiser(Trace())
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