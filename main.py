from fastapi import FastAPI
from engine import Trace, Visualiser
from schemas import Matmul, Matvec, Addvec, Dot

app = FastAPI()
trace = Trace()
visualiser = Visualiser(trace)

@app.post("/matmul")
def handle_matmul(matrices: Matmul):
    try:
        events = list(visualiser.get_events(matrices.A, matrices.B, "matmul"))
        return {"status": "success", "events": events}
    except ValueError as e:
        return {"status": "fail", "reason": str(e)}