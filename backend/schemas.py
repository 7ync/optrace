from pydantic import BaseModel, Field
from typing import Annotated

Vector = Annotated[list[int|float], Field(max_length=12)]
Matrix = Annotated[list[Vector], Field(max_length=12)]


class Matmul(BaseModel):
    A: Matrix
    B: Matrix

class Matvec(BaseModel):
    A: Matrix
    B: Vector

class Addvec(BaseModel):
    A: Vector
    B: Vector

class Dot(BaseModel):
    A: Vector
    B: Vector
