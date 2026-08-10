from pydantic import BaseModel, field_validator

class Matmul(BaseModel):
    A: list[list[int|float]]
    B: list[list[int|float]]

    @field_validator("A")
    def max_size_A(cls, A):
        if not A:
            raise ValueError("empty matrix")
        if len(A) > 12:
            raise ValueError("number of rows cannot exceed 12")
        if len(A[0]) > 12: 
            raise ValueError("number of columns cannot exceed 12")
        
        return A

    @field_validator("B")
    def max_size_B(cls, B):
        if len(B) > 12:
            raise ValueError("number of rows cannot exceed 12")
        if len(B[0]) > 12: 
            raise ValueError("number of columns cannot exceed 12")

        return B

class Matvec(BaseModel):
    ...

class Addvec(BaseModel):
    ...

class Dot(BaseModel):
    ...
