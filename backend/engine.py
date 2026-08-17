"""
Pure Python mathematical engine for matrix multiplication, matrix-vector multiplication, vector addition and dot product calculations. 
Emits step events as it calculates, each carrying running counts of multiplications, additions, scalars indexed and output writes. 
The counts are a deliberately simple logical model and describe the operations performed in this implementation, not their cost at runtime.
"""

class Trace:
    
    def __init__(self):
        self._ops = {
            "matmul": self.matmul,
            "matvec": self.matvec,
            "addvec": self.addvec,
            "dot": self.dot,
        }
        self._A: list | None = None
        self._B: list | None = None
        self._C: list | int | float | None = None
        self._op: str | None = None
        self.state = "inactive"

    def calculate(self, A, B, op):
        if self.state == "in_progress":
            raise RuntimeError
        self._A = A
        self._B = B
        self._C = 0
        self._op = op

        if self._op == None:
            raise RuntimeError

        self.muls = 0
        self.adds = 0
        self.scalars_indexed = 0
        self.writes = 0

        self.state = "in_progress"

        try:
            yield from self._ops[self._op]()

        except (Exception, GeneratorExit):
            self.state = "inactive"
            self._A = None
            self._B = None
            self._C = None
            self._op = None
            raise


    def matmul(self):
        if self._A is None:
            raise RuntimeError
        if self._B is None:
            raise RuntimeError

        Validators.validate_matrix(self._A)
        Validators.validate_matrix(self._B)

        A_rows = len(self._A)
        A_cols = len(self._A[0])
        
        B_rows = len(self._B)
        B_cols = len(self._B[0])

        # check column size for A is equal to the row size for B
        if A_cols != B_rows:
            raise ValueError("invalid size matrix")

        self._C = []

        for _ in range(len(self._A)):
            row = [0] * B_cols
            self._C.append(row)

        C_rows = A_rows
        C_cols = B_cols

        yield {
            "event": "init",
            "A_rows": A_rows,
            "A_cols": A_cols,
            "B_rows": B_rows,
            "B_cols": B_cols,
            "C_rows": C_rows,
            "C_cols": C_cols,
            }

        # compute AB
        for i in range(len(self._A)):
            for j in range(len(self._B[0])):
                result = 0
                for k in range(len(self._B)):
                    if k == 0:
                        result = self._A[i][k] * self._B[k][j]
                    else:
                        result += self._A[i][k] * self._B[k][j]
                        self.adds += 1
                    self.muls += 1
                    self.scalars_indexed += 2
                    
                    yield {
                        "event":"compute", 
                        "muls": self.muls,
                        "adds": self.adds, 
                        "scalars_indexed": self.scalars_indexed,
                        "i": i,
                        "j": j,
                        "k": k
                    }

                self._C[i][j] = result
                self.writes += 1

                yield {
                    "event": "write",
                    "writes": self.writes,
                    "i": i,
                    "j": j,
                }

        self.state = "complete"
     

    def matvec(self):
        if self._A is None:
            raise RuntimeError
        if self._B is None:
            raise RuntimeError


        Validators.validate_matrix(self._A)
        Validators.validate_vector(self._B)

        A_rows = len(self._A)
        A_cols = len(self._A[0])
        B_elements = len(self._B)

        if A_cols != B_elements:
            raise ValueError("vector element count must equal matrix column count")

        self._C = []

        yield {
            "event": "init",
            "A_rows": A_rows,
            "A_cols": A_cols,
            "B_elements": B_elements,
            "C_elements": A_rows,
        }

        for i, row in enumerate(self._A):
            matvec_iterator = zip(row, self._B)
            result = 0
            for col, (A_element, B_element) in enumerate(matvec_iterator):
                if col == 0:
                    result = A_element * B_element
                else:
                    result += A_element * B_element
                    self.adds += 1
                self.muls += 1
                self.scalars_indexed += 2

                yield {
                    "event": "compute",
                    "muls": self.muls,
                    "adds": self.adds,
                    "scalars_indexed": self.scalars_indexed,
                    "A_row": i,
                    "A_col": col,
                    "B_element": col,
                }

            self._C.append(result)
            self.writes += 1

            yield {
                "event": "write",
                "writes": self.writes,
                "C_element": i
            }

        self.state = "complete"
    

    def addvec(self):
        if self._A is None:
            raise RuntimeError
        if self._B is None:
            raise RuntimeError

        Validators.validate_vector(self._A)
        Validators.validate_vector(self._B)

        A_len = len(self._A)
        B_len = len(self._B)

        if A_len != B_len:
            raise ValueError("vectors must be the same size")

        self._C = []

        yield {
            "event": "init",
            "A_len": A_len,
            "B_len": B_len,
            "C_len": A_len,
        }

        for i, (a, b) in enumerate(zip(self._A, self._B)):
            result = a + b
            self.adds += 1
            self.scalars_indexed += 2

            yield {
                "event": "compute",
                "A_index": i,
                "B_index": i,
                "adds": self.adds,
                "scalars_indexed": self.scalars_indexed,
            }

            self._C.append(result)
            self.writes += 1

            yield {
                "event": "write",
                "C_index": i,
                "writes": self.writes,
            }

        self.state = "complete"


    def dot(self):
        if self._A is None:
            raise RuntimeError
        if self._B is None:
            raise RuntimeError

        Validators.validate_vector(self._A)
        Validators.validate_vector(self._B)

        A_len = len(self._A)
        B_len = len(self._B)

        if A_len != B_len:
            raise ValueError("vectors must be the same size")

        self._C = 0

        yield {
            "event": "init",
            "A_len": A_len,
            "B_len": B_len,
        }

        accumulator = 0

        for i, (a, b) in enumerate(zip(self._A, self._B)):
            if i == 0:
                accumulator = a * b
            else:
                accumulator += a * b
                self.adds += 1

            self.muls += 1
            self.scalars_indexed += 2

            yield {
                "event": "compute",
                "index": i,
                "muls": self.muls,
                "adds": self.adds,
                "scalars_indexed": self.scalars_indexed,
            }

        self._C = accumulator
        self.writes += 1
        yield {
            "event": "write",
            "writes": self.writes,
        }

        self.state = "complete"


    def report(self):
        if self.state != "complete":
            raise RuntimeError

        formula_count = self.get_expected_cost()
        engine_count = {"muls": self.muls, "adds": self.adds, "scalars_indexed": self.scalars_indexed, "writes": self.writes}

        if formula_count != engine_count:
           raise RuntimeError

        report = f"""
            Operation: {self._op}
            Result: {self._C}
            Multiplications: {engine_count["muls"]}
            Additions: {engine_count["adds"]}
            Scalars Indexed: {engine_count["scalars_indexed"]}
            Output Writes: {engine_count["writes"]}
        """

        print(report)


    def get_expected_cost(self):
        if self._A is None:
            raise RuntimeError
        if self._B is None:
            raise RuntimeError
        if self.state != "complete":
            raise RuntimeError

        match self._op:
            case "matmul":
                m = len(self._A)
                n = len(self._A[0])
                p = len(self._B[0])
                return {"muls": m*n*p, "adds": m*p*(n-1), "scalars_indexed":2*m*n*p, "writes": m*p}
            case "matvec":
                m = len(self._A)
                n = len(self._B)
                return {"muls": m*n, "adds": m*(n-1),"scalars_indexed": 2*(m*n), "writes": m}
            case "addvec":
                n = len(self._A)
                return {"muls": 0, "adds": n, "scalars_indexed": 2*n, "writes": n}
            case "dot":
                n = len(self._A)
                return {"muls": n, "adds": n-1, "scalars_indexed": 2*n, "writes": 1}



        return {}

    
    def C(self):
        if self.state != "complete":
            raise RuntimeError
        return self._C

    def op(self):
        return self._op



class Validators:
    @staticmethod
    def validate_matrix(matrix):
        # check matrix is a non-empty list
        if not isinstance(matrix, list) or not matrix:
            raise ValueError("empty matrix")
        # check nested lists are not empty
        if not isinstance(matrix[0], list) or not matrix[0]:
            raise ValueError("empty matrix") 

        row_size = len(matrix[0])

        for row in matrix:
            # check all rows are lists
            if not isinstance(row, list):
                raise ValueError("invalid list in matrix")
            
            # check all rows are the same length
            if len(row) != row_size:
                raise ValueError("rows must be the same size")
            
            for element in row:
                # check all elements are integers or floats
                if isinstance(element, bool) or not isinstance(element, (float, int)):
                    raise ValueError("invalid value in matrix")

    @staticmethod
    def validate_vector(vector):
        if not isinstance(vector, list) or not vector:
            raise ValueError("invalid vector input")
        for element in vector:
            if isinstance(element, bool) or not isinstance(element, (int, float)):
                raise ValueError("invalid vector value")


class Visualiser:
    def __init__(self, engine):
        self._engine = engine

    def get_events(self, A, B, op):
        self._op = op

        events = list(self._engine.calculate(A, B, op))

        return events