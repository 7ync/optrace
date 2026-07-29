"""
In-progress pure Python operation tracer for specific foundational matrix and vector calculations. 
Currently implements matrix multiplication with step events and simplified logical counts for arithmetic operations, operand reads, and output writes.
"""

class Trace:
    
    def __init__(self):
        self._ops = {
            "matmul": self.matmul,
            "matvec": self.matvec,
            "addvec": self.addvec,
            "dot": self.dot,
        }

    def calculate(self, A, B, op, visualiser=False):
        self._A = A
        self._B = B
        self.op = op
        self.visualiser = visualiser

        self._C = 0

        self.muls = 0
        self.adds = 0
        self.reads = 0
        self.writes = 0

        if self.visualiser:
            return self._ops[self.op]()

        else:
            for step in self._ops[self.op]():
                    pass
            return self.report(self.op)

    def matmul(self):

        self.validate_matrix(self._A)
        self.validate_matrix(self._B)

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
                    self.reads += 2
                    
                    yield {
                        "event":"compute", 
                        "muls": self.muls,
                        "adds": self.adds, 
                        "reads": self.reads,
                        "flops": (self.muls + self.adds),
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

    def validate_matrix(self, matrix):
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

    def validate_vector(self, vector):
        if not isinstance(vector, list) or not vector:
            raise ValueError("invalid vector input")
        for element in vector:
            if isinstance(element, bool) or not isinstance(element, (int, float)):
                raise ValueError("invalid vector value")

        

    def matvec(self):
        self.validate_matrix(self._A)
        self.validate_vector(self._B)

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
                self.reads += 2

                yield {
                    "event": "compute",
                    "muls": self.muls,
                    "reads": self.reads,
                    "adds": self.adds,
                    "flops": self.muls + self.adds,
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
    

    def addvec(self):
        self.validate_vector(self._A)
        self.validate_vector(self._B)

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
            self.reads += 2

            yield {
                "event": "compute",
                "A_index": i,
                "B_index": i,
                "adds": self.adds,
                "reads": self.reads,
            }

            self._C.append(result)
            self.writes += 1

            yield {
                "event": "write",
                "C_index": i,
                "writes": self.writes,
            }


    def dot(self):
        self.validate_vector(self._A)
        self.validate_vector(self._B)

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
            self.reads += 2

            yield {
                "event": "compute",
                "index": i,
                "muls": self.muls,
                "reads": self.reads,
                "adds": self.adds,
            }

        self._C = accumulator
        self.writes += 1
        yield {
            "event": "write",
            "writes": self.writes,
        }

    def report(self, op):

        formula_count = self.get_expected_cost(op)
        engine_count = {"muls": self.muls, "adds": self.adds, "reads": self.reads, "writes": self.writes}

        if formula_count != engine_count:
           raise RuntimeError

        report = f"""
            Operation: {op}
            Multiplications: {engine_count["muls"]}
            Additions: {engine_count["adds"]}
            Operand Reads: {engine_count["reads"]}
            Output Writes: {engine_count["writes"]}

        """

        print(report)

        print(f"Result: {self._C}")
        

        # TODO implement full report for each operation


    def get_expected_cost(self, op):

        match op:
            case "matmul":
                m = len(self._A)
                n = len(self._A[0])
                p = len(self._B[0])
                return {"muls": m*n*p, "adds": m*p*(n-1), "reads":2*m*n*p, "writes": m*p}
            case "matvec":
                m = len(self._A)
                n = len(self._B)
                return {"muls": m*n, "adds": m*(n-1),"reads": 2*(m*n), "writes": m}
            case "addvec":
                n = len(self._A)
                return {"muls": 0, "adds": n, "reads": 2*n, "writes": n}
            case "dot":
                n = len(self._A)
                return {"muls": n, "adds": n-1, "reads": 2*n, "writes": 1}



        return {}

    
    def C(self):
        return self._C



class Visualiser:
    ... # TODO  


def main():
    A = [[3,1,4,5,6],[4,46,7,8,6],[3,7,84,4,3]] # 3x5
    #B = [[1,3,3,2],[3,6,2,27],[3,1,14,8],[0,3,31,7],[11,3,13,4]] # 5x4
    #A = [1,3,1,4,1]
    B = [1,3,5,6,4]
    #B = [[1]]
    #A = [[2]]
    #B = [[4]]

    #A = [1,3,4]
    #B = [2,4,5]

    


    trace = Trace()
    trace.calculate(A, B, "matvec")

if __name__ == "__main__":
    main()