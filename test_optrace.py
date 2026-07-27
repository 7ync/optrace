import pytest
from optrace import Trace

trace = Trace()

class Test_matmul:
    
    A1 = [[1,4,4],[1,5,6],[6,2,5],[5,2,5]] # 4x3
    A2 = [[5,51,1,5,6],[4,6,7,8,4],[4,3,7,7,7]] # 3x5
    A3 = [[1,2],[5,5]] # 2x2 
    A4 = [[2]] # 1x1
    A5 = [[],[],[]] # empty
    A6 = [] # empty
    A7 = [3] # invalid
    A8 = [[3,2],[3,[]]] # invalid
    A9 = [[4,1,3],[3,4,"1"]] # invalid
    A10 = [[3,3,1],[1,4],[4,4]] # unequal rows
    A11 = [[1,3],[1,5,5],[1,4,1]] # unequal rows


    B1 = [[1,4,5,6,4,3,1],[3,5,6,7,4,3,6],[1,4,6,4,3,5,7]] # 3x7
    B2 = [[1,2],[1,7],[6,3],[9,2],[0,1]] # 5x2
    B3 = [[4,2],[2,8]] # 2x2
    B4 = [[4]] # 1x1

    def test_equal_rows(self):
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A10)
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A11)

        assert trace.validate_matrix(self.A1) == None
        assert trace.validate_matrix(self.A4) == None


    def test_empty_matrix(self):
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A5)
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A6)

        assert trace.validate_matrix(self.A3) == None
        assert trace.validate_matrix(self.B2) == None

    def test_valid_lists(self):
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A7)

        assert trace.validate_matrix(self.B4) == None

    def test_valid_AB_size(self):
        assert trace.calculate(self.A1, self.B1, "matmul") == None
        assert trace.calculate(self.A2, self.B2, "matmul") == None
        assert trace.calculate(self.A3, self.B3, "matmul") == None
        assert trace.calculate(self.A4, self.B4, "matmul") == None

    def test_valid_elements(self):
        with pytest.raises(ValueError): 
            trace.validate_matrix(self.A8)
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A9)

    def test_C_size(self):
        trace.calculate(self.A1, self.B1, "matmul")
        assert len(trace.C()) == 4 # type: ignore
        assert len(trace.C()[0]) == 7 # type: ignore

        trace.calculate(self.A2, self.B2, "matmul")
        assert len(trace.C()) == 3 # type: ignore
        assert len(trace.C()[0]) == 2 # type: ignore

        trace.calculate(self.A3, self.B3, "matmul")
        assert len(trace.C()) == 2 # type: ignore
        assert len(trace.C()[0]) == 2 # type: ignore

        trace.calculate(self.A4, self.B4, "matmul")
        assert len(trace.C()) == 1 # type: ignore
        assert len(trace.C()) == 1 # type: ignore

    def test_result(self):
        ...

    def test_trace_count(self):
        ...

    def test_yield_data(self):
        ...

