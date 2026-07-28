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

        trace.validate_matrix(self.A1)
        trace.validate_matrix(self.A4)


    def test_empty_matrix(self):
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A5)
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A6)
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A7)

        trace.validate_matrix(self.A3)
        trace.validate_matrix(self.B2)
        trace.validate_matrix(self.B4)


    def test_valid_AB_size(self):
        trace.calculate(self.A1, self.B1, "matmul")
        trace.calculate(self.A2, self.B2, "matmul")
        trace.calculate(self.A3, self.B3, "matmul")
        trace.calculate(self.A4, self.B4, "matmul")

        with pytest.raises(ValueError):
            trace.calculate(self.A1, self.B2, "matmul")
        with pytest.raises(ValueError):
            trace.calculate(self.A2, self.B3, "matmul")
        with pytest.raises(ValueError):
            trace.calculate(self.A3, self.B4, "matmul")
             

    def test_valid_elements(self):
        with pytest.raises(ValueError): 
            trace.validate_matrix(self.A8)
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A9)

    def test_calculation(self):
        trace.calculate(self.A1, self.B1, "matmul")
        cost = trace.get_expected_cost("matmul")
        assert len(trace.C()) == 4 # type: ignore
        assert len(trace.C()[0]) == 7 # type: ignore
        assert trace.C() == [[17,40,53,50,32,35,53], 
                             [22,53,71,65,42,48,73], 
                             [17,54,72,70,47,49,53], 
                             [16,50,67,64,43,46,52]]
        assert cost == {"muls": 84, "adds": 56, "reads": 168, "writes": 28}
        

        trace.calculate(self.A2, self.B2, "matmul")
        assert len(trace.C()) == 3 # type: ignore
        assert len(trace.C()[0]) == 2 # type: ignore
        assert trace.C() == [[107,386], [124,91], [112,71]]
        cost = trace.get_expected_cost("matmul")
        assert cost == {"muls": 30, "adds": 24, "reads": 60, "writes": 6}

        trace.calculate(self.A3, self.B3, "matmul")
        assert len(trace.C()) == 2 # type: ignore
        assert len(trace.C()[0]) == 2 # type: ignore
        assert trace.C() == [[8,18], [30,50]]
        cost = trace.get_expected_cost("matmul")
        assert cost == {"muls": 8, "adds": 4, "reads": 16, "writes": 4}

        trace.calculate(self.A4, self.B4, "matmul")
        assert len(trace.C()) == 1 # type: ignore
        assert len(trace.C()[0]) == 1 # type: ignore
        assert trace.C() == [[8]]
        cost = trace.get_expected_cost("matmul")
        assert cost == {"muls": 1, "adds": 0, "reads": 2, "writes": 1}

    def test_yield_data(self):
        ...

