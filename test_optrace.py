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
    A12 = [[4,9,True, 0],[0,2,4,5]]


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
        for _ in trace.calculate(self.A1, self.B1, "matmul"): pass
        for _ in trace.calculate(self.A2, self.B2, "matmul"): pass
        for _ in trace.calculate(self.A3, self.B3, "matmul"): pass
        for _ in trace.calculate(self.A4, self.B4, "matmul"): pass

        with pytest.raises(ValueError):
            for _ in trace.calculate(self.A1, self.B2, "matmul"): pass
        with pytest.raises(ValueError):
            for _ in trace.calculate(self.A2, self.B3, "matmul"): pass
        with pytest.raises(ValueError):
            for _ in trace.calculate(self.A3, self.B4, "matmul"): pass
             

    def test_valid_elements(self):
        with pytest.raises(ValueError): 
            trace.validate_matrix(self.A8)
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A9)
        with pytest.raises(ValueError):
            trace.validate_matrix(self.A12)

    def test_calculation(self):
        for _ in trace.calculate(self.A1, self.B1, "matmul"): pass
        cost = trace.get_expected_cost("matmul")
        assert len(trace.C()) == 4 # type: ignore
        assert len(trace.C()[0]) == 7 # type: ignore
        assert trace.C() == [[17,40,53,50,32,35,53], 
                             [22,53,71,65,42,48,73], 
                             [17,54,72,70,47,49,53], 
                             [16,50,67,64,43,46,52]]
        assert cost == {"muls": 84, "adds": 56, "reads": 168, "writes": 28}
        

        for _ in trace.calculate(self.A2, self.B2, "matmul"): pass
        assert len(trace.C()) == 3 # type: ignore
        assert len(trace.C()[0]) == 2 # type: ignore
        assert trace.C() == [[107,386], [124,91], [112,71]]
        cost = trace.get_expected_cost("matmul")
        assert cost == {"muls": 30, "adds": 24, "reads": 60, "writes": 6}

        for _ in trace.calculate(self.A3, self.B3, "matmul"): pass
        assert len(trace.C()) == 2 # type: ignore
        assert len(trace.C()[0]) == 2 # type: ignore
        assert trace.C() == [[8,18], [30,50]]
        cost = trace.get_expected_cost("matmul")
        assert cost == {"muls": 8, "adds": 4, "reads": 16, "writes": 4}

        for _ in trace.calculate(self.A4, self.B4, "matmul"): pass
        assert len(trace.C()) == 1 # type: ignore
        assert len(trace.C()[0]) == 1 # type: ignore
        assert trace.C() == [[8]]
        cost = trace.get_expected_cost("matmul")
        assert cost == {"muls": 1, "adds": 0, "reads": 2, "writes": 1}

    def test_yield_data(self):
        events = list(trace.calculate(self.A1, self.B1, "matmul"))

        m = len(self.A1)
        n = len(self.B1)
        p = len(self.B1[0])

        compute_events = m*n*p
        write_events = m*p

        assert len(events) == compute_events + write_events + 1

        expected_init = {
            "event": "init", 
            "A_rows": m, "A_cols": len(self.A1[0]), 
            "B_rows": n, "B_cols": p, 
            "C_rows": m, "C_cols": p,
            }

        assert events[0] == expected_init

        expected_compute_1 = {
            "event": "compute", 
            "muls": 1, "adds": 0, "reads": 2, "flops": 1, 
            "i": 0, "j": 0, "k": 0
        }

        assert events[1] == expected_compute_1

        expected_write_1 = {
            "event": "write",
            "writes": 1,
            "i": 0, "j": 0,
        }

        assert events[len(self.B1) + 1] == expected_write_1

        expected_compute_final = {
            "event": "compute",
            "muls": m*n*p,
            "adds": m*p*(n-1),
            "reads": 2*m*n*p,
            "flops": (m*p*(n-1)) + (m*n*p),
            "i": m - 1,
            "j": p - 1,
            "k": n - 1,
        }

        assert events[-2] == expected_compute_final


        events = list(trace.calculate(self.A3, self.B3, "matmul"))
        m = len(self.A3)
        n = len(self.B3)
        p = len(self.B3[0])

        expected_events = [
            {
                "event": "init", 
                "A_rows": m, "A_cols": len(self.A3[0]), 
                "B_rows": n, "B_cols": p, 
                "C_rows": m, "C_cols": p,
            },
            {
                "event": "compute",
                "muls": 1, "adds": 0, "reads": 2, "flops": 1, 
                "i": 0, "j": 0, "k": 0
            },
            {
                "event": "compute", 
                "muls": 2, "adds": 1, "reads": 4, "flops": 3, 
                "i": 0, "j": 0, "k": 1
            },
            {
                "event": "write",
                "writes": 1,
                "i": 0, "j": 0,
            },
            {
                "event": "compute", 
                "muls": 3, "adds": 1, "reads": 6, "flops": 4, 
                "i": 0, "j": 1, "k": 0
            },
            {
                "event": "compute", 
                "muls": 4, "adds": 2, "reads": 8, "flops": 6, 
                "i": 0, "j": 1, "k": 1
            },
            {
                "event": "write",
                "writes": 2,
                "i": 0, "j": 1,
            },
            {
                "event": "compute", 
                "muls": 5, "adds": 2, "reads": 10, "flops": 7, 
                "i": 1, "j": 0, "k": 0
            },
            {
                "event": "compute", 
                "muls": 6, "adds": 3, "reads": 12, "flops": 9, 
                "i": 1, "j": 0, "k": 1
            },
            {
                "event": "write",
                "writes": 3,
                "i": 1, "j": 0,
            },
            {
                "event": "compute", 
                "muls": 7, "adds": 3, "reads": 14, "flops": 10, 
                "i": 1, "j": 1, "k": 0
            },
            {
                "event": "compute", 
                "muls": 8, "adds": 4, "reads": 16, "flops": 12, 
                "i": 1, "j": 1, "k": 1
            },
            {
                "event": "write",
                "writes": 4,
                "i": 1, "j": 1,
            },
        ]

        assert events == expected_events


    

