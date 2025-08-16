import numpy as np
from utils import longest_true_segment

def test_longest_true_segment_basic():
    mask = np.array([0,1,1,1,0,1,1,0,1], dtype=bool)
    s,e = longest_true_segment(mask)
    assert (s,e) == (1,4)
