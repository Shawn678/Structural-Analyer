import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.parametric_evaluator import build_geometry_fingerprint

TRUSS_DATA = {
    "nodes": [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": 2, "x": 6.0, "y": 0.0, "z": 0.0},
        {"id": 3, "x": 3.0, "y": 3.0, "z": 0.0},
    ],
    "elements": [
        {"id": 1, "i": 1, "j": 2, "E": 200e9, "A": 0.01, "I33": 1e-4},
        {"id": 2, "i": 2, "j": 3, "E": 200e9, "A": 0.01, "I33": 1e-4},
    ],
    "supports": [
        {"node_id": 1, "ux": True, "uy": True, "uz": True, "rx": False, "ry": False, "rz": False},
        {"node_id": 2, "uy": True, "uz": True, "rx": False, "ry": False, "ux": False, "rz": False},
    ],
    "loads": [],
    "element_loads": [],
    "element_point_loads": [],
}

def test_fingerprint_n_elements():
    fp = build_geometry_fingerprint(TRUSS_DATA)
    assert fp["n_elements"] == 2

def test_fingerprint_lengths():
    fp = build_geometry_fingerprint(TRUSS_DATA)
    assert fp["elem_lengths"][0] == "6.000000"
    import math
    expected = round(math.sqrt(9 + 9), 6)
    assert fp["elem_lengths"][1] == f"{expected:.6f}"

def test_fingerprint_connections():
    fp = build_geometry_fingerprint(TRUSS_DATA)
    assert fp["connections"][0] == "1-2"
    assert fp["connections"][1] == "2-3"

def test_fingerprint_supports():
    fp = build_geometry_fingerprint(TRUSS_DATA)
    # 支承按 node_id 排序，只列出 True 的約束
    assert "1:ux,uy,uz" in fp["supports"][0]
    assert "2:uy,uz" in fp["supports"][1]

def test_fingerprint_ignores_material():
    import copy
    td2 = copy.deepcopy(TRUSS_DATA)
    td2["elements"][0]["E"] = 70e9  # 改材料
    fp1 = build_geometry_fingerprint(TRUSS_DATA)
    fp2 = build_geometry_fingerprint(td2)
    assert fp1 == fp2
