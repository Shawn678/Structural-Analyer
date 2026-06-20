import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import math
from core.materials import compute_section_props, expand_truss_data, compute_self_weight

# ── compute_section_props ──────────────────────────────────────────────────

def test_rect_solid():
    p = compute_section_props("矩形實心", {"b": 0.2, "h": 0.4})
    assert abs(p["A"]   - 0.2*0.4) < 1e-12
    assert abs(p["I33"] - 0.2*0.4**3/12) < 1e-12
    assert abs(p["I22"] - 0.4*0.2**3/12) < 1e-12
    # J > 0
    assert p["J"] > 0

def test_circle_solid():
    p = compute_section_props("圓形實心", {"d": 0.3})
    assert abs(p["A"]   - math.pi*0.3**2/4) < 1e-12
    assert abs(p["I33"] - math.pi*0.3**4/64) < 1e-12
    assert abs(p["I22"] - p["I33"]) < 1e-12
    assert abs(p["J"]   - math.pi*0.3**4/32) < 1e-12

def test_rect_tube():
    p = compute_section_props("矩形管", {"b": 0.2, "h": 0.3, "t": 0.01})
    expected_A = 0.2*0.3 - (0.2-0.02)*(0.3-0.02)
    assert abs(p["A"] - expected_A) < 1e-10
    assert p["I33"] > 0
    assert p["J"] > 0

def test_circle_tube():
    p = compute_section_props("圓管", {"d": 0.3, "t": 0.01})
    expected_A = math.pi * (0.3**2 - (0.3-0.02)**2) / 4
    assert abs(p["A"] - expected_A) < 1e-10
    assert p["I33"] > 0
    assert p["J"] > 0

def test_I_section():
    p = compute_section_props("I形", {"H": 0.3, "bf": 0.15, "tf": 0.01, "tw": 0.008})
    bw = 0.3 - 2*0.01  # 0.28
    expected_A = 2*0.15*0.01 + bw*0.008
    assert abs(p["A"] - expected_A) < 1e-10
    assert p["I33"] > 0
    assert p["I22"] > 0
    assert p["J"] > 0

def test_custom_passthrough():
    p = compute_section_props("Custom", {"A": 0.05, "I33": 1e-4, "I22": 5e-5, "J": 2e-5})
    assert p["A"]   == 0.05
    assert p["I33"] == 1e-4
    assert p["I22"] == 5e-5
    assert p["J"]   == 2e-5

# ── expand_truss_data ──────────────────────────────────────────────────────

MATERIALS = [
    {"name": "鋼材", "E": 200e9, "G": 77e9, "density": 7850},
]
SECTIONS = [
    {"name": "主樑", "material": "鋼材", "shape": "Custom",
     "A": 0.01, "I33": 1e-4, "I22": 1e-5, "J": 1e-5},
    {"name": "斜柱", "material": "鋼材", "shape": "Custom",
     "A": 0.005, "I33": 5e-5, "I22": 5e-6, "J": 5e-6},
]
TRUSS = {
    "nodes": [{"id":1,"x":0,"y":0,"z":0},{"id":2,"x":6,"y":0,"z":0}],
    "elements": [{"id":1,"i":1,"j":2,"section":"主樑"}],
    "supports": [], "loads": [], "element_loads": [], "element_point_loads": [],
}

def test_expand_fills_E_G_A():
    td = expand_truss_data(TRUSS, MATERIALS, SECTIONS)
    e = td["elements"][0]
    assert e["E"]   == 200e9
    assert e["G"]   == 77e9
    assert e["A"]   == 0.01
    assert e["I33"] == 1e-4

def test_expand_respects_override():
    import copy
    truss = copy.deepcopy(TRUSS)
    truss["elements"][0]["I33"] = 9e-4   # local override
    td = expand_truss_data(truss, MATERIALS, SECTIONS)
    assert td["elements"][0]["I33"] == 9e-4   # override 保留
    assert td["elements"][0]["E"]   == 200e9  # 其餘從 section 帶入

def test_expand_does_not_mutate_original():
    import copy
    truss = copy.deepcopy(TRUSS)
    expand_truss_data(truss, MATERIALS, SECTIONS)
    assert "E" not in truss["elements"][0]   # 原始資料不應被修改

# ── compute_self_weight ────────────────────────────────────────────────────

def test_self_weight_value():
    td_exp = expand_truss_data(TRUSS, MATERIALS, SECTIONS)
    sw = compute_self_weight(td_exp, SECTIONS, MATERIALS)
    # w = density * A * g = 7850 * 0.01 * 9.81 = 770.085 N/m，向下為負
    expected_w = -(7850 * 0.01 * 9.81)
    assert len(sw) == 1
    assert sw[0]["element_id"] == 1
    assert abs(sw[0]["w"] - expected_w) < 0.01

def test_self_weight_override_uses_overridden_A():
    import copy
    truss = copy.deepcopy(TRUSS)
    truss["elements"][0]["A"] = 0.02   # override A
    td_exp = expand_truss_data(truss, MATERIALS, SECTIONS)
    sw = compute_self_weight(td_exp, SECTIONS, MATERIALS)
    expected_w = -(7850 * 0.02 * 9.81)
    assert abs(sw[0]["w"] - expected_w) < 0.01
