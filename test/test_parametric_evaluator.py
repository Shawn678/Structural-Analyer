import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.parametric_evaluator import (
    build_geometry_fingerprint,
    evaluate_real_results,
    export_cache_to_txt,
    import_cache_from_txt,
)

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


# ── Task 2: evaluate_real_results ──────────────────────────────────────────

# 簡單靜定梁：兩端鉸支，中點集中載重 P=1，L=6m
BEAM_DATA = {
    "nodes": [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": 2, "x": 3.0, "y": 0.0, "z": 0.0},
        {"id": 3, "x": 6.0, "y": 0.0, "z": 0.0},
    ],
    "elements": [
        {"id": 1, "i": 1, "j": 2, "pin_i": True, "pin_j": False},
        {"id": 2, "i": 2, "j": 3, "pin_i": False, "pin_j": True},
    ],
    "supports": [
        {"node_id": 1, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": False},
        {"node_id": 3, "ux": False, "uy": True, "uz": True, "rx": True, "ry": True, "rz": False},
    ],
    "loads": [
        {"node_id": 2, "fx": 0.0, "fy": -1.0, "fz": 0.0, "mx": 0.0, "my": 0.0, "mz": 0.0}
    ],
    "element_loads": [],
    "element_point_loads": [],
}

REAL_PARAMS = {"E": 200e9, "A": 0.01, "I": 1e-4, "G": 77e9, "P": 1.0, "w": 0.0}

def test_evaluate_returns_structure():
    res = evaluate_real_results(BEAM_DATA, REAL_PARAMS)
    assert "node_displacements" in res
    assert "element_forces" in res
    assert "support_reactions" in res
    assert "cache_used" in res
    assert "eval_time_ms" in res

def test_evaluate_displacement_has_formula_and_value():
    res = evaluate_real_results(BEAM_DATA, REAL_PARAMS)
    nd = res["node_displacements"]
    assert len(nd) == 3
    for node in nd:
        for key in ("ux", "uy", "uz", "theta_x", "theta_y", "theta_z"):
            assert "formula" in node[key]
            assert "value" in node[key]
            assert isinstance(node[key]["value"], (float, type(None)))

def test_evaluate_cache_used_on_second_call():
    cache = {}
    res1 = evaluate_real_results(BEAM_DATA, REAL_PARAMS, symbolic_cache=cache)
    assert res1["cache_used"] is False  # 第一次：無快取，呼叫 run_symbolic_analysis
    res2 = evaluate_real_results(BEAM_DATA, REAL_PARAMS, symbolic_cache=cache)
    assert res2["cache_used"] is True   # 第二次：使用快取

def test_evaluate_reaction_sum_equals_load():
    res = evaluate_real_results(BEAM_DATA, REAL_PARAMS)
    reactions = res["support_reactions"]
    total_ry = sum(r["Ry"]["value"] for r in reactions if r["Ry"]["value"] is not None)
    # 靜力平衡：支承 Y 方向反力總和 = 外力 P=1
    assert abs(total_ry - 1.0) < 1e-3


# ── Task 3: export_cache_to_txt / import_cache_from_txt ───────────────────────

def test_export_txt_contains_sections():
    cache = {}
    evaluate_real_results(BEAM_DATA, REAL_PARAMS, symbolic_cache=cache)
    txt = export_cache_to_txt(cache)
    assert "[FINGERPRINT]" in txt
    assert "[FORMULAS]" in txt
    assert "[END]" in txt
    assert "n_elements=2" in txt

def test_export_txt_contains_formulas():
    cache = {}
    evaluate_real_results(BEAM_DATA, REAL_PARAMS, symbolic_cache=cache)
    txt = export_cache_to_txt(cache)
    assert "node_1_ux=" in txt
    assert "elem_1_N=" in txt
    assert "react_" in txt

def test_import_txt_roundtrip():
    cache = {}
    evaluate_real_results(BEAM_DATA, REAL_PARAMS, symbolic_cache=cache)
    txt = export_cache_to_txt(cache)
    imported = import_cache_from_txt(txt, BEAM_DATA)
    assert "error" not in imported
    assert "raw_result" in imported
    assert "elem_Ls" in imported
    assert "fingerprint" in imported

def test_import_txt_fingerprint_mismatch():
    cache = {}
    evaluate_real_results(BEAM_DATA, REAL_PARAMS, symbolic_cache=cache)
    txt = export_cache_to_txt(cache)
    # 修改幾何
    import copy
    different_data = copy.deepcopy(BEAM_DATA)
    different_data["nodes"][1]["x"] = 4.0  # 改中點位置
    result = import_cache_from_txt(txt, different_data)
    assert "error" in result
    assert "桿件" in result["error"] or "長度" in result["error"] or "不符" in result["error"]

def test_import_cache_enables_fast_eval():
    cache = {}
    evaluate_real_results(BEAM_DATA, REAL_PARAMS, symbolic_cache=cache)
    txt = export_cache_to_txt(cache)
    imported = import_cache_from_txt(txt, BEAM_DATA)
    res = evaluate_real_results(BEAM_DATA, REAL_PARAMS, symbolic_cache=imported)
    assert res["cache_used"] is True


# ── Task 3: per-section real_params & self-weight ─────────────────────────

from core.materials import expand_truss_data, compute_self_weight

MATS = [{"name":"鋼材","E":200e9,"G":77e9,"density":7850}]
SECS = [{"name":"主樑","material":"鋼材","shape":"Custom",
          "A":0.01,"I33":1e-4,"I22":1e-5,"J":1e-5}]
BEAM_SEC = {
    "nodes": [
        {"id":1,"x":0,"y":0,"z":0},
        {"id":2,"x":3,"y":0,"z":0},
        {"id":3,"x":6,"y":0,"z":0},
    ],
    "elements": [
        {"id":1,"i":1,"j":2,"pin_i":True,"pin_j":False,"section":"主樑"},
        {"id":2,"i":2,"j":3,"pin_i":False,"pin_j":True,"section":"主樑"},
    ],
    "supports": [
        {"node_id":1,"ux":True,"uy":True,"uz":True,"rx":True,"ry":True,"rz":False},
        {"node_id":3,"ux":False,"uy":True,"uz":True,"rx":True,"ry":True,"rz":False},
    ],
    "loads": [{"node_id":2,"fz":-1000.0}],
    "element_loads": [],
    "element_point_loads": [],
}

def test_evaluate_with_materials_sections():
    res = evaluate_real_results(BEAM_SEC, {}, materials=MATS, sections=SECS)
    assert "node_displacements" in res
    assert "support_reactions" in res

def test_self_weight_increases_reaction():
    res_no_sw  = evaluate_real_results(BEAM_SEC, {}, materials=MATS, sections=SECS,
                                        include_self_weight=False)
    res_with_sw = evaluate_real_results(BEAM_SEC, {}, materials=MATS, sections=SECS,
                                         include_self_weight=True)
    # 加入自重後，支承 Z 方向反力總和應增加
    def total_rz(res):
        return sum(r["Rz"]["value"] for r in res["support_reactions"]
                   if r.get("Rz",{}).get("value") is not None)
    # 自重向下（-Z），反力為正 Z，故 total_rz 應變大（更正）
    assert total_rz(res_with_sw) > total_rz(res_no_sw)

def test_txt_export_contains_materials_sections():
    cache = {}
    evaluate_real_results(BEAM_SEC, {}, symbolic_cache=cache, materials=MATS, sections=SECS)
    txt = export_cache_to_txt(cache)
    assert "[MATERIALS]" in txt
    assert "[SECTIONS]" in txt
    assert "鋼材" in txt
    assert "主樑" in txt

def test_txt_roundtrip_restores_materials_sections():
    cache = {}
    evaluate_real_results(BEAM_SEC, {}, symbolic_cache=cache, materials=MATS, sections=SECS)
    txt = export_cache_to_txt(cache)
    imported = import_cache_from_txt(txt, BEAM_SEC)
    assert "error" not in imported
    assert imported.get("materials") == MATS
    assert imported.get("sections")[0]["name"] == "主樑"


# ── 多斷面組測試 ──────────────────────────────────────────────────────────
import sympy as sp

MIXED_SECTION_DATA = {
    "nodes": [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": 2, "x": 6.0, "y": 0.0, "z": 0.0},
        {"id": 3, "x": 3.0, "y": 3.0, "z": 0.0},
    ],
    "elements": [
        {"id": 1, "i": 1, "j": 2, "section": "S1", "E": 200e9, "A": 0.01, "I33": 1e-4, "I22": 1e-4, "G": 77e9},
        {"id": 2, "i": 2, "j": 3, "section": "S2", "E": 200e9, "A": 0.02, "I33": 2e-4, "I22": 2e-4, "G": 77e9},
    ],
    "supports": [
        {"node_id": 1, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True},
        {"node_id": 3, "ux": False, "uy": True, "uz": True, "rx": False, "ry": False, "rz": False},
    ],
    "loads": [{"node_id": 2, "fy": -10000.0}],
    "element_loads": [],
    "element_point_loads": [],
}

SECTION_GROUP_MAP = {
    "S1": {
        "E": sp.Symbol("E_s1"), "A": sp.Symbol("A_s1"),
        "I33": sp.Symbol("I_s1"), "I22": sp.Symbol("I_s1"), "G": sp.Symbol("G_s1"),
    },
    "S2": {
        "E": sp.Symbol("E_s2"), "A": sp.Symbol("A_s2"),
        "I33": sp.Symbol("I_s2"), "I22": sp.Symbol("I_s2"), "G": sp.Symbol("G_s2"),
    },
}

def test_per_section_cache_stores_sym_names():
    """快取應儲存 section_groups 與 section_sym_names。"""
    cache = {}
    evaluate_real_results(
        MIXED_SECTION_DATA,
        {"E": 200e9, "A": 0.01, "I": 1e-4, "G": 77e9, "P": 1.0, "w": 0.0},
        symbolic_cache=cache,
        section_group_map=SECTION_GROUP_MAP,
    )
    assert "section_groups" in cache
    assert "section_sym_names" in cache
    assert "S1" in cache["section_sym_names"]
    assert cache["section_sym_names"]["S1"]["E"] == "E_s1"

def test_per_section_groups_substitution():
    """groups 格式的 real_params 應正確代入各斷面符號，位移非零。"""
    cache = {}
    # 第一次：建立快取（使用 section_group_map）
    evaluate_real_results(
        MIXED_SECTION_DATA,
        {"E": 200e9, "A": 0.01, "I": 1e-4, "G": 77e9, "P": 1.0, "w": 0.0},
        symbolic_cache=cache,
        section_group_map=SECTION_GROUP_MAP,
    )
    # 第二次：使用 groups 格式快速代入
    real_params = {
        "groups": {
            "S1": {"E": 210e9, "A": 0.012, "I": 1.2e-4, "G": 80e9},
            "S2": {"E": 200e9, "A": 0.020, "I": 2.0e-4, "G": 77e9},
        },
        "P": 1.0,
        "w": 0.0,
    }
    res = evaluate_real_results(MIXED_SECTION_DATA, real_params, symbolic_cache=cache)
    # node 2 應有非零位移（受力節點，fy 載重 → uy 位移）
    nd2 = next(n for n in res["node_displacements"] if n["node_id"] == 2)
    uy_val = nd2["uy"]["value"]
    assert uy_val is not None
    assert abs(uy_val) > 1e-10
