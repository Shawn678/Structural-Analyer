import math
import numpy as np
import sympy as sp
import time
from datetime import datetime
from io import StringIO

from core.symbolic import run_symbolic_analysis


def build_geometry_fingerprint(truss_data: dict) -> dict:
    """計算幾何+支承指紋，不含材料參數與載重數值。"""
    node_id_to_pos = {
        n["id"]: (float(n.get("x", 0)), float(n.get("y", 0)), float(n.get("z", 0)))
        for n in truss_data["nodes"]
    }

    elem_lengths = []
    connections = []
    for elem in truss_data["elements"]:
        xi, yi, zi = node_id_to_pos[elem["i"]]
        xj, yj, zj = node_id_to_pos[elem["j"]]
        Le = math.sqrt((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2)
        elem_lengths.append(f"{Le:.6f}")
        connections.append(f"{elem['i']}-{elem['j']}")

    CONSTRAINT_KEYS = ["kx", "ky", "kt", "rx", "ry", "rz", "ux", "uy", "uz"]
    supports_fp = []
    for sup in sorted(truss_data.get("supports", []), key=lambda s: s["node_id"]):
        nid = sup["node_id"]
        active = []
        for k in CONSTRAINT_KEYS:
            v = sup.get(k, 0)
            if v is True or (isinstance(v, (int, float)) and abs(float(v)) > 1e-15):
                active.append(f"{k}={v}" if k in ("kx", "ky", "kt") else k)
        if active:
            supports_fp.append(f"{nid}:{','.join(active)}")

    return {
        "n_elements": len(truss_data["elements"]),
        "elem_lengths": elem_lengths,
        "connections": connections,
        "supports": supports_fp,
    }


def _subs_value(formula_str: str, subs_dict: dict) -> float | None:
    """將公式字串代入數值，回傳 float；失敗回傳 None。"""
    try:
        expr = sp.sympify(formula_str)
        result = float(expr.subs(subs_dict))
        return result
    except Exception:
        return None


def evaluate_real_results(
    truss_data: dict,
    real_params: dict,
    symbolic_cache: dict | None = None,
) -> dict:
    """
    將實際材料/載重參數代入符號公式，回傳數值結果。
    symbolic_cache 為可變 dict：首次呼叫後會填入快取，後續呼叫直接重用。
    幾何或支承改變時請傳入空 dict {} 讓函數重新求解。
    """
    t0 = time.time()
    cache_used = False

    # ── 取得或建立快取 ────────────────────────────────────────────────────
    if symbolic_cache is not None and "raw_result" in symbolic_cache:
        cache_used = True
        raw = symbolic_cache["raw_result"]
        elem_Ls = symbolic_cache["elem_Ls"]
    else:
        raw = run_symbolic_analysis(truss_data)
        elem_Ls = []
        # 從節點座標重建桿件長度（與 symbolic.py 的 elements_info 順序一致）
        node_pos = {n["id"]: (float(n.get("x",0)), float(n.get("y",0)), float(n.get("z",0)))
                    for n in truss_data["nodes"]}
        for elem in truss_data["elements"]:
            xi, yi, zi = node_pos[elem["i"]]
            xj, yj, zj = node_pos[elem["j"]]
            elem_Ls.append(math.sqrt((xj-xi)**2+(yj-yi)**2+(zj-zi)**2))
        if symbolic_cache is not None:
            symbolic_cache["raw_result"] = raw
            symbolic_cache["elem_Ls"]    = elem_Ls
            symbolic_cache["fingerprint"] = build_geometry_fingerprint(truss_data)
            symbolic_cache["timestamp"]   = datetime.now().strftime("%Y-%m-%dT%H:%M")

    # ── 建立代入字典 ──────────────────────────────────────────────────────
    E_s, A_s, I_s, G_s = sp.symbols("E A I G", positive=True)
    P_s, w_s = sp.symbols("P w")
    L_syms = [sp.Symbol(f"L_{k+1}") for k in range(len(elem_Ls))]

    subs_dict = {
        E_s: float(real_params.get("E", 200e9)),
        A_s: float(real_params.get("A", 0.01)),
        I_s: float(real_params.get("I", 1e-4)),
        G_s: float(real_params.get("G", 77e9)),
        P_s: float(real_params.get("P", 1.0)),
        w_s: float(real_params.get("w", 0.0)),
    }
    for k, Lk_sym in enumerate(L_syms):
        subs_dict[Lk_sym] = float(elem_Ls[k])

    # ── 代入節點位移 ──────────────────────────────────────────────────────
    node_displacements = []
    for nd in raw["node_displacements"]:
        entry = {"node_id": nd["node_id"]}
        for key in ("ux", "uy", "uz", "theta_x", "theta_y", "theta_z"):
            formula = nd.get(key, "0")
            entry[key] = {"formula": formula, "value": _subs_value(formula, subs_dict)}
        node_displacements.append(entry)

    # ── 代入桿件內力 ──────────────────────────────────────────────────────
    element_forces = []
    for ef in raw["element_forces"]:
        eqs = ef.get("equations", {})
        entry = {
            "element_id": ef["element_id"],
            "nodes":      ef["nodes"],
            "i_end":      ef.get("i_end (N, V2, V3, T, M2, M3)", ""),
            "j_end":      ef.get("j_end (N, V2, V3, T, M2, M3)", ""),
        }
        for sym_key, out_key in [("N(x)","N"), ("V2(x)","V2"), ("V3(x)","V3"),
                                   ("M3(x)","M3"), ("M2(x)","M2")]:
            formula = eqs.get(sym_key, "0")
            entry[out_key] = {"formula": formula, "value": _subs_value(formula, subs_dict)}
        element_forces.append(entry)

    # ── 代入支承反力 ──────────────────────────────────────────────────────
    support_reactions = []
    for sr in raw["support_reactions"]:
        entry = {"node_id": sr["node_id"]}
        for key in ("Rx", "Ry", "Rz", "Mx", "My", "Mz"):
            formula = sr.get(key, "0")
            entry[key] = {"formula": formula, "value": _subs_value(formula, subs_dict)}
        support_reactions.append(entry)

    eval_ms = int((time.time() - t0) * 1000)
    return {
        "node_displacements": node_displacements,
        "element_forces":     element_forces,
        "support_reactions":  support_reactions,
        "cache_used":         cache_used,
        "eval_time_ms":       eval_ms,
    }
