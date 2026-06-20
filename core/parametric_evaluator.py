import copy
import math
import numpy as np
import sympy as sp
import time
from datetime import datetime

from core.symbolic import run_symbolic_analysis, run_numerical_analysis
from core.materials import expand_truss_data, compute_self_weight


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
    """將公式字串代入數值，回傳 float；失敗或結果非有限數時回傳 None。"""
    try:
        expr = sp.sympify(formula_str)
        result = float(expr.subs(subs_dict))
        return result if np.isfinite(result) else None
    except Exception:
        return None


def evaluate_real_results(
    truss_data: dict,
    real_params: dict,
    symbolic_cache: dict | None = None,
    materials: list | None = None,
    sections: list | None = None,
    include_self_weight: bool = False,
) -> dict:
    """
    將實際材料/載重參數代入符號公式，回傳數值結果。
    symbolic_cache 為可變 dict：首次呼叫後會填入快取，後續呼叫直接重用。
    幾何或支承改變時請傳入空 dict {} 讓函數重新求解。
    """
    t0 = time.time()
    cache_used = False

    # 若提供 materials/sections，先展開 truss_data
    if materials and sections:
        truss_data = expand_truss_data(truss_data, materials, sections)

    # 自重疊加
    if include_self_weight and materials and sections:
        sw_loads = compute_self_weight(truss_data, sections, materials)
        truss_data = copy.deepcopy(truss_data)
        existing = {el["element_id"]: el for el in truss_data.get("element_loads", [])}
        for sw in sw_loads:
            eid = sw["element_id"]
            if eid in existing:
                existing[eid]["w"] = existing[eid].get("w", 0.0) + sw["w"]
            else:
                truss_data["element_loads"].append({"element_id": eid, "w": sw["w"]})

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
            if materials:
                symbolic_cache["materials"] = materials
            if sections:
                symbolic_cache["sections"]  = sections

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

    # ── 代入節點位移（用符號公式，保有 E/I/L 代數式）───────────────────
    node_displacements = []
    for nd in raw["node_displacements"]:
        entry = {"node_id": nd["node_id"]}
        for key in ("ux", "uy", "uz", "theta_x", "theta_y", "theta_z"):
            formula = nd.get(key, "0")
            entry[key] = {"formula": formula, "value": _subs_value(formula, subs_dict)}
        node_displacements.append(entry)

    # ── 桿件內力與支承反力：直接數值求解，E/A/I/G 完全有效 ─────────────
    # 將 real_params 中的材料參數寫回 truss_data 的每根桿件
    # （只有在桿件本身沒設定 E/A/I33 的情況下才用全局參數）
    _E = float(real_params.get("E", 200e9))
    _A = float(real_params.get("A", 0.01))
    _I = float(real_params.get("I", 1e-4))
    _G = float(real_params.get("G", 77e9))
    _P = float(real_params.get("P", 1.0))
    _w = float(real_params.get("w", 0.0))

    # 複製 truss_data，將載重乘上倍率，並以全局材料參數覆蓋所有桿件
    # 若已提供 materials/sections，桿件材料由 expand_truss_data 填入，不以全局覆蓋
    td_num = copy.deepcopy(truss_data)
    use_per_elem = bool(materials and sections)
    for elem in td_num["elements"]:
        if not use_per_elem:
            elem["E"]   = _E
            elem["A"]   = _A
            elem["I33"] = _I
            elem["I22"] = _I
            elem["G"]   = _G
    for load in td_num["loads"]:
        for k in ("fx", "fy", "fz", "mx", "my", "mz"):
            if k in load:
                load[k] = float(load[k]) * _P
    if not use_per_elem:
        # 符號模式：element_loads 的 w 是係數，需乘上倍率
        for el in td_num["element_loads"]:
            if "w" in el:
                el["w"] = float(el["w"]) * _w
    # 若 use_per_elem，element_loads 已是實際數值（自重），直接使用
    for el in td_num["element_point_loads"]:
        if "p" in el:
            el["p"] = float(el["p"]) * _P

    num = run_numerical_analysis(td_num)

    element_forces = []
    for ef_sym, ef_num in zip(raw["element_forces"], num["element_forces"]):
        eqs = ef_sym.get("equations", {})
        entry = {
            "element_id": ef_sym["element_id"],
            "nodes":      ef_sym["nodes"],
            "i_end":      ef_sym.get("i_end (N, V2, V3, T, M2, M3)", ""),
            "j_end":      ef_sym.get("j_end (N, V2, V3, T, M2, M3)", ""),
        }
        for sym_key, out_key, num_key in [
            ("N(x)",  "N",  "N"),
            ("V2(x)", "V2", "V2"),
            ("V3(x)", "V3", "V3"),
            ("M3(x)", "M3", "M3_i"),
            ("M2(x)", "M2", "M2_i"),
        ]:
            formula = eqs.get(sym_key, "0")
            entry[out_key] = {"formula": formula, "value": ef_num.get(num_key)}
        element_forces.append(entry)

    support_reactions = []
    for sr_num in num["support_reactions"]:
        entry = {"node_id": sr_num["node_id"]}
        for key in ("Rx", "Ry", "Rz", "Mx", "My", "Mz"):
            # 從符號快取找對應公式（僅供展示）
            sr_sym = next((s for s in raw["support_reactions"]
                           if s["node_id"] == sr_num["node_id"]), {})
            entry[key] = {"formula": sr_sym.get(key, "0"), "value": sr_num.get(key, 0.0)}
        support_reactions.append(entry)

    eval_ms = int((time.time() - t0) * 1000)
    return {
        "node_displacements": node_displacements,
        "element_forces":     element_forces,
        "support_reactions":  support_reactions,
        "cache_used":         cache_used,
        "eval_time_ms":       eval_ms,
    }


def evaluate_numerical_results(truss_data: dict) -> dict:
    """
    第二階段數值分析：直接使用 truss_data 內的實際 E/A/I/G/L 與載重數值求解。
    不依賴符號快取，每次呼叫均重新計算。
    """
    t0 = time.time()
    raw = run_numerical_analysis(truss_data)
    eval_ms = int((time.time() - t0) * 1000)
    raw["eval_time_ms"] = eval_ms
    return raw


def export_cache_to_txt(symbolic_cache: dict) -> str:
    """
    將 symbolic_cache 序列化為人類可讀 TXT 字串。
    呼叫方負責寫檔或透過 Streamlit 下載。
    """
    fp = symbolic_cache.get("fingerprint", {})
    raw = symbolic_cache.get("raw_result", {})
    elem_Ls = symbolic_cache.get("elem_Ls", [])
    timestamp = symbolic_cache.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M"))

    lines = [
        "# TRUSS SYMBOLIC CACHE v1",
        f"# Generated: {timestamp}",
    ]

    # MATERIALS 區塊
    mats = symbolic_cache.get("materials", [])
    if mats:
        lines.append("[MATERIALS]")
        lines.append("name,E,G,density")
        for m in mats:
            lines.append(f"{m['name']},{m['E']},{m['G']},{m['density']}")

    # SECTIONS 區塊
    secs = symbolic_cache.get("sections", [])
    if secs:
        lines.append("[SECTIONS]")
        lines.append("name,material,shape,A,I33,I22,J")
        for s in secs:
            lines.append(
                f"{s['name']},{s.get('material','')},{s.get('shape','Custom')},"
                f"{s.get('A',0)},{s.get('I33',0)},{s.get('I22',0)},{s.get('J',0)}"
            )

    lines += [
        "[FINGERPRINT]",
        f"n_elements={fp.get('n_elements', len(elem_Ls))}",
        f"elem_lengths={','.join(fp.get('elem_lengths', [f'{L:.6f}' for L in elem_Ls]))}",
        f"connections={','.join(fp.get('connections', []))}",
        f"supports={'|'.join(fp.get('supports', []))}",
        f"timestamp={timestamp}",
        "[FORMULAS]",
    ]

    for nd in raw.get("node_displacements", []):
        nid = nd["node_id"]
        for key in ("ux", "uy", "uz", "theta_x", "theta_y", "theta_z"):
            lines.append(f"node_{nid}_{key}={nd.get(key, '0')}")

    for ef in raw.get("element_forces", []):
        eid = ef["element_id"]
        eqs = ef.get("equations", {})
        for sym_key, out_key in [("N(x)","N"), ("V2(x)","V2"), ("V3(x)","V3"),
                                   ("M3(x)","M3"), ("M2(x)","M2")]:
            lines.append(f"elem_{eid}_{out_key}={eqs.get(sym_key, '0')}")

    for sr in raw.get("support_reactions", []):
        nid = sr["node_id"]
        for key in ("Rx", "Ry", "Rz", "Mx", "My", "Mz"):
            lines.append(f"react_{nid}_{key}={sr.get(key, '0')}")

    lines.append("[END]")
    return "\n".join(lines)


def import_cache_from_txt(txt_content: str, truss_data: dict) -> dict:
    """
    從 TXT 字串重建 symbolic_cache 並驗證指紋。
    成功：回傳可直接傳入 evaluate_real_results 的 cache dict。
    失敗：回傳 {"error": "說明文字"}。
    """
    if "[FINGERPRINT]" not in txt_content or "[FORMULAS]" not in txt_content:
        return {"error": "TXT 格式無效，缺少 [FINGERPRINT] 或 [FORMULAS] 區塊"}
    if "[END]" not in txt_content:
        return {"error": "TXT 格式無效，缺少 [END] 標記（檔案可能損壞）"}

    # ── 解析區塊 ──────────────────────────────────────────────────────────
    sections = {}
    current = None
    for line in txt_content.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current:
            sections[current].append(line)

    # ── 解析指紋 ──────────────────────────────────────────────────────────
    fp_lines = {kv.split("=", 1)[0]: kv.split("=", 1)[1]
                for kv in sections.get("FINGERPRINT", []) if "=" in kv}

    cached_fp = {
        "n_elements": int(fp_lines.get("n_elements", 0)),
        "elem_lengths": fp_lines.get("elem_lengths", "").split(","),
        "connections":  fp_lines.get("connections", "").split(","),
        "supports":     [s for s in fp_lines.get("supports", "").split("|") if s],
    }

    # ── 比對指紋 ──────────────────────────────────────────────────────────
    current_fp = build_geometry_fingerprint(truss_data)

    if cached_fp["n_elements"] != current_fp["n_elements"]:
        return {"error": f"桿件數量不符：快取 {cached_fp['n_elements']} 根 vs 當前 {current_fp['n_elements']} 根"}

    for k, (ca, cu) in enumerate(zip(cached_fp["elem_lengths"], current_fp["elem_lengths"])):
        if ca != cu:
            return {"error": f"桿件 {k+1} 長度不符：快取 {ca} m vs 當前 {cu} m"}

    for k, (ca, cu) in enumerate(zip(cached_fp["connections"], current_fp["connections"])):
        if ca != cu:
            return {"error": f"桿件 {k+1} 連接不符：快取 {ca} vs 當前 {cu}"}

    cached_sup_set = set(cached_fp["supports"])
    current_sup_set = set(current_fp["supports"])
    if cached_sup_set != current_sup_set:
        diff = cached_sup_set.symmetric_difference(current_sup_set)
        return {"error": f"支承條件不符，差異：{', '.join(sorted(diff))}"}

    # ── 解析公式，重建 raw_result ─────────────────────────────────────────
    formulas = {}
    for kv in sections.get("FORMULAS", []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            formulas[k.strip()] = v.strip()

    node_ids = sorted({int(k.split("_")[1]) for k in formulas if k.startswith("node_")})
    node_displacements = []
    for nid in node_ids:
        nd = {"node_id": nid}
        for key in ("ux", "uy", "uz", "theta_x", "theta_y", "theta_z"):
            nd[key] = formulas.get(f"node_{nid}_{key}", "0")
        node_displacements.append(nd)

    elem_ids = sorted({int(k.split("_")[1]) for k in formulas if k.startswith("elem_")})
    element_forces = []
    for eid in elem_ids:
        eqs = {}
        for sym_key, out_key in [("N(x)","N"), ("V2(x)","V2"), ("V3(x)","V3"),
                                   ("M3(x)","M3"), ("M2(x)","M2")]:
            formula = formulas.get(f"elem_{eid}_{out_key}", "0")
            eqs[sym_key] = formula
        element_forces.append({"element_id": eid, "nodes": "", "equations": eqs,
                                "i_end (N, V2, V3, T, M2, M3)": "",
                                "j_end (N, V2, V3, T, M2, M3)": ""})

    react_ids = sorted({int(k.split("_")[1]) for k in formulas if k.startswith("react_")})
    support_reactions = []
    for nid in react_ids:
        sr = {"node_id": nid}
        for key in ("Rx", "Ry", "Rz", "Mx", "My", "Mz"):
            sr[key] = formulas.get(f"react_{nid}_{key}", "0")
        support_reactions.append(sr)

    elem_Ls = [float(L) for L in current_fp["elem_lengths"]]  # 從當前幾何取長度

    # 解析 MATERIALS
    materials_out = []
    mat_lines = sections.get("MATERIALS", [])
    if len(mat_lines) > 1:   # 第一行是 header
        for row in mat_lines[1:]:
            parts = row.split(",")
            if len(parts) >= 4:
                materials_out.append({
                    "name": parts[0], "E": float(parts[1]),
                    "G": float(parts[2]), "density": float(parts[3]),
                })

    # 解析 SECTIONS
    sections_out = []
    sec_lines = sections.get("SECTIONS", [])
    if len(sec_lines) > 1:
        for row in sec_lines[1:]:
            parts = row.split(",")
            if len(parts) >= 7:
                sections_out.append({
                    "name": parts[0], "material": parts[1],
                    "shape": parts[2],
                    "A":   float(parts[3]), "I33": float(parts[4]),
                    "I22": float(parts[5]), "J":   float(parts[6]),
                })

    return {
        "raw_result": {
            "node_displacements": node_displacements,
            "element_forces":     element_forces,
            "support_reactions":  support_reactions,
        },
        "elem_Ls":     elem_Ls,
        "fingerprint": current_fp,
        "timestamp":   fp_lines.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M")),
        "materials":   materials_out,
        "sections":    sections_out,
    }
