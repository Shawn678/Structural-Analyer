import math
import copy

# ── 截面幾何計算 ───────────────────────────────────────────────────────────

def compute_section_props(shape: str, params: dict) -> dict:
    if shape == "Custom":
        return {
            "A":   float(params.get("A",   0.0)),
            "I33": float(params.get("I33", 0.0)),
            "I22": float(params.get("I22", 0.0)),
            "J":   float(params.get("J",   0.0)),
        }
    if shape == "矩形實心":
        b, h = float(params["b"]), float(params["h"])
        a_big, b_small = max(b, h), min(b, h)
        J = a_big * b_small**3 * (1/3 - 0.21*(b_small/a_big)*(1 - b_small**4/(12*a_big**4)))
        return {
            "A":   b * h,
            "I33": b * h**3 / 12,
            "I22": h * b**3 / 12,
            "J":   J,
        }
    if shape == "圓形實心":
        d = float(params["d"])
        return {
            "A":   math.pi * d**2 / 4,
            "I33": math.pi * d**4 / 64,
            "I22": math.pi * d**4 / 64,
            "J":   math.pi * d**4 / 32,
        }
    if shape == "矩形管":
        b, h, t = float(params["b"]), float(params["h"]), float(params["t"])
        bi, hi = b - 2*t, h - 2*t
        J = 2*t * (b-t)**2 * (h-t)**2 / (b + h - 2*t)
        return {
            "A":   b*h - bi*hi,
            "I33": (b*h**3 - bi*hi**3) / 12,
            "I22": (h*b**3 - hi*bi**3) / 12,
            "J":   J,
        }
    if shape == "圓管":
        d, t = float(params["d"]), float(params["t"])
        di = d - 2*t
        return {
            "A":   math.pi * (d**2 - di**2) / 4,
            "I33": math.pi * (d**4 - di**4) / 64,
            "I22": math.pi * (d**4 - di**4) / 64,
            "J":   math.pi * (d**4 - di**4) / 32,
        }
    if shape == "I形":
        H  = float(params["H"])
        bf = float(params["bf"])
        tf = float(params["tf"])
        tw = float(params["tw"])
        bw = H - 2*tf
        A   = 2*bf*tf + bw*tw
        I33 = bf*H**3/12 - (bf-tw)*bw**3/12
        I22 = 2*(tf*bf**3/12) + bw*tw**3/12
        J   = (1/3) * (2*bf*tf**3 + bw*tw**3)
        return {"A": A, "I33": I33, "I22": I22, "J": J}
    raise ValueError(f"未知截面形狀: {shape}")


# ── 資料展開：將 Material/Section 注入每根桿件 ────────────────────────────

def expand_truss_data(truss_data: dict, materials: list, sections: list) -> dict:
    """回傳深拷貝的 truss_data，每根 element 補齊 E/G/A/I33/I22/J。
    local override（元素 dict 中已有的數值欄位）保留不覆蓋。"""
    mat_map = {m["name"]: m for m in materials}
    sec_map = {s["name"]: s for s in sections}

    td = copy.deepcopy(truss_data)
    for elem in td["elements"]:
        sec_name = elem.get("section")
        if not sec_name or sec_name not in sec_map:
            continue
        sec = sec_map[sec_name]
        mat = mat_map.get(sec.get("material", ""), {})

        # 截面幾何（Custom 時直接從 sec 取，非 Custom 重新計算）
        shape = sec.get("shape", "Custom")
        props = compute_section_props(shape, sec)

        # 逐欄填入，以數值比較判斷是否為 override。若 elem 中的值與 section 值不同，視為故意 override 保留；
        # 若相同或不存在，從 section 填入（允許 Streamlit UI 預先註冊所有欄位的情況）
        for key, src_val in [
            ("E",   float(mat.get("E",   0))),
            ("G",   float(mat.get("G",   0))),
            ("A",   props["A"]),
            ("I33", props["I33"]),
            ("I22", props["I22"]),
            ("J",   props["J"]),
        ]:
            if key not in elem:
                elem[key] = src_val
            else:
                current = float(elem[key]) if elem[key] is not None else 0.0
                # 只有當元素值與 section 值明顯不同時，才視為蓄意 override 並保留
                if src_val != 0 and abs(current - src_val) > abs(src_val) * 1e-9:
                    pass  # 蓄意 override — 保留 current
                else:
                    elem[key] = src_val  # 無 override — 從 section 填入
    return td


# ── 自重計算 ───────────────────────────────────────────────────────────────

def compute_self_weight(truss_data_expanded: dict, sections: list, materials: list) -> list:
    """回傳 element_loads 格式的自重清單（w 為負值，向下）。"""
    mat_map = {m["name"]: m for m in materials}
    sec_map = {s["name"]: s for s in sections}

    result = []
    for elem in truss_data_expanded["elements"]:
        sec_name = elem.get("section")
        if not sec_name or sec_name not in sec_map:
            continue
        sec = sec_map[sec_name]
        mat = mat_map.get(sec.get("material", ""), {})
        density = float(mat.get("density", 0))
        A_eff   = float(elem.get("A", 0))
        w_self  = -(density * A_eff * 9.81)
        result.append({"element_id": elem["id"], "w": w_self})
    return result
