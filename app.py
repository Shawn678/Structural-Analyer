import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import defaultdict

def _norm_id(val) -> str:
    s = str(val).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from core.parametric_evaluator import (
    build_geometry_fingerprint,
    evaluate_real_results,
    evaluate_numerical_results,
    export_cache_to_txt,
    import_cache_from_txt,
)
from core.materials import compute_section_props, expand_truss_data, compute_self_weight
from core import seismic_bridge as _sb
import os as _os
import pandas as _pd
_SEISMIC_CSV = _os.path.join(_os.path.dirname(__file__), "data", "工址反應譜.csv")

@st.cache_data(show_spinner=False)
def _load_seismic_spectrum(path: str):
    df = _pd.read_csv(path, encoding='utf-8-sig', header=None, names=['Key', '475年', '2500年'])
    params = df.iloc[0:23].set_index('Key')
    match_start = df[df['Key'].str.contains("長週期不受限制", na=False)]
    match_end   = df[df['Key'].str.contains("長週期受限制",   na=False)]
    start_idx = match_start.index[0] + 1
    end_idx   = match_end.index[0]
    spec_df = df.iloc[start_idx:end_idx].copy()
    spec_df = spec_df.apply(_pd.to_numeric, errors='coerce').dropna()
    spec_df.columns = ['T', 'Sa_475', 'Sa_2500']
    return params, spec_df

@st.cache_data(show_spinner=False)
def _compute_fingerprint(nodes_json: str, elements_json: str, supports_json: str):
    import json
    return build_geometry_fingerprint({
        "nodes": json.loads(nodes_json),
        "elements": json.loads(elements_json),
        "supports": json.loads(supports_json),
        "loads": [], "element_loads": [], "element_point_loads": [],
    })


@st.cache_data(show_spinner=False)
def _compute_overridden_ids(sections_json: str, materials_json: str, elem_records_json: str) -> frozenset:
    import json
    secs = {s["name"]: s for s in json.loads(sections_json)}
    mats = {m["name"]: m for m in json.loads(materials_json)}

    def _sec_ref(sn, field):
        if sn not in secs:
            return None
        s = secs[sn]
        if field in ("E", "G"):
            return mats.get(s.get("material", ""), {}).get(field)
        return s.get(field)

    result = set()
    for r in json.loads(elem_records_json):
        sn = r.get("section", "")
        if not sn or sn not in secs:
            continue
        for field in ("E", "G", "A", "I33", "I22", "J"):
            ref = _sec_ref(sn, field)
            val = r.get(field)
            if ref is not None and val is not None:
                try:
                    if abs(float(val) - ref) > abs(ref) * 1e-9:
                        result.add(str(r.get("id", "")))
                        break
                except (TypeError, ValueError):
                    pass
    return frozenset(result)


st.set_page_config(page_title="Structural Analysis", layout="wide")

if "sym_cache" not in st.session_state:
    st.session_state["sym_cache"] = {}
if "rigid_links" not in st.session_state:
    st.session_state["rigid_links"] = []
if "seismic_pier_selection" not in st.session_state:
    st.session_state["seismic_pier_selection"] = []
if "seismic_extracted" not in st.session_state:
    st.session_state["seismic_extracted"] = None
if "seismic_user_table" not in st.session_state:
    st.session_state["seismic_user_table"] = None
if "seismic_result" not in st.session_state:
    st.session_state["seismic_result"] = None
if "seismic_bridge_axis" not in st.session_state:
    st.session_state["seismic_bridge_axis"] = "X"
if "materials" not in st.session_state:
    st.session_state["materials"] = [
        {"name": "鋼材",   "E": 200e9, "G": 77e9, "density": 7850.0},
        {"name": "混凝土", "E":  30e9, "G": 12.5e9, "density": 2400.0},
    ]
if "sections" not in st.session_state:
    st.session_state["sections"] = []

# 注入現有的 CSS 樣式

def _section_outline_pts(sec: dict, scale: float = 1.0) -> list[tuple[float, float]]:
    """
    回傳截面輪廓的 2D 點列（局部 y-z 平面），已乘上 scale。
    回傳格式：[(y0,z0), (y1,z1), ..., (y0,z0)]（首尾相連）。
    Custom / 無形狀資訊時回傳空列表。
    """
    shape = sec.get("shape", "Custom")
    pts = []

    if shape == "矩形實心":
        b, h = float(sec.get("b", 0)), float(sec.get("h", 0))
        if b <= 0 or h <= 0: return []
        hb, hh = b/2, h/2
        pts = [(-hb,-hh),(hb,-hh),(hb,hh),(-hb,hh),(-hb,-hh)]

    elif shape == "圓形實心":
        d = float(sec.get("d", 0))
        if d <= 0: return []
        r = d / 2
        import math
        pts = [(r*math.cos(2*math.pi*i/20), r*math.sin(2*math.pi*i/20)) for i in range(21)]

    elif shape == "矩形管":
        b, h, t = float(sec.get("b",0)), float(sec.get("h",0)), float(sec.get("t",0))
        if b <= 0 or h <= 0 or t <= 0: return []
        hb, hh = b/2, h/2
        bi, hi = (b-2*t)/2, (h-2*t)/2
        pts = [(-hb,-hh),(hb,-hh),(hb,hh),(-hb,hh),(-hb,-hh),
               None,
               (-bi,-hi),(bi,-hi),(bi,hi),(-bi,hi),(-bi,-hi)]

    elif shape == "圓管":
        d, t = float(sec.get("d",0)), float(sec.get("t",0))
        if d <= 0 or t <= 0: return []
        import math
        ro, ri = d/2, d/2 - t
        outer = [(ro*math.cos(2*math.pi*i/20), ro*math.sin(2*math.pi*i/20)) for i in range(21)]
        inner = [(ri*math.cos(2*math.pi*i/20), ri*math.sin(2*math.pi*i/20)) for i in range(21)]
        pts = outer + [None] + inner

    elif shape == "I形":
        H  = float(sec.get("H",0))
        bf = float(sec.get("bf",0))
        tf = float(sec.get("tf",0))
        tw = float(sec.get("tw",0))
        if H <= 0 or bf <= 0: return []
        hH, hbf, htw = H/2, bf/2, tw/2
        pts = [
            (-hbf, -hH),(hbf, -hH),(hbf, -hH+tf),(htw, -hH+tf),
            (htw, hH-tf),(hbf, hH-tf),(hbf, hH),(-hbf, hH),
            (-hbf, hH-tf),(-htw, hH-tf),(-htw, -hH+tf),(-hbf, -hH+tf),
            (-hbf, -hH),
        ]

    elif shape == "箱涵":
        b_top = float(sec.get("b_top",0))
        b_bot = float(sec.get("b_bot",0))
        h     = float(sec.get("h",0))
        t_top = float(sec.get("t_top",0))
        t_bot = float(sec.get("t_bot",0))
        t_web = float(sec.get("t_web",0))
        if b_top <= 0 or h <= 0: return []
        # 外輪廓（梯形，頂寬 b_top 底寬 b_bot）
        pts = [
            (-b_top/2, h),( b_top/2, h),( b_bot/2, 0),(-b_bot/2, 0),(-b_top/2, h),
            None,
            (-(b_top/2 - t_web), h - t_top),( b_top/2 - t_web, h - t_top),
            ( b_bot/2 - t_web, t_bot),(-(b_bot/2 - t_web), t_bot),
            (-(b_top/2 - t_web), h - t_top),
        ]

    if not pts:
        return []
    return [(p[0]*scale, p[1]*scale) if p is not None else None for p in pts]


def _project_outline(pts2d, cx, cy, cz, ey, ez):
    """
    把 2D 截面點投影到 3D 空間。
    ey, ez：截面局部 y 和 z 方向的單位向量（np.array 3D）。
    回傳 (xs, ys, zs) 可直接傳給 Scatter3d，None 保留為斷線標記。
    """
    xs, ys, zs = [], [], []
    for p in pts2d:
        if p is None:
            xs.append(None); ys.append(None); zs.append(None)
        else:
            dy, dz = p
            xs.append(cx + dy*ey[0] + dz*ez[0])
            ys.append(cy + dy*ey[1] + dz*ez[1])
            zs.append(cz + dy*ey[2] + dz*ez[2])
    return xs, ys, zs


def _extract_val(v):
    """從符號解 dict {"value": x} 或純數值中取出 float。"""
    if isinstance(v, dict):
        return float(v.get("value", 0.0))
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _sort_member_elements(elem_forces, truss_data):
    """
    依拓撲鏈排序同一 member 的桿件，並附加 _x_offset（累積弧長）。
    無法形成唯一鏈時退回按 element_id 排序。
    """
    node_map = {
        _norm_id(str(n["id"])): n
        for n in truss_data.get("nodes", [])
    }
    elem_map = {
        _norm_id(str(e["id"])): e
        for e in truss_data.get("elements", [])
    }

    def _elem_len(eid):
        e = elem_map.get(_norm_id(str(eid)), {})
        ni = node_map.get(_norm_id(str(e.get("i", ""))), {})
        nj = node_map.get(_norm_id(str(e.get("j", ""))), {})
        if not ni or not nj:
            return 0.0
        return math.sqrt(
            (float(nj.get("x", 0)) - float(ni.get("x", 0))) ** 2 +
            (float(nj.get("y", 0)) - float(ni.get("y", 0))) ** 2 +
            (float(nj.get("z", 0)) - float(ni.get("z", 0))) ** 2
        )

    # 建立節點連接圖 (node_id -> list of eid)
    node_edges = defaultdict(list)
    eid_to_ij = {}
    for ef in elem_forces:
        eid = str(ef["element_id"])
        e = elem_map.get(_norm_id(eid), {})
        i_id = _norm_id(str(e.get("i", "")))
        j_id = _norm_id(str(e.get("j", "")))
        if i_id and j_id:
            node_edges[i_id].append(eid)
            node_edges[j_id].append(eid)
            eid_to_ij[eid] = (i_id, j_id)

    # 找度數為 1 的端點（鏈起點）
    degree = {nid: len(eids) for nid, eids in node_edges.items()}
    endpoints = [nid for nid, d in degree.items() if d == 1]
    # Sort by geometry so chain always starts from the spatially smallest endpoint (deterministic for typical beam/cable arrangements).
    endpoints.sort(key=lambda nid: (
        float(node_map.get(nid, {}).get("x", 0)),
        float(node_map.get(nid, {}).get("y", 0)),
        float(node_map.get(nid, {}).get("z", 0)),
    ))

    ordered = []
    if len(endpoints) >= 1:
        # 貪婪遍歷
        start = endpoints[0]
        visited_eids = set()
        cur_node = start
        while True:
            next_eid = None
            for eid in node_edges.get(cur_node, []):
                if eid not in visited_eids:
                    next_eid = eid
                    break
            if next_eid is None:
                break
            visited_eids.add(next_eid)
            i_id, j_id = eid_to_ij.get(next_eid, ("", ""))
            next_node = j_id if i_id == cur_node else i_id
            ef_match = next((ef for ef in elem_forces if _norm_id(str(ef["element_id"])) == _norm_id(next_eid)), None)
            if ef_match:
                ordered.append(ef_match)
            cur_node = next_node

    # 若排序結果不完整，退回按 element_id 排序
    if len(ordered) != len(elem_forces):
        ordered = sorted(elem_forces, key=lambda ef: (
            not str(ef["element_id"]).lstrip("-").isdigit(),
            str(ef["element_id"])
        ))

    # 附加累積弧長偏移
    offset = 0.0
    result = []
    for ef in ordered:
        ef2 = dict(ef)
        ef2["_x_offset"] = offset
        Le = ef.get("Le", _elem_len(ef["element_id"]))
        offset += float(Le)
        result.append(ef2)
    return result


def _render_result_tabs(res_eval, truss_data):
    tab1, tab2, tab3, tab4 = st.tabs(["📐 節點位移", "⬆️ 支承反力", "📊 桿件內力摘要", "📈 內力圖"])

    # ── Tab 1：節點位移 ─────────────────────────────────────────────────
    with tab1:
        disp_rows = []
        for nd in res_eval.get("node_displacements", []):
            disp_rows.append({
                "節點 ID": nd["node_id"],
                "ux (m)":      _extract_val(nd.get("ux", 0)),
                "uy (m)":      _extract_val(nd.get("uy", 0)),
                "uz (m)":      _extract_val(nd.get("uz", 0)),
                "rx (rad)": _extract_val(nd.get("theta_x", 0)),
                "ry (rad)": _extract_val(nd.get("theta_y", 0)),
                "rz (rad)": _extract_val(nd.get("theta_z", 0)),
            })
        disp_df = pd.DataFrame(disp_rows)
        if not disp_df.empty:
            all_nids = [str(r) for r in disp_df["節點 ID"]]
            sel_nids = st.multiselect("篩選節點 ID", all_nids, default=all_nids, key="tab1_node_filter")
            filtered = disp_df[disp_df["節點 ID"].astype(str).isin(sel_nids)] if sel_nids else pd.DataFrame(columns=disp_df.columns)
            # 摘要
            max_uz_row = disp_df.loc[disp_df["uz (m)"].abs().idxmax()]
            st.caption(
                f"最大垂直位移 |uz|（全域）= {abs(max_uz_row['uz (m)'])*1000:.3f} mm（節點 {max_uz_row['節點 ID']}）"
            )
            st.dataframe(
                filtered.style.format({
                    "ux (m)": "{:.4e}", "uy (m)": "{:.4e}", "uz (m)": "{:.4e}",
                    "rx (rad)": "{:.4e}", "ry (rad)": "{:.4e}", "rz (rad)": "{:.4e}",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("無位移資料。")

    # ── Tab 2：支承反力 ─────────────────────────────────────────────────
    with tab2:
        react_rows = []
        for sr in res_eval.get("support_reactions", []):
            react_rows.append({
                "節點 ID": sr["node_id"],
                "Fx (N)":   _extract_val(sr.get("Rx", 0)),
                "Fy (N)":   _extract_val(sr.get("Ry", 0)),
                "Fz (N)":   _extract_val(sr.get("Rz", 0)),
                "Mx (N·m)": _extract_val(sr.get("Mx", 0)),
                "My (N·m)": _extract_val(sr.get("My", 0)),
                "Mz (N·m)": _extract_val(sr.get("Mz", 0)),
            })
        react_df = pd.DataFrame(react_rows)
        if not react_df.empty:
            all_rnids = [str(r) for r in react_df["節點 ID"]]
            sel_rnids = st.multiselect("篩選節點 ID", all_rnids, default=all_rnids, key="tab2_node_filter")
            filtered_r = react_df[react_df["節點 ID"].astype(str).isin(sel_rnids)] if sel_rnids else pd.DataFrame(columns=react_df.columns)
            total_fz = react_df["Fz (N)"].sum()
            st.caption(f"垂直反力合計 ΣFz = {total_fz/1000:.3f} kN")
            st.dataframe(
                filtered_r.style.format({
                    "Fx (N)": "{:.2f}", "Fy (N)": "{:.2f}", "Fz (N)": "{:.2f}",
                    "Mx (N·m)": "{:.2f}", "My (N·m)": "{:.2f}", "Mz (N·m)": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("無支承反力資料。")

    # ── Tab 3 & 4：待實作 ───────────────────────────────────────────────
    with tab3:
        force_rows = []
        elem_member_map_t3 = {
            _norm_id(str(e.get("id", ""))): str(e.get("member", "")).strip()
            for e in st.session_state.get("elements_data", [])
        }
        for ef in res_eval.get("element_forces", []):
            eid = str(ef["element_id"])
            member = elem_member_map_t3.get(_norm_id(eid), "")
            force_rows.append({
                "桿件 ID":       eid,
                "Member":        member if member else "(未分組)",
                "N (kN)":        _extract_val(ef.get("N", 0)) / 1000,
                "V2 (kN)":       _extract_val(ef.get("V2", 0)) / 1000,
                "V3 (kN)":       _extract_val(ef.get("V3", 0)) / 1000,
                "M2_i (kN·m)":  _extract_val(ef.get("M2_i") or ef.get("M2", 0)) / 1000,
                "M2_j (kN·m)":  _extract_val(ef.get("M2_j", 0)) / 1000,
                "M3_i (kN·m)":  _extract_val(ef.get("M3_i") or ef.get("M3", 0)) / 1000,
                "M3_j (kN·m)":  _extract_val(ef.get("M3_j", 0)) / 1000,
                "Le (m)":        float(ef.get("Le", 0)),
            })
        force_df = pd.DataFrame(force_rows)
        if not force_df.empty:
            all_members_t3 = sorted(force_df["Member"].unique().tolist())
            sel_members_t3 = st.multiselect(
                "篩選 Member 群組", all_members_t3, default=all_members_t3, key="tab3_member_filter"
            )
            eid_filter = st.text_input("篩選桿件 ID（部分匹配）", value="", key="tab3_eid_filter")
            filtered_f = force_df[force_df["Member"].isin(sel_members_t3)] if sel_members_t3 else pd.DataFrame(columns=force_df.columns)
            if eid_filter.strip():
                filtered_f = filtered_f[filtered_f["桿件 ID"].str.contains(eid_filter.strip(), na=False)]
            # 摘要：最大彎矩 M3（取 M3_i 與 M3_j 的絕對值最大者）
            m3_abs = force_df[["M3_i (kN·m)", "M3_j (kN·m)"]].abs().max(axis=1)
            max_idx = m3_abs.idxmax()
            max_m3_row = force_df.loc[max_idx]
            max_m3_val = m3_abs[max_idx]
            st.caption(
                f"最大彎矩 |M3| = {max_m3_val:.3f} kN·m"
                f"（桿件 {max_m3_row['桿件 ID']}，Member {max_m3_row['Member']}）"
            )
            fmt_cols = {c: "{:.3f}" for c in force_df.columns if c not in ("桿件 ID", "Member")}
            st.dataframe(
                filtered_f.style.format(fmt_cols),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("無桿件內力資料。")
    with tab4:
        import sympy as _sp
        _x_sym = _sp.Symbol('x')

        elem_member_map_t4 = {
            _norm_id(str(e.get("id", ""))): str(e.get("member", "")).strip()
            for e in st.session_state.get("elements_data", [])
        }

        # 依 member 分組
        member_groups_t4 = {}
        for ef in res_eval.get("element_forces", []):
            eid = str(ef["element_id"])
            member = elem_member_map_t4.get(_norm_id(eid), "")
            key = member if member else "(未分組)"
            member_groups_t4.setdefault(key, []).append(ef)

        all_members_t4 = sorted(member_groups_t4.keys())
        sel_members_t4 = st.multiselect(
            "顯示構件群組", all_members_t4, default=all_members_t4, key="tab4_member_filter"
        )

        for mb_key in all_members_t4:
            if mb_key not in sel_members_t4:
                continue
            mb_elems = _sort_member_elements(member_groups_t4[mb_key], truss_data)
            total_L = sum(float(ef.get("Le", 0)) for ef in mb_elems)

            with st.expander(f"{mb_key}（共 {len(mb_elems)} 段，總長 {total_L:.2f} m）", expanded=True):
                fig_t4 = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=("軸力圖 N (kN)", "剪力圖 V2 (kN)", "彎矩圖 M3 (kN·m)")
                )
                # 接縫位置（用於垂直虛線）及節點 ID
                seam_info = []  # list of (x_pos, node_id_str)
                x_global = 0.0

                for seg_idx, ef in enumerate(mb_elems):
                    Le = float(ef.get("Le", 0))
                    x_local = np.linspace(0, Le, max(20, int(Le * 10)))
                    offset = ef["_x_offset"]
                    x_plot = x_local + offset

                    # 收集接縫（除第一段起點外）
                    if seg_idx > 0:
                        _ef_raw = next(
                            (e for e in truss_data.get("elements", [])
                             if _norm_id(str(e.get("id", ""))) == _norm_id(str(ef["element_id"]))),
                            {}
                        )
                        _seam_nid = str(_ef_raw.get("i", ""))
                        seam_info.append((offset, _seam_nid))

                    # 取得各分量的 y 值
                    def _get_y(key):
                        raw_val = ef.get(key, 0)
                        # 數值解路徑：純數字
                        if not isinstance(raw_val, dict):
                            if key in ("M3_i", "M3"):
                                m3_i = float(ef.get("M3_i", 0))
                                m3_j = float(ef.get("M3_j", 0))
                                return m3_i + (m3_j - m3_i) * x_local / Le if Le > 0 else np.zeros_like(x_local)
                            return np.full_like(x_local, float(raw_val) if raw_val is not None else 0.0)
                        # 符號解路徑：formula string
                        formula_str = raw_val.get("formula", "0")
                        try:
                            expr = _sp.parse_expr(formula_str.replace('^', '**'))
                            f_np = _sp.lambdify(_x_sym, expr, modules=[
                                'numpy', {'Heaviside': lambda x: np.where(x >= 0, 1, 0)}
                            ])
                            y = f_np(x_local)
                            if isinstance(y, (int, float, np.float64)):
                                y = np.full_like(x_local, float(y))
                            return y
                        except Exception:
                            return np.zeros_like(x_local)

                    # 軸力 N
                    y_N  = _get_y("N") / 1000
                    # 剪力 V2
                    y_V2 = _get_y("V2") / 1000
                    # 彎矩 M3：數值解用線性插值，符號解用 formula
                    if isinstance(ef.get("M3", ef.get("M3_i", 0)), dict):
                        y_M3 = _get_y("M3") / 1000
                    else:
                        m3_i = float(ef.get("M3_i", 0)) / 1000
                        m3_j = float(ef.get("M3_j", 0)) / 1000
                        y_M3 = m3_i + (m3_j - m3_i) * x_local / Le if Le > 0 else np.zeros_like(x_local)

                    show_legend = (seg_idx == 0)
                    for col_idx, (y_vals, color, name) in enumerate([
                        (y_N,  "steelblue",   "N"),
                        (y_V2, "steelblue",   "V2"),
                        (y_M3, "crimson",     "M3"),
                    ], start=1):
                        fig_t4.add_trace(
                            go.Scatter(
                                x=x_plot, y=y_vals,
                                name=name, fill='tozeroy',
                                line=dict(color=color),
                                showlegend=False,
                            ),
                            row=1, col=col_idx,
                        )

                # 接縫垂直虛線（加在所有三個子圖）及節點 ID 標籤
                for sx, snid in seam_info:
                    for col_idx in (1, 2, 3):
                        fig_t4.add_vline(
                            x=sx, line_dash="dot", line_color="gray", line_width=1,
                            row=1, col=col_idx,
                        )
                    # 節點 ID 標籤僅加在第一個子圖（避免雜亂）
                    fig_t4.add_annotation(
                        x=sx, y=0, xref="x", yref="y",
                        text=f"N{snid}", showarrow=False,
                        font=dict(size=9, color="gray"),
                        yanchor="bottom", xanchor="center",
                        row=1, col=1,
                    )

                fig_t4.update_layout(
                    height=380, showlegend=False, template="plotly_white"
                )
                fig_t4.update_xaxes(title_text="累積弧長 (m)")
                st.plotly_chart(fig_t4, use_container_width=True, key=f"ifd_t4_{mb_key}")


def create_structure_plot(nodes_df, elements_df, supports_df=None, reactions=None, rigid_links=None):
    fig = go.Figure()

    # 確保資料不包含空 ID (處理動態編輯產生的空行)
    nodes_df = nodes_df.dropna(subset=['id'])
    elements_df = elements_df.dropna(subset=['id', 'i', 'j'])
    if 'z' not in nodes_df.columns: nodes_df['z'] = 0.0

    node_pos = {str(r['id']): r for _, r in nodes_df.iterrows()}

    # 繪製桿件：所有桿件合併為單一 trace，用 None 隔開線段
    ex, ey, ez, et = [], [], [], []
    for _, elem in elements_df.iterrows():
        n1 = node_pos.get(str(elem['i']))
        n2 = node_pos.get(str(elem['j']))
        if n1 is None or n2 is None:
            continue
        ex += [n1['x'], n2['x'], None]
        ey += [n1['y'], n2['y'], None]
        ez += [n1['z'], n2['z'], None]
        et += [f"Element {elem['id']}", f"Element {elem['id']}", None]
    if ex:
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode='lines', line=dict(color='RoyalBlue', width=6),
            hoverinfo='text', text=et, name='Elements'
        ))

    # 節點：單一 trace
    if not nodes_df.empty:
        fig.add_trace(go.Scatter3d(
            x=nodes_df['x'], y=nodes_df['y'], z=nodes_df['z'],
            mode='markers', marker=dict(size=5, color='white'),
            hoverinfo='text',
            text=[str(r['id']) for _, r in nodes_df.iterrows()],
            name='Nodes'
        ))

    # 截面輪廓：依 member 分組，在整體幾何中點畫截面符號
    # member 為空的桿件以自身中點處理，視為獨立 member
    _sec_map = {s["name"]: s for s in st.session_state.get("sections", [])}
    _OUTLINE_COLORS = [
        '#00e5ff','#69ff47','#ff6d00','#d500f9','#ffea00',
        '#76ff03','#ff1744','#00b0ff','#f50057','#64dd17',
    ]
    _member_groups = {}  # member_key -> list of (n1, n2, sec_dict)
    for _, elem in elements_df.iterrows():
        n1 = node_pos.get(str(elem['i']))
        n2 = node_pos.get(str(elem['j']))
        if n1 is None or n2 is None:
            continue
        sn  = str(elem.get('section') or '')
        sec = _sec_map.get(sn, {})
        if not sec.get('shape') or sec.get('shape') == 'Custom':
            continue  # 無形狀資訊跳過
        mb = str(elem.get('member') or '').strip()
        key = mb if mb else f"__elem_{elem['id']}"
        _member_groups.setdefault(key, []).append((n1, n2, sec))

    for ci, (mb_key, segs) in enumerate(_member_groups.items()):
        color = _OUTLINE_COLORS[ci % len(_OUTLINE_COLORS)]
        # 整體幾何中點：所有端點平均
        all_pts = []
        for n1, n2, _ in segs:
            all_pts += [(float(n1['x']), float(n1['y']), float(n1['z'])),
                        (float(n2['x']), float(n2['y']), float(n2['z']))]
        cx = sum(p[0] for p in all_pts) / len(all_pts)
        cy = sum(p[1] for p in all_pts) / len(all_pts)
        cz = sum(p[2] for p in all_pts) / len(all_pts)

        # 桿件主方向（取第一段）
        n1, n2, sec = segs[0]
        dx = float(n2['x']) - float(n1['x'])
        dy_v = float(n2['y']) - float(n1['y'])
        dz_v = float(n2['z']) - float(n1['z'])
        L = (dx**2 + dy_v**2 + dz_v**2) ** 0.5
        if L < 1e-10:
            continue
        ex_v = np.array([dx, dy_v, dz_v]) / L  # 桿件軸向

        # 局部 y 方向：優先用全域 Z 叉積，若接近平行則用全域 Y
        up = np.array([0., 0., 1.])
        if abs(np.dot(ex_v, up)) > 0.95:
            up = np.array([0., 1., 0.])
        ey_v = np.cross(up, ex_v)
        ey_v /= np.linalg.norm(ey_v)
        ez_v = np.cross(ex_v, ey_v)

        # beta 滾轉角（取第一段，單位 deg）
        beta_deg = float(segs[0][0].get('beta', 0) if hasattr(segs[0][0], 'get') else 0)
        if beta_deg:
            import math
            b = math.radians(beta_deg)
            cb, sb = math.cos(b), math.sin(b)
            ey_v, ez_v = cb*ey_v + sb*ez_v, -sb*ey_v + cb*ez_v

        # 截面尺度：取桿件長度的 15% 作為輪廓縮放基準
        all_L = []
        for n1s, n2s, _ in segs:
            dxs = float(n2s['x'])-float(n1s['x'])
            dys = float(n2s['y'])-float(n1s['y'])
            dzs = float(n2s['z'])-float(n1s['z'])
            all_L.append((dxs**2+dys**2+dzs**2)**0.5)
        seg_len = sum(all_L)
        scale = seg_len * 0.08

        pts2d = _section_outline_pts(sec, scale=scale)
        if not pts2d:
            continue

        xs, ys, zs = _project_outline(pts2d, cx, cy, cz, ey_v, ez_v)
        label = mb_key if not mb_key.startswith('__elem_') else f"Element {mb_key[7:]}"
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color=color, width=3),
            hoverinfo='text',
            text=f"{label} ({sec.get('shape','')}) 截面輪廓",
            name=f"截面 {label}",
        ))

    # 繪製 Rigid Link：合併為單一 trace，slave 端另出一個 trace 標示空心圓
    if rigid_links:
        rlx, rly, rlz, rlt = [], [], [], []
        slx, sly, slz, slt = [], [], [], []
        for rl in rigid_links:
            m = node_pos.get(str(rl.get('master', '')))
            s = node_pos.get(str(rl.get('slave',  '')))
            if m is None or s is None:
                continue
            rlx += [m['x'], s['x'], None]
            rly += [m['y'], s['y'], None]
            rlz += [m['z'], s['z'], None]
            rlt += [f"RL {rl.get('id','')} ({rl.get('master')}→{rl.get('slave')})",
                    f"RL {rl.get('id','')} ({rl.get('master')}→{rl.get('slave')})", None]
            slx.append(s['x']); sly.append(s['y']); slz.append(s['z'])
            slt.append(f"slave: {rl.get('slave','')}")
        if rlx:
            fig.add_trace(go.Scatter3d(
                x=rlx, y=rly, z=rlz,
                mode='lines', line=dict(color='orange', width=3, dash='dash'),
                hoverinfo='text', text=rlt, name='Rigid Links'
            ))
        if slx:
            fig.add_trace(go.Scatter3d(
                x=slx, y=sly, z=slz,
                mode='markers', marker=dict(size=6, color='orange', symbol='circle-open'),
                hoverinfo='text', text=slt, name='RL Slaves'
            ))

    # 繪製支承
    if supports_df is not None and not supports_df.empty:
        supports_df = supports_df.dropna(subset=['node_id'])
        sup_nodes = nodes_df[nodes_df['id'].isin(supports_df['node_id'])]
        fig.add_trace(go.Scatter3d(
            x=sup_nodes['x'], y=sup_nodes['y'], z=sup_nodes['z'],
            mode='markers', marker=dict(symbol='diamond', size=10, color='red'),
            name='Supports'
        ))

    fig.update_layout(
        title="3D 結構視覺化 (遵循 SAP2000: Z 軸向上)",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=700, template="plotly_dark", showlegend=False
    )
    return fig

try:
    with open('static/css/style.css', 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title("Structural Analysis 數據編輯")

left_panel, right_panel = st.columns([1, 1.2])

with left_panel:
    st.subheader("材料 (Materials)")
    mat_df_raw = pd.DataFrame(st.session_state["materials"])
    mat_df_raw["ν (唯讀)"] = (mat_df_raw["E"] / (2 * mat_df_raw["G"]) - 1).round(4)
    mat_df = st.data_editor(
        mat_df_raw,
        column_config={
            "name":    st.column_config.TextColumn("名稱", width="small"),
            "E":       st.column_config.NumberColumn("E (Pa)",     format="%.3e"),
            "G":       st.column_config.NumberColumn("G (Pa)",     format="%.3e"),
            "density": st.column_config.NumberColumn("密度 (kg/m³)", format="%.1f"),
            "ν (唯讀)": st.column_config.NumberColumn("ν (唯讀)",   disabled=True),
        },
        num_rows="dynamic", key="mat_editor",
    )
    # 同步回 session_state（去掉唯讀欄，只在有變化時更新）
    _mat_new = mat_df.drop(columns=["ν (唯讀)"], errors="ignore").dropna(subset=["name"]).to_dict("records")
    if _mat_new != st.session_state["materials"]:
        st.session_state["materials"] = _mat_new
    mat_names = [m["name"] for m in st.session_state["materials"]]

    st.subheader("截面 (Sections)")
    SHAPE_OPTIONS = ["Custom", "矩形實心", "圓形實心", "矩形管", "圓管", "I形", "箱涵"]
    SHAPE_INPUTS  = {
        "矩形實心": [("b","寬 b (m)"),("h","高 h (m)")],
        "圓形實心": [("d","直徑 d (m)")],
        "矩形管":   [("b","寬 b (m)"),("h","高 h (m)"),("t","壁厚 t (m)")],
        "圓管":     [("d","外徑 d (m)"),("t","壁厚 t (m)")],
        "I形":      [("H","全高 H (m)"),("bf","翼板寬 bf (m)"),
                     ("tf","翼板厚 tf (m)"),("tw","腹板厚 tw (m)")],
        "箱涵": [
            ("b_top", "頂板總寬 b_top (m)"),
            ("b_bot", "底板總寬 b_bot (m)"),
            ("h",     "箱涵全高 h (m)"),
            ("t_top", "頂板厚 t_top (m)"),
            ("t_bot", "底板厚 t_bot (m)"),
            ("t_web", "外腹板厚 t_web (m)"),
            ("t_dia", "內隔板厚 t_dia (m)"),
            ("c_top", "頂板懸臂長 c_top (m)"),
        ],
    }
    BOX_DEFAULTS = {
        "b_top": 6.0, "b_bot": 5.0, "h": 2.0,
        "t_top": 0.25, "t_bot": 0.25, "t_web": 0.3,
        "t_dia": 0.2, "c_top": 0.5,
    }

    sec_df = st.data_editor(
        pd.DataFrame(st.session_state["sections"]) if st.session_state["sections"]
        else pd.DataFrame(columns=["name","material","shape","A","I33","I22","J"]),
        column_config={
            "name":     st.column_config.TextColumn("截面名稱", width="small"),
            "material": st.column_config.SelectboxColumn("材料", options=mat_names),
            "shape":    st.column_config.SelectboxColumn("形狀", options=SHAPE_OPTIONS),
            "A":        st.column_config.NumberColumn("A (m²)",   format="%.4e"),
            "I33":      st.column_config.NumberColumn("I33 (m⁴)", format="%.4e"),
            "I22":      st.column_config.NumberColumn("I22 (m⁴)", format="%.4e"),
            "J":        st.column_config.NumberColumn("J (m⁴)",   format="%.4e"),
        },
        num_rows="dynamic", key="sec_editor",
    )
    _sec_new = sec_df.dropna(subset=["name"]).to_dict("records")
    if _sec_new != st.session_state["sections"]:
        st.session_state["sections"] = _sec_new
    sec_names = [s["name"] for s in st.session_state["sections"]]

    with st.expander("從形狀計算截面參數", expanded=False):
        st.caption(
            "⚠️ 矩形實心的 J 採用 Timoshenko 近似公式；"
            "矩形管的 J 採用薄壁閉口近似；"
            "I 形截面的 J 採用薄壁開口近似；"
            "箱涵的 J 採用 Bredt 多室薄壁公式。精確值請查結構手冊。"
        )
        target_sec = st.selectbox("填入截面", options=sec_names, key="shape_target_sec")
        shape_sel  = st.selectbox("截面形狀", options=list(SHAPE_INPUTS.keys()), key="shape_sel")
        shape_vals = {}
        if shape_sel == "箱涵":
            # 箱涵：n_cell 用 selectbox，其餘參數兩欄排列
            shape_vals["n_cell"] = st.selectbox(
                "室數 n_cell", options=[1, 2, 3, 4, 5], key="sv_n_cell"
            )
            param_keys = [k for k, _ in SHAPE_INPUTS["箱涵"]]
            left_keys  = param_keys[:4]
            right_keys = param_keys[4:]
            col_l, col_r = st.columns(2)
            for k, label in [(k, lbl) for k, lbl in SHAPE_INPUTS["箱涵"] if k in left_keys]:
                shape_vals[k] = col_l.number_input(
                    label, value=BOX_DEFAULTS[k], format="%.4f", key=f"sv_{k}"
                )
            for k, label in [(k, lbl) for k, lbl in SHAPE_INPUTS["箱涵"] if k in right_keys]:
                shape_vals[k] = col_r.number_input(
                    label, value=BOX_DEFAULTS[k], format="%.4f", key=f"sv_{k}"
                )
            # ── Plotly 即時預覽 ────────────────────────────────────────────
            bv = shape_vals
            n  = bv["n_cell"]
            bt, bb_w = bv["b_top"], bv["b_bot"]
            hv = bv["h"]
            ct = bv["c_top"]
            tw, tt, tb = bv["t_web"], bv["t_top"], bv["t_bot"]

            fig = go.Figure()

            # 外廓多邊形：底板→右腹板→頂板右懸臂→折回→折回→頂板左懸臂→左腹板→底板
            # 懸臂僅有頂板厚 tt，腹板從底板頂面到頂板底面
            half_top = bt / 2
            half_bot = bb_w / 2
            outer_x = [
                -half_bot, half_bot,           # 底板
                half_bot,  half_top,            # 右腹板頂 → 頂板右端
                half_top,  half_bot,            # 頂板右端折下 → 腹板右上角
                half_bot, -half_bot,            # 頂板中段底面（右→左）
                -half_bot, -half_top,           # 腹板左上角 → 頂板左端
                -half_top, -half_bot,           # 頂板左端折下 → 左腹板頂
                -half_bot,                      # 回起點
            ]
            outer_y = [
                0,      0,                      # 底板
                hv-tt,  hv-tt,                  # 右腹板頂 → 懸臂根部底面
                hv,     hv,                     # 頂板右端上面
                hv,     hv,                     # 頂板中段上面
                hv,     hv,                     # 頂板左端上面
                hv-tt,  hv-tt,                  # 懸臂根部底面 → 左腹板頂
                0,                              # 回起點
            ]
            fig.add_trace(go.Scatter(
                x=outer_x, y=outer_y, fill="toself",
                fillcolor="rgba(70,130,180,0.65)", line=dict(color="#1a3a5c", width=2),
                showlegend=False, name="外廓",
            ))

            # 各室空腔（白色覆蓋）
            b_box = bb_w - 2 * tw
            t_dia = bv["t_dia"]
            s_void = (b_box - (n - 1) * t_dia) / n   # 每室淨寬（扣除內隔板）
            for i in range(n):
                inner_x0 = -b_box / 2 + i * (s_void + t_dia)
                inner_x1 = inner_x0 + s_void
                void_x = [inner_x0, inner_x1, inner_x1, inner_x0, inner_x0]
                void_y = [tb, tb, hv - tt, hv - tt, tb]
                fig.add_trace(go.Scatter(
                    x=void_x, y=void_y, fill="toself",
                    fillcolor="white", line=dict(color="#1a3a5c", width=1.2),
                    showlegend=False,
                ))

            # 尺寸標注：錨點在資料座標，偏移用像素，緊貼斷面
            _blue   = dict(color="#1565C0", size=11)
            _orange = dict(color="#E65100", size=11)
            annotations = [
                # b_top：頂板上方中央，向上偏移 18px
                dict(x=0, y=hv, xref="x", yref="y",
                     ax=0, ay=-18, axref="pixel", ayref="pixel",
                     text=f"<b>b_top = {bt:.3f} m</b>", showarrow=True,
                     arrowhead=0, arrowcolor="#1565C0", arrowwidth=1,
                     font=_blue, bgcolor="white", borderpad=2,
                     xanchor="center", yanchor="bottom"),
                # b_bot：底板下方中央，向下偏移 18px
                dict(x=0, y=0, xref="x", yref="y",
                     ax=0, ay=18, axref="pixel", ayref="pixel",
                     text=f"<b>b_bot = {bb_w:.3f} m</b>", showarrow=True,
                     arrowhead=0, arrowcolor="#1565C0", arrowwidth=1,
                     font=_blue, bgcolor="white", borderpad=2,
                     xanchor="center", yanchor="top"),
                # h：右腹板外側，向右偏移 18px
                dict(x=half_bot, y=hv / 2, xref="x", yref="y",
                     ax=18, ay=0, axref="pixel", ayref="pixel",
                     text=f"<b>h = {hv:.3f} m</b>", showarrow=True,
                     arrowhead=0, arrowcolor="#E65100", arrowwidth=1,
                     font=_orange, bgcolor="white", borderpad=2,
                     xanchor="left", yanchor="middle"),
                # c_top：頂板左懸臂端，向上偏移 18px
                dict(x=-half_bot - ct / 2, y=hv, xref="x", yref="y",
                     ax=0, ay=-18, axref="pixel", ayref="pixel",
                     text=f"<b>c_top = {ct:.3f} m</b>", showarrow=True,
                     arrowhead=0, arrowcolor="#E65100", arrowwidth=1,
                     font=_orange, bgcolor="white", borderpad=2,
                     xanchor="center", yanchor="bottom"),
            ]

            fig.update_layout(
                annotations=annotations,
                xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
                yaxis=dict(visible=False),
                margin=dict(l=20, r=80, t=50, b=50),
                height=420,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            cols = st.columns(len(SHAPE_INPUTS[shape_sel]))
            for col, (k, label) in zip(cols, SHAPE_INPUTS[shape_sel]):
                shape_vals[k] = col.number_input(label, value=0.1, format="%.4f", key=f"sv_{k}")
        if st.button("計算並填入", key="calc_shape"):
            try:
                props = compute_section_props(shape_sel, shape_vals)
                new_secs = []
                for s in st.session_state["sections"]:
                    if s["name"] == target_sec:
                        s = {**s, "shape": shape_sel, **shape_vals, **props}
                    new_secs.append(s)
                st.session_state["sections"] = new_secs
                st.rerun()
            except Exception as ex:
                st.error(f"計算失敗：{ex}")

    # 分析維度設定
    if "structure_dim" not in st.session_state:
        st.session_state["structure_dim"] = "auto"
    _dim_choice = st.radio(
        "分析維度",
        ["自動偵測", "2D（XZ 平面）", "2D（XY 平面）", "3D"],
        index={"auto": 0, "XZ": 1, "XY": 2, "3D": 3}.get(st.session_state["structure_dim"], 0),
        horizontal=True,
        help="2D 模式會自動補全面外自由度約束；3D 模式完全忠於邊界條件輸入。純梁橋請選「2D（XZ 平面）」。",
    )
    st.session_state["structure_dim"] = {"自動偵測": "auto", "2D（XZ 平面）": "XZ", "2D（XY 平面）": "XY", "3D": "3D"}[_dim_choice]

    st.subheader("節點 (Nodes)")
    st.caption("座標單位：**m（公尺）**")
    # 外部寫入（載入檔案、精靈生成）透過 _loaded_nodes 觸發一次性重置
    _loaded_nodes = st.session_state.pop("_loaded_nodes", None)
    if _loaded_nodes is not None:
        st.session_state["nodes_data"] = _loaded_nodes
        st.session_state.pop("nodes", None)  # 清除 widget 快取讓表格重載
    if "nodes_data" not in st.session_state:
        st.session_state["nodes_data"] = [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 4.0},
            {"id": 2, "x": 6.0, "y": 0.0, "z": 4.0},
            {"id": 3, "x": 12.0, "y": 0.0, "z": 4.0},
            {"id": 4, "x": 0.0, "y": 0.0, "z": 0.0},
        ]
    # data 固定傳 session_state 的初始值，編輯狀態由 widget key "nodes" 自行持有
    # 不在 rerun 之間讀回再寫入，消除游標重置問題
    nodes_df = st.data_editor(
        pd.DataFrame(st.session_state["nodes_data"]), num_rows="dynamic", key="nodes"
    )
    # 同步：只把當前顯示值寫回（供後續邏輯讀取），不觸發 rerun
    st.session_state["nodes_data"] = nodes_df.dropna(subset=["id"]).to_dict("records")

    st.subheader("桿件 (Elements)")
    st.caption(
        "**截面參數說明：** "
        "E=彈性模量(Pa)、G=剪切模量(Pa)、A=截面積(m²)、"
        "I33=強軸慣性矩(m⁴，控制平面內彎曲)、I22=弱軸慣性矩(m⁴)、"
        "J=抗扭常數(m⁴)、beta=滾轉角(deg，預設0)。\n"
        "pin_i / pin_j: 桿件端點釋放。若兩端皆為鉸接，程式自動忽略慣性矩 (視為二力構件)。"
    )
    import json as _json
    sec_map_ui = {s["name"]: s for s in st.session_state["sections"]}
    mat_map_ui = {m["name"]: m for m in st.session_state["materials"]}

    def _sec_val(sec_name, field):
        if sec_name not in sec_map_ui:
            return None
        s = sec_map_ui[sec_name]
        if field in ("E", "G"):
            m = mat_map_ui.get(s.get("material", ""), {})
            return m.get(field)
        return s.get(field)

    _default_elem_rows = [
        {"id":1,"i":1,"j":2,"member":"","section":"","E":None,"G":None,"A":None,
         "I33":None,"I22":None,"J":None,"beta":0.0,"dL":0.0,"pin_i":False,"pin_j":False,"status":""},
        {"id":2,"i":2,"j":3,"member":"","section":"","E":None,"G":None,"A":None,
         "I33":None,"I22":None,"J":None,"beta":0.0,"dL":0.0,"pin_i":False,"pin_j":False,"status":""},
        {"id":3,"i":4,"j":1,"member":"","section":"","E":None,"G":None,"A":None,
         "I33":None,"I22":None,"J":None,"beta":0.0,"dL":0.0,"pin_i":False,"pin_j":False,"status":""},
    ]
    if "elements_data" not in st.session_state:
        st.session_state["elements_data"] = _default_elem_rows
    if "elem_prev_section" not in st.session_state:
        st.session_state["elem_prev_section"] = {}

    # section 帶入：在顯示前先把已知的 section 填值套入 elements_data
    # 這樣 data_editor 的 data 永遠是最新的，不需要 rerun 讓 widget 重載
    _pre_rows = []
    _section_changed = False
    for r in [dict(row) for row in st.session_state["elements_data"]]:
        elem_id = str(r.get("id", ""))
        sn = r.get("section") or ""
        prev_sn = st.session_state["elem_prev_section"].get(elem_id)
        if sn and sn in sec_map_ui and sn != prev_sn:
            for field in ("E", "G", "A", "I33", "I22", "J"):
                ref = _sec_val(sn, field)
                if ref is not None:
                    r[field] = ref
            _section_changed = True
        st.session_state["elem_prev_section"][elem_id] = sn
        _pre_rows.append(r)
    if _section_changed:
        st.session_state["elements_data"] = _pre_rows

    # override 偵測（在 data_editor 之前計算，結果套入 status 欄）
    _secs_json  = _json.dumps(st.session_state["sections"],  sort_keys=True)
    _mats_json  = _json.dumps(st.session_state["materials"], sort_keys=True)
    _elems_json = _json.dumps(st.session_state["elements_data"], sort_keys=True)
    _overridden_ids = _compute_overridden_ids(_secs_json, _mats_json, _elems_json)
    for r in st.session_state["elements_data"]:
        r["status"] = "● 已修改" if str(r.get("id", "")) in _overridden_ids else ""

    elements_df = st.data_editor(
        pd.DataFrame(st.session_state["elements_data"]),
        column_config={
            "member":  st.column_config.TextColumn("構件", width="small"),
            "section": st.column_config.SelectboxColumn(
                "截面", options=[""] + sec_names, width="small"
            ),
            "status": st.column_config.TextColumn("狀態", disabled=True, width="small"),
        },
        num_rows="dynamic", key="elements",
    )
    # 同步回 session_state，讓下一輪 section 帶入邏輯能讀到最新編輯
    st.session_state["elements_data"] = elements_df.to_dict("records")

    st.subheader("支承 (Supports)")
    st.caption(
        "**支承說明：**\n"
        "- **ux, uy, uz**: 限制沿著 X, Y, Z 軸的位移 (Translation)。\n"
        "- **rx, ry, rz**: 限制繞著 X, Y, Z 軸的轉動 (Rotation)。\n"
        "- **彈簧剛度**: 若要使用 kx/ky/kt (扭轉剛度)，請取消勾選對應的位移/轉動約束。"
    )
    _supports_default = st.session_state.pop("_loaded_supports", None) or [
        {"node_id": 4, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True},
        {"node_id": 2, "ux": False, "uy": True, "uz": True, "rx": True, "ry": False, "rz": True},
        {"node_id": 3, "ux": False, "uy": True, "uz": True, "rx": True, "ry": False, "rz": True},
    ]
    supports_df = st.data_editor(
        pd.DataFrame(_supports_default), num_rows="dynamic", key="supports"
    )

    # 供載重欄位使用的 ID 選單
    _all_elem_ids = sorted(
        {str(e.get("id", "")) for e in st.session_state.get("elements_data", []) if e.get("id", "") != ""},
        key=lambda x: (not x.startswith("gen"), x)
    )
    _all_node_ids = sorted(
        {str(n.get("id", "")) for n in st.session_state.get("nodes_data", []) if n.get("id", "") != ""},
        key=lambda x: (not x.lstrip("-").isdigit(), x)
    )

    st.subheader("載重 (Loads)")
    st.caption("fx / fy / fz：**N（牛頓）**；mx / my / mz：**N·m（牛頓·公尺）**")
    _loads_default = (
        st.session_state.pop("_loaded_loads", None)
        or st.session_state.get("_pt_accumulated_loads")
        or [{"node_id": "", "fx": 0.0, "fy": 0.0, "fz": 0.0}]
    )
    _loads_df_raw = pd.DataFrame(_loads_default)
    if "node_id" in _loads_df_raw.columns:
        _loads_df_raw["node_id"] = _loads_df_raw["node_id"].astype(str).str.replace(r"^(nan|None)$", "", regex=True)
    loads_df = st.data_editor(
        _loads_df_raw, num_rows="dynamic", key="loads",
        column_config={
            "node_id": st.column_config.SelectboxColumn("節點 ID", options=_all_node_ids, width="medium"),
        }
    )

    st.subheader("桿件載重 (Element Loads)")
    st.caption("w：均佈載重，單位 **N/m**，向下為負（與 Z 軸正方向相反）")

    # ── 按 member 快捷套用均佈載重 ──
    with st.expander("快捷：按構件名稱套用均佈載重", expanded=False):
        _all_members = sorted({
            str(e.get("member", "")) for e in st.session_state.get("elements_data", [])
            if e.get("member", "")
        })
        _wiz_member = st.selectbox("構件 (member)", options=[""] + _all_members, key="_quick_load_member")
        _wiz_w = st.number_input("w (N/m，向下為負)", value=0.0, key="_quick_load_w")
        if st.button("套用至桿件載重表", key="_quick_load_apply"):
            if _wiz_member:
                _target_ids = [
                    str(e.get("id", "")) for e in st.session_state.get("elements_data", [])
                    if str(e.get("member", "")) == _wiz_member and e.get("id", "") != ""
                ]
                _cur = st.session_state.get("_eloads_override",
                       st.session_state.get("_loaded_element_loads") or [{"element_id": 1, "w": -5.0}])
                # 移除同 member 的舊記錄，再追加新的
                _keep = [r for r in _cur if str(r.get("element_id", "")) not in set(_target_ids)]
                _new  = [{"element_id": eid, "w": _wiz_w} for eid in _target_ids]
                st.session_state["_eloads_override"] = _keep + _new
                st.rerun()

    _eloads_default = (
        st.session_state.pop("_eloads_override", None)
        or st.session_state.pop("_loaded_element_loads", None)
        or [{"element_id": "", "w": 0.0}]
    )
    _eloads_df_raw = pd.DataFrame(_eloads_default)
    if "element_id" in _eloads_df_raw.columns:
        _eloads_df_raw["element_id"] = _eloads_df_raw["element_id"].astype(str).str.replace(r"^(nan|None)$", "", regex=True)
    e_loads_df = st.data_editor(
        _eloads_df_raw, num_rows="dynamic", key="element_loads",
        column_config={
            "element_id": st.column_config.SelectboxColumn("桿件 ID", options=_all_elem_ids, width="medium"),
        }
    )

    st.subheader("桿件集中載重 (Element Point Loads)")
    st.caption("p：集中力 **N**；a：距 i 端距離 **m**")
    _eptloads_default = st.session_state.pop("_loaded_element_point_loads", None) or [
        {"element_id": "", "p": 0.0, "a": 0.0}
    ]
    _eptloads_df_raw = pd.DataFrame(_eptloads_default)
    if "element_id" in _eptloads_df_raw.columns:
        _eptloads_df_raw["element_id"] = _eptloads_df_raw["element_id"].astype(str).str.replace(r"^(nan|None)$", "", regex=True)
    e_pt_loads_df = st.data_editor(
        _eptloads_df_raw, num_rows="dynamic", key="element_point_loads",
        column_config={
            "element_id": st.column_config.SelectboxColumn("桿件 ID", options=_all_elem_ids, width="medium"),
        }
    )

    # ── Rigid Link 表格 ───────────────────────────────────────────────────
    with st.expander("剛性連桿 Rigid Links", expanded=False):
        _rl_list = st.session_state.get("rigid_links", [])
        rl_df_raw = pd.DataFrame(_rl_list) if _rl_list else pd.DataFrame(
            columns=["id", "master", "slave", "group"])
        for _col in ("id", "master", "slave", "group"):
            if _col in rl_df_raw.columns:
                rl_df_raw[_col] = rl_df_raw[_col].fillna("").astype(str)

        _node_pos = {str(n.get("id")): n for n in nodes_df.dropna(subset=["id"]).to_dict("records")}

        def _ecc_str(row):
            m = _node_pos.get(str(row.get("master", "")), {})
            s = _node_pos.get(str(row.get("slave",  "")), {})
            if not m or not s:
                return ""
            dx = float(s.get("x", 0)) - float(m.get("x", 0))
            dy = float(s.get("y", 0)) - float(m.get("y", 0))
            dz = float(s.get("z", 0)) - float(m.get("z", 0))
            return f"({dx:.3f}, {dy:.3f}, {dz:.3f})"

        rl_df_raw["偏心向量(m)"] = rl_df_raw.apply(_ecc_str, axis=1) if not rl_df_raw.empty else ""
        rl_df = st.data_editor(
            rl_df_raw,
            column_config={
                "id":          st.column_config.TextColumn("ID",       width="small"),
                "master":      st.column_config.TextColumn("Master 節點"),
                "slave":       st.column_config.TextColumn("Slave 節點"),
                "group":       st.column_config.TextColumn("群組",     width="small"),
                "偏心向量(m)": st.column_config.TextColumn("偏心向量", disabled=True),
            },
            num_rows="dynamic", key="rl_editor",
        )
        st.session_state["rigid_links"] = (
            rl_df.drop(columns=["偏心向量(m)"], errors="ignore")
                 .dropna(subset=["master", "slave"])
                 .to_dict("records")
        )
        _existing_grps = sorted({
            rl.get("group", "") for rl in st.session_state["rigid_links"]
            if isinstance(rl.get("group"), str) and rl.get("group")
        })
        if _existing_grps:
            _del_grp = st.selectbox("刪除整組", options=[""] + _existing_grps, key="rl_del_grp")
            if st.button("刪除群組", key="rl_del_btn") and _del_grp:
                st.session_state["rigid_links"] = [
                    rl for rl in st.session_state["rigid_links"] if rl.get("group") != _del_grp]
                st.session_state["nodes_data"] = [
                    n for n in st.session_state.get("nodes_data", []) if n.get("group") != _del_grp]
                st.session_state["elements_data"] = [
                    e for e in st.session_state["elements_data"] if e.get("group") != _del_grp]
                for _k in ("nodes", "elements", "rl_editor"):
                    st.session_state.pop(_k, None)
                st.rerun()

    # ── 索面精靈 ──────────────────────────────────────────────────────────
    with st.expander("索面精靈 Cable Face Wizard", expanded=False):
        from core.cable_wizard import generate_cable_face
        st.caption("填寫索面參數，自動生成主梁中心節點、偏心錨點、Rigid Link 與索構件。")
        # 收集桿件表格中已定義的 member 名稱（供主梁選單用）
        _wiz_member_names = sorted({
            str(r.get("member", "")) for r in st.session_state.get("elements_data", [])
            if isinstance(r.get("member"), str) and r.get("member", "").strip()
        })

        _wiz_c1, _wiz_c2 = st.columns(2)
        with _wiz_c1:
            wiz_group    = st.text_input("群組名稱", value="tower1_left",  key="wiz_group")
            wiz_tower_id = st.text_input("塔頂節點 ID", value="",          key="wiz_tower_id")
            wiz_n        = st.number_input("根數", min_value=1, max_value=50, value=7, step=1, key="wiz_n")
            wiz_ecc_y       = st.number_input("橋面偏心距 y (m) 右+左-", value=3.0, format="%.3f", key="wiz_ecc_y")
            wiz_tower_ecc_y = st.number_input("塔側偏心距 y (m) 右+左-（0=不偏移）", value=0.0, format="%.3f", key="wiz_tower_ecc_y")
            wiz_deck_z   = st.number_input("主梁 z 座標 (m)", value=0.0,   format="%.3f", key="wiz_deck_z")
        with _wiz_c2:
            wiz_t_off    = st.number_input("塔側起始偏移 (m)", value=-0.5, format="%.3f", key="wiz_t_off")
            wiz_t_sp     = st.number_input("塔側間距 (m)",     value=-0.5, format="%.3f", key="wiz_t_sp")
            wiz_dx_start = st.number_input("橋面起始 x (m)",   value=0.0,  format="%.3f", key="wiz_dx_start")
            wiz_dx_sp    = st.number_input("橋面間距 往跨中+ 往橋台-", value=5.0, format="%.3f", key="wiz_dx_sp")
            wiz_beam_member = st.selectbox(
                "主梁 member（自動拆分用，留空跳過）",
                options=[""] + _wiz_member_names,
                key="wiz_beam_member",
            )

        def _wiz_nid(v):
            s = str(v).strip()
            try:
                f = float(s)
                if f == int(f):
                    return str(int(f))
            except (ValueError, OverflowError):
                pass
            return s

        if st.button("生成索面", key="wiz_generate"):
            _node_ids_existing = {str(n.get("id")) for n in nodes_df.dropna(subset=["id"]).to_dict("records")}
            if str(wiz_tower_id) not in _node_ids_existing:
                st.error(f"塔頂節點 '{wiz_tower_id}' 不存在於節點表格中。")
            else:
                _tw_row = nodes_df[nodes_df["id"].astype(str) == str(wiz_tower_id)].iloc[0]
                _wiz_params = {
                    "group_name":         wiz_group,
                    "tower_node_id":      str(wiz_tower_id),
                    "tower_node_pos":     {"x": float(_tw_row["x"]), "y": float(_tw_row["y"]), "z": float(_tw_row["z"])},
                    "tower_offset_start": float(wiz_t_off),
                    "tower_spacing":      float(wiz_t_sp),
                    "deck_x_start":       float(wiz_dx_start),
                    "deck_spacing":       float(wiz_dx_sp),
                    "n_cables":           int(wiz_n),
                    "eccentricity_y":     float(wiz_ecc_y),
                    "tower_eccentricity_y": float(wiz_tower_ecc_y),
                    "deck_z":             float(wiz_deck_z),
                }
                _existing_nodes_list = nodes_df.dropna(subset=["id"]).to_dict("records")
                _gen = generate_cable_face(_wiz_params, _existing_nodes_list)

                _new_nodes = [{k: v for k, v in n.items() if k != "_role"} for n in _gen["nodes"]]
                _new_elems = [{k: v for k, v in e.items() if k != "_role"} for e in _gen["elements"]]

                _existing_node_ids = {_wiz_nid(n.get("id")) for n in st.session_state.get("nodes_data", [])}
                for _n in _new_nodes:
                    if _wiz_nid(_n["id"]) not in _existing_node_ids:
                        st.session_state.setdefault("nodes_data", []).append(_n)
                        _existing_node_ids.add(_wiz_nid(_n["id"]))

                st.session_state["elements_data"].extend(_new_elems)
                st.session_state["rigid_links"].extend(_gen["rigid_links"])

                # ── 主梁自動拆分 ──────────────────────────────────────────
                _split_count = 0
                if wiz_beam_member:
                    from core.cable_wizard import split_beam_at_nodes
                    _dc_nodes = [n for n in _gen["nodes"] if n.get("_role") == "deck_center"]
                    _node_map = {_wiz_nid(n.get("id")): n for n in st.session_state.get("nodes_data", [])}
                    # dc_nodes 裡找不到的節點補進 node_map（用 _gen["nodes"] 原始座標）
                    for _dcn in _dc_nodes:
                        _dcn_id = _wiz_nid(_dcn["id"])
                        if _dcn_id not in _node_map:
                            _node_map[_dcn_id] = _dcn

                    _beam_elems = [e for e in st.session_state["elements_data"] if str(e.get("member","")) == wiz_beam_member]

                    st.session_state["elements_data"], _split_count = split_beam_at_nodes(
                        elements=st.session_state["elements_data"],
                        beam_member=wiz_beam_member,
                        split_nodes=_dc_nodes,
                        node_map=_node_map,
                    )

                for _k in ("nodes", "elements", "rl_editor"):
                    st.session_state.pop(_k, None)
                _msg = (
                    f"已生成 {len(_new_nodes)} 個節點、"
                    f"{len(_new_elems)} 根索構件、"
                    f"{len(_gen['rigid_links'])} 個 Rigid Link。"
                )
                if _split_count:
                    _msg += f" 自動拆分主梁 {_split_count} 段。"
                st.success(_msg)
                st.rerun()

    # ── 索力初始設定 ─────────────────────────────────────────────────────────
    # checkbox 必須在 expander 之前渲染，否則同一 run 內 expander 讀不到勾選值
    include_sw = st.checkbox("含自重（Self-Weight）", value=False, key="include_sw")

    with st.expander("⚙️ 索力初始設定 Cable Pretension",
                     expanded=st.session_state.get("_pt_expander_open", False)):
        from core.cable_wizard import compute_cable_pretension_guess

        st.caption(
            "基於主梁標稱彎矩容量，自動計算初始索力猜測值。"
            "計算採三彎矩方程（假設 EI 均一），結果作為設計起點，請依彎矩圖微調。"
        )

        _pt_apply_msg = st.session_state.pop("_pt_apply_msg", "")
        if _pt_apply_msg:
            st.success(_pt_apply_msg)

        # 蒐集所有索群組（pin_i=True 且 pin_j=True 且有 group 標籤）
        _pt_cable_elems = [
            e for e in st.session_state.get("elements_data", [])
            if e.get("pin_i") and e.get("pin_j") and str(e.get("group", "")).startswith("gen:")
        ]
        _pt_groups = sorted({str(e["group"]) for e in _pt_cable_elems})

        if not _pt_groups:
            st.info("尚未生成任何索構件，請先使用 Cable Face Wizard 生成索面。")
        else:
            _pt_grp_labels = [g.replace("gen:", "") for g in _pt_groups]
            _pt_sel_label = st.selectbox(
                "選擇索群組", _pt_grp_labels, key="pt_sel_group"
            )
            _pt_sel_grp = f"gen:{_pt_sel_label}"
            _pt_grp_elems = [e for e in _pt_cable_elems if str(e.get("group")) == _pt_sel_grp]

            # 取得節點位置 map
            _pt_node_map = {
                str(n.get("id")): n
                for n in st.session_state.get("nodes_data", [])
            }

            # 計算各索傾角（從塔側偏心節點到橋面偏心節點）
            _pt_thetas = []
            _pt_anchor_xs = []
            _pt_anchor_node_ids = []
            _pt_elem_ids = []
            for e in _pt_grp_elems:
                ni = _pt_node_map.get(str(e.get("i")), {})
                nj = _pt_node_map.get(str(e.get("j")), {})
                if not ni or not nj:
                    continue
                xi, zi = float(ni.get("x", 0)), float(ni.get("z", 0))
                xj, zj = float(nj.get("x", 0)), float(nj.get("z", 0))
                dx = abs(xj - xi)
                dz = abs(zj - zi)
                theta = math.atan2(dz, dx) if dx > 1e-10 else math.pi / 2
                _pt_thetas.append(theta)
                # j 端（橋面偏心節點）的 x 作為錨點位置
                _pt_anchor_xs.append(xj)
                _pt_anchor_node_ids.append(str(e.get("j")))
                _pt_elem_ids.append(str(e.get("id")))

            # 從 supports 取真實支承 x 座標（uz=True 的節點）
            _pt_supports_raw = supports_df.dropna(subset=["node_id"]).to_dict("records")
            _pt_support_xs = []
            for sup in _pt_supports_raw:
                if sup.get("uz", False):
                    nid = str(int(float(sup["node_id"]))) if sup["node_id"] else ""
                    sn = _pt_node_map.get(nid, {})
                    if sn:
                        _pt_support_xs.append(float(sn.get("x", 0)))

            # 主梁 member 選單（與 Cable Face Wizard 邏輯一致）
            _pt_all_members = sorted({
                str(e.get("member", "")) for e in st.session_state.get("elements_data", [])
                if isinstance(e.get("member"), str) and e.get("member", "").strip()
                and not e.get("pin_i") and not e.get("pin_j")
            })
            _pt_beam_member = st.selectbox(
                "主梁 member 名稱",
                options=[""] + _pt_all_members,
                key="pt_beam_member",
                help="選擇主梁所屬的構件群組名稱，用於自動估算線重（與 Cable Face Wizard 的主梁 member 相同）",
            )

            # 估算主梁線重（從 compute_self_weight 結果，或 fallback 手動輸入）
            _sw_available = False
            _pt_w_auto = 0.0
            if st.session_state.get("include_sw") and st.session_state.get("sections") and _pt_beam_member:
                try:
                    from core.materials import compute_self_weight, expand_truss_data
                    _td_sw = expand_truss_data(
                        {
                            "nodes": nodes_df.dropna(subset=["id"]).fillna(0).to_dict("records"),
                            "elements": elements_df.dropna(subset=["id","i","j"]).fillna(0).to_dict("records"),
                            "supports": [], "loads": [], "element_loads": [], "element_point_loads": [],
                        },
                        st.session_state["materials"],
                        st.session_state["sections"],
                    )
                    _sw_res = compute_self_weight(_td_sw, st.session_state["sections"], st.session_state["materials"])
                    _sw_items = _sw_res.get("element_loads", [])
                    _sw_elem_map = {_norm_id(x["element_id"]): float(x.get("w", 0)) for x in _sw_items}
                    _main_ws, _main_lens = [], []
                    for _, row in elements_df.iterrows():
                        eid = row.get("id")
                        if str(row.get("member", "")).strip() != _pt_beam_member:
                            continue
                        if _norm_id(eid) not in _sw_elem_map:
                            continue
                        ni_ = _pt_node_map.get(_norm_id(row["i"]), {})
                        nj_ = _pt_node_map.get(_norm_id(row["j"]), {})
                        if ni_ and nj_:
                            Le_ = ((float(nj_["x"])-float(ni_["x"]))**2 +
                                   (float(nj_.get("z",0))-float(ni_.get("z",0)))**2)**0.5
                            _main_ws.append(abs(_sw_elem_map[_norm_id(eid)]))
                            _main_lens.append(Le_)
                    if _main_ws and sum(_main_lens) > 0:
                        _pt_w_auto = sum(w_*l_ for w_, l_ in zip(_main_ws, _main_lens)) / sum(_main_lens)
                        _sw_available = True
                except Exception:
                    pass

            if _sw_available:
                st.success(f"主梁等效線重（自重）：{_pt_w_auto/1000:.2f} kN/m（自動估算，member='{_pt_beam_member}'）")
                _pt_w = _pt_w_auto
            else:
                if st.session_state.get("include_sw") and _pt_beam_member:
                    st.warning(f"member='{_pt_beam_member}' 未取得自重資料，請確認截面已設定密度。")
                else:
                    st.warning("請勾選「含自重」並選擇主梁 member，或手動輸入主梁線重。")
                _pt_w_input = st.number_input(
                    "主梁線重 w（kN/m，向下）", min_value=0.0, value=50.0,
                    format="%.1f", key="pt_w_manual"
                )
                _pt_w = _pt_w_input * 1000.0  # kN/m → N/m

            _pt_M_allow_kNm = st.number_input(
                "主梁標稱彎矩容量 M_allow（kN·m）",
                min_value=0.0, value=0.0, format="%.1f", key="pt_M_allow"
            )
            _pt_M_allow = _pt_M_allow_kNm * 1000.0  # kN·m → N·m

            _pt_calc_btn = st.button("計算初始索力", key="pt_calc_btn",
                                     disabled=(not _pt_anchor_xs or not _pt_support_xs))

            if not _pt_support_xs:
                st.error("未偵測到有垂直約束（uz=True）的支承節點，請先設定支承。")

            if _pt_calc_btn and _pt_anchor_xs and _pt_support_xs:
                try:
                    _pt_result = compute_cable_pretension_guess(
                        support_xs=sorted(_pt_support_xs),
                        anchor_xs=_pt_anchor_xs,
                        thetas=_pt_thetas,
                        w=_pt_w,
                        M_allow=_pt_M_allow,
                    )
                    st.session_state.setdefault("pretension_state", {})
                    st.session_state["pretension_state"][_pt_sel_grp] = {
                        "global_ratio": st.session_state.get("pretension_state", {}).get(
                            _pt_sel_grp, {}).get("global_ratio", 1.0),
                        "individual_ratios": {
                            eid: 1.0 for eid in _pt_elem_ids
                        },
                        "base_forces": {
                            eid: float(f)
                            for eid, f in zip(_pt_elem_ids, _pt_result["cable_forces"])
                        },
                        "anchor_node_ids": _pt_anchor_node_ids,
                        "elem_ids": _pt_elem_ids,
                        "thetas_deg": [math.degrees(t) for t in _pt_thetas],
                        "M_max": _pt_result["M_max"],
                        "V_uniform": _pt_result["V_uniform"],
                        "feasible": _pt_result["feasible"],
                    }
                    st.rerun()
                except Exception as _pt_ex:
                    st.error(f"計算失敗：{_pt_ex}")

            # 顯示調整介面（若已計算過）
            _pt_state = st.session_state.get("pretension_state", {}).get(_pt_sel_grp)
            if _pt_state:
                M_max_kNm = _pt_state["M_max"] / 1000.0
                V_kN      = _pt_state["V_uniform"] / 1000.0

                if not _pt_state["feasible"]:
                    st.info(
                        f"無索狀態 M_max = {M_max_kNm:.1f} kN·m，已在容量範圍內。"
                        "可輸入較小的 M_allow 以獲得保守的索力初始值，或直接跳過此步驟。"
                    )
                else:
                    st.success(
                        f"無索 M_max = {M_max_kNm:.1f} kN·m｜"
                        f"每根索均勻垂直力 V = {V_kN:.2f} kN"
                    )

                    # Layer 1：全局倍率
                    _pt_global = st.slider(
                        "全局倍率（影響所有索）",
                        min_value=0.0, max_value=3.0, step=0.05,
                        value=float(_pt_state.get("global_ratio", 1.0)),
                        key=f"pt_global_{_pt_sel_grp}"
                    )
                    st.session_state["pretension_state"][_pt_sel_grp]["global_ratio"] = _pt_global

                    # Layer 2：逐根倍率表格
                    _pt_base  = _pt_state["base_forces"]
                    _pt_iratios = _pt_state["individual_ratios"]
                    _pt_thetas_deg = _pt_state["thetas_deg"]
                    _pt_eids  = _pt_state["elem_ids"]

                    _pt_table_rows = []
                    for idx, eid in enumerate(_pt_eids):
                        base_f = _pt_base.get(eid, 0.0)
                        ind_r  = _pt_iratios.get(eid, 1.0)
                        final_f = base_f * _pt_global * ind_r
                        _pt_table_rows.append({
                            "索 ID":      eid,
                            "θ (deg)":   round(_pt_thetas_deg[idx], 2) if idx < len(_pt_thetas_deg) else 0.0,
                            "初始力 F (kN)": round(base_f / 1000.0, 2),
                            "個別倍率":   ind_r,
                            "最終力 (kN)": round(final_f / 1000.0, 2),
                        })

                    import pandas as _pd_pt
                    _pt_df_in = _pd_pt.DataFrame(_pt_table_rows)
                    _pt_df_out = st.data_editor(
                        _pt_df_in,
                        column_config={
                            "索 ID":         st.column_config.TextColumn("索 ID",      disabled=True),
                            "θ (deg)":       st.column_config.NumberColumn("θ (deg)",  disabled=True, format="%.2f"),
                            "初始力 F (kN)": st.column_config.NumberColumn("初始力 (kN)", disabled=True, format="%.2f"),
                            "個別倍率":      st.column_config.NumberColumn("個別倍率",  min_value=0.0, max_value=5.0, step=0.05, format="%.2f"),
                            "最終力 (kN)":   st.column_config.NumberColumn("最終力 (kN)", disabled=True, format="%.2f"),
                        },
                        use_container_width=True,
                        hide_index=True,
                        key=f"pt_table_{_pt_sel_grp}",
                    )

                    # 更新個別倍率到 session_state，並即時更新最終力欄
                    for _, row in _pt_df_out.iterrows():
                        eid = str(row["索 ID"])
                        new_ratio = float(row["個別倍率"])
                        st.session_state["pretension_state"][_pt_sel_grp]["individual_ratios"][eid] = new_ratio

                    # 套用按鈕
                    if st.button("套用索力到載重表", key=f"pt_apply_{_pt_sel_grp}", type="primary"):
                        # 以累積載重為基底，再疊加本群組的錨點力
                        _loads_map = {
                            _norm_id(ld["node_id"]): dict(ld)
                            for ld in (st.session_state.get("_pt_accumulated_loads") or [])
                            if str(ld.get("node_id", "")).strip() not in ("", "nan", "None")
                        }

                        _applied_eids = []
                        _final_forces_kN = []
                        _cur_iratios = st.session_state["pretension_state"][_pt_sel_grp]["individual_ratios"]
                        _cur_global  = st.session_state["pretension_state"][_pt_sel_grp]["global_ratio"]
                        _anchor_nids = _pt_state["anchor_node_ids"]

                        for idx, eid in enumerate(_pt_eids):
                            base_f   = _pt_base.get(eid, 0.0)
                            ind_r    = _cur_iratios.get(eid, 1.0)
                            final_fz = base_f * _cur_global * ind_r  # 向上（正 fz）
                            anid = _anchor_nids[idx] if idx < len(_anchor_nids) else None
                            if anid is None:
                                continue
                            anid_key = _norm_id(anid)
                            if anid_key in _loads_map:
                                _loads_map[anid_key]["fz"] = final_fz
                            else:
                                _loads_map[anid_key] = {"node_id": anid, "fz": final_fz}
                            _applied_eids.append(eid)
                            _final_forces_kN.append(final_fz / 1000.0)

                        # 合併現有載重表（保留其他群組已套用的資料）再覆蓋本群組錨點
                        _existing_loads = {
                            _norm_id(ld["node_id"]): dict(ld)
                            for ld in (st.session_state.get("_pt_accumulated_loads") or [])
                            if str(ld.get("node_id", "")).strip() not in ("", "nan", "None")
                        }
                        _existing_loads.update(_loads_map)
                        st.session_state["_pt_accumulated_loads"] = list(_existing_loads.values())
                        st.session_state.pop("loads", None)
                        st.session_state["_loaded_loads"] = list(_existing_loads.values())
                        st.session_state["_pt_expander_open"] = True
                        st.session_state["_pt_apply_msg"] = (
                            f"已套用 {len(_applied_eids)} 根索｜"
                            f"最大 {max(_final_forces_kN):.2f} kN｜"
                            f"最小 {min(_final_forces_kN):.2f} kN"
                        ) if _final_forces_kN else ""
                        st.rerun()

    run_btn = st.button("執行分析（符號解）", type="primary", use_container_width=True)
    num_btn = st.button("執行數值分析（直接代入）", use_container_width=True)

    cache_valid = bool(st.session_state["sym_cache"].get("raw_result"))
    if cache_valid:
        import json as _json
        current_fp = _compute_fingerprint(
            _json.dumps(nodes_df.dropna(subset=['id']).fillna(0).to_dict('records'), sort_keys=True),
            _json.dumps(elements_df.dropna(subset=['id','i','j']).fillna(0).to_dict('records'), sort_keys=True),
            _json.dumps(supports_df.dropna(subset=['node_id']).fillna(False).to_dict('records'), sort_keys=True),
        )
        cache_valid = (current_fp == st.session_state["sym_cache"].get("fingerprint"))

    if cache_valid:
        _current_sec_groups = sorted(
            s.get("name", "") for s in st.session_state.get("sections", [])
            if s.get("name")
        )
        _cached_sec_groups = st.session_state["sym_cache"].get("section_groups", [])
        if _current_sec_groups != _cached_sec_groups:
            cache_valid = False

    st.caption("快速代入：幾何與支承不變時，直接代入新材料/載重參數，無需重新求符號解。")
    with st.expander("⚡ 快速代入參數", expanded=False):
        # 斷面組清單（依 elements_data 中實際使用的斷面）
        _qf_sec_names = sorted(set(
            r.get("section", "") for r in st.session_state.get("elements_data", [])
            if isinstance(r.get("section"), str) and r.get("section")
        ))
        _qf_sec_key = str(_qf_sec_names)

        if ("quickfill_overrides" not in st.session_state
                or st.session_state.get("quickfill_sec_key") != _qf_sec_key):
            # 從 sections + materials 取預設值
            _qf_defaults = {}
            for sn in _qf_sec_names:
                sec = sec_map_ui.get(sn, {})
                mat = mat_map_ui.get(sec.get("material", ""), {})
                _qf_defaults[sn] = {
                    "E":   float(mat.get("E",   200e9)),
                    "A":   float(sec.get("A",   0.01)),
                    "I33": float(sec.get("I33", 1e-4)),
                    "G":   float(mat.get("G",   77e9)),
                }
            st.session_state["quickfill_overrides"] = _qf_defaults
            st.session_state["quickfill_sec_key"]   = _qf_sec_key

        if _qf_sec_names:
            _qf_rows = [
                {"斷面名稱": sn,
                 "E (Pa)":   st.session_state["quickfill_overrides"][sn]["E"],
                 "A (m²)":   st.session_state["quickfill_overrides"][sn]["A"],
                 "I33 (m⁴)": st.session_state["quickfill_overrides"][sn]["I33"],
                 "G (Pa)":   st.session_state["quickfill_overrides"][sn]["G"]}
                for sn in _qf_sec_names
            ]
            _qf_df = st.data_editor(
                pd.DataFrame(_qf_rows),
                column_config={
                    "斷面名稱": st.column_config.TextColumn("斷面名稱", disabled=True),
                    "E (Pa)":   st.column_config.NumberColumn("E (Pa)",   format="%.3e"),
                    "A (m²)":   st.column_config.NumberColumn("A (m²)",   format="%.4e"),
                    "I33 (m⁴)": st.column_config.NumberColumn("I33 (m⁴)", format="%.4e"),
                    "G (Pa)":   st.column_config.NumberColumn("G (Pa)",   format="%.3e"),
                },
                num_rows="fixed",
                key="qf_editor",
                use_container_width=True,
            )
            # 將編輯後的值寫回 quickfill_overrides
            for _, row in _qf_df.iterrows():
                sn = row["斷面名稱"]
                if sn in st.session_state["quickfill_overrides"]:
                    st.session_state["quickfill_overrides"][sn] = {
                        "E":   float(row["E (Pa)"]),
                        "A":   float(row["A (m²)"]),
                        "I33": float(row["I33 (m⁴)"]),
                        "G":   float(row["G (Pa)"]),
                    }
            if len(_qf_sec_names) > 5:
                st.warning("超過 5 個斷面組，表格可能較長，請捲動查看。")
        else:
            st.info("請先在桿件表格中指派斷面，才能使用快速代入。")

        with st.expander("從形狀計算截面參數（填入快速代入表格）", expanded=False):
            st.caption(
                "計算結果**只填入上方快速代入表格**，不影響左側的截面定義。\n"
                "⚠️ J 計算公式同左側截面管理區（矩形實心用 Timoshenko 近似等）。"
            )
            _qf_calc_target = st.selectbox(
                "填入斷面", options=_qf_sec_names, key="qf_calc_target"
            ) if _qf_sec_names else None

            _qf_shape_sel = st.selectbox(
                "截面形狀", options=list(SHAPE_INPUTS.keys()), key="qf_shape_sel"
            )
            _qf_shape_vals = {}

            if _qf_shape_sel == "箱涵":
                _qf_shape_vals["n_cell"] = st.selectbox(
                    "室數 n_cell", options=[1, 2, 3, 4, 5], key="qf_sv_n_cell"
                )
                param_keys = [k for k, _ in SHAPE_INPUTS["箱涵"]]
                left_keys  = param_keys[:4]
                right_keys = param_keys[4:]
                qf_col_l, qf_col_r = st.columns(2)
                for k, label in [(k, lbl) for k, lbl in SHAPE_INPUTS["箱涵"] if k in left_keys]:
                    _qf_shape_vals[k] = qf_col_l.number_input(
                        label, value=BOX_DEFAULTS[k], format="%.4f", key=f"qf_sv_{k}"
                    )
                for k, label in [(k, lbl) for k, lbl in SHAPE_INPUTS["箱涵"] if k in right_keys]:
                    _qf_shape_vals[k] = qf_col_r.number_input(
                        label, value=BOX_DEFAULTS[k], format="%.4f", key=f"qf_sv_{k}"
                    )
                # ── 箱涵 Plotly 即時預覽（同左側計算器，複製邏輯）────────────────
                bv = _qf_shape_vals
                n_qf  = bv["n_cell"]
                bt_qf = bv["b_top"]; bb_qf = bv["b_bot"]
                hv_qf = bv["h"];     ct_qf = bv["c_top"]
                tw_qf = bv["t_web"]; tt_qf = bv["t_top"]; tb_qf = bv["t_bot"]; tdia_qf = bv["t_dia"]

                fig_qf = go.Figure()
                half_top_qf = bt_qf / 2
                half_bot_qf = bb_qf / 2
                fig_qf.add_trace(go.Scatter(
                    x=[-half_bot_qf, half_bot_qf,
                       half_bot_qf,  half_top_qf,
                       half_top_qf,  half_bot_qf,
                       half_bot_qf, -half_bot_qf,
                       -half_bot_qf, -half_top_qf,
                       -half_top_qf, -half_bot_qf,
                       -half_bot_qf],
                    y=[0,           0,
                       hv_qf-tt_qf, hv_qf-tt_qf,
                       hv_qf,       hv_qf,
                       hv_qf,       hv_qf,
                       hv_qf,       hv_qf,
                       hv_qf-tt_qf, hv_qf-tt_qf,
                       0],
                    fill="toself", fillcolor="rgba(70,130,180,0.65)",
                    line=dict(color="#1a3a5c", width=2), showlegend=False,
                ))
                b_box_qf = bb_qf - 2 * tw_qf
                s_void_qf = (b_box_qf - (n_qf - 1) * tdia_qf) / n_qf
                for i in range(n_qf):
                    ix0 = -b_box_qf / 2 + i * (s_void_qf + tdia_qf)
                    ix1 = ix0 + s_void_qf
                    fig_qf.add_trace(go.Scatter(
                        x=[ix0, ix1, ix1, ix0, ix0],
                        y=[tb_qf, tb_qf, hv_qf - tt_qf, hv_qf - tt_qf, tb_qf],
                        fill="toself", fillcolor="white",
                        line=dict(color="#1a3a5c", width=1.2), showlegend=False,
                    ))
                _qf_blue   = dict(color="#1565C0", size=10)
                _qf_orange = dict(color="#E65100", size=10)
                fig_qf.update_layout(
                    annotations=[
                        dict(x=0, y=hv_qf, xref="x", yref="y",
                             ax=0, ay=-18, axref="pixel", ayref="pixel",
                             text=f"<b>b_top = {bt_qf:.3f} m</b>", showarrow=True,
                             arrowhead=0, arrowcolor="#1565C0", arrowwidth=1,
                             font=_qf_blue, bgcolor="white", borderpad=2,
                             xanchor="center", yanchor="bottom"),
                        dict(x=0, y=0, xref="x", yref="y",
                             ax=0, ay=18, axref="pixel", ayref="pixel",
                             text=f"<b>b_bot = {bb_qf:.3f} m</b>", showarrow=True,
                             arrowhead=0, arrowcolor="#1565C0", arrowwidth=1,
                             font=_qf_blue, bgcolor="white", borderpad=2,
                             xanchor="center", yanchor="top"),
                        dict(x=half_bot_qf, y=hv_qf / 2, xref="x", yref="y",
                             ax=18, ay=0, axref="pixel", ayref="pixel",
                             text=f"<b>h = {hv_qf:.3f} m</b>", showarrow=True,
                             arrowhead=0, arrowcolor="#E65100", arrowwidth=1,
                             font=_qf_orange, bgcolor="white", borderpad=2,
                             xanchor="left", yanchor="middle"),
                    ],
                    xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
                    yaxis=dict(visible=False),
                    margin=dict(l=10, r=100, t=45, b=45),
                    height=320, plot_bgcolor="white",
                )
                st.plotly_chart(fig_qf, use_container_width=True)
            else:
                _qf_cols = st.columns(len(SHAPE_INPUTS[_qf_shape_sel]))
                for col, (k, label) in zip(_qf_cols, SHAPE_INPUTS[_qf_shape_sel]):
                    _qf_shape_vals[k] = col.number_input(
                        label, value=0.1, format="%.4f", key=f"qf_sv_{k}"
                    )

            if st.button("計算並填入快速代入表格", key="qf_calc_btn") and _qf_calc_target:
                try:
                    _qf_props = compute_section_props(_qf_shape_sel, _qf_shape_vals)
                    st.session_state["quickfill_overrides"][_qf_calc_target].update({
                        "A":   _qf_props["A"],
                        "I33": _qf_props["I33"],
                    })
                    st.success(
                        f"已填入 {_qf_calc_target}：A={_qf_props['A']:.4e} m²，"
                        f"I33={_qf_props['I33']:.4e} m⁴（G 維持原值）。"
                    )
                    st.session_state.pop("qf_editor", None)
                    st.rerun()
                except Exception as ex:
                    st.error(f"計算失敗：{ex}")

        _qf_col1, _qf_col2 = st.columns(2)
        pe_P = _qf_col1.number_input("P 倍率", value=1.0, key="pe_P")
        pe_w = _qf_col2.number_input("w 倍率", value=0.0, key="pe_w")
        st.caption(
            "**位移**：從符號公式直接代入 E/A/I/G/L，結果包含材料依賴性。\n"
            "**內力與反力**：以輸入的 E/A/I/G 重新數值求解（使用快取跳過符號分析），"
            "P 與 w 為載重倍率（相對於輸入表格中的數值）。\n"
            "幾何或支承改變時請重新執行完整分析（符號解）以更新快取。"
        )
        fast_btn = st.button(
            "⚡ 代入參數（快速）",
            disabled=not cache_valid,
            use_container_width=True,
            key="fast_btn",
            help="請先執行完整分析以建立快取" if not cache_valid else "代入新參數至已有的符號公式",
        )

# ── 共用 truss_data 建構 ──────────────────────────────────────────────
print(f"[APP] e_loads_df dtypes={e_loads_df.dtypes.to_dict()}, head=\n{e_loads_df.head(3)}")
_e_loads_records = e_loads_df.dropna(subset=['element_id']).fillna(0).to_dict('records')
print(f"[APP] dropna後={len(_e_loads_records)}筆")
truss_data = {
    "nodes":               nodes_df.dropna(subset=['id']).fillna(0).to_dict('records'),
    "elements":            elements_df.dropna(subset=['id','i','j']).fillna(0).to_dict('records'),
    "supports":            supports_df.dropna(subset=['node_id']).fillna(False).to_dict('records'),
    "loads":               loads_df.dropna(subset=['node_id']).fillna(0).to_dict('records'),
    "element_loads":       _e_loads_records,
    "element_point_loads": e_pt_loads_df.dropna(subset=['element_id']).fillna(0).to_dict('records'),
    "rigid_links":         st.session_state.get("rigid_links", []),
}

with right_panel:
    st.subheader("分析結果輸出")

    # ── 存檔 / 載入 ──────────────────────────────────────────────────────
    with st.expander("💾 存檔 / 載入", expanded=False):
        import json as _json_save

        # 存檔
        _save_data = {
            "version": 1,
            "materials":            st.session_state.get("materials", []),
            "sections":             st.session_state.get("sections", []),
            "elements_data":        st.session_state.get("elements_data", []),
            "nodes":                truss_data["nodes"],
            "supports":             truss_data["supports"],
            "loads":                truss_data["loads"],
            "element_loads":        truss_data["element_loads"],
            "element_point_loads":  truss_data["element_point_loads"],
            "rigid_links":          st.session_state.get("rigid_links", []),
        }
        st.download_button(
            label="💾 存檔（JSON）",
            data=_json_save.dumps(_save_data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="truss_project.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()

        # 載入
        _load_file = st.file_uploader("載入存檔（JSON）", type=["json"], key="project_upload")
        if _load_file is not None:
            try:
                _loaded = _json_save.loads(_load_file.read().decode("utf-8"))
                if _loaded.get("version") != 1:
                    st.error("不支援的存檔版本，請確認檔案正確。")
                else:
                    st.session_state["materials"]           = _loaded.get("materials", [])
                    st.session_state["sections"]            = _loaded.get("sections", [])
                    st.session_state["elements_data"]       = _loaded.get("elements_data", [])
                    st.session_state["rigid_links"]         = _loaded.get("rigid_links", [])
                    st.session_state["elem_prev_section"]   = {}
                    st.session_state["sym_cache"]           = {}
                    st.session_state.pop("quickfill_overrides", None)
                    st.session_state.pop("quickfill_sec_key", None)
                    # 清除 data_editor 快取，強制從 session_state 重新載入
                    for _k in ("nodes", "elements", "supports", "loads",
                               "element_loads", "element_point_loads", "rl_editor"):
                        st.session_state.pop(_k, None)
                    # 將節點、支承、載重寫入 nodes_data / session_state 供下次 render 使用
                    st.session_state["nodes_data"]                  = _loaded.get("nodes", [])
                    st.session_state["_loaded_supports"]            = _loaded.get("supports", [])
                    st.session_state["_loaded_loads"]               = _loaded.get("loads", [])
                    st.session_state["_loaded_element_loads"]       = _loaded.get("element_loads", [])
                    st.session_state["_loaded_element_point_loads"] = _loaded.get("element_point_loads", [])
                    st.success("存檔已載入，代數解快取已清除，請重新執行分析。")
                    st.rerun()
            except Exception as _ex:
                st.error(f"載入失敗：{_ex}")

    # ── 快取狀態顯示 ─────────────────────────────────────────────────────
    with st.expander("代數解快取管理", expanded=False):
        if st.session_state["sym_cache"].get("raw_result"):
            fp = st.session_state["sym_cache"].get("fingerprint", {})
            ts = st.session_state["sym_cache"].get("timestamp", "—")
            n_e = fp.get("n_elements", "?")
            st.success(f"快取有效：{n_e} 根桿件，分析時間 {ts}")
        else:
            st.warning("尚無快取，請先執行完整分析（符號解）。")

        # 匯出
        if st.session_state["sym_cache"].get("raw_result"):
            txt_content = export_cache_to_txt(st.session_state["sym_cache"])
            st.download_button(
                label="匯出代數解 TXT",
                data=txt_content.encode("utf-8"),
                file_name="symbolic_cache.txt",
                mime="text/plain",
            )

        # 匯入
        uploaded = st.file_uploader("匯入代數解 TXT", type=["txt"], key="cache_upload")
        if uploaded is not None:
            txt_str = uploaded.read().decode("utf-8")
            import_result = import_cache_from_txt(txt_str, truss_data)
            if "error" in import_result:
                st.error(f"匯入失敗：{import_result['error']}")
            else:
                st.session_state["sym_cache"] = import_result
                if import_result.get("materials"):
                    st.session_state["materials"] = import_result["materials"]
                if import_result.get("sections"):
                    st.session_state["sections"] = import_result["sections"]
                st.success("指紋一致，快取已載入，材料與截面定義已還原。")

    output_area = st.empty()
    _any_analysis_btn = num_btn or fast_btn or run_btn
    if st.session_state.get("last_result") and not _any_analysis_btn:
        with output_area.container():
            _render_result_tabs(st.session_state["last_result"], truss_data)
    res_eval = None

    if num_btn:
        # 截面完整性驗證：數值分析前確認所有被使用的截面都有有效 A
        _sec_map_chk = {s["name"]: s for s in st.session_state.get("sections", [])}
        _bad_secs = []
        for _elem in st.session_state.get("elements_data", []):
            _sn = _elem.get("section", "")
            if not _sn:
                continue
            _sec = _sec_map_chk.get(_sn, {})
            _shape = _sec.get("shape", "Custom")
            try:
                _props = compute_section_props(_shape, _sec)
                _A_eff = float(_elem.get("A") or 0) or _props.get("A", 0)
            except Exception:
                _A_eff = 0.0
            if _A_eff <= 0 and _sn not in _bad_secs:
                _bad_secs.append(_sn)
        if _bad_secs:
            st.error(
                f"數值分析中止：以下截面缺少幾何參數（A=0），"
                f"請至左側「截面」區塊選取截面並點擊「計算並填入」後再分析：\n\n"
                + "\n".join(f"- **{s}**" for s in _bad_secs)
            )
            st.stop()

        with st.status("數值分析中...", expanded=True) as _status:
            try:
                st.write("建立結構模型...")
                _td_num = expand_truss_data(truss_data, st.session_state["materials"], st.session_state["sections"]) if st.session_state["sections"] else truss_data
                if include_sw and st.session_state["sections"]:
                    import copy as _copy
                    st.write("計算自重...")
                    _td_num = _copy.deepcopy(_td_num)
                    _sw = compute_self_weight(_td_num, st.session_state["sections"], st.session_state["materials"])
                    _existing_el = {el["element_id"]: el for el in _td_num.get("element_loads", [])}
                    for _swl in _sw["element_loads"]:
                        _eid = _swl["element_id"]
                        if _eid in _existing_el:
                            _existing_el[_eid]["w"] = _existing_el[_eid].get("w", 0.0) + _swl["w"]
                        else:
                            _td_num["element_loads"].append({"element_id": _eid, "w": _swl["w"]})
                    _existing_nl = {ld["node_id"]: ld for ld in _td_num.get("loads", [])}
                    for _nl in _sw["node_loads"]:
                        _nid = _nl["node_id"]
                        if _nid in _existing_nl:
                            _existing_nl[_nid]["fz"] = _existing_nl[_nid].get("fz", 0.0) + _nl["fz"]
                        else:
                            _td_num["loads"].append({"node_id": _nid, "fz": _nl["fz"]})
                st.write("組裝剛度矩陣並求解...")
                res_eval = evaluate_numerical_results(_td_num, force_2d=st.session_state.get("structure_dim", "auto"))
                _status.update(label=f"數值分析完成（耗時 {res_eval['eval_time_ms']} ms）", state="complete")
                st.session_state["last_result"] = res_eval
                with output_area.container():
                    _render_result_tabs(res_eval, truss_data)
            except Exception as e:
                _status.update(label="分析失敗", state="error")
                if "singular" in str(e).lower() or "not invertible" in str(e).lower():
                    st.error(f"分析失敗：結構不穩定 (奇異矩陣)。請檢查支承是否足夠。\n\n詳細錯誤: {e}")
                else:
                    st.error(f"數值分析發生錯誤：{e}")
                st.stop()

    if fast_btn:
        _group_vals = {
            sn: {"E": v["E"], "A": v["A"], "I": v["I33"], "G": v["G"]}
            for sn, v in st.session_state.get("quickfill_overrides", {}).items()
        }
        real_params = {"groups": _group_vals, "P": pe_P, "w": pe_w}
        try:
            res_eval = evaluate_real_results(
                truss_data, real_params,
                symbolic_cache=st.session_state["sym_cache"],
                materials=st.session_state["materials"],
                sections=st.session_state["sections"],
                include_self_weight=include_sw,
            )
            st.session_state["last_result"] = res_eval
            with output_area.container():
                _render_result_tabs(res_eval, truss_data)
        except Exception as e:
            st.error(f"代入失敗：{e}")

    if run_btn:
        with st.status("符號解分析中（可能需要數十秒）...", expanded=True) as _status:
            try:
                # 清除舊快取，強制重新求解
                st.session_state["sym_cache"] = {}

                # 建立各斷面組的符號對應
                st.write("建立符號變數...")
                _sec_group_map = {}
                for gi, sec in enumerate(st.session_state["sections"], start=1):
                    sn = sec.get("name", "")
                    if not sn:
                        continue
                    _sec_group_map[sn] = {
                        "E":   sp.Symbol(f"E_s{gi}"),
                        "A":   sp.Symbol(f"A_s{gi}"),
                        "I33": sp.Symbol(f"I_s{gi}"),
                        "I22": sp.Symbol(f"I_s{gi}"),
                        "G":   sp.Symbol(f"G_s{gi}"),
                    }

                st.write("組裝符號剛度矩陣並求解（SymPy）...")
                _run_group_vals = {
                    sn: {"E": v["E"], "A": v["A"], "I": v["I33"], "G": v["G"]}
                    for sn, v in st.session_state.get("quickfill_overrides", {}).items()
                }
                real_params = {"groups": _run_group_vals, "P": pe_P, "w": pe_w}
                res_eval = evaluate_real_results(
                    truss_data, real_params,
                    symbolic_cache=st.session_state["sym_cache"],
                    materials=st.session_state["materials"],
                    sections=st.session_state["sections"],
                    include_self_weight=include_sw,
                    section_group_map=_sec_group_map if _sec_group_map else None,
                )
                st.write("代入數值並整理結果...")
                res = st.session_state["sym_cache"]["raw_result"]
                _status.update(label=f"符號解分析完成（耗時 {res_eval['eval_time_ms']} ms）", state="complete")
                st.session_state["last_result"] = res_eval
                with output_area.container():
                    _render_result_tabs(res_eval, truss_data)
            except Exception as e:
                import traceback as _tb
                _status.update(label="分析失敗", state="error")
                if "singular" in str(e).lower() or "not invertible" in str(e).lower():
                    st.error(f"分析失敗：結構不穩定 (出現奇異矩陣)。請檢查是否有足夠的支承，或者是否有機構 (Mechanism) 產生！\n\n詳細錯誤: {e}")
                    # 印出所有桿件的截面參數供診斷
                    print("=== [SINGULAR DEBUG] 桿件截面參數 ===")
                    for _e in truss_data.get("elements", []):
                        print(f"  elem {_e.get('id')}: i={_e.get('i')} j={_e.get('j')} "
                              f"E={_e.get('E')} A={_e.get('A')} I33={_e.get('I33')} "
                              f"pin_i={_e.get('pin_i')} pin_j={_e.get('pin_j')}")
                    print("=== [SINGULAR DEBUG] 節點清單 ===")
                    for _n in truss_data.get("nodes", []):
                        print(f"  node {_n.get('id')}: x={_n.get('x')} y={_n.get('y')} z={_n.get('z')}")
                    print("=== [SINGULAR DEBUG] 支承 ===")
                    for _s in truss_data.get("supports", []):
                        print(f"  support node={_s.get('node_id')} ux={_s.get('ux')} uy={_s.get('uy')} uz={_s.get('uz')} rx={_s.get('rx')} ry={_s.get('ry')} rz={_s.get('rz')}")
                    print("=== [SINGULAR DEBUG] Rigid Links ===")
                    for _rl in truss_data.get("rigid_links", []):
                        print(f"  RL {_rl.get('id')}: master={_rl.get('master')} slave={_rl.get('slave')}")
                else:
                    st.error(f"分析時發生錯誤：{e}")
                    st.code(_tb.format_exc(), language="python")
                st.stop()

    st.divider()
    st.subheader("結構預覽與分析圖形")
    _rl_state = st.session_state.get("rigid_links", [])
    if run_btn:
        # 分析後的結果圖
        fig = create_structure_plot(nodes_df, elements_df, supports_df, res_eval.get('support_reactions'), rigid_links=_rl_state)
        st.plotly_chart(fig, use_container_width=True, key="struct_plot_result")
    else:
        # 純輸入預覽
        fig = create_structure_plot(nodes_df, elements_df, supports_df, rigid_links=_rl_state)
        st.plotly_chart(fig, use_container_width=True, key="struct_plot_preview")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — 地震力分析
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("🌋 地震力分析（MDOF）")

_last = st.session_state.get("last_result")
_has_fem = bool(_last and "K_red_diagonal" in _last)

if not _has_fem:
    st.info("請先執行 **FEM 數值分析**，才能進行地震力計算。")
else:
    _seismic_tabs = st.tabs(["⚙️ 輸入設定", "📊 模態分析", "🌋 地震力與載重組合"])

    # ── 子 Tab A：輸入設定 ─────────────────────────────────────────────────
    with _seismic_tabs[0]:
        st.subheader("橋墩識別與參數設定")

        # 收集所有非空 member 名稱
        _all_members = sorted({
            (e.get("member") or "").strip()
            for e in st.session_state.get("elements_data", [])
            if (e.get("member") or "").strip()
        })

        if not _all_members:
            st.warning("元素表中尚無 member 標記。請在元素表的「構件」欄位填入橋墩群組名稱後再回來。")
        else:
            col_a1, col_a2 = st.columns([2, 1])
            with col_a1:
                _pier_sel = st.multiselect(
                    "選擇橋墩構件群組（可複選）",
                    options=_all_members,
                    default=st.session_state["seismic_pier_selection"],
                    key="_seismic_pier_sel_widget",
                    help="只選取**橋墩（垂直構件）**的 member 群組；橋面構件不要選入。"
                )
                st.session_state["seismic_pier_selection"] = _pier_sel
            with col_a2:
                _axis = st.radio(
                    "橋軸方向（縱向）",
                    ["X 軸", "Y 軸"],
                    index=0 if st.session_state["seismic_bridge_axis"] == "X" else 1,
                    help="選擇橋梁縱向對應的全域座標軸，用於萃取側向勁度。",
                )
                st.session_state["seismic_bridge_axis"] = "X" if _axis == "X 軸" else "Y"

            # SDL 附加靜載倍率
            if "seismic_sdl_ratio" not in st.session_state:
                st.session_state["seismic_sdl_ratio"] = 0.0
            st.session_state["seismic_sdl_ratio"] = st.number_input(
                "SDL 附加靜載倍率",
                min_value=0.0, max_value=5.0, step=0.05,
                value=float(st.session_state["seismic_sdl_ratio"]),
                format="%.2f",
                help="附加靜載（非結構質量）佔結構自重的比例。"
                     "例如 0.20 代表 SDL = 20% 自重。最終質量 = 自重質量 × (1 + 倍率)。",
            )

            if _pier_sel:
                if st.button("🔍 從 FEM 提取質量與勁度", type="primary"):
                    try:
                        _pier_tops = _sb.identify_pier_tops(
                            _pier_sel,
                            st.session_state.get("elements_data", []),
                            truss_data,
                        )
                        _masses = _sb.extract_pier_masses(
                            _pier_tops,
                            st.session_state.get("elements_data", []),
                            truss_data,
                            st.session_state.get("sections", []),
                            st.session_state.get("materials", []),
                        )
                        # 縱向勁度（橋軸方向）與橫向勁度（垂直橋軸水平方向）
                        _long_axis = st.session_state["seismic_bridge_axis"]       # "X" or "Y"
                        _lat_axis  = "Y" if _long_axis == "X" else "X"
                        _stiffs_L = _sb.extract_pier_stiffnesses(_pier_tops, _last, bridge_axis=_long_axis)
                        _stiffs_T = _sb.extract_pier_stiffnesses(_pier_tops, _last, bridge_axis=_lat_axis)
                        _warnings = _sb.validate_mdof_assumptions(
                            _pier_tops, truss_data, _last, _stiffs_L
                        )

                        # 組成可編輯表格（SDL 倍率套用在自重質量上）
                        import pandas as pd
                        _sdl = float(st.session_state.get("seismic_sdl_ratio", 0.0))
                        _rows = []
                        for _pn in _pier_sel:
                            if _pn not in _pier_tops:
                                continue
                            _m_sw = _masses.get(_pn, {}).get("mass_Mg", 0)
                            _rows.append({
                                "橋墩":           _pn,
                                "頂點節點":        _pier_tops[_pn]["top_node"],
                                "質量 (Mg)":       round(_m_sw * (1 + _sdl), 4),
                                f"縱向勁度 {_long_axis} (kN/m)": round(_stiffs_L.get(_pn, {}).get("stiffness_kNm", 0), 2),
                                f"橫向勁度 {_lat_axis} (kN/m)": round(_stiffs_T.get(_pn, {}).get("stiffness_kNm", 0), 2),
                            })
                        st.session_state["seismic_extracted"] = {
                            "pier_tops": _pier_tops,
                            "masses":    _masses,
                            "stiffs_L":  _stiffs_L,
                            "stiffs_T":  _stiffs_T,
                            "long_axis": _long_axis,
                            "lat_axis":  _lat_axis,
                        }
                        st.session_state["seismic_user_table"] = pd.DataFrame(_rows)
                        st.session_state["seismic_warnings"] = _warnings
                        st.success("提取完成！請確認下方表格數值，可直接修改後再執行分析。")
                    except Exception as _e:
                        import traceback
                        st.error(f"提取失敗：{_e}")
                        st.code(traceback.format_exc())

        # 顯示驗證警告
        for _w in st.session_state.get("seismic_warnings", []):
            if _w["level"] == "error":
                st.error(_w["message"])
            else:
                st.warning(_w["message"])

        # 可編輯質量/勁度表
        if st.session_state["seismic_user_table"] is not None:
            st.subheader("橋墩參數（可手動修改）")
            import pandas as pd
            _edited = st.data_editor(
                st.session_state["seismic_user_table"],
                column_config={
                    "橋墩":       st.column_config.TextColumn("橋墩", disabled=True),
                    "頂點節點":   st.column_config.TextColumn("頂點節點", disabled=True),
                    "質量 (Mg)":  st.column_config.NumberColumn("質量 (Mg)",  min_value=0.0, format="%.4f"),
                    "勁度 (kN/m)": st.column_config.NumberColumn("勁度 (kN/m)", min_value=0.0, format="%.2f"),
                },
                use_container_width=True,
                hide_index=True,
                key="seismic_table_editor",
            )
            st.session_state["seismic_user_table"] = _edited

        # 地震參數輸入
        st.subheader("地震設計參數")
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            _mode_label = st.selectbox("分析等級", ["L1", "475年", "2500年"], index=1)
        with _c2:
            _i_coeff = st.number_input("用途係數 I", value=1.2, min_value=1.0, max_value=1.5, step=0.1)
        with _c3:
            _alpha_y = st.number_input("起始降伏放大倍數 αy", value=1.0, min_value=0.1, step=0.1)
        _c4, _c5 = st.columns(2)
        with _c4:
            _r_val = st.number_input("韌性容量 R", value=2.0, min_value=1.0, step=0.5)
        with _c5:
            _analysis_dir = st.selectbox("計算方向", ["縱向與橫向均計算", "僅縱向", "僅橫向"])

        st.session_state["seismic_params"] = {
            "mode_label": _mode_label,
            "i_coeff":    _i_coeff,
            "alpha_y":    _alpha_y,
            "r_val":      _r_val,
            "analysis_dir": _analysis_dir,
        }

    # ── 子 Tab B：模態分析 ─────────────────────────────────────────────────
    with _seismic_tabs[1]:
        st.subheader("MDOF 模態分析結果")

        if st.session_state["seismic_user_table"] is None:
            st.info("請先在「輸入設定」頁面提取橋墩參數。")
        else:
            if st.button("▶ 執行模態分析", type="primary"):
                try:
                    from core import seismic_logic as _sl

                    _df_tbl = st.session_state["seismic_user_table"]
                    _ext    = st.session_state.get("seismic_extracted", {})
                    _long_ax = _ext.get("long_axis", "X")
                    _lat_ax  = _ext.get("lat_axis",  "Y")
                    _L_col = f"縱向勁度 {_long_ax} (kN/m)"
                    _T_col = f"橫向勁度 {_lat_ax} (kN/m)"

                    _masses_list   = _df_tbl["質量 (Mg)"].tolist()
                    _stiffs_L_list = _df_tbl[_L_col].tolist() if _L_col in _df_tbl.columns else []
                    _stiffs_T_list = _df_tbl[_T_col].tolist() if _T_col in _df_tbl.columns else []

                    if any(m <= 0 for m in _masses_list):
                        st.error("質量值必須大於 0，請確認橋墩參數。")
                    else:
                        _results = {}
                        if _stiffs_L_list and all(k > 0 for k in _stiffs_L_list):
                            _results["L"] = _sl.calculate_mdof_periods(_masses_list, _stiffs_L_list)
                        if _stiffs_T_list and all(k > 0 for k in _stiffs_T_list):
                            _results["T"] = _sl.calculate_mdof_periods(_masses_list, _stiffs_T_list)
                        st.session_state["seismic_result"] = _results
                        st.success("模態分析完成！")
                except Exception as _e:
                    import traceback
                    st.error(f"模態分析失敗：{_e}")
                    st.code(traceback.format_exc())

            _mdof_results = st.session_state.get("seismic_result")
            if _mdof_results:
                import pandas as pd
                import plotly.graph_objects as go
                _ext     = st.session_state.get("seismic_extracted", {})
                _long_ax = _ext.get("long_axis", "X")
                _lat_ax  = _ext.get("lat_axis",  "Y")

                for _dir_key, _dir_label in [("L", f"縱向（{_long_ax}）"), ("T", f"橫向（{_lat_ax}）")]:
                    _res = _mdof_results.get(_dir_key)
                    if _res is None:
                        st.warning(f"{_dir_label} 勁度為 0，跳過模態分析。")
                        continue
                    st.markdown(f"**{_dir_label} 模態**")
                    _mode_rows = [{"模態": _m.mode_number, "T (s)": round(_m.period, 4),
                                   "Γ": round(_m.gamma, 4), "M_eff (Mg)": round(_m.m_eff, 3),
                                   "M_eff 比 (%)": round(_m.m_eff_ratio, 1)} for _m in _res.modes]
                    st.dataframe(pd.DataFrame(_mode_rows), use_container_width=True, hide_index=True)

                # 主導模態 info（縱向）
                _res_L = _mdof_results.get("L")
                if _res_L:
                    _dom = max(_res_L.modes, key=lambda m: m.m_eff_ratio)
                    st.info(
                        f"**縱向主導模態**：模態 {_dom.mode_number}　｜　"
                        f"T_L = **{_dom.period:.4f} s**　｜　"
                        f"有效質量比 = **{_dom.m_eff_ratio:.1f}%**"
                    )
                _res_T = _mdof_results.get("T")
                if _res_T:
                    _dom_t = max(_res_T.modes, key=lambda m: m.m_eff_ratio)
                    st.info(
                        f"**橫向主導模態**：模態 {_dom_t.mode_number}　｜　"
                        f"T_T = **{_dom_t.period:.4f} s**　｜　"
                        f"有效質量比 = **{_dom_t.m_eff_ratio:.1f}%**"
                    )

    # ── 子 Tab C：地震力與載重組合 ────────────────────────────────────────
    with _seismic_tabs[2]:
        st.subheader("設計地震力")
        st.info("ℹ️ 目前地震力分析**僅適用於正交橋**（支承線垂直橋軸）。斜交橋的縱橫向耦合效應尚未處理，請自行確認適用性。")

        _mdof_results = st.session_state.get("seismic_result")
        _params       = st.session_state.get("seismic_params", {})
        _df_tbl       = st.session_state.get("seismic_user_table")

        if not _mdof_results or _df_tbl is None:
            st.info("請先完成「輸入設定」與「模態分析」步驟。")
        else:
            try:
                from core import seismic_logic as _sl3
                from core.seismic_logic import Mode as _Mode

                # 載入工址反應譜
                _df_params, _df_spec = _load_seismic_spectrum(_SEISMIC_CSV)

                _mode_label = _params.get("mode_label", "475年")
                _i_coeff    = _params.get("i_coeff", 1.2)
                _alpha_y    = _params.get("alpha_y", 1.0)
                _r_val      = _params.get("r_val", 2.0)
                _dir        = _params.get("analysis_dir", "縱向與橫向均計算")

                _mode_enum = {"L1": _Mode.L1, "475年": _Mode.LEVEL_475, "2500年": _Mode.LEVEL_2500}[_mode_label]
                _site_p = _sl3.get_site_params(_mode_enum, _df_params)
                _ss_val = _sl3.get_ss_for_mode(_mode_enum, _df_params)

                # 縱向/橫向主導模態週期（各自取有效質量比最高者）
                _res_L = _mdof_results.get("L")
                _res_T = _mdof_results.get("T")
                _T_L = max(_res_L.modes, key=lambda m: m.m_eff_ratio).period if _res_L else 0.0
                _T_T = max(_res_T.modes, key=lambda m: m.m_eff_ratio).period if _res_T else _T_L

                # 總重量（kN）
                _W_total_kN = float(_df_tbl["質量 (Mg)"].sum()) * 9.81

                # 上部結構重（橋面 tributary）vs 下部結構重（墩柱自重）
                _extracted  = st.session_state.get("seismic_extracted", {})
                _masses_raw = _extracted.get("masses", {})
                _sdl_r      = float(st.session_state.get("seismic_sdl_ratio", 0.0))
                _W_sup_kN = sum(v.get("deck_mass_Mg", 0) for v in _masses_raw.values()) * 9.81 * (1 + _sdl_r)
                _W_sub_kN = sum(v.get("pier_mass_Mg", 0) for v in _masses_raw.values()) * 9.81 * (1 + _sdl_r)

                # 縱橫週期耦合警告（兩個方向都有結果才比較）
                if _res_L and _res_T and _T_T > 0 and 0.8 <= _T_L / _T_T <= 1.2:
                    st.warning(
                        f"⚠️ 縱向週期 T_L={_T_L:.3f}s 與橫向週期 T_T={_T_T:.3f}s 相近"
                        f"（比值 {_T_L/_T_T:.2f}），兩模態方向辨識可能不可靠，"
                        "建議確認模態振型或調整結構設計。"
                    )

                # 計算水平地震力
                _sa_col = "Sa_2500" if _mode_enum == _Mode.LEVEL_2500 else "Sa_475"
                _spec_data = _df_spec[["T", _sa_col]].rename(columns={_sa_col: "Sa"}) if _mode_enum != _Mode.L1 else None

                _v_l, _v_t = 0.0, 0.0
                _sa_l = _sa_t = _fu_l = _fu_t = _cd_l = _cd_t = 0.0
                _fu_text_l = _fu_text_t = ""

                if _dir in ("縱向與橫向均計算", "僅縱向"):
                    _sa_l, _fu_l, _cd_l, _v_l, _fu_text_l = _sb.compute_horizontal_force(
                        _T_L, _mode_enum, _W_total_kN, _i_coeff, _alpha_y, _r_val, _site_p, _spec_data
                    )
                if _dir in ("縱向與橫向均計算", "僅橫向"):
                    _sa_t, _fu_t, _cd_t, _v_t, _fu_text_t = _sb.compute_horizontal_force(
                        _T_T, _mode_enum, _W_total_kN, _i_coeff, _alpha_y, _r_val, _site_p, _spec_data
                    )

                # 垂直地震力
                _vv_res = _sl3.calculate_vertical_seismic_force(
                    _W_sup_kN, _W_sub_kN, _i_coeff, _alpha_y, _ss_val
                )

                # 顯示主要結果
                st.markdown("#### 水平設計地震力")
                _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                _mc1.metric("縱向 T_L (s)", f"{_T_L:.4f}")
                _mc2.metric("縱向 Sa", f"{_sa_l:.4f} g" if _v_l else "—")
                _mc3.metric("縱向 Cd", f"{_cd_l:.4f}" if _v_l else "—")
                _mc4.metric("縱向 V_L (kN)", f"{_v_l:.1f}" if _v_l else "—")

                _mc5, _mc6, _mc7, _mc8 = st.columns(4)
                _mc5.metric("橫向 T_T (s)", f"{_T_T:.4f}")
                _mc6.metric("橫向 Sa", f"{_sa_t:.4f} g" if _v_t else "—")
                _mc7.metric("橫向 Cd", f"{_cd_t:.4f}" if _v_t else "—")
                _mc8.metric("橫向 V_T (kN)", f"{_v_t:.1f}" if _v_t else "—")

                st.markdown("#### 垂直設計地震力")
                _vc1, _vc2, _vc3 = st.columns(3)
                _vc1.metric("上部結構 Vv_sup (kN)", f"{_vv_res.Vv_sup:.1f}")
                _vc2.metric("下部結構 Vv_sub (kN)", f"{_vv_res.Vv_sub:.1f}")
                _vc3.metric("垂直合計 Vv (kN)",     f"{_vv_res.Vv_total:.1f}")

                # 載重組合
                st.markdown("#### 載重組合（規範 2.7 節）")
                st.warning(
                    "⚠️ 目前載重組合為**總地震力直接疊加**（V_L、V_T、V_v 純量組合），"
                    "尚未依規範 2.7 節將地震力施加回結構並提取**構材內力**後再組合。"
                    "下表數值僅供確認地震力量值參考，不可直接用於構材設計。"
                )
                _combo_res = _sl3.calculate_load_combinations(_v_l, _v_t, _vv_res.Vv_total)
                import pandas as pd
                _combo_rows = []
                for _cb in _combo_res.combinations:
                    _combo_rows.append({
                        "組合":     _cb.name,
                        "公式":     _cb.formula,
                        "合計 (kN)": round(_cb.total, 1),
                    })
                _df_combo = pd.DataFrame(_combo_rows)

                # 控制組合加粗（用 st.dataframe + column_config 無法直接加粗，改用 markdown 表格）
                _ctrl_name = _combo_res.max_combo.name
                _md_rows = ["| 組合 | 公式 | 合計 (kN) |", "|---|---|---|"]
                for _cb in _combo_res.combinations:
                    _bold = "**" if _cb.name == _ctrl_name else ""
                    _md_rows.append(
                        f"| {_bold}{_cb.name}{_bold} | {_bold}{_cb.formula}{_bold} | "
                        f"{_bold}{_cb.total:.1f}{_bold} |"
                    )
                st.markdown("\n".join(_md_rows))
                st.success(f"✅ 控制組合：**{_ctrl_name}**，合計 {_combo_res.max_combo.total:.1f} kN")

                # TODO: 注入節點載重 — 待確認規範施力點位與方向後實作

            except Exception as _e:
                import traceback
                st.error(f"地震力計算失敗：{_e}")
                st.code(traceback.format_exc())