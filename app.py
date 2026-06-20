import streamlit as st
import pandas as pd
import numpy as np
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

@st.cache_data(show_spinner=False)
def _compute_fingerprint(nodes_json: str, elements_json: str, supports_json: str):
    import json
    return build_geometry_fingerprint({
        "nodes": json.loads(nodes_json),
        "elements": json.loads(elements_json),
        "supports": json.loads(supports_json),
        "loads": [], "element_loads": [], "element_point_loads": [],
    })


st.set_page_config(page_title="Structural Analysis", layout="wide")

if "sym_cache" not in st.session_state:
    st.session_state["sym_cache"] = {}
if "materials" not in st.session_state:
    st.session_state["materials"] = [
        {"name": "鋼材",   "E": 200e9, "G": 77e9, "density": 7850.0},
        {"name": "混凝土", "E":  30e9, "G": 12.5e9, "density": 2400.0},
    ]
if "sections" not in st.session_state:
    st.session_state["sections"] = []

# 注入現有的 CSS 樣式

def create_structure_plot(nodes_df, elements_df, supports_df=None, reactions=None):
    fig = go.Figure()
    
    # 確保資料不包含空 ID (處理動態編輯產生的空行)
    nodes_df = nodes_df.dropna(subset=['id'])
    elements_df = elements_df.dropna(subset=['id', 'i', 'j'])
    if 'z' not in nodes_df.columns: nodes_df['z'] = 0.0

    # 繪製桿件
    for _, elem in elements_df.iterrows():
        matched_i = nodes_df[nodes_df['id'] == elem['i']]
        matched_j = nodes_df[nodes_df['id'] == elem['j']]
        if matched_i.empty or matched_j.empty:
            continue
        n1, n2 = matched_i.iloc[0], matched_j.iloc[0]
        fig.add_trace(go.Scatter3d(
            # 繪圖座標保持與數據一致，Plotly 預設 Z 向上
            x=[n1['x'], n2['x']], y=[n1['y'], n2['y']], z=[n1['z'], n2['z']], 
            mode='lines+markers', line=dict(color='RoyalBlue', width=6),
            marker=dict(size=8, color='black'), hoverinfo='text',
            text=f"Element {elem['id']}", name=f"Elem {elem['id']}"
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
    # 同步回 session_state（去掉唯讀欄）
    st.session_state["materials"] = mat_df.drop(columns=["ν (唯讀)"], errors="ignore").dropna(subset=["name"]).to_dict("records")
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
    st.session_state["sections"] = sec_df.dropna(subset=["name"]).to_dict("records")
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
            BOX_DEFAULTS = {
                "b_top": 6.0, "b_bot": 5.0, "h": 2.0,
                "t_top": 0.25, "t_bot": 0.25, "t_web": 0.3,
                "t_dia": 0.2, "c_top": 0.5,
            }
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

            # 外廓多邊形（梯形，頂寬 b_top，底寬 b_bot）
            half_top = bt / 2
            half_bot = bb_w / 2
            outer_x = [-half_top, half_top, half_bot, -half_bot, -half_top]
            outer_y = [hv,        hv,       0,         0,          hv]
            fig.add_trace(go.Scatter(
                x=outer_x, y=outer_y, fill="toself",
                fillcolor="rgba(150,150,150,0.4)", line=dict(color="black", width=1.5),
                showlegend=False, name="外廓",
            ))

            # 各室空腔（白色覆蓋）
            b_box = bb_w - 2 * tw
            s_cell = b_box / n
            for i in range(n):
                inner_x0 = -b_box / 2 + i * s_cell
                inner_x1 = inner_x0 + s_cell
                void_x = [inner_x0, inner_x1, inner_x1, inner_x0, inner_x0]
                void_y = [tb, tb, hv - tt, hv - tt, tb]
                fig.add_trace(go.Scatter(
                    x=void_x, y=void_y, fill="toself",
                    fillcolor="white", line=dict(color="rgba(100,100,100,0.5)", width=0.8),
                    showlegend=False,
                ))

            # 尺寸標注
            annotations = [
                dict(x=0, y=hv + 0.05 * hv, ax=bt / 2, ay=hv + 0.05 * hv,
                     xref="x", yref="y", axref="x", ayref="y",
                     text=f"b_top={bt:.3f}m", showarrow=True, arrowhead=2,
                     font=dict(size=11), arrowcolor="royalblue", arrowwidth=1.5),
                dict(x=0, y=-0.08 * hv, ax=bb_w / 2, ay=-0.08 * hv,
                     xref="x", yref="y", axref="x", ayref="y",
                     text=f"b_bot={bb_w:.3f}m", showarrow=True, arrowhead=2,
                     font=dict(size=11), arrowcolor="royalblue", arrowwidth=1.5),
                dict(x=half_top + 0.05 * bt, y=hv / 2, ax=half_top + 0.05 * bt, ay=0,
                     xref="x", yref="y", axref="x", ayref="y",
                     text=f"h={hv:.3f}m", showarrow=True, arrowhead=2,
                     font=dict(size=11), arrowcolor="seagreen", arrowwidth=1.5),
                dict(x=half_top - ct / 2, y=hv + 0.12 * hv, ax=0, ay=0,
                     xref="x", yref="y", axref="x", ayref="y",
                     text=f"c_top={ct:.3f}m", showarrow=True, arrowhead=2,
                     font=dict(size=10), arrowcolor="darkorange", arrowwidth=1.5),
            ]

            fig.update_layout(
                annotations=annotations,
                xaxis=dict(visible=False, scaleanchor="y"),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
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

    st.subheader("節點 (Nodes)")
    st.caption("座標單位：**m（公尺）**")
    nodes_df = st.data_editor(
        pd.DataFrame([
            {"id": 1, "x": 0.0, "y": 0.0, "z": 4.0}, # 梁左端
            {"id": 2, "x": 6.0, "y": 0.0, "z": 4.0}, # 梁中點
            {"id": 3, "x": 12.0, "y": 0.0, "z": 4.0}, # 梁右端
            {"id": 4, "x": 0.0, "y": 0.0, "z": 0.0}, # 柱底
        ]), num_rows="dynamic", key="nodes"
    )

    st.subheader("桿件 (Elements)")
    st.caption(
        "**截面參數說明：** "
        "E=彈性模量(Pa)、G=剪切模量(Pa)、A=截面積(m²)、"
        "I33=強軸慣性矩(m⁴，控制平面內彎曲)、I22=弱軸慣性矩(m⁴)、"
        "J=抗扭常數(m⁴)、beta=滾轉角(deg，預設0)。\n"
        "pin_i / pin_j: 桿件端點釋放。若兩端皆為鉸接，程式自動忽略慣性矩 (視為二力構件)。"
    )
    # section 帶入 + override 偵測
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
        {"id":1,"i":1,"j":2,"section":"","E":None,"G":None,"A":None,
         "I33":None,"I22":None,"J":None,"beta":0.0,"dL":0.0,"pin_i":False,"pin_j":False,"status":""},
        {"id":2,"i":2,"j":3,"section":"","E":None,"G":None,"A":None,
         "I33":None,"I22":None,"J":None,"beta":0.0,"dL":0.0,"pin_i":False,"pin_j":False,"status":""},
        {"id":3,"i":4,"j":1,"section":"","E":None,"G":None,"A":None,
         "I33":None,"I22":None,"J":None,"beta":0.0,"dL":0.0,"pin_i":False,"pin_j":False,"status":""},
    ]
    if "elements_data" not in st.session_state:
        st.session_state["elements_data"] = _default_elem_rows
    if "elem_prev_section" not in st.session_state:
        st.session_state["elem_prev_section"] = {}

    elements_df = st.data_editor(
        pd.DataFrame(st.session_state["elements_data"]),
        column_config={
            "section": st.column_config.SelectboxColumn(
                "截面", options=[""] + sec_names, width="small"
            ),
            "status": st.column_config.TextColumn("狀態", disabled=True, width="small"),
        },
        num_rows="dynamic", key="elements",
    )

    # 斷面帶入：section 改變時自動填入對應參數
    _need_rerun = False
    updated_rows = []
    for _, row in elements_df.iterrows():
        r = row.to_dict()
        elem_id = r.get("id", "")
        sn = r.get("section") or ""
        prev_sn = st.session_state["elem_prev_section"].get(str(elem_id), None)

        if sn and sn in sec_map_ui and sn != prev_sn:
            for field in ("E", "G", "A", "I33", "I22", "J"):
                ref = _sec_val(sn, field)
                if ref is not None:
                    r[field] = ref
            _need_rerun = True

        st.session_state["elem_prev_section"][str(elem_id)] = sn
        updated_rows.append(r)

    # override 偵測：數值與斷面預設值不同時標記
    _secs_json  = _json.dumps(st.session_state["sections"],  sort_keys=True)
    _mats_json  = _json.dumps(st.session_state["materials"], sort_keys=True)
    _elems_json = _json.dumps(updated_rows, sort_keys=True)

    def _compute_overridden_ids(sections_json, materials_json, elem_records_json):
        import json
        secs = {s["name"]: s for s in json.loads(sections_json)}
        mats = {m["name"]: m for m in json.loads(materials_json)}
        def sec_val(sn, field):
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
                ref = sec_val(sn, field)
                val = r.get(field)
                if ref is not None and val is not None:
                    try:
                        if abs(float(val) - ref) > abs(ref) * 1e-9:
                            result.add(str(r.get("id", "")))
                            break
                    except (TypeError, ValueError):
                        pass
        return result

    _overridden_ids = _compute_overridden_ids(_secs_json, _mats_json, _elems_json)
    for r in updated_rows:
        r["status"] = "● 已修改" if str(r.get("id", "")) in _overridden_ids else ""

    st.session_state["elements_data"] = updated_rows
    elements_df = pd.DataFrame(updated_rows)

    if _need_rerun:
        # 清除 data_editor 的快取 key，讓下次 rerun 從 elements_data 重新載入
        st.session_state.pop("elements", None)
        st.rerun()

    st.subheader("支承 (Supports)")
    st.caption(
        "**支承說明：**\n"
        "- **ux, uy, uz**: 限制沿著 X, Y, Z 軸的位移 (Translation)。\n"
        "- **rx, ry, rz**: 限制繞著 X, Y, Z 軸的轉動 (Rotation)。\n"
        "- **彈簧剛度**: 若要使用 kx/ky/kt (扭轉剛度)，請取消勾選對應的位移/轉動約束。"
    )
    supports_df = st.data_editor(
        pd.DataFrame([
            # SAP2000 XZ 平面：彎矩釋放應在 ry。固定柱底 (node 4)
            {"node_id": 4, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True},
            {"node_id": 2, "ux": False, "uy": True, "uz": True, "rx": True, "ry": False, "rz": True},
            {"node_id": 3, "ux": False, "uy": True, "uz": True, "rx": True, "ry": False, "rz": True}
        ]), num_rows="dynamic", key="supports"
    )

    st.subheader("載重 (Loads)")
    st.caption("fx / fy / fz：**N（牛頓）**；mx / my / mz：**N·m（牛頓·公尺）**")
    loads_df = st.data_editor(
        pd.DataFrame([
            {"node_id": 2, "fx": 0.0, "fy": 0.0, "fz": -10.0}
        ]), num_rows="dynamic", key="loads"
    )

    st.subheader("桿件載重 (Element Loads)")
    st.caption("w：均佈載重，單位 **N/m**，向下為負（與 Z 軸正方向相反）")
    e_loads_df = st.data_editor(
        pd.DataFrame([
            {"element_id": 1, "w": -5.0}
        ]), num_rows="dynamic", key="element_loads"
    )

    st.subheader("桿件集中載重 (Element Point Loads)")
    st.caption("p：集中力 **N**；a：距 i 端距離 **m**")
    e_pt_loads_df = st.data_editor(
        pd.DataFrame([
            {"element_id": 1, "p": 0.0, "a": 0.0}
        ]), num_rows="dynamic", key="element_point_loads"
    )

    include_sw = st.checkbox("含自重（Self-Weight）", value=False, key="include_sw")
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
        _current_sec_groups = sorted(set(
            r.get("section", "") for r in st.session_state.get("elements_data", [])
            if r.get("section")
        ))
        _cached_sec_groups = st.session_state["sym_cache"].get("section_groups", [])
        if _current_sec_groups != _cached_sec_groups:
            cache_valid = False

    st.caption("快速代入：幾何與支承不變時，直接代入新材料/載重參數，無需重新求符號解。")
    with st.expander("⚡ 快速代入參數", expanded=False):
        # 斷面組清單（依 elements_data 中實際使用的斷面）
        _qf_sec_names = sorted(set(
            r.get("section", "") for r in st.session_state.get("elements_data", [])
            if r.get("section")
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
truss_data = {
    "nodes":               nodes_df.dropna(subset=['id']).fillna(0).to_dict('records'),
    "elements":            elements_df.dropna(subset=['id','i','j']).fillna(0).to_dict('records'),
    "supports":            supports_df.dropna(subset=['node_id']).fillna(False).to_dict('records'),
    "loads":               loads_df.dropna(subset=['node_id']).fillna(0).to_dict('records'),
    "element_loads":       e_loads_df.dropna(subset=['element_id']).fillna(0).to_dict('records'),
    "element_point_loads": e_pt_loads_df.dropna(subset=['element_id']).fillna(0).to_dict('records'),
}

with right_panel:
    st.subheader("分析結果輸出")

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
    res_eval = None

    if num_btn:
        try:
            _td_num = expand_truss_data(truss_data, st.session_state["materials"], st.session_state["sections"]) if st.session_state["sections"] else truss_data
            if include_sw and st.session_state["sections"]:
                import copy as _copy
                _td_num = _copy.deepcopy(_td_num)
                sw = compute_self_weight(_td_num, st.session_state["sections"], st.session_state["materials"])
                _existing = {el["element_id"]: el for el in _td_num.get("element_loads", [])}
                for _sw in sw:
                    _eid = _sw["element_id"]
                    if _eid in _existing:
                        _existing[_eid]["w"] = _existing[_eid].get("w", 0.0) + _sw["w"]
                    else:
                        _td_num["element_loads"].append({"element_id": _eid, "w": _sw["w"]})
            res_eval = evaluate_numerical_results(_td_num)
            output_area.json(res_eval)
            st.info(f"數值分析完成，耗時 {res_eval['eval_time_ms']} ms（直接代入實際參數）。")
        except Exception as e:
            if "singular" in str(e).lower() or "not invertible" in str(e).lower():
                st.error(f"分析失敗：結構不穩定 (奇異矩陣)。請檢查支承是否足夠。\n\n詳細錯誤: {e}")
            else:
                st.error(f"數值分析發生錯誤：{e}")
            st.stop()

        st.divider()
        st.subheader("內力分佈圖 (Internal Force Diagrams)")
        for force in res_eval['element_forces']:
            with st.expander(f"桿件 {force['element_id']} 內力圖表", expanded=True):
                Le = force['Le']
                x_vals = np.linspace(0, Le, 100)
                # N, V2 為常數；M3 由 i 端線性插值至 j 端
                N_vals  = np.full_like(x_vals, force['N'])
                V2_vals = np.full_like(x_vals, force['V2'])
                M3_vals = force['M3_i'] + (force['M3_j'] - force['M3_i']) * x_vals / Le

                fig = make_subplots(rows=1, cols=3, subplot_titles=("軸力圖 (ND)", "剪力圖 (V2)", "彎矩圖 (M3)"))
                for col_idx, (y_vals, title, color) in enumerate([
                    (N_vals,  "Axial",  "blue"),
                    (V2_vals, "Shear",  "blue"),
                    (M3_vals, "Moment", "red"),
                ], start=1):
                    fig.add_trace(
                        go.Scatter(x=x_vals, y=y_vals, name=title, fill='tozeroy',
                                   line=dict(color=color)),
                        row=1, col=col_idx
                    )
                fig.update_layout(height=400, showlegend=False, template="plotly_white")
                fig.update_xaxes(title_text="Position (x)")
                st.plotly_chart(fig, use_container_width=True)

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
            output_area.json(res_eval)
            st.info(f"快速代入完成，耗時 {res_eval['eval_time_ms']} ms（使用快取符號解）。")
        except Exception as e:
            st.error(f"代入失敗：{e}")

    if run_btn:
        try:
            # 清除舊快取，強制重新求解
            st.session_state["sym_cache"] = {}

            # 建立各斷面組的符號對應
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
            res = st.session_state["sym_cache"]["raw_result"]
            output_area.json(res_eval)
        except Exception as e:
            if "singular" in str(e).lower() or "not invertible" in str(e).lower():
                st.error(f"分析失敗：結構不穩定 (出現奇異矩陣)。請檢查是否有足夠的支承，或者是否有機構 (Mechanism) 產生！\n\n詳細錯誤: {e}")
            else:
                st.error(f"分析時發生錯誤：{e}")
            st.stop()

        # 新增：繪製內力圖 (ND, VD, MD)
        st.divider()
        st.subheader("內力分佈圖 (Internal Force Diagrams)")

        # 預先定義繪圖用的符號
        x_sym = sp.Symbol('x')

        for force in res_eval['element_forces']:
            with st.expander(f"桿件 {force['element_id']} 內力圖表", expanded=True):
                # 取得該桿件的實際長度
                elem_id = force['element_id']
                elem_raw = next(e for e in truss_data['elements'] if e['id'] == elem_id)
                node_i = next(n for n in truss_data['nodes'] if n['id'] == elem_raw['i'])
                node_j = next(n for n in truss_data['nodes'] if n['id'] == elem_raw['j'])
                actual_L = np.sqrt(
                    (node_j['x']-node_i['x'])**2 +
                    (node_j['y']-node_i['y'])**2 +
                    (float(node_j.get('z', 0)) - float(node_i.get('z', 0)))**2
                )

                x_vals = np.linspace(0, actual_L, 100)
                fig = make_subplots(rows=1, cols=3, subplot_titles=("軸力圖 (ND)", "剪力圖 (V2)", "彎矩圖 (M3)"))

                for idx, (key, title) in enumerate([("N", "Axial"), ("V2", "Shear"), ("M3", "Moment")]):
                    expr_str = force[key]['formula']
                    try:
                        expr = sp.parse_expr(expr_str.replace('^', '**'))
                        f_np = sp.lambdify(x_sym, expr, modules=['numpy', {'Heaviside': lambda x: np.where(x >= 0, 1, 0)}])
                        y_vals = f_np(x_vals)
                        if isinstance(y_vals, (int, float, np.float64)):
                            y_vals = np.full_like(x_vals, float(y_vals))
                    except Exception as e:
                        st.error(f"解析 {key} 失敗: {e}")
                        y_vals = np.zeros_like(x_vals)

                    fig.add_trace(
                        go.Scatter(x=x_vals, y=y_vals, name=title, fill='tozeroy', line=dict(color='blue' if key != 'M3' else 'red')),
                        row=1, col=idx+1
                    )

                fig.update_layout(height=400, showlegend=False, template="plotly_white")
                fig.update_xaxes(title_text="Position (x)")
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("結構預覽與分析圖形")
    if run_btn:
        # 分析後的結果圖
        fig = create_structure_plot(nodes_df, elements_df, supports_df, res_eval.get('support_reactions'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 純輸入預覽
        fig = create_structure_plot(nodes_df, elements_df, supports_df)
        st.plotly_chart(fig, use_container_width=True)