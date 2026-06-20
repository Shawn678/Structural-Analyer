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
    SHAPE_OPTIONS = ["Custom", "矩形實心", "圓形實心", "矩形管", "圓管", "I形"]
    SHAPE_INPUTS  = {
        "矩形實心": [("b","寬 b (m)"),("h","高 h (m)")],
        "圓形實心": [("d","直徑 d (m)")],
        "矩形管":   [("b","寬 b (m)"),("h","高 h (m)"),("t","壁厚 t (m)")],
        "圓管":     [("d","外徑 d (m)"),("t","壁厚 t (m)")],
        "I形":      [("H","全高 H (m)"),("bf","翼板寬 bf (m)"),
                     ("tf","翼板厚 tf (m)"),("tw","腹板厚 tw (m)")],
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
            "I 形截面的 J 採用薄壁開口近似。精確值請查結構手冊。"
        )
        target_sec = st.selectbox("填入截面", options=sec_names, key="shape_target_sec")
        shape_sel  = st.selectbox("截面形狀", options=list(SHAPE_INPUTS.keys()), key="shape_sel")
        shape_vals = {}
        cols = st.columns(len(SHAPE_INPUTS[shape_sel]))
        for col, (k, label) in zip(cols, SHAPE_INPUTS[shape_sel]):
            shape_vals[k] = col.number_input(label, value=0.1, format="%.4f", key=f"sv_{k}")
        if st.button("計算並填入", key="calc_shape"):
            try:
                props = compute_section_props(shape_sel, shape_vals)
                new_secs = []
                for s in st.session_state["sections"]:
                    if s["name"] == target_sec:
                        s = {**s, "shape": shape_sel, **props}
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
    elements_df = st.data_editor(
        pd.DataFrame([
            {"id":1,"i":1,"j":2,"section":"","E":200e9,"G":77e9,"A":0.01,
             "I33":1e-4,"I22":1e-5,"J":1e-5,"beta":0.0,"dL":0.0,
             "pin_i":False,"pin_j":False,"status":""},
            {"id":2,"i":2,"j":3,"section":"","E":200e9,"G":77e9,"A":0.01,
             "I33":1e-4,"I22":1e-5,"J":1e-5,"beta":0.0,"dL":0.0,
             "pin_i":False,"pin_j":False,"status":""},
            {"id":3,"i":4,"j":1,"section":"","E":200e9,"G":77e9,"A":0.01,
             "I33":1e-4,"I22":1e-5,"J":1e-5,"beta":0.0,"dL":0.0,
             "pin_i":False,"pin_j":False,"status":""},
        ]),
        column_config={
            "section": st.column_config.SelectboxColumn(
                "截面", options=[""] + sec_names, width="small"
            ),
            "status": st.column_config.TextColumn("狀態", disabled=True, width="small"),
        },
        num_rows="dynamic", key="elements",
    )

    # section 帶入 + override 偵測
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

    def _is_override(row):
        sn = row.get("section", "")
        if not sn or sn not in sec_map_ui:
            return False
        for field in ("E", "G", "A", "I33", "I22", "J"):
            ref = _sec_val(sn, field)
            if ref is not None and abs(float(row.get(field, ref)) - ref) > ref * 1e-9:
                return True
        return False

    # Initialize tracking dict if needed
    if "elem_prev_section" not in st.session_state:
        st.session_state["elem_prev_section"] = {}

    updated_rows = []
    for _, row in elements_df.iterrows():
        r = row.to_dict()
        elem_id = r.get("id", "")
        sn = r.get("section", "")
        prev_sn = st.session_state["elem_prev_section"].get(str(elem_id), None)

        # Auto-fill when section changes (including first selection)
        if sn and sn in sec_map_ui and sn != prev_sn:
            for field in ("E", "G", "A", "I33", "I22", "J"):
                ref = _sec_val(sn, field)
                if ref is not None:
                    r[field] = ref

        # Update tracking
        st.session_state["elem_prev_section"][str(elem_id)] = sn

        r["status"] = "【修改】" if _is_override(r) else ""
        updated_rows.append(r)

    elements_df_styled = pd.DataFrame(updated_rows)

    def _highlight_override(row):
        return ["background-color: #FFE0B2" if row.get("status") == "【修改】" else "" for _ in row]

    st.dataframe(
        elements_df_styled.style.apply(_highlight_override, axis=1),
        use_container_width=True,
    )
    elements_df = elements_df_styled  # 後續分析使用帶入後的版本

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
        current_fp = build_geometry_fingerprint({
            "nodes": nodes_df.dropna(subset=['id']).fillna(0).to_dict('records'),
            "elements": elements_df.dropna(subset=['id','i','j']).fillna(0).to_dict('records'),
            "supports": supports_df.dropna(subset=['node_id']).fillna(False).to_dict('records'),
            "loads": [], "element_loads": [], "element_point_loads": [],
        })
        cache_valid = (current_fp == st.session_state["sym_cache"].get("fingerprint"))

    st.caption("快速代入：幾何與支承不變時，直接代入新材料/載重參數，無需重新求符號解。")
    with st.expander("⚡ 快速代入參數", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            pe_E = st.number_input("E (Pa)", value=200e9, format="%.3e", key="pe_E")
            pe_I = st.number_input("I (m⁴)", value=1e-4,  format="%.3e", key="pe_I")
            pe_P = st.number_input("P 倍率", value=1.0,   key="pe_P")
        with col_b:
            pe_A = st.number_input("A (m²)", value=0.01,  format="%.4f", key="pe_A")
            pe_G = st.number_input("G (Pa)", value=77e9,  format="%.3e", key="pe_G")
            pe_w = st.number_input("w 倍率", value=0.0,   key="pe_w")
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
                st.success("指紋一致，快取已載入，可使用快速代入。")

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
        real_params = {"E": pe_E, "A": pe_A, "I": pe_I, "G": pe_G, "P": pe_P, "w": pe_w}
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
            real_params = {"E": pe_E, "A": pe_A, "I": pe_I, "G": pe_G, "P": pe_P, "w": pe_w}
            res_eval = evaluate_real_results(
                truss_data, real_params,
                symbolic_cache=st.session_state["sym_cache"],
                materials=st.session_state["materials"],
                sections=st.session_state["sections"],
                include_self_weight=include_sw,
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