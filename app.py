import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from core.parametric_evaluator import (
    build_geometry_fingerprint,
    evaluate_real_results,
    export_cache_to_txt,
    import_cache_from_txt,
)

st.set_page_config(page_title="Structural Analysis", layout="wide")

if "sym_cache" not in st.session_state:
    st.session_state["sym_cache"] = {}

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
    st.subheader("節點 (Nodes)")
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
            {"id": 1, "i": 1, "j": 2, "E": 200e9, "G": 77e9, "A": 0.01, "I33": 1e-4, "I22": 1e-5, "J": 1e-5, "beta": 0.0, "dL": 0.0, "pin_i": False, "pin_j": False},
            {"id": 2, "i": 2, "j": 3, "E": 200e9, "G": 77e9, "A": 0.01, "I33": 1e-4, "I22": 1e-5, "J": 1e-5, "beta": 0.0, "dL": 0.0, "pin_i": False, "pin_j": False},
            {"id": 3, "i": 4, "j": 1, "E": 200e9, "G": 77e9, "A": 0.01, "I33": 1e-4, "I22": 1e-5, "J": 1e-5, "beta": 0.0, "dL": 0.0, "pin_i": False, "pin_j": False}
        ]), num_rows="dynamic", key="elements"
    )

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
    loads_df = st.data_editor(
        pd.DataFrame([
            {"node_id": 2, "fx": 0.0, "fy": 0.0, "fz": -10.0} # 集中載重在 Z 向
        ]), num_rows="dynamic", key="loads"
    )

    st.subheader("桿件載重 (Element Loads)")
    e_loads_df = st.data_editor(
        pd.DataFrame([
            {"element_id": 1, "w": -5.0} # 均佈載重向下 (Z向)
        ]), num_rows="dynamic", key="element_loads"
    )

    st.subheader("桿件集中載重 (Element Point Loads)")
    e_pt_loads_df = st.data_editor(
        pd.DataFrame([
            {"element_id": 1, "p": 0.0, "a": 0.0}
        ]), num_rows="dynamic", key="element_point_loads"
    )

    run_btn = st.button("執行分析（符號解）", type="primary", use_container_width=True)

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
        st.caption("假設所有桿件使用相同截面參數。若各桿件截面不同，請執行完整分析。")
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

    if fast_btn:
        real_params = {"E": pe_E, "A": pe_A, "I": pe_I, "G": pe_G, "P": pe_P, "w": pe_w}
        try:
            res_eval = evaluate_real_results(truss_data, real_params,
                                             symbolic_cache=st.session_state["sym_cache"])
            output_area.json(res_eval)
            st.info(f"快速代入完成，耗時 {res_eval['eval_time_ms']} ms（使用快取符號解）。")
        except Exception as e:
            st.error(f"代入失敗：{e}")

    if run_btn:
        try:
            # 清除舊快取，強制重新求解
            st.session_state["sym_cache"] = {}
            real_params = {"E": pe_E, "A": pe_A, "I": pe_I, "G": pe_G, "P": pe_P, "w": pe_w}
            res_eval = evaluate_real_results(truss_data, real_params,
                                             symbolic_cache=st.session_state["sym_cache"])
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
                actual_L = np.sqrt((node_j['x']-node_i['x'])**2 + (node_j['y']-node_i['y'])**2)

                x_vals = np.linspace(0, actual_L, 100)
                fig = make_subplots(rows=1, cols=3, subplot_titles=("軸力圖 (ND)", "剪力圖 (V2)", "彎矩圖 (M3)"))

                for idx, (key, title) in enumerate([("N", "Axial"), ("V2", "Shear"), ("M3", "Moment")]):
                    expr_str = force[key]['formula']
                    try:
                        expr = sp.parse_expr(expr_str.replace('^', '**'))
                        expr_num = expr.subs({x_sym: x_sym})  # formula is already fully substituted
                        f_np = sp.lambdify(x_sym, expr_num, modules=['numpy', {'Heaviside': lambda x: np.where(x >= 0, 1, 0)}])
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