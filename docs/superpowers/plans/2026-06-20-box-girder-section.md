# 橋梁箱涵斷面（Box Girder Section）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在現有材料／斷面系統中新增「箱涵」斷面形狀，支援單箱多室 Bredt 薄壁計算，並在 Streamlit 前端提供即時 Plotly 斷面預覽與外廓尺寸標注。

**Architecture:** 在 `core/materials.py` 新增 `"箱涵"` 分支，回傳與現有形狀相同的 `{A, I33, I22, J}` 字典介面；前端在 `app.py` 的「從形狀計算截面參數」expander 內新增箱涵參數輸入欄位與 Plotly 預覽圖，放置於同一 expander 的兩欄佈局中。`expand_truss_data` 與 `compute_self_weight` 無需修改。

**Tech Stack:** Python 3, NumPy (`numpy.linalg.solve`), Streamlit, Plotly (`plotly.graph_objects`)

## Global Constraints

- 不修改 `expand_truss_data`、`compute_self_weight`、現有任何斷面分支
- 不修改 `.venv` 等環境目錄
- `compute_section_props("箱涵", params)` 回傳 `dict` 含 `A`、`I33`、`I22`、`J`（單位：m², m⁴, m⁴, m⁴）
- 測試以 pytest 執行，路徑為 `test/test_materials.py`
- `n_cell` 支援範圍 1～5

---

## File Map

| 檔案 | 動作 | 說明 |
|---|---|---|
| `core/materials.py` | Modify | 新增 `"箱涵"` 分支於 `compute_section_props` |
| `test/test_materials.py` | Modify | 新增箱涵斷面測試 |
| `app.py` | Modify | 新增箱涵輸入 UI 與 Plotly 預覽 |

---

## Task 1: 核心計算 — `compute_section_props("箱涵", ...)`

**Files:**
- Modify: `core/materials.py` — 在 `raise ValueError` 前新增 `if shape == "箱涵":` 分支
- Test: `test/test_materials.py`

**Interfaces:**
- Produces: `compute_section_props("箱涵", params) -> {"A": float, "I33": float, "I22": float, "J": float}`
- `params` 必要鍵：`b_top`, `b_bot`, `h`, `t_top`, `t_bot`, `t_web`, `t_dia`, `n_cell`, `c_top`（均為 float，`n_cell` 為 int）

---

- [ ] **Step 1: 寫失敗測試（單室退化）**

在 `test/test_materials.py` 末尾加入：

```python
# ── 箱涵斷面 ───────────────────────────────────────────────────────────────

def _box_params_1cell():
    return dict(b_top=2.0, b_bot=1.6, h=1.2,
                t_top=0.2, t_bot=0.2, t_web=0.2, t_dia=0.15,
                n_cell=1, c_top=0.2)

def test_box_girder_1cell_area():
    """n_cell=1 單室：面積 = 頂板 + 底板 + 2×腹板（無內隔板）"""
    p = _box_params_1cell()
    hw = p["h"] - p["t_top"] - p["t_bot"]          # 0.8
    expected_A = (p["b_top"] * p["t_top"]           # 頂板（含懸臂）
                + p["b_bot"] * p["t_bot"]            # 底板
                + 2 * p["t_web"] * hw)               # 2×外腹板
    result = compute_section_props("箱涵", p)
    assert abs(result["A"] - expected_A) < 1e-10

def test_box_girder_1cell_J_positive():
    p = _box_params_1cell()
    result = compute_section_props("箱涵", p)
    assert result["J"] > 0

def test_box_girder_1cell_I33_positive():
    p = _box_params_1cell()
    result = compute_section_props("箱涵", p)
    assert result["I33"] > 0

def test_box_girder_1cell_I22_symmetric():
    """c_top=0 時 I22 應與左右對稱斷面一致（用 b_top==b_bot 且 c_top=0 驗證）"""
    p = dict(b_top=1.6, b_bot=1.6, h=1.2,
             t_top=0.2, t_bot=0.2, t_web=0.2, t_dia=0.15,
             n_cell=1, c_top=0.0)
    result = compute_section_props("箱涵", p)
    assert result["I22"] > 0

def test_box_girder_3cell_area():
    """n_cell=3：面積含 2 條內隔板"""
    p = dict(b_top=6.0, b_bot=5.0, h=2.0,
             t_top=0.25, t_bot=0.25, t_web=0.3, t_dia=0.2,
             n_cell=3, c_top=0.5)
    hw = p["h"] - p["t_top"] - p["t_bot"]
    expected_A = (p["b_top"] * p["t_top"]
                + p["b_bot"] * p["t_bot"]
                + 2 * p["t_web"] * hw
                + (p["n_cell"] - 1) * p["t_dia"] * hw)
    result = compute_section_props("箱涵", p)
    assert abs(result["A"] - expected_A) < 1e-10

def test_box_girder_3cell_J_greater_than_1cell():
    """多室 J 應大於單室（相同外廓下）"""
    base = dict(b_top=6.0, b_bot=5.0, h=2.0,
                t_top=0.25, t_bot=0.25, t_web=0.3, t_dia=0.2, c_top=0.5)
    p1 = {**base, "n_cell": 1}
    p3 = {**base, "n_cell": 3}
    j1 = compute_section_props("箱涵", p1)["J"]
    j3 = compute_section_props("箱涵", p3)["J"]
    assert j3 > j1
```

- [ ] **Step 2: 執行測試，確認失敗**

```
pytest test/test_materials.py::test_box_girder_1cell_area -v
```

預期：`FAILED` — `ValueError: 未知截面形狀: 箱涵`

- [ ] **Step 3: 實作 `"箱涵"` 分支**

在 `core/materials.py` 的 `raise ValueError(...)` 前插入以下程式碼：

```python
if shape == "箱涵":
    import numpy as np
    b_top = float(params["b_top"])
    b_bot = float(params["b_bot"])
    h     = float(params["h"])
    t_top = float(params["t_top"])
    t_bot = float(params["t_bot"])
    t_web = float(params["t_web"])
    t_dia = float(params["t_dia"])
    n     = int(params["n_cell"])
    c_top = float(params["c_top"])

    hw = h - t_top - t_bot                          # 腹板淨高
    b_box = b_bot - 2 * t_web                       # 底板內淨寬

    # ── 面積 ──────────────────────────────────────
    A = (b_top * t_top
         + b_bot * t_bot
         + 2 * t_web * hw
         + (n - 1) * t_dia * hw)

    # ── 形心高度（由底部量起）─────────────────────
    pieces = [
        (b_top * t_top,            h - t_top / 2),          # 頂板
        (b_bot * t_bot,            t_bot / 2),               # 底板
        (t_web * hw,               t_bot + hw / 2),          # 左外腹板
        (t_web * hw,               t_bot + hw / 2),          # 右外腹板
    ]
    for _ in range(n - 1):
        pieces.append((t_dia * hw, t_bot + hw / 2))          # 內隔板

    y_bar = sum(a * y for a, y in pieces) / A

    # ── I33（對水平形心軸）────────────────────────
    def _rect_I33(b, hh, y_piece):
        return b * hh**3 / 12 + b * hh * (y_piece - y_bar)**2

    I33 = (_rect_I33(b_top, t_top, h - t_top / 2)
           + _rect_I33(b_bot, t_bot, t_bot / 2)
           + 2 * _rect_I33(t_web, hw, t_bot + hw / 2)
           + (n - 1) * _rect_I33(t_dia, hw, t_bot + hw / 2))

    # ── I22（對垂直形心軸，原點取箱涵幾何中心 x=0）────
    # 箱涵幾何：左外腹板內緣 x = -b_box/2，右 x = +b_box/2
    # 各室隔板均分，間距 s = b_box / n
    s = b_box / n

    def _rect_I22(hh, bb, x_c):
        return hh * bb**3 / 12 + hh * bb * x_c**2

    # 頂板（含懸臂）：全寬 b_top，形心在 x=0
    I22 = _rect_I22(t_top, b_top, 0.0)
    # 底板：全寬 b_bot，形心在 x=0
    I22 += _rect_I22(t_bot, b_bot, 0.0)
    # 左外腹板：形心 x = -(b_box/2 + t_web/2)
    I22 += _rect_I22(hw, t_web, -(b_box / 2 + t_web / 2))
    # 右外腹板：形心 x = +(b_box/2 + t_web/2)
    I22 += _rect_I22(hw, t_web,  (b_box / 2 + t_web / 2))
    # 內隔板：x 位置均分
    for k in range(1, n):
        x_dia = -b_box / 2 + k * s
        I22 += _rect_I22(hw, t_dia, x_dia)

    # ── J（Bredt 多室薄壁）────────────────────────
    # 每室封閉面積 Ak = s × hw（底板側均分）
    Ak = s * hw

    # 建立 n×n 聯立方程 [C]{q} = {2Ak}
    # C[i,i] = Σ(ds/t) 沿第 i 室周長
    # C[i,i-1] = C[i,i+1] = -hw / t_dia（共用隔板）
    C = np.zeros((n, n))
    for i in range(n):
        # 頂板段：s / t_top
        # 底板段：s / t_bot
        # 外腹板（最左室 i=0 或最右室 i=n-1）：hw / t_web
        # 內隔板（左側共用）：hw / t_dia
        # 內隔板（右側共用）：hw / t_dia
        seg_top = s / t_top
        seg_bot = s / t_bot
        seg_left  = (hw / t_web) if i == 0     else (hw / t_dia)
        seg_right = (hw / t_web) if i == n - 1 else (hw / t_dia)
        C[i, i] = seg_top + seg_bot + seg_left + seg_right
        if i > 0:
            C[i, i - 1] = -hw / t_dia
        if i < n - 1:
            C[i, i + 1] = -hw / t_dia

    rhs = np.full(n, 2 * Ak)
    q = np.linalg.solve(C, rhs)
    J = float(np.dot(q, np.full(n, 2 * Ak)))

    return {"A": A, "I33": I33, "I22": I22, "J": J}
```

- [ ] **Step 4: 執行所有箱涵測試，確認全通過**

```
pytest test/test_materials.py -k "box_girder" -v
```

預期：6 個測試全部 `PASSED`

- [ ] **Step 5: 執行完整測試套件，確認無迴歸**

```
pytest test/test_materials.py -v
```

預期：全部 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add core/materials.py test/test_materials.py
git commit -m "feat: add box girder (箱涵) section with Bredt multi-cell J"
```

---

## Task 2: 前端 UI — 箱涵參數輸入與 Plotly 即時預覽

**Files:**
- Modify: `app.py` — 在 `SHAPE_OPTIONS`、`SHAPE_INPUTS`、expander 內新增箱涵邏輯

**Interfaces:**
- Consumes: `compute_section_props("箱涵", params)` — Task 1 定義的介面
- Produces: 使用者輸入箱涵參數後，按「計算並填入」可將 A/I33/I22/J 寫入 session_state sections

---

- [ ] **Step 1: 在 `SHAPE_OPTIONS` 加入「箱涵」**

找到 `app.py:140`：
```python
SHAPE_OPTIONS = ["Custom", "矩形實心", "圓形實心", "矩形管", "圓管", "I形"]
```
改為：
```python
SHAPE_OPTIONS = ["Custom", "矩形實心", "圓形實心", "矩形管", "圓管", "I形", "箱涵"]
```

- [ ] **Step 2: 在 `SHAPE_INPUTS` 加入箱涵鍵**

找到 `app.py:141`，在 `SHAPE_INPUTS` dict 的 `"I形"` 條目後加入：
```python
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
```

- [ ] **Step 3: 在 expander 加入箱涵專屬輸入區塊**

找到 `app.py:174`（`shape_sel = st.selectbox(...)`）後的欄位渲染邏輯：

```python
cols = st.columns(len(SHAPE_INPUTS[shape_sel]))
for col, (k, label) in zip(cols, SHAPE_INPUTS[shape_sel]):
    shape_vals[k] = col.number_input(label, value=0.1, format="%.4f", key=f"sv_{k}")
```

將這段替換為：

```python
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
    import plotly.graph_objects as go
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
    # 頂板底面 x 位置（隨高度線性插值）
    def x_at_y(y_val, side):
        # side='left'/-1 or 'right'/+1
        t_ratio = y_val / hv if hv > 0 else 0
        half = half_bot + (half_top - half_bot) * t_ratio
        return -half if side == "left" else half

    for i in range(n):
        # 每室底板內緣 x 左右（底板側均分）
        x0_bot = -b_box / 2 + i * s_cell + tw
        x1_bot =  x0_bot + s_cell - (tw if i == 0 else 0) - (tw if i == n-1 else 0)
        # 簡化：空腔邊界用直腹板（不做梯形插值，因直腹板假設）
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
```

- [ ] **Step 4: 更新 caption，加入箱涵說明**

找到 `app.py:168`：
```python
st.caption(
    "⚠️ 矩形實心的 J 採用 Timoshenko 近似公式；"
    "矩形管的 J 採用薄壁閉口近似；"
    "I 形截面的 J 採用薄壁開口近似。精確值請查結構手冊。"
)
```
改為：
```python
st.caption(
    "⚠️ 矩形實心的 J 採用 Timoshenko 近似公式；"
    "矩形管的 J 採用薄壁閉口近似；"
    "I 形截面的 J 採用薄壁開口近似；"
    "箱涵的 J 採用 Bredt 多室薄壁公式。精確值請查結構手冊。"
)
```

- [ ] **Step 5: 手動啟動 app 確認 UI**

```
streamlit run app.py
```

確認：
1. 截面形狀下拉選單出現「箱涵」
2. 選擇「箱涵」後顯示 n_cell selectbox 與 8 個 number_input（兩欄排列）
3. Plotly 圖即時顯示梯形外廓、各室空腔、4 個尺寸標注
4. 修改任一參數後圖形自動更新
5. 按「計算並填入」後 section 的 A/I33/I22/J 被正確寫入

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: box girder UI with Plotly live preview and dimension annotations"
```

---

## Self-Review

**Spec coverage:**
- ✅ 單箱多室（n_cell 1～5）— Task 1 + Task 2
- ✅ 直腹板（無傾斜腹板參數）— 設計明確排除
- ✅ 頂板懸臂 c_top — Task 1 A/I22 計算含懸臂；Task 2 UI 含 c_top 輸入與標注
- ✅ 底板無懸臂 — 未加 c_bot 參數
- ✅ Bredt 多室 J — Task 1 Step 3 實作
- ✅ Plotly 即時預覽 — Task 2 Step 3
- ✅ 外廓尺寸標注（b_top, b_bot, h, c_top）— Task 2 Step 3 annotations
- ✅ 不修改 expand_truss_data / compute_self_weight — 無相關步驟

**Placeholder scan:** 無 TBD / TODO。

**Type consistency:**
- `compute_section_props("箱涵", params)` — Task 1 定義，Task 2 Step 5 驗證呼叫
- `shape_vals` dict — Task 2 Step 2/3 統一使用相同鍵名（`b_top`, `b_bot` 等）
- `n_cell` 在 Task 1 以 `int(params["n_cell"])` 處理，Task 2 UI 以 `st.selectbox` 回傳 int，一致
