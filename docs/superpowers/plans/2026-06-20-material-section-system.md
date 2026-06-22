# Material / Section 管理系統 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立三層 Material → Section → Element 管理系統，讓使用者統一定義材料與截面後指派給桿件，符號解以 per-section 符號變數運算，支援自重自動疊加。

**Architecture:** 新增 `core/materials.py` 處理截面幾何計算與資料展開邏輯；修改 `core/symbolic.py` 支援 per-section 符號變數；修改 `core/parametric_evaluator.py` 擴充 `real_params` 格式與 TXT 快取；修改 `app.py` 新增 UI 表格與自重 checkbox。

**Tech Stack:** Python 3.12, Streamlit, SymPy, NumPy, pandas

## Global Constraints

- 不得讀取或修改 `.venv/` 目錄
- 修改程式碼只提供修改部分，不完整重寫大檔案
- 符號變數命名規則：`E_s0`, `A_s0`... (section index)；override 用 `E_e3`... (element id)
- 標色使用 `#FFE0B2`（淡橘色）
- 測試使用 pytest，檔案置於 `test/`
- Section name 為前端唯一 key；後端統一轉 index

---

## File Map

| 檔案 | 動作 | 說明 |
|------|------|------|
| `core/materials.py` | 建立 | 截面幾何計算、資料展開、自重計算 |
| `core/symbolic.py` | 修改 | `_assemble_K_np` 與符號版接受 per-section 符號 |
| `core/parametric_evaluator.py` | 修改 | `evaluate_real_results` 擴充 real_params；TXT 加 MATERIALS/SECTIONS 區塊 |
| `app.py` | 修改 | 材料表、截面表、桿件表 section 下拉、自重 checkbox、整列標色 |
| `test/test_materials.py` | 建立 | 截面幾何計算與 expand_truss_data 的單元測試 |

---

## Task 1: core/materials.py — 截面計算與資料展開

**Files:**
- Create: `core/materials.py`
- Create: `test/test_materials.py`

**Interfaces:**
- Produces:
  - `compute_section_props(shape: str, params: dict) -> dict`
    回傳 `{"A": float, "I33": float, "I22": float, "J": float}`
  - `expand_truss_data(truss_data: dict, materials: list[dict], sections: list[dict]) -> dict`
    回傳深拷貝的 truss_data，每根 element 補齊 E/G/A/I33/I22/J（尊重 local override）
  - `compute_self_weight(truss_data_expanded: dict, sections: list[dict], materials: list[dict]) -> list[dict]`
    回傳 element_loads list（格式同 truss_data["element_loads"]），只含自重疊加量

- [ ] **Step 1: 寫 test_materials.py 的截面幾何測試（先讓它們失敗）**

建立 `test/test_materials.py`：

```python
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
```

- [ ] **Step 2: 確認測試失敗**

```
pytest test/test_materials.py -v
```

預期：`ModuleNotFoundError: No module named 'core.materials'`

- [ ] **Step 3: 建立 core/materials.py**

```python
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

        # 逐欄填入，只在 elem 中尚未存在該欄位時才填（保留 override）
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
```

- [ ] **Step 4: 執行測試確認全過**

```
pytest test/test_materials.py -v
```

預期：全部 PASS

- [ ] **Step 5: Commit**

```
git add core/materials.py test/test_materials.py
git commit -m "feat: add core/materials.py with section geometry and expand_truss_data"
```

---

## Task 2: symbolic.py — per-section 符號變數支援

**Files:**
- Modify: `core/symbolic.py`（`_assemble_K_np` 函數及符號版對應函數）

**Interfaces:**
- Consumes: `truss_data` 中的 element 已由 `expand_truss_data` 補齊 E/G/A/I33/I22/J（Task 1）
- Produces:
  - `run_symbolic_analysis(truss_data, section_sym_map=None)` 新增選用參數
  - `section_sym_map: dict` 格式：`{elem_id: {"E": sp.Symbol, "A": sp.Symbol, ...}}`
    呼叫端（parametric_evaluator）傳入，symbolic.py 按此 map 決定每根桿件用哪個符號

**注意：** 目前 `run_symbolic_analysis` 在 `symbolic.py` 中的確切行號請在實作前用 `grep -n "def run_symbolic_analysis" core/symbolic.py` 確認。

- [ ] **Step 1: 確認現有符號組裝邏輯的位置**

```
grep -n "def run_symbolic_analysis\|def _assemble_K_sym\|E_sym\|A_sym\|I_sym" core/symbolic.py
```

記下行號，了解目前全域符號的定義位置。

- [ ] **Step 2: 寫測試（先讓它失敗）**

在 `test/test_materials.py` 底部新增：

```python
# ── Task 2: symbolic per-section ──────────────────────────────────────────

from core.symbolic import run_symbolic_analysis
import sympy as sp

BEAM_2SEC = {
    "nodes": [
        {"id":1,"x":0,"y":0,"z":0},
        {"id":2,"x":3,"y":0,"z":0},
        {"id":3,"x":6,"y":0,"z":0},
    ],
    "elements": [
        {"id":1,"i":1,"j":2,"E":200e9,"G":77e9,"A":0.01,"I33":1e-4,"I22":1e-5,"J":1e-5,
         "pin_i":False,"pin_j":False,"beta":0,"dL":0,"section":"主樑"},
        {"id":2,"i":2,"j":3,"E":100e9,"G":40e9,"A":0.005,"I33":5e-5,"I22":5e-6,"J":5e-6,
         "pin_i":False,"pin_j":False,"beta":0,"dL":0,"section":"斜柱"},
    ],
    "supports": [
        {"node_id":1,"ux":True,"uy":True,"uz":True,"rx":True,"ry":True,"rz":True},
        {"node_id":3,"ux":False,"uy":True,"uz":True,"rx":True,"ry":True,"rz":False},
    ],
    "loads": [{"node_id":2,"fz":-1000.0}],
    "element_loads": [],
    "element_point_loads": [],
}

E_s0, A_s0, I33_s0 = sp.symbols("E_s0 A_s0 I33_s0", positive=True)
E_s1, A_s1, I33_s1 = sp.symbols("E_s1 A_s1 I33_s1", positive=True)

SECTION_SYM_MAP = {
    1: {"E": E_s0, "G": sp.Symbol("G_s0",positive=True),
        "A": A_s0, "I33": I33_s0,
        "I22": sp.Symbol("I22_s0",positive=True), "J": sp.Symbol("J_s0",positive=True)},
    2: {"E": E_s1, "G": sp.Symbol("G_s1",positive=True),
        "A": A_s1, "I33": I33_s1,
        "I22": sp.Symbol("I22_s1",positive=True), "J": sp.Symbol("J_s1",positive=True)},
}

def test_symbolic_per_section_formula_contains_both_symbols():
    raw = run_symbolic_analysis(BEAM_2SEC, section_sym_map=SECTION_SYM_MAP)
    # 至少一個節點位移公式應包含兩個 section 的符號
    all_formulas = " ".join(
        nd.get("ux","0") + nd.get("uz","0")
        for nd in raw["node_displacements"]
    )
    assert "E_s0" in all_formulas or "A_s0" in all_formulas
    assert "E_s1" in all_formulas or "A_s1" in all_formulas

def test_symbolic_no_section_sym_map_still_works():
    # 不傳 section_sym_map 時，行為與原來相同（全域 E/A/I/G 符號）
    raw = run_symbolic_analysis(BEAM_2SEC)
    assert "node_displacements" in raw
    assert len(raw["node_displacements"]) == 3
```

- [ ] **Step 3: 執行確認測試失敗**

```
pytest test/test_materials.py::test_symbolic_per_section_formula_contains_both_symbols -v
```

預期：FAIL（`run_symbolic_analysis` 不接受 `section_sym_map` 參數）

- [ ] **Step 4: 修改 run_symbolic_analysis 接受 section_sym_map**

在 `core/symbolic.py` 找到 `run_symbolic_analysis` 的定義，修改簽名：

```python
def run_symbolic_analysis(truss_data: dict, section_sym_map: dict | None = None) -> dict:
```

在函數內找到組裝符號剛度矩陣的位置（目前全域定義 `E_sym = sp.Symbol('E', positive=True)` 等），將每根桿件取用符號的邏輯改為：

```python
# 原本：
#   elem_E_sym = E_sym  （所有桿件共用）
# 改為：
if section_sym_map and elem["id"] in section_sym_map:
    sym = section_sym_map[elem["id"]]
    elem_E_sym   = sym["E"]
    elem_A_sym   = sym["A"]
    elem_I_sym   = sym["I33"]
    elem_I22_sym = sym["I22"]
    elem_J_sym   = sym["J"]
    elem_G_sym   = sym["G"]
else:
    elem_E_sym   = E_sym   # 原全域符號
    elem_A_sym   = A_sym
    elem_I_sym   = I_sym
    elem_I22_sym = I22_sym
    elem_J_sym   = J_sym
    elem_G_sym   = G_sym
```

（確切插入位置以 Step 1 grep 結果為準）

- [ ] **Step 5: 執行測試確認通過**

```
pytest test/test_materials.py -v
```

預期：全部 PASS，包含 Task 2 的兩個新測試

- [ ] **Step 6: 執行既有測試確認無回歸**

```
pytest test/test_parametric_evaluator.py -v
```

預期：全部 PASS

- [ ] **Step 7: Commit**

```
git add core/symbolic.py test/test_materials.py
git commit -m "feat: symbolic.py supports per-section symbol map"
```

---

## Task 3: parametric_evaluator.py — 擴充 real_params 與 TXT 格式

**Files:**
- Modify: `core/parametric_evaluator.py`
- Modify: `test/test_parametric_evaluator.py`（新增測試）

**Interfaces:**
- Consumes:
  - `expand_truss_data(truss_data, materials, sections)` from `core/materials.py`
  - `compute_self_weight(td_exp, sections, materials)` from `core/materials.py`
  - `run_symbolic_analysis(truss_data, section_sym_map)` from `core/symbolic.py`
- `evaluate_real_results` 新簽名：
  ```python
  def evaluate_real_results(
      truss_data: dict,
      real_params: dict,
      symbolic_cache: dict | None = None,
      materials: list | None = None,
      sections: list | None = None,
      include_self_weight: bool = False,
  ) -> dict
  ```
- `export_cache_to_txt(cache)` → 新增 `[MATERIALS]`、`[SECTIONS]` 區塊
- `import_cache_from_txt(txt, truss_data)` → 回傳 cache 加入 `materials`、`sections` 欄位

- [ ] **Step 1: 在 test_parametric_evaluator.py 底部新增測試**

```python
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
```

- [ ] **Step 2: 確認測試失敗**

```
pytest test/test_parametric_evaluator.py::test_evaluate_with_materials_sections -v
```

預期：FAIL（`evaluate_real_results` 不接受 `materials`/`sections` 參數）

- [ ] **Step 3: 修改 evaluate_real_results 簽名與邏輯**

在 `core/parametric_evaluator.py` 頂部新增 import：

```python
from core.materials import expand_truss_data, compute_self_weight
```

修改 `evaluate_real_results` 函數簽名（約第 57 行）：

```python
def evaluate_real_results(
    truss_data: dict,
    real_params: dict,
    symbolic_cache: dict | None = None,
    materials: list | None = None,
    sections: list | None = None,
    include_self_weight: bool = False,
) -> dict:
```

在函數開頭（t0 = time.time() 之後）新增：

```python
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
```

在 `symbolic_cache` 存入邏輯（約第 85-89 行）中，額外儲存 materials/sections：

```python
        if symbolic_cache is not None:
            symbolic_cache["raw_result"]  = raw
            symbolic_cache["elem_Ls"]     = elem_Ls
            symbolic_cache["fingerprint"] = build_geometry_fingerprint(truss_data)
            symbolic_cache["timestamp"]   = datetime.now().strftime("%Y-%m-%dT%H:%M")
            if materials:
                symbolic_cache["materials"] = materials
            if sections:
                symbolic_cache["sections"]  = sections
```

- [ ] **Step 4: 修改 export_cache_to_txt 加入 MATERIALS/SECTIONS 區塊**

在 `export_cache_to_txt` 函數（約第 199 行）的 `lines = [...]` 建立後、`[FINGERPRINT]` 前插入：

```python
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
```

- [ ] **Step 5: 修改 import_cache_from_txt 解析 MATERIALS/SECTIONS**

在 `import_cache_from_txt` 的回傳 dict 建立前（約第 297 行附近），解析兩個新區塊：

```python
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
```

在最終 return dict 加入：

```python
    return {
        "raw_result":  raw_result,
        "elem_Ls":     elem_Ls,
        "fingerprint": cached_fp,
        "timestamp":   fp_lines.get("timestamp", "—"),
        "materials":   materials_out,
        "sections":    sections_out,
    }
```

- [ ] **Step 6: 執行所有測試**

```
pytest test/test_parametric_evaluator.py test/test_materials.py -v
```

預期：全部 PASS

- [ ] **Step 7: Commit**

```
git add core/parametric_evaluator.py test/test_parametric_evaluator.py
git commit -m "feat: parametric_evaluator supports per-section params, self-weight, and TXT materials/sections"
```

---

## Task 4: app.py — 材料表、截面表、桿件表 UI

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes:
  - `compute_section_props(shape, params)` from `core/materials.py`
  - `evaluate_real_results(..., materials, sections, include_self_weight)` 新簽名（Task 3）
  - `export_cache_to_txt`, `import_cache_from_txt` 已含 MATERIALS/SECTIONS（Task 3）

**注意：** 這個 task 是純 UI 修改，無自動化測試。完成後需手動啟動 Streamlit 驗證。

- [ ] **Step 1: 在 app.py 頂部新增 import**

在現有 `from core.parametric_evaluator import (...)` 區塊後新增：

```python
from core.materials import compute_section_props, expand_truss_data
```

- [ ] **Step 2: 新增 session_state 初始化**

在 `if "sym_cache" not in st.session_state:` 後新增：

```python
if "materials" not in st.session_state:
    st.session_state["materials"] = [
        {"name": "鋼材",   "E": 200e9, "G": 77e9, "density": 7850.0},
        {"name": "混凝土", "E":  30e9, "G": 12.5e9, "density": 2400.0},
    ]
if "sections" not in st.session_state:
    st.session_state["sections"] = []
```

- [ ] **Step 3: 在 left_panel 最頂端插入材料定義表**

在 `with left_panel:` 的 `st.subheader("節點 (Nodes)")` 之前插入：

```python
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
```

- [ ] **Step 4: 插入截面定義表（含從形狀計算展開區塊）**

緊接在材料表後插入：

```python
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
```

- [ ] **Step 5: 修改桿件表加入 section 下拉與 status 欄**

找到現有 `elements_df = st.data_editor(...)` 的 DataFrame 定義，將預設 DataFrame 和 column_config 改為：

```python
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
```

- [ ] **Step 6: 新增 section 選取後自動帶入 + override 標色邏輯**

在 `elements_df = st.data_editor(...)` 之後（桿件表 key="elements" 下方）插入：

```python
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

    updated_rows = []
    for _, row in elements_df.iterrows():
        r = row.to_dict()
        sn = r.get("section", "")
        # 自動帶入（只在值為預設 0 或與上一次截面相同時帶入，避免覆蓋 override）
        if sn and sn in sec_map_ui:
            for field in ("E", "G", "A", "I33", "I22", "J"):
                ref = _sec_val(sn, field)
                if ref is not None and r.get(field, 0) == 0:
                    r[field] = ref
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
```

- [ ] **Step 7: 在分析按鈕區加入自重 checkbox**

找到現有 `run_btn = st.button(...)` 前，插入：

```python
    include_sw = st.checkbox("含自重（Self-Weight）", value=False, key="include_sw")
```

- [ ] **Step 8: 修改三個分析呼叫傳入新參數**

找到 `run_btn` 區塊中 `evaluate_real_results` 的呼叫，改為：

```python
        res_eval = evaluate_real_results(
            truss_data, real_params,
            symbolic_cache=st.session_state["sym_cache"],
            materials=st.session_state["materials"],
            sections=st.session_state["sections"],
            include_self_weight=include_sw,
        )
```

同樣修改 `fast_btn` 與 `num_btn` 的呼叫（`num_btn` 不用 symbolic_cache，但傳 materials/sections/include_self_weight）。

- [ ] **Step 9: 手動驗證 UI**

```
streamlit run app.py
```

驗證清單：
1. 材料表顯示鋼材/混凝土預設值，ν 欄位自動計算且無法編輯
2. 截面表可新增截面，material 欄位為下拉選單只能選現有材料
3. 「從形狀計算」展開後選矩形實心，輸入 b=0.2 h=0.3，按「計算並填入」後截面的 A/I33/I22/J 更新
4. 桿件表的 section 欄為下拉選單，選截面後 E/G/A/I33/I22/J 自動帶入
5. 手動改一個桿件的 I33 後，整列變橘色且 status 欄顯示「【修改】」，改回原值後恢復
6. 勾選「含自重」後執行分析，支承反力數值增加

- [ ] **Step 10: Commit**

```
git add app.py
git commit -m "feat: app.py material/section UI, section dropdown, override highlight, self-weight checkbox"
```

---

## Task 5: 匯出/匯入 TXT 同步 materials/sections 至 UI

**Files:**
- Modify: `app.py`（匯入後同步 session_state）

**Interfaces:**
- Consumes: `import_cache_from_txt` 回傳 dict 中的 `materials`、`sections` 欄位（Task 3）

- [ ] **Step 1: 修改匯入後的 session_state 同步邏輯**

找到現有匯入區塊（約 `if uploaded is not None:` 處）：

```python
        if uploaded is not None:
            txt_str = uploaded.read().decode("utf-8")
            import_result = import_cache_from_txt(txt_str, truss_data)
            if "error" in import_result:
                st.error(f"匯入失敗：{import_result['error']}")
            else:
                st.session_state["sym_cache"] = import_result
                # 新增：同步 materials/sections
                if import_result.get("materials"):
                    st.session_state["materials"] = import_result["materials"]
                if import_result.get("sections"):
                    st.session_state["sections"] = import_result["sections"]
                st.success("指紋一致，快取已載入，材料與截面定義已還原。")
```

- [ ] **Step 2: 手動驗證匯出/匯入**

1. 定義材料（加「鋁材」E=70e9）與截面，執行分析
2. 匯出 TXT，用文字編輯器確認含 `[MATERIALS]`、`[SECTIONS]` 區塊
3. 重新整理頁面（清空 session），匯入剛才的 TXT
4. 確認材料表與截面表自動還原

- [ ] **Step 3: Commit**

```
git add app.py
git commit -m "feat: import TXT restores materials and sections to session_state"
```

---

## Self-Review

### Spec Coverage

| Spec 需求 | 對應 Task |
|-----------|----------|
| Material：E, G, density, ν 唯讀 | Task 4 Step 3 |
| 內建鋼材/混凝土預設 | Task 4 Step 2 |
| Section：name, material 下拉, A/I33/I22/J | Task 4 Step 4 |
| 5 種截面形狀計算 | Task 1 Step 3 |
| 近似公式 UI 警示 | Task 4 Step 4 (`st.caption`) |
| Element：section 下拉，選後自動帶入 | Task 4 Step 5-6 |
| Local override 整列淡橘色 + 【修改】 | Task 4 Step 6 |
| 恢復原值後標色消除 | Task 4 Step 6 (`_is_override`) |
| 符號解 per-section 符號變數 (E_s0...) | Task 2 |
| Fingerprint 不變 | 未修改 `build_geometry_fingerprint`（維持現狀） |
| 自重 checkbox，疊加均佈線載重 | Task 3 Step 3, Task 4 Step 7-8 |
| TXT 加 MATERIALS/SECTIONS 區塊 | Task 3 Step 4-5 |
| 匯入後還原 materials/sections | Task 5 |

### 無 Placeholder

所有步驟均有完整程式碼或指令。

### 型別一致性

- `compute_section_props` → `dict` with keys A/I33/I22/J：Task 1 定義，Task 4 Step 4 `st.button("計算並填入")` 呼叫相同 key
- `expand_truss_data` 回傳 deep copy，elem 含 E/G/A/I33/I22/J：Task 1 定義，Task 3/4 呼叫時均使用此回傳值
- `evaluate_real_results` 新簽名 materials/sections/include_self_weight：Task 3 Step 3 定義，Task 4 Step 8 呼叫一致
- `import_cache_from_txt` 回傳含 materials/sections：Task 3 Step 5 定義，Task 5 Step 1 讀取相同 key
