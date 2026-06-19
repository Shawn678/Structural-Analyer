# Parametric Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `core/parametric_evaluator.py` 實現代數解快取與快速參數代入，並修改 `app.py` 加入對應 UI，讓使用者在幾何與支承不變時跳過昂貴的符號分析。

**Architecture:** `parametric_evaluator.py` 提供三個獨立 API（指紋建立、快速代入、TXT 匯出/匯入），`app.py` 在 `st.session_state['sym_cache']` 中快取符號分析結果，「代入參數（快速）」按鈕先比對指紋再代入數值。

**Tech Stack:** Python 3.x, SymPy, NumPy, Streamlit

## Global Constraints

- 絕對不修改 `core/symbolic.py`（只呼叫它）
- 絕對不讀取、搜尋或修改 `.venv` 目錄
- 測試放在 `test/` 目錄下
- 公式字串格式：SymPy 標準（`**` 次方、`*` 乘），不用 `^` 或 `·`
- `elem_Ls` 順序必須與 `run_symbolic_analysis` 回傳的 `elements_info` 順序一致
- 指紋比對容差：桿件長度四捨五入至小數點後 6 位（1e-6 m）

---

## File Map

| 檔案 | 動作 | 責任 |
|---|---|---|
| `core/parametric_evaluator.py` | 新增 | 指紋、代入、TXT 匯出/匯入 |
| `core/__init__.py` | 修改 | 匯出新 API |
| `app.py` | 修改 | 快取邏輯 + 快速代入 UI + TXT 管理 UI |
| `test/test_parametric_evaluator.py` | 新增 | 單元測試 |

---

## Task 1: `build_geometry_fingerprint` + 基礎測試

**Files:**
- Create: `core/parametric_evaluator.py`
- Create: `test/test_parametric_evaluator.py`

**Interfaces:**
- Produces: `build_geometry_fingerprint(truss_data: dict) -> dict`
  ```python
  # 回傳結構：
  {
      "n_elements": 3,
      "elem_lengths": ["6.000000", "6.000000", "4.000000"],
      "connections": ["1-2", "2-3", "4-1"],
      "supports": ["2:uy,uz,rx,rz", "3:uy,uz,rx,rz", "4:rx,ry,rz,ux,uy,uz"],
  }
  ```

- [ ] **Step 1: 建立 `test/test_parametric_evaluator.py`，寫指紋測試**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.parametric_evaluator import build_geometry_fingerprint

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
```

- [ ] **Step 2: 確認測試失敗**

```
cd "c:/Users/USER/Desktop/One_Piece/TRUSS ANALYER"
.venv\Scripts\python.exe -m pytest test/test_parametric_evaluator.py -v
```
Expected: ImportError 或 ModuleNotFoundError（`parametric_evaluator` 尚未存在）

- [ ] **Step 3: 建立 `core/parametric_evaluator.py`，實作 `build_geometry_fingerprint`**

```python
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
```

- [ ] **Step 4: 執行測試確認通過**

```
.venv\Scripts\python.exe -m pytest test/test_parametric_evaluator.py::test_fingerprint_n_elements test/test_parametric_evaluator.py::test_fingerprint_lengths test/test_parametric_evaluator.py::test_fingerprint_connections test/test_parametric_evaluator.py::test_fingerprint_supports test/test_parametric_evaluator.py::test_fingerprint_ignores_material -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add core/parametric_evaluator.py test/test_parametric_evaluator.py
git commit -m "feat: add build_geometry_fingerprint with tests"
```

---

## Task 2: `evaluate_real_results`

**Files:**
- Modify: `core/parametric_evaluator.py`
- Modify: `test/test_parametric_evaluator.py`

**Interfaces:**
- Consumes: `build_geometry_fingerprint`, `run_symbolic_analysis` (from `core.symbolic`)
- Produces: `evaluate_real_results(truss_data, real_params, symbolic_cache=None) -> dict`
  ```python
  # real_params 結構：
  {"E": float, "A": float, "I": float, "G": float, "P": float, "w": float}

  # 回傳結構：
  {
    "node_displacements": [
      {"node_id": 1, "ux": {"formula": "...", "value": 0.0}, "uy": {...}, ...}
    ],
    "element_forces": [
      {"element_id": 1, "nodes": "N1 - N2",
       "N":  {"formula": "...", "value": 0.0},
       "V2": {"formula": "...", "value": 0.0},
       "V3": {"formula": "...", "value": 0.0},
       "M3": {"formula": "...", "value": 0.0},
       "M2": {"formula": "...", "value": 0.0},
       "i_end": "...", "j_end": "..."}
    ],
    "support_reactions": [
      {"node_id": 4, "Rx": {"formula": "...", "value": 0.0}, ...}
    ],
    "cache_used": True,
    "eval_time_ms": 12,
  }
  ```

- [ ] **Step 1: 在 `test/test_parametric_evaluator.py` 新增代入測試**

在現有測試檔末尾加入：

```python
from core.parametric_evaluator import evaluate_real_results

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
```

- [ ] **Step 2: 確認新測試失敗**

```
.venv\Scripts\python.exe -m pytest test/test_parametric_evaluator.py::test_evaluate_returns_structure -v
```
Expected: ImportError（`evaluate_real_results` 未定義）

- [ ] **Step 3: 在 `core/parametric_evaluator.py` 實作 `evaluate_real_results`**

在 `build_geometry_fingerprint` 之後加入：

```python
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
```

- [ ] **Step 4: 執行所有代入測試**

```
.venv\Scripts\python.exe -m pytest test/test_parametric_evaluator.py::test_evaluate_returns_structure test/test_parametric_evaluator.py::test_evaluate_displacement_has_formula_and_value test/test_parametric_evaluator.py::test_evaluate_cache_used_on_second_call test/test_parametric_evaluator.py::test_evaluate_reaction_sum_equals_load -v
```
Expected: 4 passed（`test_evaluate_reaction_sum_equals_load` 若結構對稱，Ry 各 0.5，總和 1.0）

- [ ] **Step 5: Commit**

```bash
git add core/parametric_evaluator.py test/test_parametric_evaluator.py
git commit -m "feat: add evaluate_real_results with cache reuse"
```

---

## Task 3: `export_cache_to_txt` + `import_cache_from_txt`

**Files:**
- Modify: `core/parametric_evaluator.py`
- Modify: `test/test_parametric_evaluator.py`

**Interfaces:**
- Consumes: `build_geometry_fingerprint`, `symbolic_cache` dict（Task 2 產出）
- Produces:
  - `export_cache_to_txt(symbolic_cache: dict) -> str`（回傳 TXT 字串，由呼叫方寫檔或下載）
  - `import_cache_from_txt(txt_content: str, truss_data: dict) -> dict`（成功回傳 cache dict，失敗回傳 `{"error": "..."}`）

- [ ] **Step 1: 新增 TXT 匯出/匯入測試**

在 `test/test_parametric_evaluator.py` 末尾加入：

```python
from core.parametric_evaluator import (
    evaluate_real_results,
    export_cache_to_txt,
    import_cache_from_txt,
)

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
```

- [ ] **Step 2: 確認新測試失敗**

```
.venv\Scripts\python.exe -m pytest test/test_parametric_evaluator.py::test_export_txt_contains_sections -v
```
Expected: ImportError（函數未定義）

- [ ] **Step 3: 在 `core/parametric_evaluator.py` 實作 `export_cache_to_txt`**

在 `evaluate_real_results` 之後加入：

```python
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
        "[FINGERPRINT]",
        f"n_elements={fp.get('n_elements', len(elem_Ls))}",
        f"elem_lengths={','.join(fp.get('elem_lengths', [f'{L:.6f}' for L in elem_Ls]))}",
        f"connections={','.join(fp.get('connections', []))}",
        f"supports={'|'.join(fp.get('supports', []))}",
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
```

- [ ] **Step 4: 實作 `import_cache_from_txt`**

緊接在 `export_cache_to_txt` 之後加入：

```python
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

    return {
        "raw_result": {
            "node_displacements": node_displacements,
            "element_forces":     element_forces,
            "support_reactions":  support_reactions,
        },
        "elem_Ls":     elem_Ls,
        "fingerprint": current_fp,
        "timestamp":   fp_lines.get("generated", datetime.now().strftime("%Y-%m-%dT%H:%M")),
    }
```

- [ ] **Step 5: 執行所有 TXT 相關測試**

```
.venv\Scripts\python.exe -m pytest test/test_parametric_evaluator.py::test_export_txt_contains_sections test/test_parametric_evaluator.py::test_export_txt_contains_formulas test/test_parametric_evaluator.py::test_import_txt_roundtrip test/test_parametric_evaluator.py::test_import_txt_fingerprint_mismatch test/test_parametric_evaluator.py::test_import_cache_enables_fast_eval -v
```
Expected: 5 passed

- [ ] **Step 6: 執行全部測試確認無迴歸**

```
.venv\Scripts\python.exe -m pytest test/test_parametric_evaluator.py -v
```
Expected: 全部 passed

- [ ] **Step 7: Commit**

```bash
git add core/parametric_evaluator.py test/test_parametric_evaluator.py
git commit -m "feat: add export/import cache TXT with fingerprint validation"
```

---

## Task 4: 更新 `core/__init__.py` 並整合 `app.py`

**Files:**
- Modify: `core/__init__.py`
- Modify: `app.py`

**Interfaces:**
- Consumes: 所有 Task 1-3 的公開函數

- [ ] **Step 1: 更新 `core/__init__.py`**

將檔案內容改為：

```python
from .symbolic import run_symbolic_analysis
from .parametric_evaluator import (
    build_geometry_fingerprint,
    evaluate_real_results,
    export_cache_to_txt,
    import_cache_from_txt,
)
```

- [ ] **Step 2: 在 `app.py` 頂部更新 import**

找到：
```python
from core.symbolic import run_symbolic_analysis
```
替換為：
```python
from core.symbolic import run_symbolic_analysis
from core.parametric_evaluator import (
    build_geometry_fingerprint,
    evaluate_real_results,
    export_cache_to_txt,
    import_cache_from_txt,
)
```

- [ ] **Step 3: 在 `app.py` 初始化 session_state**

在 `st.set_page_config(...)` 之後、`def create_structure_plot` 之前加入：

```python
if "sym_cache" not in st.session_state:
    st.session_state["sym_cache"] = {}
```

- [ ] **Step 4: 修改左側面板的按鈕區**

找到：
```python
    run_btn = st.button("執行分析", type="primary", use_container_width=True)
```
替換為：
```python
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
```

- [ ] **Step 5: 修改右側面板的分析觸發邏輯**

找到：
```python
    if run_btn:
        truss_data = {
```
替換為：
```python
    # ── 共用 truss_data 建構 ──────────────────────────────────────────────
    truss_data = {
        "nodes":               nodes_df.dropna(subset=['id']).fillna(0).to_dict('records'),
        "elements":            elements_df.dropna(subset=['id','i','j']).fillna(0).to_dict('records'),
        "supports":            supports_df.dropna(subset=['node_id']).fillna(False).to_dict('records'),
        "loads":               loads_df.dropna(subset=['node_id']).fillna(0).to_dict('records'),
        "element_loads":       e_loads_df.dropna(subset=['element_id']).fillna(0).to_dict('records'),
        "element_point_loads": e_pt_loads_df.dropna(subset=['element_id']).fillna(0).to_dict('records'),
    }

    if run_btn:
```

- [ ] **Step 6: 修改完整分析的執行區塊**

找到（在 `if run_btn:` 內）：
```python
        truss_data = {
            "nodes": nodes_df.dropna(subset=['id']).fillna(0).to_dict('records'),
            "elements": elements_df.dropna(subset=['id', 'i', 'j']).fillna(0).to_dict('records'),
            "supports": supports_df.dropna(subset=['node_id']).fillna(False).to_dict('records'),
            "loads": loads_df.dropna(subset=['node_id']).fillna(0).to_dict('records'),
            "element_loads": e_loads_df.dropna(subset=['element_id']).fillna(0).to_dict('records'),
            "element_point_loads": e_pt_loads_df.dropna(subset=['element_id']).fillna(0).to_dict('records'),
        }

        try:
            res = run_symbolic_analysis(truss_data)
            output_area.json(res)
```
替換為：
```python
        try:
            # 清除舊快取，強制重新求解
            st.session_state["sym_cache"] = {}
            real_params = {"E": pe_E, "A": pe_A, "I": pe_I, "G": pe_G, "P": pe_P, "w": pe_w}
            res_eval = evaluate_real_results(truss_data, real_params,
                                             symbolic_cache=st.session_state["sym_cache"])
            res = st.session_state["sym_cache"]["raw_result"]
            output_area.json(res_eval)
```

- [ ] **Step 7: 在右側面板加入快速代入觸發與快取管理 UI**

在 `with right_panel:` 的 `st.subheader("分析結果輸出")` 之後、`output_area = st.empty()` 之前加入：

```python
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
```

- [ ] **Step 8: 加入快速代入觸發區塊**

在 `output_area = st.empty()` 之後加入：

```python
    if fast_btn:
        real_params = {"E": pe_E, "A": pe_A, "I": pe_I, "G": pe_G, "P": pe_P, "w": pe_w}
        try:
            res_eval = evaluate_real_results(truss_data, real_params,
                                             symbolic_cache=st.session_state["sym_cache"])
            output_area.json(res_eval)
            st.info(f"快速代入完成，耗時 {res_eval['eval_time_ms']} ms（使用快取符號解）。")
        except Exception as e:
            st.error(f"代入失敗：{e}")
```

- [ ] **Step 9: 手動測試 Streamlit app**

```
.venv\Scripts\python.exe -m streamlit run app.py
```

執行以下測試流程：
1. 點「執行分析（符號解）」→ 確認結果正常顯示，快取狀態顯示「快取有效」
2. 開啟「代數解快取管理」→ 點「匯出代數解 TXT」→ 確認檔案下載
3. 修改 E 值 → 點「⚡ 代入參數（快速）」→ 確認結果更新，顯示耗時毫秒
4. 修改節點座標 → 確認「代入參數」按鈕變灰（快取失效）
5. 匯入剛才下載的 TXT → 確認「指紋一致，快取已載入」

- [ ] **Step 10: Commit**

```bash
git add core/__init__.py app.py
git commit -m "feat: integrate parametric evaluator into app with cache UI and TXT export/import"
```

---

## Self-Review

**Spec coverage 確認：**

| 需求 | 實作任務 |
|---|---|
| `evaluate_real_results(truss_data, real_params)` | Task 2 |
| 呼叫 `run_symbolic_analysis` 並快取 | Task 2 Step 3 |
| `subs_dict` 含 E/A/I/G/P/w + L_k | Task 2 Step 3 |
| TXT 匯出 | Task 3 Step 3 |
| TXT 匯入 + 指紋驗證 | Task 3 Step 4 |
| 幾何+支承指紋（含彈簧） | Task 1 Step 3 |
| `formula` + `value` 雙欄位回傳 | Task 2 Step 3 |
| `cache_used` + `eval_time_ms` | Task 2 Step 3 |
| Streamlit 快取按鈕 + 狀態顯示 | Task 4 Step 4–8 |
| 匯出下載 + 匯入驗證 UI | Task 4 Step 7 |
| 支承改變觸發重算 | Task 1（指紋含支承）+ Task 4 Step 4 |

**Placeholder scan：** 無 TBD、無「類似 Task N」、所有程式碼塊均完整。

**Type consistency：**
- `symbolic_cache` 為 `dict`，Task 2/3/4 均一致
- `export_cache_to_txt` 回傳 `str`，Task 4 Step 7 用 `.encode("utf-8")` 傳給 `st.download_button` ✓
- `import_cache_from_txt` 簽名：`(txt_content: str, truss_data: dict) -> dict` ，Task 3/4 均一致 ✓
- `_subs_value` 回傳 `float | None`，Task 2 回傳結構中 `"value"` 欄位一致 ✓
