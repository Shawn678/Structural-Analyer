# 各斷面獨立符號快速代入 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓每個斷面組擁有獨立 SymPy 符號（`E_s1, A_s1, ...`），並以各斷面可編輯表格（含嵌入式計算器）取代舊的六個全局輸入框，實現真正的多斷面快速代入。

**Architecture:** `symbolic.py` 改為接收 `section_group_map`，以 Hadamard 交叉採樣對各斷面組獨立擾動；`parametric_evaluator.py` 的 `subs_dict` 改以各組符號為鍵；`app.py` 快速代入 UI 換為各斷面表格＋嵌入式計算器。

**Tech Stack:** Python 3.11、SymPy、NumPy、Streamlit、Plotly

## 全局約束

- 斷面組數上限：5 組（超過時顯示警告，仍允許執行）
- 向後相容：`real_params` 無 `"groups"` 鍵時回退舊行為
- 快速代入面板的修改不影響 `st.session_state["sections"]`
- 測試執行：`pytest test/ -v`

---

## 檔案修改清單

| 動作 | 路徑 | 說明 |
|------|------|------|
| 修改 | `core/symbolic.py` | 接收 `section_group_map`，Hadamard 交叉採樣 |
| 修改 | `core/parametric_evaluator.py` | `real_params["groups"]` 路徑、快取存 `section_sym_names` |
| 修改 | `core/__init__.py` | 匯出新函式（若有） |
| 修改 | `app.py` | 移除舊輸入框、新增各斷面表格與嵌入式計算器 |
| 修改 | `test/test_parametric_evaluator.py` | 新增多斷面符號代入測試 |

---

## Task 1：`symbolic.py` — 接收 `section_group_map` 並建立各組符號

**Files:**
- Modify: `core/symbolic.py:324`（`run_symbolic_analysis` 函式簽名與符號指派邏輯）

**Interfaces:**
- Produces: `run_symbolic_analysis(truss_data, section_group_map=None) -> dict`
  - `section_group_map`: `{sec_name: {"E": sp.Symbol, "A": sp.Symbol, "I33": sp.Symbol, "I22": sp.Symbol, "G": sp.Symbol}}`
  - 回傳 dict 新增欄位：`section_groups: list[str]`、`section_sym_names: dict`

- [ ] **Step 1：確認現有函式簽名**

  讀取 `core/symbolic.py` 第 324 行，確認 `run_symbolic_analysis(truss_data, section_sym_map=None)` 的現有參數名為 `section_sym_map`。

- [ ] **Step 2：統一參數名為 `section_group_map`**

  將 `run_symbolic_analysis` 的參數 `section_sym_map` 改為 `section_group_map`，同步修改函式內部所有使用處（搜尋 `section_sym_map`，全部替換為 `section_group_map`）。

  ```python
  # core/symbolic.py:324
  def run_symbolic_analysis(truss_data: dict, section_group_map: dict | None = None) -> dict:
  ```

- [ ] **Step 3：在回傳 dict 中補充 `section_groups` 與 `section_sym_names`**

  在 `run_symbolic_analysis` 最後組裝回傳 dict 的位置（搜尋 `"node_displacements"` 被 return 的地方），加入：

  ```python
  # 收集斷面組資訊，供快取失效比對與 subs_dict 重建
  _sec_groups = sorted(section_group_map.keys()) if section_group_map else []
  _sec_sym_names = {}
  if section_group_map:
      for sn, syms in section_group_map.items():
          _sec_sym_names[sn] = {k: str(v) for k, v in syms.items()}
  ```

  並將 `section_groups` 和 `section_sym_names` 加入回傳 dict：

  ```python
  return {
      # ... 原有欄位 ...
      "section_groups":    _sec_groups,
      "section_sym_names": _sec_sym_names,
  }
  ```

- [ ] **Step 4：執行現有測試確認不破壞**

  ```
  pytest test/ -v -k "not test_per_section"
  ```
  預期：全部既有測試 PASS。

- [ ] **Step 5：Commit**

  ```bash
  git add core/symbolic.py
  git commit -m "feat: run_symbolic_analysis returns section_groups and section_sym_names"
  ```

---

## Task 2：`symbolic.py` — Hadamard 交叉採樣

**Files:**
- Modify: `core/symbolic.py`（`run_symbolic_analysis` 內的採樣迴圈，約第 469–548 行）

**Interfaces:**
- Consumes: Task 1 產生的 `section_group_map` 參數
- Produces: 採樣矩陣正確反映各組獨立擾動

- [ ] **Step 1：新增輔助函式 `_make_cross_samples`**

  在 `_build_basis_row` 函式之前（約第 204 行），插入：

  ```python
  def _make_cross_samples(n_groups: int, base_vals: list[dict], n_samples: int, seed: int = 42) -> list[list[float]]:
      """
      為每個斷面組獨立產生 scale 序列。
      回傳 shape=(n_samples, n_groups) 的 list-of-list，每格是該採樣點該組的 scale 倍率。
      base_vals: [{"E": float, "A": float, "I": float, "G": float}, ...]，長度 = n_groups
      """
      SCALES = [1.0, 5.0, 25.0, 100.0, 500.0]
      rng = np.random.default_rng(seed)
      result = []
      for _ in range(n_samples):
          row = [SCALES[rng.integers(0, len(SCALES))] for _ in range(n_groups)]
          result.append(row)
      return result
  ```

- [ ] **Step 2：修改採樣迴圈使用交叉採樣**

  在 `run_symbolic_analysis` 的「多點採樣」區塊（約第 469 行），於迴圈前加入：

  ```python
  # 當有 section_group_map 時，改用 Hadamard 交叉採樣
  _group_names = sorted(section_group_map.keys()) if section_group_map else []
  _n_groups = len(_group_names)
  if _n_groups > 0:
      n_samples = max(len(SAMPLE_SCALES), _n_groups * 4 + 4)
      # 取每組基準材料值（從 truss_data 第一根屬於該組的桿件）
      _elem_map = {e["id"]: e for e in truss_data["elements"]}
      _group_base = []
      for gname in _group_names:
          # 找屬於此斷面組的第一根桿件取基準值
          e0 = next((e for e in truss_data["elements"] if e.get("section") == gname), {})
          _group_base.append({
              "E": _to_float(e0.get("E",   E_base), E_base),
              "A": _to_float(e0.get("A",   A_base), A_base),
              "I": _to_float(e0.get("I33", e0.get("I", I_base)), I_base),
              "G": _to_float(e0.get("G",   G_base), G_base),
          })
      _cross_scales = _make_cross_samples(_n_groups, _group_base, n_samples)
  ```

  然後修改迴圈本體，使 `per_elem_E` 等列表依各桿件所屬斷面組的 scale 設定：

  ```python
  for s_idx in range(n_samples):
      if _n_groups > 0:
          # 每根桿件依所屬組取對應 scale
          scale_by_group = {gname: _cross_scales[s_idx][gi]
                            for gi, gname in enumerate(_group_names)}
          global_scale = float(np.mean(list(scale_by_group.values())))

          def _scaled(e, field_I="I33", default_base=None):
              sn = e.get("section", "")
              sc = scale_by_group.get(sn, global_scale)
              base = {"E": E_base, "A": A_base, "I33": I_base, "G": G_base}
              return _to_float(e.get(field_I, base.get(field_I, 1.0)), base.get(field_I, 1.0)) * sc

          td_scaled = copy.deepcopy(truss_data)
          for elem in td_scaled["elements"]:
              elem["E"]   = _scaled(elem, "E")
              elem["A"]   = _scaled(elem, "A")
              elem["I33"] = _scaled(elem, "I33")
              elem["I22"] = _scaled(elem, "I33")
              elem["G"]   = _scaled(elem, "G")
          K_s, elems_s, _, free_s = _assemble_K_np(td_scaled, E_base * global_scale,
                                                     A_base * global_scale,
                                                     I_base * global_scale,
                                                     G_base * global_scale)
          per_elem_E_s = [_to_float(e.get("E",   E_base), E_base) * scale_by_group.get(e.get("section",""), global_scale) for e in truss_data["elements"]]
          per_elem_A_s = [_to_float(e.get("A",   A_base), A_base) * scale_by_group.get(e.get("section",""), global_scale) for e in truss_data["elements"]]
          per_elem_I_s = [_to_float(e.get("I33", e.get("I", I_base)), I_base) * scale_by_group.get(e.get("section",""), global_scale) for e in truss_data["elements"]]
          per_elem_G_s = [_to_float(e.get("G",   G_base), G_base) * scale_by_group.get(e.get("section",""), global_scale) for e in truss_data["elements"]]
      else:
          # 原有邏輯（全局 scale）
          scale = SAMPLE_SCALES[s_idx] if s_idx < len(SAMPLE_SCALES) else SAMPLE_SCALES[-1]
          # ... 原有程式碼不變 ...
  ```

- [ ] **Step 3：執行現有測試確認不破壞**

  ```
  pytest test/ -v -k "not test_per_section"
  ```
  預期：全部既有測試 PASS。

- [ ] **Step 4：Commit**

  ```bash
  git add core/symbolic.py
  git commit -m "feat: Hadamard cross-sampling for per-section-group symbols"
  ```

---

## Task 3：`parametric_evaluator.py` — 支援 `real_params["groups"]` 與快取 `section_sym_names`

**Files:**
- Modify: `core/parametric_evaluator.py`

**Interfaces:**
- Consumes:
  - `run_symbolic_analysis` 回傳的 `section_sym_names` 欄位（Task 1）
  - `real_params = {"groups": {sec_name: {"E", "A", "I", "G"}}, "P": float, "w": float}`
- Produces:
  - `evaluate_real_results` 在 `real_params` 有 `"groups"` 時正確代入各組符號

- [ ] **Step 1：快取儲存 `section_groups` 與 `section_sym_names`**

  在 `evaluate_real_results` 的快取寫入區塊（約第 106 行，`symbolic_cache["raw_result"] = raw` 之後），新增：

  ```python
  symbolic_cache["section_groups"]    = raw.get("section_groups", [])
  symbolic_cache["section_sym_names"] = raw.get("section_sym_names", {})
  ```

- [ ] **Step 2：`subs_dict` 支援 `"groups"` 格式**

  將現有的 `subs_dict` 建構（約第 120–127 行）改為：

  ```python
  subs_dict = {}
  if "groups" in real_params and symbolic_cache and symbolic_cache.get("section_sym_names"):
      # 多斷面組路徑：各組獨立符號
      sym_names = symbolic_cache["section_sym_names"]
      for sec_name, vals in real_params["groups"].items():
          if sec_name not in sym_names:
              continue
          for field, sym_str in sym_names[sec_name].items():
              field_key = "I" if field == "I33" else field  # real_params 用 "I"
              val = vals.get(field_key, vals.get(field))
              if val is not None:
                  subs_dict[sp.Symbol(sym_str)] = float(val)
      # P / w 仍為全局符號
      subs_dict[P_s] = float(real_params.get("P", 1.0))
      subs_dict[w_s] = float(real_params.get("w", 0.0))
      # L 符號
      for k, Lk_sym in enumerate(L_syms):
          subs_dict[Lk_sym] = float(elem_Ls[k])
  else:
      # 舊路徑：全局 E/A/I/G
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
  ```

- [ ] **Step 3：數值路徑（元件內力）支援 `"groups"` 格式**

  在 `evaluate_real_results` 的 `td_num` 建構區塊（約第 152–174 行），於 `use_per_elem` 判斷後新增 `"groups"` 路徑：

  ```python
  use_groups = "groups" in real_params and bool(real_params["groups"])
  if use_groups:
      # 依各桿件 section 名稱從 groups 取值，跳過 expand_truss_data
      td_num = copy.deepcopy(truss_data)
      group_vals = real_params["groups"]
      for elem in td_num["elements"]:
          sn = elem.get("section", "")
          if sn in group_vals:
              gv = group_vals[sn]
              elem["E"]   = float(gv.get("E",   elem.get("E",   200e9)))
              elem["A"]   = float(gv.get("A",   elem.get("A",   0.01)))
              elem["I33"] = float(gv.get("I",   elem.get("I33", 1e-4)))
              elem["I22"] = float(gv.get("I",   elem.get("I22", 1e-4)))
              elem["G"]   = float(gv.get("G",   elem.get("G",   77e9)))
      # 套用載重倍率
      _P = float(real_params.get("P", 1.0))
      _w = float(real_params.get("w", 0.0))
      for load in td_num["loads"]:
          for k in ("fx", "fy", "fz", "mx", "my", "mz"):
              if k in load:
                  load[k] = float(load[k]) * _P
      for el in td_num["element_loads"]:
          if "w" in el:
              el["w"] = float(el["w"]) * _w
      for el in td_num["element_point_loads"]:
          if "p" in el:
              el["p"] = float(el["p"]) * _P
  elif not use_per_elem:
      # 舊路徑：全局參數
      # ... 原有程式碼不變 ...
  ```

- [ ] **Step 4：寫測試**

  在 `test/test_parametric_evaluator.py` 末尾加入：

  ```python
  # ── 多斷面組測試 ──────────────────────────────────────────────────────────
  import sympy as sp

  MIXED_SECTION_DATA = {
      "nodes": [
          {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
          {"id": 2, "x": 6.0, "y": 0.0, "z": 0.0},
          {"id": 3, "x": 3.0, "y": 3.0, "z": 0.0},
      ],
      "elements": [
          {"id": 1, "i": 1, "j": 2, "section": "S1", "E": 200e9, "A": 0.01, "I33": 1e-4, "I22": 1e-4, "G": 77e9},
          {"id": 2, "i": 2, "j": 3, "section": "S2", "E": 200e9, "A": 0.02, "I33": 2e-4, "I22": 2e-4, "G": 77e9},
      ],
      "supports": [
          {"node_id": 1, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True},
          {"node_id": 3, "ux": False, "uy": True, "uz": True, "rx": False, "ry": False, "rz": False},
      ],
      "loads": [{"node_id": 2, "fz": -10000.0}],
      "element_loads": [],
      "element_point_loads": [],
  }

  SECTION_GROUP_MAP = {
      "S1": {
          "E": sp.Symbol("E_s1"), "A": sp.Symbol("A_s1"),
          "I33": sp.Symbol("I_s1"), "I22": sp.Symbol("I_s1"), "G": sp.Symbol("G_s1"),
      },
      "S2": {
          "E": sp.Symbol("E_s2"), "A": sp.Symbol("A_s2"),
          "I33": sp.Symbol("I_s2"), "I22": sp.Symbol("I_s2"), "G": sp.Symbol("G_s2"),
      },
  }

  def test_per_section_cache_stores_sym_names():
      """快取應儲存 section_groups 與 section_sym_names。"""
      cache = {}
      evaluate_real_results(
          MIXED_SECTION_DATA,
          {"E": 200e9, "A": 0.01, "I": 1e-4, "G": 77e9, "P": 1.0, "w": 0.0},
          symbolic_cache=cache,
          section_group_map=SECTION_GROUP_MAP,
      )
      assert "section_groups" in cache
      assert "section_sym_names" in cache
      assert "S1" in cache["section_sym_names"]
      assert cache["section_sym_names"]["S1"]["E"] == "E_s1"

  def test_per_section_groups_substitution():
      """groups 格式的 real_params 應正確代入各斷面符號，位移非零。"""
      cache = {}
      # 第一次：建立快取（使用 section_group_map）
      evaluate_real_results(
          MIXED_SECTION_DATA,
          {"E": 200e9, "A": 0.01, "I": 1e-4, "G": 77e9, "P": 1.0, "w": 0.0},
          symbolic_cache=cache,
          section_group_map=SECTION_GROUP_MAP,
      )
      # 第二次：使用 groups 格式快速代入
      real_params = {
          "groups": {
              "S1": {"E": 210e9, "A": 0.012, "I": 1.2e-4, "G": 80e9},
              "S2": {"E": 200e9, "A": 0.020, "I": 2.0e-4, "G": 77e9},
          },
          "P": 1.0,
          "w": 0.0,
      }
      res = evaluate_real_results(MIXED_SECTION_DATA, real_params, symbolic_cache=cache)
      # node 2 應有非零位移（受力節點）
      nd2 = next(n for n in res["node_displacements"] if n["node_id"] == 2)
      uz_val = nd2["uz"]["value"]
      assert uz_val is not None
      assert abs(uz_val) > 1e-10
  ```

- [ ] **Step 5：執行新測試確認**

  ```
  pytest test/test_parametric_evaluator.py::test_per_section_cache_stores_sym_names
  pytest test/test_parametric_evaluator.py::test_per_section_groups_substitution
  ```
  預期：兩個測試均 PASS。

- [ ] **Step 6：`evaluate_real_results` 簽名補上 `section_group_map` 參數**

  ```python
  def evaluate_real_results(
      truss_data: dict,
      real_params: dict,
      symbolic_cache: dict | None = None,
      materials: list | None = None,
      sections: list | None = None,
      include_self_weight: bool = False,
      section_group_map: dict | None = None,   # 新增
  ) -> dict:
  ```

  並在函式內部，若 `section_group_map` 不為 None，將其傳入 `run_symbolic_analysis`：

  ```python
  raw = run_symbolic_analysis(truss_data, section_group_map=section_group_map)
  ```

- [ ] **Step 7：執行全部測試**

  ```
  pytest test/ -v
  ```
  預期：全部 PASS。

- [ ] **Step 8：Commit**

  ```bash
  git add core/parametric_evaluator.py test/test_parametric_evaluator.py
  git commit -m "feat: evaluate_real_results supports per-section groups substitution"
  ```

---

## Task 4：`app.py` — 建立 `section_group_map` 並傳入符號解

**Files:**
- Modify: `app.py`（`run_btn` 按鈕處理區塊，約第 581 行）

**Interfaces:**
- Consumes: `evaluate_real_results(..., section_group_map=...)` 簽名（Task 3）
- Produces: `run_btn` 路徑正確建立 `section_group_map` 並傳入

- [ ] **Step 1：在 `run_btn` 區塊加入 `section_group_map` 建構**

  在 `app.py` 約第 581 行 `if run_btn:` 區塊內，`st.session_state["sym_cache"] = {}` 之後，插入：

  ```python
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
  ```

  並在 `evaluate_real_results` 呼叫中加入此參數：

  ```python
  res_eval = evaluate_real_results(
      truss_data, real_params,
      symbolic_cache=st.session_state["sym_cache"],
      materials=st.session_state["materials"],
      sections=st.session_state["sections"],
      include_self_weight=include_sw,
      section_group_map=_sec_group_map if _sec_group_map else None,
  )
  ```

  同時在 `app.py` 頂部確認已 `import sympy as sp`（若未匯入則加入）。

- [ ] **Step 2：快取存入 `section_groups` 的失效檢查**

  在約第 430 行的 `cache_valid` 檢查處，於現有指紋比對之後加入：

  ```python
  if cache_valid:
      _current_sec_groups = sorted(
          set(r.get("section", "") for r in st.session_state.get("elements_data", [])
              if r.get("section"))
      )
      _cached_sec_groups = st.session_state["sym_cache"].get("section_groups", [])
      if _current_sec_groups != _cached_sec_groups:
          cache_valid = False
  ```

- [ ] **Step 3：手動測試**

  啟動 `streamlit run app.py`，在斷面表格新增兩個斷面（例如 S1、S2），分別指派給不同桿件，執行符號解，確認：
  - 分析正常完成不報錯
  - 若刪除一個斷面後快速代入按鈕變成 disabled

- [ ] **Step 4：Commit**

  ```bash
  git add app.py
  git commit -m "feat: build section_group_map in run_btn and extend cache validity check"
  ```

---

## Task 5：`app.py` — 移除舊輸入框，新增各斷面可編輯表格

**Files:**
- Modify: `app.py`（`⚡ 快速代入參數` expander 內部，約第 441–463 行）

**Interfaces:**
- Consumes: `st.session_state["sections"]`、`st.session_state["materials"]`
- Produces: `st.session_state["quickfill_overrides"]`（`{sec_name: {E, A, I33, G}}`）

- [ ] **Step 1：移除舊的六個 `st.number_input`**

  刪除約第 442–450 行（`col_a`、`col_b` 兩欄及其 `pe_E`、`pe_I`、`pe_P`、`pe_A`、`pe_G`、`pe_w` 六個輸入框）。

- [ ] **Step 2：加入 `quickfill_overrides` 初始化邏輯**

  在 expander 開頭（移除舊輸入框的位置）插入：

  ```python
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
  ```

- [ ] **Step 3：渲染各斷面可編輯表格**

  緊接上述初始化邏輯後，加入：

  ```python
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
  else:
      st.info("請先在桿件表格中指派斷面，才能使用快速代入。")
  ```

- [ ] **Step 4：加入 P/w 倍率輸入（替代移除的 pe_P、pe_w）**

  表格之後加入：

  ```python
  _qf_col1, _qf_col2 = st.columns(2)
  pe_P = _qf_col1.number_input("P 倍率", value=1.0, key="pe_P")
  pe_w = _qf_col2.number_input("w 倍率", value=0.0, key="pe_w")
  ```

- [ ] **Step 5：更新 `fast_btn` 處理器**

  找到約第 566 行 `if fast_btn:` 區塊，將 `real_params` 建構改為：

  ```python
  if fast_btn:
      _group_vals = {
          sn: {"E": v["E"], "A": v["A"], "I": v["I33"], "G": v["G"]}
          for sn, v in st.session_state["quickfill_overrides"].items()
      }
      real_params_fast = {"groups": _group_vals, "P": pe_P, "w": pe_w}
      try:
          res_eval = evaluate_real_results(
              truss_data, real_params_fast,
              symbolic_cache=st.session_state["sym_cache"],
              materials=st.session_state["materials"],
              sections=st.session_state["sections"],
              include_self_weight=include_sw,
          )
          output_area.json(res_eval)
          st.info(f"快速代入完成，耗時 {res_eval['eval_time_ms']} ms（使用快取符號解）。")
      except Exception as e:
          st.error(f"代入失敗：{e}")
  ```

- [ ] **Step 6：手動測試**

  啟動 `streamlit run app.py`，確認：
  - 快速代入 expander 顯示各斷面表格（含 E/A/I33/G 可編輯）
  - P 倍率、w 倍率輸入框存在
  - 執行符號解後點擊「代入參數（快速）」，輸出正確且耗時顯示

- [ ] **Step 7：Commit**

  ```bash
  git add app.py
  git commit -m "feat: replace global quick-fill inputs with per-section editable table"
  ```

---

## Task 6：`app.py` — 嵌入式斷面性質計算器

**Files:**
- Modify: `app.py`（快速代入 expander 內，接在各斷面表格之後）

**Interfaces:**
- Consumes: `compute_section_props`（已在 `core/materials.py` 定義）、`SHAPE_INPUTS`、`BOX_DEFAULTS`（已在 `app.py` 上方定義）
- Produces: 計算結果寫入 `st.session_state["quickfill_overrides"][sn]`，不動 `st.session_state["sections"]`

- [ ] **Step 1：確認 `SHAPE_INPUTS`、`BOX_DEFAULTS`、Plotly 箱涵預覽程式碼的作用域**

  `SHAPE_INPUTS` 和 `BOX_DEFAULTS` 定義在約第 115–131 行（上方斷面管理區），為模組層級變數，快速代入區可直接使用，**無需重複定義**。

- [ ] **Step 2：加入嵌入式計算器 expander**

  在 Task 5 的各斷面表格渲染之後（`_qf_df` 之後）、P/w 倍率輸入之前，插入：

  ```python
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
          tw_qf = bv["t_web"]; tt_qf = bv["t_top"]; tb_qf = bv["t_bot"]

          fig_qf = go.Figure()
          half_top_qf = bt_qf / 2
          half_bot_qf = bb_qf / 2
          fig_qf.add_trace(go.Scatter(
              x=[-half_top_qf, half_top_qf, half_bot_qf, -half_bot_qf, -half_top_qf],
              y=[hv_qf, hv_qf, 0, 0, hv_qf],
              fill="toself", fillcolor="rgba(150,150,150,0.4)",
              line=dict(color="black", width=1.5), showlegend=False,
          ))
          b_box_qf = bb_qf - 2 * tw_qf
          s_cell_qf = b_box_qf / n_qf
          for i in range(n_qf):
              ix0 = -b_box_qf / 2 + i * s_cell_qf
              ix1 = ix0 + s_cell_qf
              fig_qf.add_trace(go.Scatter(
                  x=[ix0, ix1, ix1, ix0, ix0],
                  y=[tb_qf, tb_qf, hv_qf - tt_qf, hv_qf - tt_qf, tb_qf],
                  fill="toself", fillcolor="white",
                  line=dict(color="rgba(100,100,100,0.5)", width=0.8), showlegend=False,
              ))
          fig_qf.update_layout(
              xaxis=dict(visible=False, scaleanchor="y"),
              yaxis=dict(visible=False),
              margin=dict(l=10, r=10, t=10, b=10),
              height=250, plot_bgcolor="white",
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
              # 將 G 維持不變（計算器不輸出 G），只更新 A/I33
              st.success(
                  f"已填入 {_qf_calc_target}：A={_qf_props['A']:.4e} m²，"
                  f"I33={_qf_props['I33']:.4e} m⁴（G 維持原值）。"
              )
              st.session_state.pop("qf_editor", None)  # 清除 data_editor 快取以觸發重繪
              st.rerun()
          except Exception as ex:
              st.error(f"計算失敗：{ex}")
  ```

- [ ] **Step 3：手動測試**

  啟動 `streamlit run app.py`，確認：
  - 快速代入 expander 內有「從形狀計算截面參數」子 expander
  - 選 I 形或箱涵，輸入尺寸，點擊「計算並填入快速代入表格」
  - 表格對應列的 A/I33 更新，左側截面定義不受影響
  - 箱涵顯示 Plotly 預覽圖

- [ ] **Step 4：Commit**

  ```bash
  git add app.py
  git commit -m "feat: embedded section calculator in quick-fill panel"
  ```

---

## Self-Review

### Spec 覆蓋檢查

| Spec 要求 | 對應 Task |
|---|---|
| 各斷面組獨立符號 `E_s1, A_s1, ...` | Task 1、2 |
| Hadamard 交叉採樣，最多 5 組 | Task 2 |
| 快取新增 `section_groups`、`section_sym_names` | Task 1、3 |
| `real_params["groups"]` 格式 + 向後相容 | Task 3 |
| 數值路徑依 `groups` 替換各桿件 E/A/I33/G | Task 3 |
| 移除舊六個輸入框 | Task 5 |
| 各斷面可編輯表格（`num_rows="fixed"`） | Task 5 |
| `quickfill_overrides` session state | Task 5 |
| 斷面組變更時快取失效 | Task 4 |
| `run_btn` 路徑建立 `section_group_map` | Task 4 |
| 嵌入式計算器（不動 `sections`） | Task 6 |
| 計算器填入後觸發表格重繪 | Task 6 |
| 超過 5 組顯示警告 | ⚠️ 未涵蓋，補充於下方 |

**補充**：Task 5 Step 3 的 expander 開頭加入：

```python
if len(_qf_sec_names) > 5:
    st.warning(f"斷面組數 ({len(_qf_sec_names)}) 超過 5，符號擬合精度可能下降。")
```

### Placeholder 掃描

無 TBD / TODO / 未完成段落。

### 型別一致性

- `real_params["groups"][sn]["I"]` — Task 3 `subs_dict` 建構時用 `field_key = "I" if field == "I33" else field` 轉換，與 Task 5 `_group_vals` 使用 `"I": v["I33"]` 一致。
- `section_group_map` 在 Task 1 回傳 `section_sym_names`，Task 3 以 `sp.Symbol(sym_str)` 重建，型別正確。
- `quickfill_overrides` 鍵為 `"E", "A", "I33", "G"`（Task 5），Task 5 Step 5 轉換為 `"I": v["I33"]` 再傳入 `real_params["groups"]`，一致。
