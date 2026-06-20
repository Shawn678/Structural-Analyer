# 各斷面獨立符號快速代入設計文件

**日期：** 2026-06-21  
**狀態：** 已核准

---

## 概述

升級符號分析與快速代入 UI，讓每個唯一斷面組擁有獨立符號（`E_s1, A_s1, I_s1, G_s1`、`E_s2, ...`），取代現有的單一全局符號 `E, A, I, G`。快速代入面板改為各斷面獨立的可編輯表格，搭配 P/w 載重倍率輸入。

---

## 背景與動機

現有的快速代入區塊有六個全局輸入框（`E, A, I, G, P倍率, w倍率`）。在所有桿件共用同一材料時尚可使用，但引入各桿件獨立斷面/材料系統後，`evaluate_real_results` 在 `use_per_elem=True` 時完全忽略全局 E/A/I/G 輸入，使這些輸入框實際失效。位移公式同樣只用單一全局符號，在混合斷面下只是近似值。

定位：初步設計工具，斷面組數上限為 5 組。

---

## 架構

三個層次皆需修改：

```
app.py（UI 層）
  └─ 建立 section_group_map + group_vals
       └─ symbolic.py（run_symbolic_analysis）
            └─ 各斷面組獨立符號、Hadamard 交叉採樣
       └─ parametric_evaluator.py（evaluate_real_results）
            └─ subs_dict 以各斷面組符號為鍵
```

---

## 一、`symbolic.py` 修改

### 1.1 新增參數：`section_group_map`

```python
run_symbolic_analysis(truss_data, section_group_map=None)
```

`section_group_map` 將每個唯一斷面名稱對應至一組 SymPy 符號：

```python
{
  "H300": {"E": Symbol("E_s1"), "A": Symbol("A_s1"),
            "I33": Symbol("I_s1"), "I22": Symbol("I_s1"), "G": Symbol("G_s1")},
  "I200": {"E": Symbol("E_s2"), ...},
}
```

由 `app.py` 在呼叫求解器前根據 `st.session_state["sections"]` 建立。斷面組數 `G = len(section_group_map)`。

未指定斷面的桿件回退使用全局 `E, A, I, G` 符號（維持現有行為）。

### 1.2 各桿件符號指派

在 `run_symbolic_analysis` 建立 `elements_info` 後，對每根桿件查詢其 `section` 欄位，從 `section_group_map` 取得對應符號，填入 `elem_E_syms[k]`、`elem_A_syms[k]` 等列表。無斷面名稱者沿用全局符號。

### 1.3 交叉採樣策略（Hadamard）

**現行做法**：所有桿件同步乘以同一個 `scale` 因子。  
**新做法**：每個斷面組各自取獨立的 scale，在採樣矩陣中各組互相獨立變化。

```
SCALES = [1.0, 5.0, 25.0, 100.0, 500.0]
n_samples = max(len(SCALES), G * 4 + 4)   # 例如 G=5 → 24 個採樣點
```

每個採樣列 `s`：
- 每個斷面組 `g` 獨立從 `SCALES` 中取一個 scale 值
- 該組所有桿件的材料參數乘上 `scale_g`
- 無斷面的桿件使用固定 scale（各組平均值或 1.0）

組合序列以 `np.random.default_rng(seed=42)` 確定性生成，保證每次結果一致。`G ≤ 5` 且 `len(SCALES) = 5`，維度充足。

### 1.4 基底與符號表達式

`_build_basis_row` 已支援 `elem_E_list` 等參數，**無需修改**。  
`_fit_and_symbolize` 已從 `sym_vars` 讀取 `elem_E_syms`，**無需修改**。

升級後的位移公式範例：
```
uz = c1·P·L_1³/(E_s1·I_s1) + c2·P·L_2³/(E_s2·I_s2) + ...
```

### 1.5 快取相容性

快取新增兩個欄位：

- `section_groups`：斷面組名稱的排序列表，例如 `["H300", "I200"]`。用於快取失效比對。
- `section_sym_names`：`{斷面名稱: {"E": "E_s1", "A": "A_s1", "I33": "I_s1", "G": "G_s1"}, ...}`——求解時使用的 SymPy 符號名稱字串。載入快取時，`evaluate_real_results` 透過 `sp.Symbol(name)` 重建符號物件，再建立 `subs_dict`。

若當前桿件的唯一斷面名稱集合與 `cache["section_groups"]` 不符，快取失效，強制重新執行完整分析。

---

## 二、`parametric_evaluator.py` 修改

### 2.1 `real_params` 格式升級

**舊格式**：`{"E": 200e9, "A": 0.01, "I": 1e-4, "G": 77e9, "P": 1.0, "w": 0.0}`

**新格式**：
```python
{
  "groups": {
    "H300": {"E": 210e9, "A": 0.015, "I": 2.4e-4, "G": 80e9},
    "I200": {"E": 206e9, "A": 0.010, "I": 1.2e-4, "G": 77e9},
  },
  "P": 1.0,
  "w": 0.0,
}
```

**向後相容**：若 `"groups"` 鍵不存在，回退至舊的平坦格式（全局 E/A/I/G 套用所有桿件），保留 `run_btn` 路徑現有行為不受影響。

### 2.2 `subs_dict` 建構

```python
if "groups" in real_params:
    for sec_name, vals in real_params["groups"].items():
        sym = {k: sp.Symbol(v) for k, v in cache["section_sym_names"][sec_name].items()}
        subs_dict[sym["E"]]   = float(vals["E"])
        subs_dict[sym["A"]]   = float(vals["A"])
        subs_dict[sym["I33"]] = float(vals["I"])
        subs_dict[sym["G"]]   = float(vals["G"])
else:
    # 舊路徑
    subs_dict[E_s] = float(real_params.get("E", 200e9))
    ...
```

### 2.3 數值路徑（桿件內力）

快速代入的數值求解路徑（內力、反力）：當使用者在快速代入表格中覆蓋參數時，跳過 `expand_truss_data`，改為直接依各桿件的 `section` 名稱從 `real_params["groups"]` 取值，填入 `td_num` 各桿件的 E/A/I33/G，再呼叫 `run_numerical_analysis`。載重倍率 P/w 套用方式不變。

---

## 三、`app.py` UI 修改

### 3.1 移除舊輸入框

移除 `⚡ 快速代入參數` expander 內的六個 `st.number_input`（`pe_E`, `pe_A`, `pe_I`, `pe_G`, `pe_P`, `pe_w`）及其兩欄佈局。

### 3.2 新增各斷面參數表格

```
⚡ 快速代入參數（可展開）
┌──────────────────────────────────────────────────────┐
│ 斷面名稱 │ E (Pa)  │ A (m²)  │ I33 (m⁴) │ G (Pa)  │
│ H300    │ 200e9  │ 0.015   │ 2.4e-4   │ 77e9   │  ← 可編輯
│ I200    │ 206e9  │ 0.010   │ 1.2e-4   │ 79e9   │  ← 可編輯
├──────────────────────────────────────────────────────┤
│  P 倍率 [1.0]             w 倍率 [0.0]               │
│  [⚡ 代入參數（快速）]  ← 快取無效時停用              │
└──────────────────────────────────────────────────────┘
```

預設值從 `st.session_state["sections"]` + `materials` 讀取（邏輯同 `_sec_val`）。表格使用 `st.data_editor`，`num_rows="fixed"`（列數 = 唯一斷面數，不可新增/刪除）。`斷面名稱` 欄設為唯讀；E/A/I33/G 欄可編輯。

**Session state 鍵**：`"quickfill_overrides"` — `{斷面名稱: {E, A, I33, G}}`，跨 rerun 保留使用者修改值。

重置規則：
- 首次渲染（鍵不存在）：從 `sections`+`materials` 初始化。
- 當前斷面名稱排序列表與 `st.session_state.get("quickfill_sec_key")` 不同時：重新初始化並更新 `quickfill_sec_key`（涵蓋新增/刪除/重命名斷面的情況）。
- 使用者在表格內的編輯直接透過 `st.data_editor` 回傳值寫回 `quickfill_overrides`。

### 3.3 快取失效邏輯

快取有效性檢查新增條件：除幾何指紋外，額外比對 `sorted(elements 中的唯一斷面名稱)` 與 `cache["section_groups"]`。不符時 `cache_valid = False`，快速代入按鈕停用，提示「斷面組已變更，請重新執行符號解」。

### 3.4 按鈕處理器

```python
if fast_btn:
    group_vals = {
        row["斷面名稱"]: {"E": row["E"], "A": row["A"], "I": row["I33"], "G": row["G"]}
        for _, row in qf_df.iterrows()
    }
    real_params = {"groups": group_vals, "P": pe_P, "w": pe_w}
    res_eval = evaluate_real_results(
        truss_data, real_params,
        symbolic_cache=st.session_state["sym_cache"],
        ...
    )
```

### 3.5 `run_btn` 路徑（符號解）

使用者點擊「執行分析（符號解）」時，`app.py` 從當前斷面建立 `section_group_map` 並傳入 `run_symbolic_analysis`。快取儲存 `section_sym_names`（符號名稱字串形式），供後續快速代入時重建 SymPy 符號物件。

---

## 四、錯誤處理

| 情境 | 處理方式 |
|---|---|
| 代入時 `section_group_map` 中找不到該斷面 | 回退使用全局 E/A/I/G；記錄警告 |
| `n_groups > 5` | 在快速代入面板顯示 `st.warning`；仍允許分析，但提示精度可能下降 |
| `lstsq` 相對誤差 > 1e-3 | 在輸出中標記 `[近似]`；現有 `any_invalid` 旗標已處理此情況 |
| 快取 `section_groups` 不符 | 停用快速代入按鈕，提示重新執行符號解 |

---

## 五、不在範圍內

- 在快速代入面板修改參數**不會**更動 `st.session_state["sections"]`，僅影響此次快速代入的計算。
- 不支援各桿件獨立符號覆蓋，僅支援各斷面組層級。
- 快速代入面板內不提供新增斷面的 UI。
