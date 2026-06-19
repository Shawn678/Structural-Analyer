# Parametric Evaluator 設計文件

**日期：** 2026-06-19  
**狀態：** 已審核

---

## 背景與目標

`symbolic.py` 的 `run_symbolic_analysis` 採用多點採樣 + 力學基底 lstsq 擬合，對大型結構耗時可達數十秒。當使用者**僅修改材料參數（E, A, I, G）或載重大小（P, w）**而幾何拓樸與支承條件不變時，代數解完全可以重用——只需將新的數值代入已有的符號公式即可，速度接近瞬時。

本功能的目標是：
1. 提供 `evaluate_real_results` 讓調參變得極快。
2. 用 `st.session_state` 在同一 session 內快取代數解。
3. 支援匯出/匯入 TXT，讓快取跨 session 持久化。

---

## 架構

```
core/
  symbolic.py               (現有，不修改)
  parametric_evaluator.py   (新增)
app.py                      (修改：快取邏輯 + 匯出/匯入 UI)
```

### 流程圖

```
使用者點「執行分析（符號解）」
  └─→ run_symbolic_analysis(truss_data)
      └─→ 結果存入 st.session_state['sym_cache']
          └─→ 可匯出 TXT

使用者修改材料/載重 → 點「代入參數（快速）」
  └─→ build_geometry_fingerprint(truss_data) 比對快取指紋
      ├─ 一致 → evaluate_real_results(truss_data, real_params, cache)  ← 毫秒級
      └─ 不一致 → 提示「幾何/支承已變更，請先執行完整分析」

使用者匯入 TXT
  └─→ import_cache_from_txt(filepath)
      ├─ 指紋比對通過 → 載入快取，啟用「代入參數」按鈕
      └─ 指紋不符 → 顯示具體差異說明（桿件數/長度/支承）
```

---

## `core/parametric_evaluator.py` API

### `build_geometry_fingerprint(truss_data) -> dict`

計算幾何 + 支承的結構指紋，用於快取有效性比對。

**包含：**
- 桿件數量
- 每根桿件長度（四捨五入至 1e-6 m，避免浮點抖動）
- 每根桿件的 i/j 節點連接對
- 支承節點 ID + 約束方向（ux/uy/uz/rx/ry/rz）+ 彈簧剛度（kx/ky/kt）

**不包含：** 材料參數（E/A/I/G）、載重數值（P/w）

---

### `evaluate_real_results(truss_data, real_params, symbolic_cache) -> dict`

主函數。將實際參數代入代數解，回傳數值結果。

**參數：**
```python
real_params = {
    "E": 2e11,    # 彈性模量 (Pa)
    "A": 0.01,    # 截面積 (m²)
    "I": 1e-4,    # 慣性矩 (m⁴)
    "G": 77e9,    # 剪切模量 (Pa)
    "P": 10000,   # 集中載重倍率
    "w": 5000,    # 均佈載重倍率
}
```

**內部流程：**
1. 若 `symbolic_cache` 為 None，呼叫 `run_symbolic_analysis` 並快取結果。
2. 建立 `subs_dict`：將 E, A, I, G, P, w 及 L_1, L_2... 填入對應數值。
   - L_k 從 `symbolic_cache['elem_Ls']` 取得實際幾何長度。
3. 對每個公式字串執行 `sp.sympify(expr_str).subs(subs_dict)`，轉 `float`。
4. 回傳結構：

```python
{
  "node_displacements": [
    {
      "node_id": 1,
      "ux": {"formula": "P·L_1^3/(48·E·I)", "value": 0.00125},
      "uy": {"formula": "0", "value": 0.0},
      ...
    }
  ],
  "element_forces": [...],   # 同結構，含 formula + value
  "support_reactions": [...],
  "cache_used": True,        # 是否使用了快取（告知前端）
  "eval_time_ms": 12,        # 代入計算耗時
}
```

---

### `export_cache_to_txt(symbolic_cache, filepath) -> None`

將快取序列化為人類可讀 TXT。

**TXT 格式：**
```
# TRUSS SYMBOLIC CACHE v1
# Generated: 2026-06-19T10:30:00
[FINGERPRINT]
n_elements=3
elem_lengths=6.000000,6.000000,4.000000
connections=1-2,2-3,4-1
supports=4:ux,uy,uz,rx,ry,rz|2:uy,uz,rx,rz|3:uy,uz,rx,rz
[FORMULAS]
node_1_ux=P*L_1**3/(48*E*I)
node_1_uy=0
node_1_uz=0
...
elem_1_N=P/2
elem_1_V2=-P/2
elem_1_M3=P*L_1/4
...
react_4_Rx=P/2
react_4_Ry=0
...
[END]
```

- 公式使用 SymPy 標準字串（`**` 次方、`*` 乘），匯入時直接 `sympify`。
- `[FINGERPRINT]` 區塊供指紋比對使用。

---

### `import_cache_from_txt(filepath, truss_data) -> dict | None`

讀取 TXT，重建快取並驗證指紋。

**流程：**
1. 解析 `[FINGERPRINT]` 區塊，重建指紋 dict。
2. 對當前 `truss_data` 呼叫 `build_geometry_fingerprint`，比對兩者。
3. 不一致時回傳 `{"error": "桿件數量不符：快取 3 根 vs 當前 4 根"}` 類型的錯誤訊息。
4. 一致時解析 `[FORMULAS]` 區塊，用 `sp.sympify` 重建所有 SymPy Expression。
5. 回傳與 `run_symbolic_analysis` 相同結構的 cache dict（加上 `elem_Ls` 欄位）。

---

## `app.py` UI 修改

### 左側面板底部（按鈕區）

```
[ 執行分析（符號解）]          ← 原有，完整重算
[ ⚡ 代入參數（快速）]         ← 新增；快取無效時顯示 disabled + tooltip
```

### 右側面板（分析結果下方）

```
── 代數解快取管理 ──
狀態：✅ 快取有效（3 根桿件，上次分析：10:30）
      或 ⚠️ 尚無快取，請先執行完整分析

[ 匯出代數解 TXT ]             ← 有快取才啟用，下載按鈕
[ 匯入代數解 TXT ]             ← st.file_uploader，上傳後自動驗證
  └─ 驗證結果：✅ 指紋一致，已載入  /  ❌ 桿件數不符（快取3根 vs 當前4根）
```

### 材料參數快速代入區（「代入參數」按鈕觸發）

顯示獨立的數字輸入欄，與桿件表格中的 E/A/I 分離，代表「全域統一代入值」：

```
E (Pa):  [200e9]   A (m²): [0.01]
I (m⁴):  [1e-4]   G (Pa): [77e9]
P 倍率:  [1.0]     w 倍率: [1.0]
```

> **說明：** 此處數值為代入符號公式的實際參數，不影響結構幾何與支承設定。若需修改幾何或支承，請使用「執行分析（符號解）」重新分析。

---

## 快取資料結構（`st.session_state['sym_cache']`）

```python
{
    "fingerprint": { ... },          # build_geometry_fingerprint 的輸出
    "elem_Ls": [6.0, 6.0, 4.0],     # 桿件長度，供 L_k 代入
    "raw_result": { ... },           # run_symbolic_analysis 的完整輸出
    "timestamp": "2026-06-19T10:30", # 供 UI 顯示
}
```

---

## 邊界條件與錯誤處理

| 情況 | 處理方式 |
|---|---|
| `sympify` 解析公式字串失敗 | 該欄位 value 回傳 `None`，formula 保留原字串，UI 顯示警告 |
| 代入後出現除以零（如 E=0） | `float()` 拋出例外，統一捕捉回傳 `{"error": "代入參數無效：E 不可為零"}` |
| TXT 格式損壞（缺少區塊） | `import_cache_from_txt` 回傳 `{"error": "TXT 格式無效，缺少 [FORMULAS] 區塊"}` |
| 指紋比對部分不符 | 回傳最具體的差異描述，列出第一個不符的欄位 |
| 快取存在但使用者強制重算 | 點「執行分析（符號解）」直接覆蓋 session_state，不詢問確認 |
