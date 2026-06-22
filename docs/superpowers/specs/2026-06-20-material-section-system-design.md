# Material / Section 管理系統設計文件

**日期：** 2026-06-20  
**狀態：** 待實作

---

## 背景與目標

目前程式每根桿件在 `elements_df` 直接填入 E、G、A、I33、I22、J，參數管理分散。
本次擴充目標：整合成三層結構（Material → Section → Element），讓使用者像市售軟體（SAP2000）一樣統一管理材料與截面定義，再指派給桿件。

符號解定位為**概念設計探索工具**（保留各參數的代數形式，便於敏感度分析），純數值解為未來另開模組的工作。

---

## 資料模型

### Material

| 欄位 | 型別 | 說明 |
|------|------|------|
| name | str | 唯一 key，使用者自訂 |
| E | float | 彈性模量 (Pa) |
| G | float | 剪切模量 (Pa) |
| density | float | 密度 (kg/m³) |
| ν | 唯讀 | 自動計算：`E / (2G) - 1` |

內建預設（可編輯/刪除）：
- **鋼材**：E=200e9, G=77e9, ρ=7850
- **混凝土**：E=30e9, G=12.5e9, ρ=2400

### Section

| 欄位 | 型別 | 說明 |
|------|------|------|
| name | str | 唯一 key，使用者自訂（如「主樑斷面」、「斜索」） |
| material | str | 下拉選單，選已定義 Material.name |
| shape | str | 截面形狀：Custom / 矩形實心 / 圓形實心 / 矩形管 / 圓管 / I形 |
| *形狀輸入參數* | float | 依 shape 顯示對應輸入欄位（見截面計算公式） |
| A | float | 截面積 (m²)，Custom 時直接填，其餘自動計算 |
| I33 | float | 強軸慣性矩 (m⁴) |
| I22 | float | 弱軸慣性矩 (m⁴) |
| J | float | 抗扭常數 (m⁴) |

### Element（調整後）

原有的 E、G、A、I33、I22、J 直接輸入欄位保留，但新增：

| 欄位 | 說明 |
|------|------|
| section | 下拉選單（SelectboxColumn），選已定義 Section.name，選後自動帶入對應 E/G/A/I33/I22/J |
| status | 唯讀文字欄，無 override 時空白，有任一參數被手動修改時顯示「【修改】」 |

有 override 的桿件整列背景變**淡橘色**（`#FFE0B2`）；
改回與 section 定義相同數值時，標色與「【修改】」自動消除。

---

## 截面幾何計算公式

### 矩形實心（輸入：b, h，h 為強軸方向高度）

```
A   = b × h
I33 = b × h³ / 12
I22 = h × b³ / 12
J   ≈ a × b³ × [1/3 - 0.21(b/a)(1 - b⁴/(12a⁴))]   （a = max(b,h), b = min(b,h)）
```
> ⚠️ J 為 Timoshenko 近似公式，適用矩形實心截面，精確值請查結構手冊。

### 圓形實心（輸入：d）

```
A   = π d² / 4
I33 = π d⁴ / 64
I22 = π d⁴ / 64
J   = π d⁴ / 32
```

### 矩形管（輸入：b, h, t，t 為壁厚）

```
A   = b×h - (b-2t)×(h-2t)
I33 = [b×h³ - (b-2t)×(h-2t)³] / 12
I22 = [h×b³ - (h-2t)×(b-2t)³] / 12
J   ≈ 2t × (b-t)² × (h-t)² / (b+h-2t)              （薄壁閉口近似）
```
> ⚠️ J 為薄壁閉口截面近似公式。

### 圓管（輸入：d, t，t 為壁厚）

```
A   = π [d² - (d-2t)²] / 4
I33 = π [d⁴ - (d-2t)⁴] / 64
I22 = π [d⁴ - (d-2t)⁴] / 64
J   = π [d⁴ - (d-2t)⁴] / 32
```

### I 形截面（輸入：H, bf, tf, tw）

```
bw  = H - 2×tf                                       （自動推算）
A   = 2×bf×tf + bw×tw
I33 = bf×H³/12 - (bf-tw)×bw³/12
I22 = 2×(tf×bf³/12) + bw×tw³/12
J   ≈ (1/3) × (2×bf×tf³ + bw×tw³)                   （薄壁開口近似）
```
> ⚠️ J 為薄壁開口截面近似公式，I 形鋼實際 J 值請查型鋼手冊。

---

## 符號解調整

### 符號變數命名策略

以 Section 的後端 index（0-based）為單位分配符號，前端顯示名稱（如「主樑斷面」）在傳入後端前轉換為 index。同截面的桿件共用同一組符號：

```python
# Section index=0（前端名稱「主樑斷面」）
E_s0, G_s0, A_s0, I33_s0, I22_s0, J_s0 = sp.symbols('E_s0 G_s0 A_s0 I33_s0 I22_s0 J_s0')

# Section index=1（前端名稱「斜柱」）
E_s1, G_s1, A_s1, I33_s1, I22_s1, J_s1 = sp.symbols('E_s1 G_s1 A_s1 I33_s1 I22_s1 J_s1')
```

有 local override 的桿件，被 override 的欄位改用獨立符號（以桿件 id 區分）：

```python
# 桿件 id=3 的 I33 被手動修改
I33_e3 = sp.Symbol('I33_e3')
```

### Fingerprint 不變

仍只考慮幾何拓樸（節點座標、桿件連接、支承條件）。
截面/材料改變走**快速代入**通道，不需重跑符號解。

### 快速代入擴充

`evaluate_real_results` 的 `real_params` 由現有的全域 scalar 擴充為 per-section dict：

```python
real_params = {
    "sections": {
        0: {"E": 200e9, "G": 77e9, "A": 0.01, "I33": 1e-4, "I22": 1e-5, "J": 1e-5},  # 主樑斷面
        1: {"E": 200e9, "G": 77e9, "A": 0.005, "I33": 5e-5, "I22": 5e-6, "J": 5e-6}, # 斜柱
    },
    "overrides": {
        3: {"I33": 2e-4},   # 桿件 id=3 的 local override
    }
}
```

---

## 自重功能

### UI

右側結果區（分析按鈕附近）新增：

```
☐ 含自重（Self-Weight）
```

### 計算邏輯

```python
if include_self_weight:
    for elem in elements:
        sec = sections[elem['section']]
        mat = materials[sec['material']]
        A_eff = elem.get('A', sec['A'])   # 尊重 local override
        w_self = mat['density'] * A_eff * 9.81  # N/m，向下
        # 疊加到 element_loads（-Z 方向，符合現有 w 欄位慣例）
        elem_loads[elem['id']] = elem_loads.get(elem['id'], 0.0) + (-w_self)
```

自重只在分析時動態疊加，不寫入使用者的 element_loads 表格。

---

## UI 佈局（左側面板）

```
[材料定義表]                ← 新增
  • st.data_editor
  • 唯讀欄：ν（自動計算）

[截面定義表]                ← 新增
  • st.data_editor（Section 主表）
  • 展開區塊「從形狀計算」：下拉選形狀 → 輸入幾何參數 → 帶入 A/I/J
  • st.caption 標註近似公式警示

[節點表]                    ← 不變
[桿件表]                    ← 新增 section 下拉、status 欄、整列標色
[支承表]                    ← 不變
[載重表]                    ← 不變
[桿件均佈載重表]            ← 不變
[桿件集中載重表]            ← 不變
```

右側面板：「含自重」checkbox 置於分析按鈕上方。

---

## TXT 快取格式擴充

在現有 `[FINGERPRINT]`、`[SYMBOLIC]` 區塊前新增：

```
[MATERIALS]
name,E,G,density
鋼材,200000000000,77000000000,7850
混凝土,30000000000,12500000000,2400

[SECTIONS]
name,material,A,I33,I22,J
主樑斷面,鋼材,0.01,0.0001,0.00001,0.00001
斜柱,鋼材,0.005,0.00005,0.000005,0.000005

[FINGERPRINT]
...（現有不變）

[SYMBOLIC]
...（現有不變）
```

匯入時一併還原 materials/sections 定義；fingerprint 驗證邏輯不變。

---

## 不在本次範圍

- 純數值解模組（未來另開 `core/numerical_solver.py`）
- 截面形狀資料庫（H 型鋼、T 型鋼等型鋼手冊查詢）
- 3D 視覺化截面渲染
