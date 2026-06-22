# 設計文件：3D 建模擴充 — Rigid Link 與索面參數化精靈

**日期：** 2026-06-22  
**背景：** 現有模型將所有構件視為一維線段，節點只有中心線座標，無法表達截面的物理寬度。要分析脊背橋時，橋塔柱腳和斜張索錨點需要鎖定在箱涵頂板兩側緣（偏心位置），現有機制無法處理。

---

## 1. 核心概念：Rigid Link

### 定義
Rigid Link 是一個**運動拘束關係**，不是真實構件。它宣告一個「slave 節點」的 6 個自由度完全由「master 節點」決定：

```
u_slave = u_master + θ_master × d
```

其中 `d` 是從 master 到 slave 的偏心向量（由兩節點座標差自動計算）。

### 與剛接/鉸接的區別
- **Rigid Link**：描述構件本身不變形（傳遞幾何關係）
- **剛接 / 鉸接**：描述接頭處轉角是否連續（傳不傳彎矩）

兩者獨立設定，不衝突。例如：索兩端設鉸接（不傳彎矩），但索端點透過 Rigid Link 附屬在主梁上（正確傳遞偏心幾何）。

### 數值實作
求解前將 slave 自由度用線性拘束方程式消去，等效成 master 節點上的額外剛度貢獻，**不插入虛擬高剛度構件**（避免數值病態）。

---

## 2. 新增資料結構

### 2a. Rigid Link 表格（`rigid_links`）

```json
{
  "rigid_links": [
    {
      "id": "rl_1",
      "master": "N1",
      "slave": "N4",
      "group": "gen:tower1_left"
    }
  ]
}
```

- `master`：主節點 ID（運動主導方）
- `slave`：從屬節點 ID（運動被決定方）
- 偏心向量 `d` 由兩節點座標差自動計算，不需手動填入
- `group`：可選，標記來源群組

### 2b. 節點與構件的群組標籤（`group` 欄位）

所有由參數化精靈生成的節點、構件、Rigid Link，均帶有 `group` 欄位，格式為 `gen:<名稱>`。手動定義的元素 `group` 為空。

---

## 3. 參數化索面精靈

### 適用場景
脊背橋（Extradosed Bridge）的索面建模。一次填寫一組索面參數，自動生成該索面所有相關元素。

### 輸入參數

| 參數 | 型別 | 說明 |
|------|------|------|
| `group_name` | string | 群組名稱，例如 `tower1_left` |
| `tower_node` | node ID | 塔頂 master 節點 |
| `tower_offset_start` | float (m) | 從塔頂的第一根索偏移量（負值往下，例如 -0.5） |
| `tower_spacing` | float (m) | 塔側相鄰索間距（負值往下，例如 -0.5） |
| `deck_x_start` | float (m) | 橋面側第一根索的 x 座標（絕對座標） |
| `deck_spacing` | float (m) | 橋面側相鄰索間距（正值往跨中，負值往橋台） |
| `n_cables` | int | 根數 |
| `eccentricity_y` | float (m) | 偏心距，正值為右側，負值為左側 |
| `deck_z` | float (m) | 主梁中心線 z 座標（通常為 0） |

### 生成元素（以 `n_cables=7` 為例）

1. **主梁中心線節點**（7 個）  
   座標：`(deck_x_start + i*deck_spacing, 0, deck_z)`，i = 0..6  
   若該座標已存在節點則複用，不重複生成。

2. **橋面偏心錨點**（7 個，slave 節點）  
   座標：`(deck_x_start + i*deck_spacing, eccentricity_y, deck_z)`

3. **塔側偏心錨點**（7 個，slave 節點）  
   座標：塔頂節點座標 + `(0, eccentricity_y, tower_offset_start + i*tower_spacing)`

4. **Rigid Link**（14 個）  
   橋面：master = 主梁中心線節點，slave = 橋面偏心錨點  
   塔側：master = `tower_node`，slave = 塔側偏心錨點

5. **索構件**（7 根）  
   i 端 = 塔側偏心錨點，j 端 = 橋面偏心錨點  
   預設 `pin_i=True, pin_j=True`（純軸力構件）

所有生成元素帶 `group = "gen:<group_name>"`。

### 前側 / 背側處理
`deck_spacing` 正值 = 往跨中（前側），負值 = 往橋台（背側）。不另設模式。

---

## 4. UI 設計

### Rigid Link 表格
- 位置：現有節點/構件表格旁，獨立分頁或展開區
- 欄位：`id`、`master`、`slave`、`group`（唯讀顯示偏心向量供參考）
- 顏色：`group` 非空的列以淡色底色標示為精靈生成

### 索面精靈面板
- 位置：側邊欄展開區（`st.expander`）
- 填寫參數後按「生成索面」按鈕
- 生成前檢查：`tower_node` 是否存在、`deck_x_start` 是否在主梁範圍內
- 生成後：元素寫入 session_state，表格即時更新並以顏色標示群組

### 群組管理
- 每個群組可整組刪除（清除該 `group` 的所有節點、構件、Rigid Link）
- 不提供整組鎖定（允許手動微調）

---

## 5. 分析流程修改（`core/symbolic.py`）

1. 正常組裝所有節點與構件的全域剛度矩陣 K
2. 讀取 `rigid_links`，對每個 slave 節點建立拘束方程式
3. 用靜態凝縮（static condensation）消去 slave 自由度
4. 求解自由度
5. 反推 slave 節點位移與反力

Rigid Link 不影響截面性質計算，只影響自由度組裝。

---

## 6. 不在本次範圍內

- 索的預拉力設定（後續再擴充）
- 截面自動生成偏心節點（非本次設計）
- 橋塔幾何參數化（橋塔仍手動定義構件）
