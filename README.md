# Truss Analyzer 結構分析工具

這是一個基於 Python 與 Streamlit 開發的二維框架/桁架結構分析工具，支援數值計算與符號代數分析。

## 🚀 啟動與操作流程

### 1. 環境準備

請確保安裝了 Python 3.8+，並安裝必要依賴庫：

```bash
pip install streamlit pandas numpy scipy sympy
```

### 2. 啟動程式

於終端機進入專案根目錄，執行：

```bash
streamlit run app.py
```

### 3. 操作步驟

1. **節點編輯**：在左側面板「Nodes」表格定義坐標。
2. **桿件定義**：在「Elements」表格輸入材料屬性（E, A, I）及起訖節點。
3. **邊界條件**：在「Supports」表格勾選位移約束（ux, uy, theta）。
4. **施加載重**：在「Loads」表格輸入外力或彎矩。
5. **執行分析**：點擊「執行分析」按鈕，結果將以 JSON 格式顯示於右側。

## 📝 更新與維護日誌

### v2.0.0 (當前版本)

- **架構遷移**：移除 Flask、HTML、JavaScript 舊架構，全面採用 Streamlit 實現宣告式 UI。
- **維度提升**：由原先的 2 自由度 (ux, uy) 桁架分析升級為 **3 自由度 (ux, uy, θ)** 框架分析，支援彎矩 (Moment) 與轉角 (Rotation) 計算。
- **數據交互**：引入 `st.data_editor` 實現即時數據增刪，取代原有的動態表格。

### v1.0.0

- 基礎桁架分析功能，使用 Flask 與原生 Canvas 預覽。

## ⚠️ 現行功能限制

1. **視覺化限制**：目前介面右側的「結構預覽」區域僅為佔位符。原本的 HTML Canvas 繪圖邏輯尚未移植。建議後續使用 `Plotly` 或 `Matplotlib` 進行動態繪圖。
2. **符號運算規模**：代數分析 (Symbolic Mode) 在節點數量超過 5 個時，計算複雜度會大幅增加，可能導致響應緩慢。
3. **檔案管理**：目前 CSV 載入功能主要保留在 `main.py` (CLI)，`app.py` 尚未完全整合 CSV 上傳自動填充功能。

---

**維護者備註**：

- 原有的 `static/` 與 `templates/` 資料夾已無功能，可隨時刪除。
- 核心分析邏輯位於 `core/analyzer.py` 中。
