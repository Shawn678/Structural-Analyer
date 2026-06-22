# Structural Analyzer 結構分析工具

基於 Python 與 Streamlit 的**二維框架結構分析工具**，支援符號代數推導、多截面材料系統、剛性連桿靜力凝縮，以及纜索面板參數化精靈，定位為橋梁概念設計探索平台。

## 功能概覽

### 核心分析
- **3 自由度框架分析**：每節點含 ux、uy、θ，支援軸力、剪力、彎矩計算
- **符號代數分析**：以 SymPy 自動推導位移公式，保留各參數代數形式，適合敏感度分析
- **各截面獨立符號**：每個截面群組擁有獨立符號（`E_s1, A_s1, I_s1...`），混合截面模型精確求解

### 截面與材料系統
- **三層結構**：Material → Section → Element，類似 SAP2000 的管理方式
- **截面形狀計算器**：支援矩形實心、圓形實心、矩形管、圓管、I 形、**箱涵（Box Girder）**
- **箱涵截面**：單箱多室（1～5 室）、Bredt 多室薄壁扭轉常數、Plotly 即時斷面預覽
- **自重計算**：依材料密度與截面積自動疊加均佈自重

### 剛性連桿（Rigid Link）
- 宣告 master-slave 節點運動拘束關係，表達截面偏心錨點
- 靜力凝縮消去 slave 自由度，**不插入虛擬高剛度構件**（避免數值病態）
- 自動整合至符號分析流程

### 纜索面板精靈（Cable Face Wizard）
- 針對脊背橋（Extradosed Bridge）索面建模
- 輸入塔頂節點、間距、偏心距等參數，自動生成索、偏心錨點及 Rigid Link
- 支援前側 / 背側索面，所有生成元素帶群組標籤可整組刪除

### 快速代入（Parametric Quick-Fill）
- 符號解完成後，各截面參數可獨立調整並即時代入，無需重跑符號分析
- 嵌入式截面計算器，計算結果可直接填入快速代入表格
- 載重倍率（P / w）獨立調整

### 專案管理
- **JSON 存取**：完整儲存節點、桿件、截面、材料、剛性連桿等所有狀態
- **TXT 快取匯出**：符號解結果快取，避免重複計算

## 快速啟動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 操作流程

1. **材料定義**：在材料表設定 E、G、密度（內建鋼材與混凝土預設值）
2. **截面定義**：選擇截面形狀，計算 A、I33、I22、J，指派材料
3. **節點**：在 Nodes 表格定義 x、y 座標
4. **桿件**：在 Elements 表格指派截面，支援個別參數覆蓋
5. **支承**：在 Supports 表格勾選位移約束（ux, uy, theta）
6. **載重**：在 Loads 表格輸入集中力或彎矩；均佈載重另有獨立表格
7. **剛性連桿**（選用）：在 Rigid Links 表格定義 master-slave 節點對
8. **纜索精靈**（選用）：展開 Cable Face Wizard，填寫索面參數後一鍵生成
9. **執行分析**：符號解或數值解，結果顯示於右側面板
10. **快速代入**：符號解完成後，調整各截面參數即時查看結果

## 專案結構

```
app.py                        # Streamlit 主程式與 UI
main.py                       # CLI 入口（測試用）
requirements.txt
core/
  symbolic.py                 # 符號分析引擎（SymPy，含各截面獨立符號）
  rigid_link.py               # 剛性連桿靜力凝縮
  cable_wizard.py             # 纜索面板參數化產生器
  parametric_evaluator.py     # 符號結果數值代入
  materials.py                # 截面形狀計算（含箱涵 Bredt 公式）
test/
  truss_test.py
  beam_test.py
  portal_frame_test.py
```

## 依賴套件

| 套件 | 版本 |
|------|------|
| streamlit | 1.58.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| sympy | 1.14.0 |
| plotly | 6.8.0 |
| matplotlib | 3.11.0 |

## 已知限制

- 符號分析在節點數超過 5 個或截面組數超過 5 組時，計算複雜度大幅上升
- CSV 上傳自動填充目前僅在 `main.py` (CLI) 完整支援
- 索的預拉力設定、橋塔幾何參數化尚未實作
