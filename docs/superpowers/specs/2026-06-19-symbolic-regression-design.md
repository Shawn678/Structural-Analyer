# 設計文件：多點採樣符號回歸（Symbolic Regression via Basis Fitting）

**日期**：2026-06-19  
**檔案**：`core/symbolic.py`  
**方案**：方案 B — 完整力學基底 + 稀疏篩選

---

## 目標

將現有的純數值分析結果，重構為包含 `E, A, I, L_k, P, w` 的全代數符號公式輸出。使用多點數值採樣與結構力學基底擬合，在不 hard-code 任何特定公式的前提下，於 runtime 動態反推每個 DOF 的符號解。

---

## 架構

### 假設與約束

- 結構拓樸（節點數、桿件數、支承位置）每次都不同，必須在 runtime 動態處理。
- 所有桿件共用一組符號材料參數 `E, A, I`（選項 B）。
- 每根桿件的長度 `L_k` 從節點座標計算出數值，但在符號層各自宣告為 `sp.Symbol('L_1')`, `sp.Symbol('L_2')`, ...，保留符號（選項 B2）。
- `P`（集中載重幅值）與 `w`（均佈載重幅值）保留為符號，分量分開擬合。

---

## 三個階段

### 第一階段：數值採樣（6 組）

產生 6 組不同量級的材料參數組合：

```python
SAMPLE_SCALES = [1.0, 5.0, 25.0, 100.0, 500.0, 2000.0]
E_base = 200e9  # 基準值
A_base = 1e-3
I_base = 1e-5
G_base = E_base / 2.6

# 每組 s：E_s = E_base * scale, A_s = A_base * scale, I_s = I_base * scale
# J_s = 2 * I_s, I22_s = I_s（保持比例關係）
```

每組採樣：
1. 用採樣的 `E_s, A_s, I_s, G_s` 重新組裝 `K_np`（完整重跑）。
2. 用 `np.linalg.solve` 解出 P 分量 `U_P_s` 與 w 分量 `U_w_s`（各為長度 `total_dof` 的向量）。
3. 儲存結果：`samples_P[s, :] = U_P_s`，`samples_w[s, :] = U_w_s`。

形狀：`samples_P` 和 `samples_w` 均為 `(6, total_dof)`。

---

### 第二階段：候選基底建立 + lstsq 擬合

#### 基底定義

對每根桿件 `k`（長度數值 `Lk_val`），定義以下純數值基底值（共 7 種 × n_elem 根桿件）：

| 基底符號形式 | 數值計算（每組採樣 s） | 物理對應 |
|---|---|---|
| `P·L_k³/(E·I)` | `Lk_val³ / (E_s·I_s)` | EI33 彎曲主項 |
| `P·L_k²/(E·I)` | `Lk_val² / (E_s·I_s)` | EI33 彎曲次項 |
| `P·L_k/(E·A)`  | `Lk_val  / (E_s·A_s)` | EA 軸向 |
| `P·L_k/(G·J)`  | `Lk_val  / (G_s·J_s)` | GJ 扭轉 |
| `P·L_k³/(E·I22)` | `Lk_val³ / (E_s·I22_s)` | EI22 面外彎曲 |
| `w·L_k⁴/(E·I)` | `Lk_val⁴ / (E_s·I_s)` | 均佈載重主項 |
| `w·L_k³/(E·I)` | `Lk_val³ / (E_s·I_s)` | 均佈載重次項 |

P 分量與 w 分量各自使用對應的基底子集擬合（P 分量不含 w 基底，反之亦然）。

#### 基底矩陣

對每個 free DOF `d`：

```
B_P  shape = (6, n_elem * 5)   # 5 個 P 相關基底 × n_elem 根桿
b_P  shape = (6,)              # samples_P[:, d]
c_P  = lstsq(B_P, b_P)        # 擬合係數

B_w  shape = (6, n_elem * 2)   # 2 個 w 相關基底 × n_elem 根桿
b_w  shape = (6,)
c_w  = lstsq(B_w, b_w)
```

#### 稀疏篩選

```python
threshold = 1e-6 * np.max(np.abs(c))
c[np.abs(c) < threshold] = 0.0
```

---

### 第三階段：符號組裝

#### 符號變數宣告

```python
E, A, I, G_sym = sp.symbols('E A I G', positive=True)
P, w = sp.symbols('P w')
L_syms = [sp.Symbol(f'L_{k+1}') for k in range(n_elem)]
```

#### 基底的符號對應表

| 數值基底 | 符號表達式 |
|---|---|
| `Lk³/(E_s·I_s)` | `L_k**3 / (E * I)` |
| `Lk²/(E_s·I_s)` | `L_k**2 / (E * I)` |
| `Lk/(E_s·A_s)`  | `L_k / (E * A)` |
| `Lk/(G_s·J_s)`  | `L_k / (2*G*I)` （J=2I，保留 G 符號以與 E 區分） |
| `Lk³/(E_s·I22_s)` | `L_k**3 / (E * I)` （I22=I，與 EI33 主項合併） |
| `Lk⁴/(E_s·I_s)` | `L_k**4 / (E * I)` |
| `Lk³/(E_s·I_s)` | `L_k**3 / (E * I)` （w 次項） |

#### 係數轉換

```python
c_rational = sp.nsimplify(float_coeff, tolerance=1e-6, rational=True)
```

#### 最終表達式

```python
expr_P = sum(c_P[j] * basis_sym_P[j] for j if c_P[j] != 0) * P
expr_w = sum(c_w[j] * basis_sym_w[j] for j if c_w[j] != 0) * w
expr_total = sp.simplify(expr_P + expr_w)
```

---

## 修改範圍

只修改 `core/symbolic.py`，新增兩個私有函式並重構 `run_symbolic_analysis`：

| 函式 | 職責 |
|---|---|
| `_sample_and_solve(truss_data, E_s, A_s, I_s, G_s)` | 用指定材料參數重組 K_np 並求解，回傳 `(coeff_P_np, coeff_w_np)` |
| `_build_basis_matrix(elem_Ls, E_s, A_s, I_s, G_s, mode)` | 建立單組採樣的基底列向量（mode='P' 或 'w'） |
| `_fit_and_symbolize(samples_P, samples_w, elem_Ls, dof_idx)` | 對單一 DOF 擬合並回傳 SymPy 表達式 |

`run_symbolic_analysis` 的主流程改為：

1. 組裝原始結構拓樸（固定，不含材料數值）
2. 執行 6 次 `_sample_and_solve`
3. 對每個 DOF 執行 `_fit_and_symbolize`
4. 格式化輸出

---

## 輸出格式範例

```
uy_node2 = (1/48)·P·L₁³/(E·I) + (5/384)·w·L₁⁴/(E·I)
Rx_node1 = -(1/2)·P - (1/2)·w·L₁
```

`fmt()` 輸出保持現有 `.replace('**','^').replace('*','·')` 風格。

---

## 風險與限制

1. **欠定問題**：當桿件數 n_elem 很大時，基底矩陣的列數（n_elem × 5 for P，n_elem × 2 for w）會超過採樣組數，造成欠定 → 動態調整採樣組數至 `max(6, n_elem*5 + 2)`，確保方程數 ≥ 未知數。
2. **數值共線**：若所有桿件長度相同，L_1 = L_2 = ... = L_k，基底矩陣列間高度相關，lstsq 係數不唯一 → 擬合後需驗證殘差，若殘差 > 1e-4 則退回純數值輸出並警告。
3. **GJ=0 的純桁架**：扭轉基底貢獻為零，稀疏篩選自然歸零，無需特殊處理。
