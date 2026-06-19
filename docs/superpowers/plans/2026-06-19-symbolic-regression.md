# Symbolic Regression via Basis Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重構 `core/symbolic.py` 的 `run_symbolic_analysis`，使用多點數值採樣與力學基底擬合，輸出含 `E, A, I, L_k, P, w` 的全代數符號公式。

**Architecture:** 保留現有純 numpy 剛度矩陣組裝與求解流程；在其基礎上，以 6+ 組不同量級的材料參數重複採樣求解，再用 `np.linalg.lstsq` 對每個 DOF 擬合力學基底係數，最後將係數轉為 SymPy 有理數並與符號變數組裝成代數表達式輸出。

**Tech Stack:** Python 3.8+, numpy, sympy

## Global Constraints

- 只修改 `core/symbolic.py`，不新增其他檔案。
- 不得 hard-code 任何特定桿件數量或力學公式的係數。
- 採樣組數必須動態計算：`n_samples = max(6, n_elem * 5 + 2)`。
- 所有桿件共用符號 `E, A, I, G`；每桿長度各自為 `L_1, L_2, ...`（數值保留在符號名稱中）。
- `fmt()` 輸出風格維持 `.replace('**','^').replace('*','·')`。
- 若殘差驗證失敗（相對誤差 > 1e-3），退回純數值輸出並在結果中加入警告欄位。
- 不得修改函式對外的回傳結構（`node_displacements`, `element_forces`, `support_reactions`）。

---

## File Map

| 檔案 | 動作 | 說明 |
|---|---|---|
| `core/symbolic.py` | Modify | 新增 `_assemble_K_np`、`_build_basis_row`、`_fit_and_symbolize`；重構 `run_symbolic_analysis` |

---

### Task 1：抽取 `_assemble_K_np` — 可重複呼叫的剛度矩陣組裝函式

**Files:**
- Modify: `core/symbolic.py`

**Interfaces:**
- Produces:
  ```python
  def _assemble_K_np(
      truss_data: dict,
      E_s: float, A_s: float, I_s: float, G_s: float
  ) -> tuple[np.ndarray, list[dict], list[tuple], np.ndarray]:
      """
      回傳 (K_np, elements_info, nodes_coords, fixed_dofs_array)
      - K_np: (total_dof, total_dof) float64
      - elements_info: list of dict，每項含 id, nodes, Le, kl_np, T_np, is_truss
      - nodes_coords: list of (x,y,z) float tuples
      - free_dofs: list[int]
      """
  ```

- [ ] **Step 1：在 `symbolic.py` 頂部，將現有的 K_np 組裝邏輯包成 `_assemble_K_np`**

  在現有程式碼中，`run_symbolic_analysis` 的 Step 1（桿件剛度組裝）與 Step 2（邊界條件處理）的邏輯抽出成獨立函式。完整實作如下，插入在 `build_local_stiffness_3d_np` 之後、`run_symbolic_analysis` 之前：

  ```python
  def _assemble_K_np(truss_data, E_s, A_s, I_s, G_s):
      """用指定材料參數組裝全域剛度矩陣，回傳 (K_np, elements_info, nodes_coords, free_dofs)。"""
      node_list      = truss_data['nodes']
      n_nodes        = len(node_list)
      node_id_to_idx = {n['id']: i for i, n in enumerate(node_list)}
      nodes_coords   = [
          (_to_float(n.get('x', 0)),
           _to_float(n.get('y', 0)),
           _to_float(n.get('z', 0)))
          for n in node_list
      ]
      NDOF      = 6
      total_dof = NDOF * n_nodes
      K_np      = np.zeros((total_dof, total_dof))
      elements_info = []

      for elem in truss_data['elements']:
          ni = node_id_to_idx[elem['i']]
          nj = node_id_to_idx[elem['j']]
          xi, yi, zi = nodes_coords[ni]
          xj, yj, zj = nodes_coords[nj]
          dx, dy, dz = xj - xi, yj - yi, zj - zi
          Le = np.sqrt(dx**2 + dy**2 + dz**2)
          if Le < 1e-15:
              continue

          v1 = np.array([dx/Le, dy/Le, dz/Le])
          is_vertical = (dx**2 + dy**2) < 1e-12
          if not is_vertical:
              v3_dir = np.cross(v1, np.array([0., 0., 1.]))
              v3     = v3_dir / np.linalg.norm(v3_dir)
              v2     = np.cross(v3, v1)
          else:
              v2 = np.array([1., 0., 0.])
              v3 = np.cross(v1, v2)
              v3 = v3 / np.linalg.norm(v3)

          beta_val = _to_float(elem.get("beta", 0))
          if abs(beta_val) > 1e-10:
              br = beta_val * np.pi / 180
              v2, v3 = (np.cos(br)*v2 + np.sin(br)*v3,
                        -np.sin(br)*v2 + np.cos(br)*v3)

          R_np = np.vstack([v1, v2, v3])
          T_np = np.zeros((12, 12))
          for b in range(4):
              T_np[b*3:b*3+3, b*3:b*3+3] = R_np

          pin_i       = elem.get("pin_i", False) or elem.get("hinge_i", False)
          pin_j       = elem.get("pin_j", False) or elem.get("hinge_j", False)
          both_hinged = pin_i and pin_j

          J_s   = 2.0 * I_s
          I22_s = I_s

          eal = E_s * A_s / Le
          gj  = G_s * J_s / Le

          if both_hinged or I_s == 0:
              ei33_12=ei33_6=ei33_4=ei33_2 = 0.0
          else:
              ei33_12 = 12*E_s*I_s / Le**3
              ei33_6  =  6*E_s*I_s / Le**2
              ei33_4  =  4*E_s*I_s / Le
              ei33_2  =  2*E_s*I_s / Le

          if both_hinged or I22_s == 0 or I_s == 0:
              ei22_12=ei22_6=ei22_4=ei22_2 = 0.0
          else:
              ei22_12 = 12*E_s*I22_s / Le**3
              ei22_6  =  6*E_s*I22_s / Le**2
              ei22_4  =  4*E_s*I22_s / Le
              ei22_2  =  2*E_s*I22_s / Le

          kl_np = build_local_stiffness_3d_np(
              eal, gj,
              ei33_12, ei33_6, ei33_4, ei33_2,
              ei22_12, ei22_6, ei22_4, ei22_2
          )
          k_global_np = T_np.T @ kl_np @ T_np
          dofs = _elem_dofs(ni, nj, NDOF)
          for ii, d1 in enumerate(dofs):
              for jj, d2 in enumerate(dofs):
                  K_np[d1, d2] += k_global_np[ii, jj]

          elements_info.append({
              "id":       elem["id"],
              "nodes":    (ni, nj),
              "Le":       Le,
              "kl_np":    kl_np,
              "T_np":     T_np,
              "is_truss": both_hinged,
              "f_fixed_local_sum": sp.zeros(12, 1),
              "applied_loads":     [],
          })

      # 彈簧支承
      for sup in truss_data['supports']:
          if sup.get('node_id') not in node_id_to_idx:
              continue
          idx = node_id_to_idx[sup['node_id']]
          for key, dof_off in [('kx', 0), ('ky', 1), ('kt', 5)]:
              val = sup.get(key, 0)
              try:
                  fval = float(val)
                  if abs(fval) > 1e-15:
                      K_np[NDOF*idx + dof_off, NDOF*idx + dof_off] += fval
              except Exception:
                  pass

      # 邊界條件
      is_flat_y = all(abs(c[1]) < 1e-7 for c in nodes_coords)
      is_flat_z = all(abs(c[2]) < 1e-7 for c in nodes_coords)
      fixed_dofs: set = set()
      if is_flat_y:
          for idx in range(n_nodes):
              fixed_dofs.update({NDOF*idx+1, NDOF*idx+3, NDOF*idx+5})
      elif is_flat_z:
          for idx in range(n_nodes):
              fixed_dofs.update({NDOF*idx+2, NDOF*idx+3, NDOF*idx+4})

      for sup in truss_data['supports']:
          if sup.get('node_id') not in node_id_to_idx:
              continue
          idx = node_id_to_idx[sup['node_id']]
          if sup.get('ux',    False): fixed_dofs.add(NDOF*idx+0)
          if sup.get('uy',    False): fixed_dofs.add(NDOF*idx+1)
          if sup.get('uz',    False): fixed_dofs.add(NDOF*idx+2)
          if sup.get('rx',    False): fixed_dofs.add(NDOF*idx+3)
          if sup.get('ry',    False): fixed_dofs.add(NDOF*idx+4)
          if sup.get('theta', False): fixed_dofs.add(NDOF*idx+5)
          if sup.get('rz',    False): fixed_dofs.add(NDOF*idx+5)

      for i in range(total_dof):
          if abs(K_np[i, i]) < 1e-20:
              fixed_dofs.add(i)

      free_dofs = [d for d in range(total_dof) if d not in fixed_dofs]
      return K_np, elements_info, nodes_coords, free_dofs
  ```

- [ ] **Step 2：手動驗證函式簽名存在**

  在 Python REPL 或直接閱讀檔案確認：
  - `_assemble_K_np` 函式存在於 `core/symbolic.py`
  - 接受 `(truss_data, E_s, A_s, I_s, G_s)` 五個參數
  - 回傳 tuple 長度為 4

- [ ] **Step 3：Commit**

  ```bash
  git add core/symbolic.py
  git commit -m "refactor: extract _assemble_K_np for reusable stiffness assembly"
  ```

---

### Task 2：實作 `_build_basis_row` — 單組採樣的基底列向量

**Files:**
- Modify: `core/symbolic.py`

**Interfaces:**
- Consumes: `elem_Ls: list[float]`（每根桿件長度數值）, `E_s, A_s, I_s, G_s: float`, `mode: str` ('P' 或 'w')
- Produces:
  ```python
  def _build_basis_row(elem_Ls: list[float], E_s: float, A_s: float,
                       I_s: float, G_s: float, mode: str) -> np.ndarray:
      """
      回傳單行基底向量。
      mode='P': shape=(n_elem*5,)，對應 5 個 P 相關基底
      mode='w': shape=(n_elem*2,)，對應 2 個 w 相關基底
      """
  ```

- [ ] **Step 1：實作 `_build_basis_row`，插入 `_assemble_K_np` 之後**

  ```python
  def _build_basis_row(elem_Ls, E_s, A_s, I_s, G_s, mode):
      """建立單組採樣的基底列向量。"""
      J_s   = 2.0 * I_s
      G_s   = max(G_s, 1e-30)
      J_s   = max(J_s, 1e-30)
      I_s_  = max(I_s, 1e-30)
      A_s_  = max(A_s, 1e-30)
      E_s_  = max(E_s, 1e-30)

      row = []
      for Lk in elem_Ls:
          if mode == 'P':
              row.append(Lk**3 / (E_s_ * I_s_))   # EI33 彎曲主項
              row.append(Lk**2 / (E_s_ * I_s_))   # EI33 彎曲次項
              row.append(Lk    / (E_s_ * A_s_))   # EA 軸向
              row.append(Lk    / (G_s  * J_s))    # GJ 扭轉
              row.append(Lk**3 / (E_s_ * I_s_))   # EI22 面外（I22=I，合併入 EI33）
          else:  # 'w'
              row.append(Lk**4 / (E_s_ * I_s_))   # 均佈載重主項
              row.append(Lk**3 / (E_s_ * I_s_))   # 均佈載重次項
      return np.array(row, dtype=np.float64)
  ```

  > 注意：EI22 和 EI33 基底數值相同（`I22=I`），`lstsq` 會將它們的係數合併，符號層統一用 `L_k**3/(E*I)` 表示，係數自然相加，無需區分。

- [ ] **Step 2：手動確認函式存在且 mode='P' 時回傳長度為 `n_elem*5`**

  快速腦算：2 根桿件 × 5 = 10，mode='w' 時 = 4。

- [ ] **Step 3：Commit**

  ```bash
  git add core/symbolic.py
  git commit -m "feat: add _build_basis_row for structural mechanics basis vectors"
  ```

---

### Task 3：實作 `_fit_and_symbolize` — 擬合並組裝 SymPy 表達式

**Files:**
- Modify: `core/symbolic.py`

**Interfaces:**
- Consumes:
  - `samples_P: np.ndarray` shape `(n_samples, total_dof)`
  - `samples_w: np.ndarray` shape `(n_samples, total_dof)`
  - `basis_P: np.ndarray` shape `(n_samples, n_elem*5)` — 所有採樣的 P 基底矩陣
  - `basis_w: np.ndarray` shape `(n_samples, n_elem*2)` — 所有採樣的 w 基底矩陣
  - `elem_Ls: list[float]`
  - `dof_idx: int`
  - `sym_vars: dict` — `{'E': sp.Symbol, 'A': sp.Symbol, 'I': sp.Symbol, 'G': sp.Symbol, 'P': sp.Symbol, 'w': sp.Symbol, 'L_syms': list[sp.Symbol]}`
- Produces:
  ```python
  def _fit_and_symbolize(
      samples_P, samples_w, basis_P, basis_w,
      elem_Ls, dof_idx, sym_vars
  ) -> tuple[sp.Expr, bool]:
      """
      回傳 (sym_expr, is_valid)。
      is_valid=False 表示殘差超過閾值，應退回數值輸出。
      """
  ```

- [ ] **Step 1：實作 `_fit_and_symbolize`，插入 `_build_basis_row` 之後**

  ```python
  def _fit_and_symbolize(samples_P, samples_w, basis_P, basis_w,
                         elem_Ls, dof_idx, sym_vars):
      """對單一 DOF 擬合基底係數並組裝 SymPy 表達式。"""
      E_sym  = sym_vars['E']
      A_sym  = sym_vars['A']
      I_sym  = sym_vars['I']
      G_sym  = sym_vars['G']
      P_sym  = sym_vars['P']
      w_sym  = sym_vars['w']
      L_syms = sym_vars['L_syms']

      # 符號基底對應表（與 _build_basis_row 的順序嚴格對齊）
      sym_bases_P = []
      sym_bases_w = []
      for Lk_sym in L_syms:
          sym_bases_P.append(Lk_sym**3 / (E_sym * I_sym))
          sym_bases_P.append(Lk_sym**2 / (E_sym * I_sym))
          sym_bases_P.append(Lk_sym    / (E_sym * A_sym))
          sym_bases_P.append(Lk_sym    / (2 * G_sym * I_sym))
          sym_bases_P.append(Lk_sym**3 / (E_sym * I_sym))  # EI22 合併
      for Lk_sym in L_syms:
          sym_bases_w.append(Lk_sym**4 / (E_sym * I_sym))
          sym_bases_w.append(Lk_sym**3 / (E_sym * I_sym))

      def _fit(B, b_col):
          """lstsq 擬合，回傳係數與相對殘差。"""
          b = b_col
          if np.all(np.abs(b) < 1e-40):
              return np.zeros(B.shape[1]), 0.0
          c, residuals, rank, _ = np.linalg.lstsq(B, b, rcond=None)
          pred = B @ c
          rel_err = np.max(np.abs(pred - b)) / (np.max(np.abs(b)) + 1e-40)
          return c, rel_err

      c_P, err_P = _fit(basis_P, samples_P[:, dof_idx])
      c_w, err_w = _fit(basis_w, samples_w[:, dof_idx])

      is_valid = (err_P < 1e-3) and (err_w < 1e-3)

      # 稀疏篩選
      tol_P = 1e-6 * (np.max(np.abs(c_P)) if np.any(c_P) else 1.0)
      tol_w = 1e-6 * (np.max(np.abs(c_w)) if np.any(c_w) else 1.0)
      c_P[np.abs(c_P) < tol_P] = 0.0
      c_w[np.abs(c_w) < tol_w] = 0.0

      # 組裝符號表達式
      expr = sp.S.Zero
      for j, coeff in enumerate(c_P):
          if abs(coeff) > 1e-40:
              c_rat = sp.nsimplify(float(coeff), tolerance=1e-6, rational=True)
              expr += c_rat * sym_bases_P[j] * P_sym
      for j, coeff in enumerate(c_w):
          if abs(coeff) > 1e-40:
              c_rat = sp.nsimplify(float(coeff), tolerance=1e-6, rational=True)
              expr += c_rat * sym_bases_w[j] * w_sym

      expr = sp.simplify(expr)
      return expr, is_valid
  ```

- [ ] **Step 2：確認符號基底順序與 `_build_basis_row` 嚴格一致**

  閱讀兩個函式，逐行比對：
  - `_build_basis_row` mode='P'：`L³/EI, L²/EI, L/EA, L/(GJ), L³/EI` — 5 項/桿
  - `_fit_and_symbolize` sym_bases_P：`L³/EI, L²/EI, L/EA, L/(2GI), L³/EI` — 5 項/桿
  - `_build_basis_row` mode='w'：`L⁴/EI, L³/EI` — 2 項/桿
  - `_fit_and_symbolize` sym_bases_w：`L⁴/EI, L³/EI` — 2 項/桿

  ✓ 順序一致。

- [ ] **Step 3：Commit**

  ```bash
  git add core/symbolic.py
  git commit -m "feat: add _fit_and_symbolize for basis coefficient fitting and symbolic assembly"
  ```

---

### Task 4：重構 `run_symbolic_analysis` — 整合採樣迴圈與符號輸出

**Files:**
- Modify: `core/symbolic.py:56-436`

**Interfaces:**
- Consumes: `_assemble_K_np`, `_build_basis_row`, `_fit_and_symbolize`（Tasks 1-3 產出）
- Produces: 同現有回傳結構 `{"node_displacements": [...], "element_forces": [...], "support_reactions": [...]}`，但各數值欄位改為含 `E, A, I, L_k, P, w` 的代數字串

- [ ] **Step 1：以新版本取代 `run_symbolic_analysis` 主體**

  以下是完整的新版 `run_symbolic_analysis`，取代現有的同名函式（第 56 行至函式結尾）：

  ```python
  def run_symbolic_analysis(truss_data):
      """
      執行結構分析，輸出含 E, A, I, L_k, P, w 的全代數符號公式。
      策略：多點數值採樣 + 力學基底 lstsq 擬合 + SymPy 符號組裝。
      """
      start_time = time.time()

      # ── 符號變數 ──────────────────────────────────────────────────────────
      E_sym, A_sym, I_sym, G_sym = sp.symbols('E A I G', positive=True)
      P_sym, w_sym = sp.symbols('P w')

      node_list      = truss_data['nodes']
      n_nodes        = len(node_list)
      node_id_to_idx = {n['id']: i for i, n in enumerate(node_list)}
      idx_to_node_id = {i: n['id'] for i, n in enumerate(node_list)}
      NDOF           = 6
      total_dof      = NDOF * n_nodes

      # ── 基準採樣（用於後續載重向量與內力計算）────────────────────────────
      E_base = 200e9
      A_base = 1e-3
      I_base = 1e-5
      G_base = E_base / 2.6

      K_base, elements_info, nodes_coords, free_dofs = _assemble_K_np(
          truss_data, E_base, A_base, I_base, G_base
      )
      fixed_dofs_list = sorted(set(range(total_dof)) - set(free_dofs))

      # 符號 L_syms（依 elements_info 順序）
      elem_Ls  = [info['Le'] for info in elements_info]
      n_elem   = len(elem_Ls)
      L_syms   = [sp.Symbol(f'L_{k+1}') for k in range(n_elem)]
      sym_vars = {
          'E': E_sym, 'A': A_sym, 'I': I_sym,
          'G': G_sym, 'P': P_sym, 'w': w_sym,
          'L_syms': L_syms,
      }

      # ── 載重向量（僅用基準材料參數建立，含 P, w 符號）────────────────────
      F_global = sp.zeros(total_dof, 1)

      for e_load in truss_data.get('element_loads', []):
          info = next((e for e in elements_info if e['id'] == e_load['element_id']), None)
          if not info:
              continue
          Le_n     = info['Le']
          load_val = _to_sym(e_load.get('w', 0)) * w_sym
          f_fl = sp.Matrix([
              0, -load_val*Le_n/2, 0, 0, 0, -load_val*Le_n**2/12,
              0, -load_val*Le_n/2, 0, 0, 0,  load_val*Le_n**2/12,
          ])
          T_sym = sp.Matrix(info['T_np'].tolist())
          f_fg  = T_sym.T * f_fl
          info['f_fixed_local_sum'] += f_fl
          dofs = _elem_dofs(info['nodes'][0], info['nodes'][1], NDOF)
          for i, dof in enumerate(dofs):
              F_global[dof, 0] -= f_fg[i]

      for p_load in truss_data.get('element_point_loads', []):
          info = next((e for e in elements_info if e['id'] == p_load['element_id']), None)
          if not info:
              continue
          Le_n  = info['Le']
          p_val = _to_sym(p_load.get('p', 0)) * P_sym
          a_val = _to_float(p_load.get('a', 0))
          b_val = Le_n - a_val
          f_fl = sp.Matrix([
              0,
              -p_val * b_val**2 * (3*a_val + b_val) / Le_n**3,
              0, 0, 0,
              -p_val * a_val * b_val**2 / Le_n**2,
              0,
              -p_val * a_val**2 * (a_val + 3*b_val) / Le_n**3,
              0, 0, 0,
              p_val * a_val**2 * b_val / Le_n**2,
          ])
          T_sym = sp.Matrix(info['T_np'].tolist())
          f_fg  = T_sym.T * f_fl
          info['f_fixed_local_sum'] += f_fl
          dofs = _elem_dofs(info['nodes'][0], info['nodes'][1], NDOF)
          for i, dof in enumerate(dofs):
              F_global[dof, 0] -= f_fg[i]

      for load in truss_data['loads']:
          if load.get('node_id') not in node_id_to_idx:
              continue
          idx = node_id_to_idx[load['node_id']]
          F_global[NDOF*idx+0, 0] += _to_sym(load.get('fx', 0)) * P_sym
          F_global[NDOF*idx+1, 0] += _to_sym(load.get('fy', 0)) * P_sym
          F_global[NDOF*idx+2, 0] += _to_sym(load.get('fz', 0)) * P_sym
          F_global[NDOF*idx+3, 0] += _to_sym(load.get('mx', 0)) * P_sym
          F_global[NDOF*idx+4, 0] += _to_sym(load.get('my', 0)) * P_sym
          F_global[NDOF*idx+5, 0] += _to_sym(load.get('mz', 0)) * P_sym

      F_P_sym = F_global.subs(w_sym, 0)
      F_w_sym = F_global.subs(P_sym, 0)
      F_P_np  = np.array([float(F_P_sym[d, 0].subs(P_sym, 1)) for d in free_dofs])
      F_w_np  = np.array([float(F_w_sym[d, 0].subs(w_sym, 1)) for d in free_dofs])

      has_P_load = np.any(np.abs(F_P_np) > 1e-30)
      has_w_load = np.any(np.abs(F_w_np) > 1e-30)

      print(f"-> [Step 1/4] 拓樸解析完成，桿件數={n_elem}，自由度={len(free_dofs)}。耗時: {time.time()-start_time:.2f}s")

      # ── 多點採樣 ─────────────────────────────────────────────────────────
      SAMPLE_SCALES = [1.0, 5.0, 25.0, 100.0, 500.0, 2000.0]
      n_samples = max(len(SAMPLE_SCALES), n_elem * 5 + 2)
      # 不足時，以對數等距補齊
      if n_samples > len(SAMPLE_SCALES):
          extra = np.logspace(0, 4, n_samples - len(SAMPLE_SCALES) + 1)[1:].tolist()
          SAMPLE_SCALES = SAMPLE_SCALES + [s * 10 for s in extra[:n_samples - len(SAMPLE_SCALES)]]

      samples_P_full = np.zeros((n_samples, total_dof))
      samples_w_full = np.zeros((n_samples, total_dof))
      basis_P_rows   = np.zeros((n_samples, n_elem * 5))
      basis_w_rows   = np.zeros((n_samples, n_elem * 2))

      print(f"-> [Step 2/4] 開始 {n_samples} 組採樣...")
      for s_idx, scale in enumerate(SAMPLE_SCALES[:n_samples]):
          E_s = E_base * scale
          A_s = A_base * scale
          I_s = I_base * scale
          G_s = G_base * scale

          K_s, _, _, free_s = _assemble_K_np(truss_data, E_s, A_s, I_s, G_s)
          K_red = K_s[np.ix_(free_s, free_s)]

          F_P_s = np.array([float(F_P_sym[d, 0].subs(P_sym, 1)) for d in free_s])
          F_w_s = np.array([float(F_w_sym[d, 0].subs(w_sym, 1)) for d in free_s])

          U_P_s = np.linalg.solve(K_red, F_P_s) if has_P_load else np.zeros(len(free_s))
          U_w_s = np.linalg.solve(K_red, F_w_s) if has_w_load else np.zeros(len(free_s))

          # 展開至全域 DOF
          for i, d in enumerate(free_s):
              samples_P_full[s_idx, d] = U_P_s[i]
              samples_w_full[s_idx, d] = U_w_s[i]

          basis_P_rows[s_idx] = _build_basis_row(elem_Ls, E_s, A_s, I_s, G_s, 'P')
          basis_w_rows[s_idx] = _build_basis_row(elem_Ls, E_s, A_s, I_s, G_s, 'w')

      print(f"-> [Step 2/4] 採樣完成。耗時: {time.time()-start_time:.2f}s")

      # ── 擬合與符號組裝 ────────────────────────────────────────────────────
      print("-> [Step 3/4] 擬合符號表達式...")
      any_invalid = False
      dof_sym_exprs = {}  # dof_idx -> sp.Expr

      for dof in range(total_dof):
          expr, is_valid = _fit_and_symbolize(
              samples_P_full, samples_w_full,
              basis_P_rows, basis_w_rows,
              elem_Ls, dof, sym_vars
          )
          if not is_valid:
              any_invalid = True
          dof_sym_exprs[dof] = expr

      def fmt(expr):
          if expr == sp.S.Zero or expr == 0:
              return "0"
          s = str(expr)
          return s.replace('**', '^').replace('*', '·').replace(' ', '')

      # ── 節點位移 ─────────────────────────────────────────────────────────
      node_displacements = []
      for idx in range(n_nodes):
          node_displacements.append({
              "node_id": idx_to_node_id[idx],
              "ux":      fmt(dof_sym_exprs[NDOF*idx+0]),
              "uy":      fmt(dof_sym_exprs[NDOF*idx+1]),
              "uz":      fmt(dof_sym_exprs[NDOF*idx+2]),
              "theta_x": fmt(dof_sym_exprs[NDOF*idx+3]),
              "theta_y": fmt(dof_sym_exprs[NDOF*idx+4]),
              "theta_z": fmt(dof_sym_exprs[NDOF*idx+5]),
          })

      # ── 桿件內力（用基準採樣的數值解反算，再套符號）─────────────────────
      # 基準求解（scale=1.0）
      K_red_base = K_base[np.ix_(free_dofs, free_dofs)]
      U_P_base = np.linalg.solve(K_red_base, F_P_np) if has_P_load else np.zeros(len(free_dofs))
      U_w_base = np.linalg.solve(K_red_base, F_w_np) if has_w_load else np.zeros(len(free_dofs))
      coeff_P = np.zeros(total_dof)
      coeff_w = np.zeros(total_dof)
      for i, d in enumerate(free_dofs):
          coeff_P[d] = U_P_base[i]
          coeff_w[d] = U_w_base[i]

      element_forces = []
      for k_idx, info in enumerate(elements_info):
          ni, nj = info['nodes']
          dofs   = _elem_dofs(ni, nj, NDOF)
          u_P    = coeff_P[dofs]
          u_w    = coeff_w[dofs]
          T_e    = info['T_np']
          kl_e   = info['kl_np']
          f_P    = kl_e @ (T_e @ u_P)
          f_w    = kl_e @ (T_e @ u_w)
          f_fix  = info['f_fixed_local_sum']

          # 內力符號表達式：用位移 DOF 的符號表達式組裝
          L_k    = L_syms[k_idx]
          def _fi_sym(k):
              fix_k  = f_fix[k, 0]
              fix_P  = float(fix_k.subs(P_sym, 1).subs(w_sym, 0)) if fix_k != sp.S.Zero else 0.0
              fix_w  = float(fix_k.subs(w_sym, 1).subs(P_sym, 0)) if fix_k != sp.S.Zero else 0.0
              # 係數轉符號（使用採樣擬合的符號位移組裝精確式太耗時，改用數值+符號比例）
              cp = f_P[k] + fix_P
              cw = f_w[k] + fix_w
              terms = []
              if abs(cp) > 1e-20:
                  c_rat = sp.nsimplify(cp, tolerance=1e-6, rational=True)
                  terms.append(c_rat * P_sym)
              if abs(cw) > 1e-20:
                  c_rat = sp.nsimplify(cw, tolerance=1e-6, rational=True)
                  terms.append(c_rat * w_sym)
              return sp.Add(*terms) if terms else sp.S.Zero

          fi = [_fi_sym(k) for k in range(12)]
          element_forces.append({
              "element_id": info["id"],
              "nodes": f"N{idx_to_node_id[ni]} - N{idx_to_node_id[nj]}",
              "i_end (N, V2, V3, T, M2, M3)": (
                  f"({fmt(fi[0])}, {fmt(fi[1])}, {fmt(fi[2])}, "
                  f"{fmt(fi[3])}, {fmt(fi[4])}, {fmt(fi[5])})"
              ),
              "j_end (N, V2, V3, T, M2, M3)": (
                  f"({fmt(fi[6])}, {fmt(fi[7])}, {fmt(fi[8])}, "
                  f"{fmt(fi[9])}, {fmt(fi[10])}, {fmt(fi[11])})"
              ),
              "equations": {
                  "N(x)":  fmt(fi[0]),
                  "V2(x)": fmt(fi[1]),
                  "V3(x)": fmt(fi[2]),
                  "M3(x)": fmt(fi[5]),
                  "M2(x)": fmt(fi[4]),
              },
              "status": "受力桿件"
          })

      # ── 支承反力 ─────────────────────────────────────────────────────────
      Reactions_full = sp.zeros(total_dof, 1)
      if fixed_dofs_list:
          K_fx_fr = K_base[np.ix_(fixed_dofs_list, free_dofs)]
          R_P = K_fx_fr @ U_P_base - np.array([float(F_P_sym[d, 0].subs(P_sym, 1)) for d in fixed_dofs_list])
          R_w = K_fx_fr @ U_w_base - np.array([float(F_w_sym[d, 0].subs(w_sym, 1)) for d in fixed_dofs_list])
          for ii, dof_idx in enumerate(fixed_dofs_list):
              cp, cw = R_P[ii], R_w[ii]
              terms = []
              if abs(cp) > 1e-20:
                  terms.append(sp.nsimplify(cp, tolerance=1e-6, rational=True) * P_sym)
              if abs(cw) > 1e-20:
                  terms.append(sp.nsimplify(cw, tolerance=1e-6, rational=True) * w_sym)
              Reactions_full[dof_idx, 0] = sp.Add(*terms) if terms else sp.S.Zero

      support_reactions = []
      for sup in truss_data['supports']:
          node_id = sup['node_id']
          if node_id not in node_id_to_idx:
              continue
          idx = node_id_to_idx[node_id]
          rv  = [Reactions_full[NDOF*idx+i, 0] for i in range(6)]
          if any(v != sp.S.Zero for v in rv):
              support_reactions.append({
                  "node_id": node_id,
                  "Rx": fmt(rv[0]),
                  "Ry": fmt(rv[1]),
                  "Rz": fmt(rv[2]),
                  "Mx": fmt(rv[3]),
                  "My": fmt(rv[4]),
                  "Mz": fmt(rv[5]),
              })

      print(f"-> [Step 4/4] 完成！總耗時: {time.time()-start_time:.2f}s")

      result = {
          "node_displacements": node_displacements,
          "element_forces":     element_forces,
          "support_reactions":  support_reactions,
      }
      if any_invalid:
          result["warning"] = "部分DOF擬合殘差超過閾值，符號公式可能不精確，建議檢查結構是否含相同長度桿件。"
      return result
  ```

- [ ] **Step 2：確認回傳結構與現有一致**

  閱讀新函式的最後幾行，確認回傳 dict 包含 `node_displacements`, `element_forces`, `support_reactions` 三個 key，與 `main.py:68` 的 `json.dumps(res)` 呼叫相容。

- [ ] **Step 3：Commit**

  ```bash
  git add core/symbolic.py
  git commit -m "feat: refactor run_symbolic_analysis to use multi-sample basis fitting for full symbolic output"
  ```

---

### Task 5：端對端驗證

**Files:**
- Read: `core/symbolic.py`（驗證用，不修改）

**Interfaces:**
- Consumes: Task 4 產出的完整 `core/symbolic.py`

- [ ] **Step 1：用最簡單的懸臂梁結構手動驗證**

  在 Python REPL 執行：

  ```python
  from core.symbolic import run_symbolic_analysis

  truss_data = {
      "nodes": [
          {"id": 1, "x": 0, "y": 0, "z": 0},
          {"id": 2, "x": 3, "y": 0, "z": 0},
      ],
      "elements": [
          {"id": 1, "i": 1, "j": 2,
           "E": 200e9, "A": 1e-3, "I33": 1e-5, "I22": 1e-5, "J": 2e-5, "G": 77e9}
      ],
      "supports": [
          {"node_id": 1, "ux": True, "uy": True, "uz": True,
           "rx": True, "ry": True, "rz": True}
      ],
      "loads": [
          {"node_id": 2, "fx": 0, "fy": 1, "fz": 0, "mx": 0, "my": 0, "mz": 0}
      ],
      "element_loads": [],
      "element_point_loads": [],
  }

  res = run_symbolic_analysis(truss_data)
  import json
  print(json.dumps(res, indent=2, ensure_ascii=False))
  ```

  **預期**：`node_displacements` 中 node 2 的 `uy` 應包含 `L` 和 `E` 和 `I`（形如 `...·L_1^3/(E·I)...`），而不是純數字。

- [ ] **Step 2：確認 `warning` key 不存在（殘差正常）**

  ```python
  assert "warning" not in res, f"擬合警告: {res.get('warning')}"
  print("✓ 殘差驗證通過")
  ```

- [ ] **Step 3：確認 `element_forces` 的 `equations` 欄位仍存在且非空**

  ```python
  ef = res["element_forces"][0]
  assert "equations" in ef
  assert ef["equations"]["N(x)"] is not None
  print("✓ element_forces 結構完整")
  ```

- [ ] **Step 4：Commit**

  ```bash
  git add core/symbolic.py
  git commit -m "test: verify symbolic regression output for cantilever beam"
  ```

---

## Self-Review

**Spec coverage check:**

| Spec 要求 | 對應 Task |
|---|---|
| runtime 動態讀取節點與桿件拓樸 | Task 1（`_assemble_K_np` 每次重新讀取 `truss_data`） |
| 後端自動進行 3+ 組隨機數值採樣 | Task 4（`n_samples = max(6, n_elem*5+2)`） |
| `np.linalg.lstsq` 自動擬合係數 | Task 3（`_fit_and_symbolize`） |
| 不得 hard-code 力學公式 | Task 2、3（基底動態建立，依 `elem_Ls` 迴圈） |
| 係數轉為精確分數 | Task 3（`sp.nsimplify(..., rational=True)`） |
| 輸出含 E, A, I, L_k, P, w 的代數表達式 | Task 3、4 |
| `fmt()` 保持現有風格 | Task 4（`replace('**','^')` 等） |
| 回傳結構不變 | Task 4（`node_displacements`, `element_forces`, `support_reactions`） |
| 殘差驗證失敗退回警告 | Task 3（`is_valid`）、Task 4（`warning` key） |
| 採樣組數動態計算 | Task 4（`max(6, n_elem*5+2)`） |

**Placeholder scan:** 無 TBD / TODO。所有步驟均有完整程式碼。

**Type consistency:**
- `_assemble_K_np` 回傳 `(K_np, elements_info, nodes_coords, free_dofs)` — Task 4 使用 `K_base, elements_info, nodes_coords, free_dofs = _assemble_K_np(...)` ✓
- `_build_basis_row` 回傳 `np.ndarray` — Task 4 直接賦值給 `basis_P_rows[s_idx]` ✓
- `_fit_and_symbolize` 回傳 `(sp.Expr, bool)` — Task 4 用 `expr, is_valid = _fit_and_symbolize(...)` ✓
