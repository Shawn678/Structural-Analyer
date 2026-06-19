import sympy as sp
import numpy as np
import time

def _to_sym(val, default=0):
    """安全地將輸入轉換為 SymPy 數值，處理 None, NaN 或空字串。"""
    if val is None:
        return sp.S(default)
    try:
        if isinstance(val, (float, np.float64)) and np.isnan(val):
            return sp.S(default)
        s = str(val).strip()
        if s.lower() in ('none', 'nan', ''):
            return sp.S(default)
        return sp.nsimplify(sp.sympify(s), tolerance=1e-12)
    except:
        return sp.S(default)

def _to_float(val, default=0.0):
    """安全地將輸入轉換為 Python float。"""
    try:
        f = float(_to_sym(val, default))
        return f if np.isfinite(f) else default
    except:
        return default

# ==============================================================================
# numpy 局部剛度矩陣 (數值版，用於快速組裝與求解)
# ==============================================================================

def build_local_stiffness_3d_np(eal, gj,
                                  ei33_12, ei33_6, ei33_4, ei33_2,
                                  ei22_12, ei22_6, ei22_4, ei22_2):
    """組裝 12×12 局部剛度矩陣，返回 numpy array。"""
    k = np.zeros((12, 12))
    # 軸向
    k[0,0]=k[6,6]=eal;  k[0,6]=k[6,0]=-eal
    # 扭轉
    k[3,3]=k[9,9]=gj;   k[3,9]=k[9,3]=-gj
    # I33 彎曲 (1-2 平面)
    k[1,1]=k[7,7]=ei33_12;  k[1,7]=k[7,1]=-ei33_12
    k[1,5]=k[5,1]=k[1,11]=k[11,1]=ei33_6
    k[7,5]=k[5,7]=k[7,11]=k[11,7]=-ei33_6
    k[5,5]=k[11,11]=ei33_4; k[5,11]=k[11,5]=ei33_2
    # I22 彎曲 (1-3 平面)
    k[2,2]=k[8,8]=ei22_12;  k[2,8]=k[8,2]=-ei22_12
    k[2,4]=k[4,2]=k[2,10]=k[10,2]=-ei22_6
    k[8,4]=k[4,8]=k[8,10]=k[10,8]=ei22_6
    k[4,4]=k[10,10]=ei22_4; k[4,10]=k[10,4]=ei22_2
    return k

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

        # Per-element properties, fall back to sampling scalar when not specified
        elem_E  = _to_float(elem.get('E',   E_s),  E_s)
        elem_A  = _to_float(elem.get('A',   A_s),  A_s)
        elem_I  = _to_float(elem.get('I33', elem.get('I', I_s)), I_s)
        elem_I22= _to_float(elem.get('I22', elem_I), elem_I)
        elem_J  = _to_float(elem.get('J',   2.0 * elem_I), 2.0 * elem_I)
        elem_G  = _to_float(elem.get('G',   G_s),  G_s)

        eal = elem_E * elem_A / Le
        gj  = elem_G * elem_J / Le

        if both_hinged or elem_I == 0:
            ei33_12=ei33_6=ei33_4=ei33_2 = 0.0
        else:
            ei33_12 = 12*elem_E*elem_I / Le**3
            ei33_6  =  6*elem_E*elem_I / Le**2
            ei33_4  =  4*elem_E*elem_I / Le
            ei33_2  =  2*elem_E*elem_I / Le

        if both_hinged or elem_I22 == 0 or elem_I == 0:
            ei22_12=ei22_6=ei22_4=ei22_2 = 0.0
        else:
            ei22_12 = 12*elem_E*elem_I22 / Le**3
            ei22_6  =  6*elem_E*elem_I22 / Le**2
            ei22_4  =  4*elem_E*elem_I22 / Le
            ei22_2  =  2*elem_E*elem_I22 / Le

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
    if is_flat_z:
        # 結構在 XY 平面（z=0），固定面外自由度 uz, rx, ry
        for idx in range(n_nodes):
            fixed_dofs.update({NDOF*idx+2, NDOF*idx+3, NDOF*idx+4})
    elif is_flat_y:
        # 結構在 XZ 平面（y=0），固定面外自由度 uy, rx, rz
        for idx in range(n_nodes):
            fixed_dofs.update({NDOF*idx+1, NDOF*idx+3, NDOF*idx+5})

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


def _build_basis_row(elem_Ls, E_s, A_s, I_s, G_s, mode):
    """建立單組採樣的基底列向量。"""
    J_s   = 2.0 * I_s
    J_s   = max(J_s, 1e-30)
    G_s_  = max(G_s, 1e-30)
    I_s_  = max(I_s, 1e-30)
    A_s_  = max(A_s, 1e-30)
    E_s_  = max(E_s, 1e-30)

    row = []
    for Lk in elem_Ls:
        Lk = max(float(Lk), 0.0)
        if mode == 'P':
            row.append(Lk**3 / (E_s_ * I_s_))   # EI33 彎曲主項
            row.append(Lk**2 / (E_s_ * I_s_))   # EI33 彎曲次項
            row.append(Lk    / (E_s_ * A_s_))   # EA 軸向
            row.append(Lk    / (G_s_ * J_s))    # GJ 扭轉
            row.append(Lk**3 / (E_s_ * I_s_))   # EI22 面外（I22=I，合併入 EI33）
        else:  # 'w'
            row.append(Lk**4 / (E_s_ * I_s_))   # 均佈載重主項
            row.append(Lk**3 / (E_s_ * I_s_))   # 均佈載重次項
    return np.array(row, dtype=np.float64)


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
        sym_bases_P.append(Lk_sym    / (2 * G_sym * I_sym))  # J = 2I 假設，與 _build_basis_row 數值側一致
        sym_bases_P.append(Lk_sym**3 / (E_sym * I_sym))  # EI22 合併
    for Lk_sym in L_syms:
        sym_bases_w.append(Lk_sym**4 / (E_sym * I_sym))
        sym_bases_w.append(Lk_sym**3 / (E_sym * I_sym))

    def _fit(B, b_col):
        """lstsq 擬合，回傳係數與相對殘差。"""
        if np.all(np.abs(b_col) < 1e-40):
            return np.zeros(B.shape[1]), 0.0
        c, _, rank, _ = np.linalg.lstsq(B, b_col, rcond=None)
        c = c.copy()
        pred = B @ c
        rel_err = np.max(np.abs(pred - b_col)) / (np.max(np.abs(b_col)) + 1e-40)
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

# ==============================================================================
# 主分析函式
# ==============================================================================

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
        assert set(free_s) == set(free_dofs), f"採樣 {s_idx}: free_dof 集合在縮放下改變，請檢查結構輸入"
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
        # 內力係數僅保留數值×P/w，不展開 L/E/I 符號（精確展開耗時過長）
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


# ==============================================================================
# 工具函式
# ==============================================================================

def build_local_stiffness_3d(eal, gj,
                              ei33_12, ei33_6, ei33_4, ei33_2,
                              ei22_12, ei22_6, ei22_4, ei22_2):
    """12×12 局部剛度矩陣 (SymPy 版，保留以供外部呼叫)。"""
    Z = sp.S.Zero
    return sp.Matrix([
        [ eal,   Z,         Z,         Z,     Z,         Z,       -eal,   Z,          Z,         Z,     Z,         Z      ],
        [ Z,     ei33_12,   Z,         Z,     Z,         ei33_6,  Z,     -ei33_12,    Z,         Z,     Z,         ei33_6 ],
        [ Z,     Z,         ei22_12,   Z,    -ei22_6,   Z,        Z,      Z,         -ei22_12,   Z,    -ei22_6,   Z      ],
        [ Z,     Z,         Z,         gj,    Z,         Z,        Z,      Z,          Z,        -gj,    Z,         Z      ],
        [ Z,     Z,        -ei22_6,    Z,     ei22_4,   Z,        Z,      Z,          ei22_6,    Z,     ei22_2,   Z      ],
        [ Z,     ei33_6,    Z,         Z,     Z,         ei33_4,  Z,     -ei33_6,     Z,         Z,     Z,         ei33_2 ],
        [-eal,   Z,         Z,         Z,     Z,         Z,        eal,   Z,          Z,         Z,     Z,         Z      ],
        [ Z,    -ei33_12,   Z,         Z,     Z,        -ei33_6,  Z,      ei33_12,    Z,         Z,     Z,        -ei33_6 ],
        [ Z,     Z,        -ei22_12,   Z,     ei22_6,   Z,        Z,      Z,          ei22_12,   Z,     ei22_6,   Z      ],
        [ Z,     Z,         Z,        -gj,    Z,         Z,        Z,      Z,          Z,         gj,   Z,         Z      ],
        [ Z,     Z,        -ei22_6,    Z,     ei22_2,   Z,        Z,      Z,          ei22_6,    Z,     ei22_4,   Z      ],
        [ Z,     ei33_6,    Z,         Z,     Z,         ei33_2,  Z,     -ei33_6,     Z,         Z,     Z,         ei33_4 ],
    ])


def _elem_dofs(ni: int, nj: int, ndof: int = 6) -> list:
    """回傳一根桿件的 12 個全域 DOF 索引。"""
    return [ndof*ni+k for k in range(ndof)] + [ndof*nj+k for k in range(ndof)]
