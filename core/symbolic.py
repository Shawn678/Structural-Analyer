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

# ==============================================================================
# 主分析函式
# ==============================================================================

def run_symbolic_analysis(truss_data):
    """
    執行結構分析。

    策略：使用 numpy 進行高效數值求解，保留 P, w 作為符號以輸出符號結果。
    全域 DOF 排列 (每節點): [Ux(0), Uy(1), Uz(2), Rx(3), Ry(4), Rz(5)]
    """
    start_time = time.time()

    # 符號變數 (僅保留載重幅值符號)
    P, w = sp.symbols('P w')

    node_list      = truss_data['nodes']
    n_nodes        = len(node_list)
    node_id_to_idx = {n['id']: i for i, n in enumerate(node_list)}
    idx_to_node_id = {i: n['id'] for i, n in enumerate(node_list)}

    # 節點座標 (直接轉為 float，避免後續 SymPy 矩陣運算)
    nodes_coords = [
        (_to_float(n.get('x', 0)),
         _to_float(n.get('y', 0)),
         _to_float(n.get('z', 0)))
        for n in node_list
    ]

    NDOF      = 6
    total_dof = NDOF * n_nodes

    # 全域剛度矩陣 (numpy float64，速度快)
    K_np      = np.zeros((total_dof, total_dof))
    # 全域載重向量 (SymPy，含 P, w 符號)
    F_global  = sp.zeros(total_dof, 1)
    elements_info = []

    # ── 3. 逐桿件組裝全域剛度矩陣 ──────────────────────────────────────────
    for elem in truss_data['elements']:
        ni = node_id_to_idx[elem['i']]
        nj = node_id_to_idx[elem['j']]

        xi, yi, zi = nodes_coords[ni]
        xj, yj, zj = nodes_coords[nj]
        dx, dy, dz  = xj - xi, yj - yi, zj - zi
        Le = np.sqrt(dx**2 + dy**2 + dz**2)
        if Le < 1e-15:
            continue

        # ── WP 3.3: 方向餘弦矩陣 (numpy) ──
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
            br   = beta_val * np.pi / 180
            v2, v3 = (np.cos(br)*v2 + np.sin(br)*v3,
                      -np.sin(br)*v2 + np.cos(br)*v3)

        R_np = np.vstack([v1, v2, v3])   # 3×3
        T_np = np.zeros((12, 12))
        for b in range(4):
            T_np[b*3:b*3+3, b*3:b*3+3] = R_np

        # ── 截面常數 (WP 3.1，使用實際數值) ──
        pin_i       = elem.get("pin_i", False) or elem.get("hinge_i", False)
        pin_j       = elem.get("pin_j", False) or elem.get("hinge_j", False)
        both_hinged = pin_i and pin_j

        raw_A    = _to_float(elem.get("A",   1.0),  1.0)
        raw_I33  = _to_float(elem.get("I33", elem.get("I", 1e-4)), 1e-4)
        raw_I22  = _to_float(elem.get("I22", 0.0))
        raw_J    = _to_float(elem.get("J",   0.0))
        raw_G    = _to_float(elem.get("G",   77e9),  77e9)
        raw_E_v  = _to_float(elem.get("E",   200e9), 200e9)

        eal = raw_E_v * raw_A / Le
        gj  = raw_G * raw_J / Le

        if both_hinged or raw_I33 == 0:
            ei33_12=ei33_6=ei33_4=ei33_2 = 0.0
        else:
            ei33_12 = 12*raw_E_v*raw_I33 / Le**3
            ei33_6  =  6*raw_E_v*raw_I33 / Le**2
            ei33_4  =  4*raw_E_v*raw_I33 / Le
            ei33_2  =  2*raw_E_v*raw_I33 / Le

        if both_hinged or raw_I22 == 0 or raw_I33 == 0:
            ei22_12=ei22_6=ei22_4=ei22_2 = 0.0
        else:
            ei22_12 = 12*raw_E_v*raw_I22 / Le**3
            ei22_6  =  6*raw_E_v*raw_I22 / Le**2
            ei22_4  =  4*raw_E_v*raw_I22 / Le
            ei22_2  =  2*raw_E_v*raw_I22 / Le

        # ── WP 3.2: 12×12 局部剛度矩陣 (numpy) ──
        kl_np = build_local_stiffness_3d_np(
            eal, gj,
            ei33_12, ei33_6, ei33_4, ei33_2,
            ei22_12, ei22_6, ei22_4, ei22_2
        )

        # 轉至全域並組裝
        k_global_np = T_np.T @ kl_np @ T_np
        dofs = _elem_dofs(ni, nj, NDOF)
        for ii, d1 in enumerate(dofs):
            for jj, d2 in enumerate(dofs):
                K_np[d1, d2] += k_global_np[ii, jj]

        elements_info.append({
            "id":    elem["id"],
            "nodes": (ni, nj),
            "Le":    Le,
            "kl_np": kl_np,
            "T_np":  T_np,
            "is_truss": both_hinged,
            "f_fixed_local_sum": sp.zeros(12, 1),
            "applied_loads": []
        })

    print(f"-> [Step 1/4] 全域剛度矩陣組裝完成。耗時: {time.time()-start_time:.2f}s")

    # ── 4. 桿件均佈載重 ──────────────────────────────────────────────────────
    for e_load in truss_data.get('element_loads', []):
        info = next((e for e in elements_info if e['id'] == e_load['element_id']), None)
        if not info:
            continue
        Le_n     = info['Le']
        load_val = _to_sym(e_load.get('w', 0)) * w   # 保留 w 符號

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

    # ── 5. 桿件集中載重 ──────────────────────────────────────────────────────
    for p_load in truss_data.get('element_point_loads', []):
        info = next((e for e in elements_info if e['id'] == p_load['element_id']), None)
        if not info:
            continue
        Le_n  = info['Le']
        p_val = _to_sym(p_load.get('p', 0)) * P   # 保留 P 符號
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

    # ── 6. 節點外力向量 ──────────────────────────────────────────────────────
    for load in truss_data['loads']:
        if load.get('node_id') not in node_id_to_idx:
            continue
        idx = node_id_to_idx[load['node_id']]
        F_global[NDOF*idx+0, 0] += _to_sym(load.get('fx', 0)) * P
        F_global[NDOF*idx+1, 0] += _to_sym(load.get('fy', 0)) * P
        F_global[NDOF*idx+2, 0] += _to_sym(load.get('fz', 0)) * P
        F_global[NDOF*idx+3, 0] += _to_sym(load.get('mx', 0)) * P
        F_global[NDOF*idx+4, 0] += _to_sym(load.get('my', 0)) * P
        F_global[NDOF*idx+5, 0] += _to_sym(load.get('mz', 0)) * P

    # ── 7. 支承與彈簧邊界條件 ────────────────────────────────────────────────
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

    # ── 7.5 自動檢測平面結構 ─────────────────────────────────────────────────
    is_flat_y = all(abs(c[1]) < 1e-7 for c in nodes_coords)
    is_flat_z = all(abs(c[2]) < 1e-7 for c in nodes_coords)

    has_out_of_plane_load = any(
        float(F_global[NDOF*i+2].subs([(P, 1), (w, 1)])) != 0 or
        float(F_global[NDOF*i+3].subs([(P, 1), (w, 1)])) != 0 or
        float(F_global[NDOF*i+4].subs([(P, 1), (w, 1)])) != 0
        for i in range(n_nodes)
    )

    fixed_dofs: set = set()
    if is_flat_y:
        for idx in range(n_nodes):
            fixed_dofs.update({NDOF*idx+1, NDOF*idx+3, NDOF*idx+5})
    elif is_flat_z and not has_out_of_plane_load:
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

    # 自動固定零對角線 DOF (純桁架旋轉 DOF、2D 面外 DOF)
    for i in range(total_dof):
        if abs(K_np[i, i]) < 1e-20:
            fixed_dofs.add(i)

    free_dofs       = [d for d in range(total_dof) if d not in fixed_dofs]
    fixed_dofs_list = sorted(fixed_dofs)

    print(f"-> [Step 2/4] 邊界條件處理完成。待解自由度數量: {len(free_dofs)}")

    # ── 8. 數值求解 (numpy，取代 SymPy LDLsolve) ─────────────────────────────
    print(f"-> [Step 3/4] 正在進行數值矩陣求解 (numpy.linalg.solve)...")
    solve_start = time.time()

    K_red_np = K_np[np.ix_(free_dofs, free_dofs)]

    # 將 F 分解為 P 分量與 w 分量
    F_P_sym = F_global.subs(w, 0)
    F_w_sym = F_global.subs(P, 0)

    F_P_np = np.array([float(F_P_sym[d, 0].subs(P, 1)) for d in free_dofs])
    F_w_np = np.array([float(F_w_sym[d, 0].subs(w, 1)) for d in free_dofs])

    has_P_load = np.any(np.abs(F_P_np) > 1e-30)
    has_w_load = np.any(np.abs(F_w_np) > 1e-30)

    U_P_np = np.linalg.solve(K_red_np, F_P_np) if has_P_load else np.zeros(len(free_dofs))
    U_w_np = np.linalg.solve(K_red_np, F_w_np) if has_w_load else np.zeros(len(free_dofs))

    print(f"-> 求解完成。耗時: {time.time()-solve_start:.4f}s")

    # 將數值解重組為 P, w 符號向量
    coeff_P = np.zeros(total_dof)
    coeff_w = np.zeros(total_dof)
    for i, dof in enumerate(free_dofs):
        coeff_P[dof] = U_P_np[i]
        coeff_w[dof] = U_w_np[i]

    def _build_sym_expr(cp, cw, tol=1e-20):
        terms = []
        if abs(cp) > tol:
            terms.append(sp.Float(cp, 8) * P)
        if abs(cw) > tol:
            terms.append(sp.Float(cw, 8) * w)
        return sp.Add(*terms) if terms else sp.S.Zero

    # ── 8.5 計算支承反力 ──────────────────────────────────────────────────────
    print("-> 正在計算支承反力...")
    Reactions_full = sp.zeros(total_dof, 1)
    if fixed_dofs_list:
        K_fx_fr = K_np[np.ix_(fixed_dofs_list, free_dofs)]
        R_P = K_fx_fr @ U_P_np - np.array([float(F_P_sym[d,0].subs(P,1)) for d in fixed_dofs_list])
        R_w = K_fx_fr @ U_w_np - np.array([float(F_w_sym[d,0].subs(w,1)) for d in fixed_dofs_list])
        for ii, dof_idx in enumerate(fixed_dofs_list):
            Reactions_full[dof_idx, 0] = _build_sym_expr(R_P[ii], R_w[ii])

    # ── 9. 格式化輸出 ─────────────────────────────────────────────────────────
    print("-> [Step 4/4] 正在格式化輸出結果...")

    def fmt(expr):
        if expr == sp.S.Zero or expr == 0:
            return "0"
        s = str(expr)
        return s.replace('**', '^').replace('*', '·').replace(' ', '')

    # 節點位移
    node_displacements = []
    for idx in range(n_nodes):
        node_displacements.append({
            "node_id": idx_to_node_id[idx],
            "ux":      fmt(_build_sym_expr(coeff_P[NDOF*idx+0], coeff_w[NDOF*idx+0])),
            "uy":      fmt(_build_sym_expr(coeff_P[NDOF*idx+1], coeff_w[NDOF*idx+1])),
            "uz":      fmt(_build_sym_expr(coeff_P[NDOF*idx+2], coeff_w[NDOF*idx+2])),
            "theta_x": fmt(_build_sym_expr(coeff_P[NDOF*idx+3], coeff_w[NDOF*idx+3])),
            "theta_y": fmt(_build_sym_expr(coeff_P[NDOF*idx+4], coeff_w[NDOF*idx+4])),
            "theta_z": fmt(_build_sym_expr(coeff_P[NDOF*idx+5], coeff_w[NDOF*idx+5])),
        })

    # 桿件內力 (數值剛度 × 符號位移 + 符號固端力)
    element_forces = []
    for info in elements_info:
        ni, nj = info['nodes']
        dofs   = _elem_dofs(ni, nj, NDOF)

        u_P    = coeff_P[dofs]
        u_w    = coeff_w[dofs]
        T_e    = info['T_np']
        kl_e   = info['kl_np']

        f_P    = kl_e @ (T_e @ u_P)   # 各分量的 P 係數
        f_w    = kl_e @ (T_e @ u_w)   # 各分量的 w 係數

        # 加上固端力的符號分量
        f_fix  = info['f_fixed_local_sum']
        def _fi(k):
            fix_k  = f_fix[k, 0]
            fix_P  = float(fix_k.subs(P, 1).subs(w, 0)) if fix_k != sp.S.Zero else 0.0
            fix_w  = float(fix_k.subs(w, 1).subs(P, 0)) if fix_k != sp.S.Zero else 0.0
            return _build_sym_expr(f_P[k] + fix_P, f_w[k] + fix_w)

        fi = [_fi(k) for k in range(12)]

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

    print(f"-> 分析全部完成！總耗時: {time.time()-start_time:.2f}s")

    # 支承反力
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

    return {
        "node_displacements": node_displacements,
        "element_forces":     element_forces,
        "support_reactions":  support_reactions,
    }


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
