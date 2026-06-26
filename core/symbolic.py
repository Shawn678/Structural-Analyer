import copy
import sympy as sp
import numpy as np
import time
from core.rigid_link import apply_rigid_links, recover_slave_displacements

def _norm_id(val) -> str:
    """統一節點/桿件 ID 為字串：1.0 → '1'，'1.0' → '1'，'gen_...' → 'gen_...'"""
    s = str(val).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s


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
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        pass
    try:
        return float(_to_sym(val, default))
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

def _assemble_K_np(truss_data, E_s, A_s, I_s, G_s, _tag="", force_2d: str | None = None):
    """用指定材料參數組裝全域剛度矩陣，回傳 (K_np, elements_info, nodes_coords, free_dofs)。"""
    node_list      = truss_data['nodes']
    n_nodes        = len(node_list)
    node_id_to_idx = {_norm_id(n['id']): i for i, n in enumerate(node_list)}
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
        ni = node_id_to_idx[_norm_id(elem['i'])]
        nj = node_id_to_idx[_norm_id(elem['j'])]
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
        # Note: fillna(0) in app.py may set missing props to 0; treat 0 as missing
        def _ep(val, fb):
            v = _to_float(val, fb)
            return v if v != 0.0 else fb
        elem_E  = _ep(elem.get('E'),   E_s)
        elem_A  = _ep(elem.get('A'),   A_s)
        elem_I  = _ep(elem.get('I33', elem.get('I')), I_s)
        elem_I22= _ep(elem.get('I22'), elem_I)
        elem_J  = _ep(elem.get('J'),   2.0 * elem_I)
        elem_G  = _ep(elem.get('G'),   G_s)

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
        ix = np.array(dofs)
        K_np[np.ix_(ix, ix)] += k_global_np

        elements_info.append({
            "id":       elem["id"],
            "nodes":    (ni, nj),
            "Le":       Le,
            "kl_np":    kl_np,
            "T_np":     T_np,
            "is_truss": both_hinged,
            "f_fixed_local_sum": None,
            "applied_loads":     [],
        })

    # 彈簧支承
    for sup in truss_data['supports']:
        if _norm_id(sup.get('node_id')) not in node_id_to_idx:
            continue
        idx = node_id_to_idx[_norm_id(sup['node_id'])]
        for key, dof_off in [('kx', 0), ('ky', 1), ('kt', 5)]:
            val = sup.get(key, 0)
            try:
                fval = float(val)
                if abs(fval) > 1e-15:
                    K_np[NDOF*idx + dof_off, NDOF*idx + dof_off] += fval
            except Exception:
                pass

    # 邊界條件
    # force_2d="XZ" → 強制套用 XZ 平面面外約束；force_2d="XY" → XY 平面；None → 自動偵測
    is_flat_y = all(abs(c[1]) < 1e-7 for c in nodes_coords)
    is_flat_z = all(abs(c[2]) < 1e-7 for c in nodes_coords)
    _apply_flat_z = (force_2d == "XY") or (force_2d is None and is_flat_z)
    _apply_flat_y = (force_2d == "XZ") or (force_2d is None and is_flat_y and not is_flat_z)
    fixed_dofs: set = set()
    if _apply_flat_z:
        # 結構在 XY 平面（z=0），固定面外自由度 uz, rx, ry
        for idx in range(n_nodes):
            fixed_dofs.update({NDOF*idx+2, NDOF*idx+3, NDOF*idx+4})
    elif _apply_flat_y:
        # 2D 結構在 XZ 平面，補全面外約束：uy, rx, rz
        for idx in range(n_nodes):
            fixed_dofs.update({NDOF*idx+1, NDOF*idx+3, NDOF*idx+5})

    for sup in truss_data['supports']:
        if _norm_id(sup.get('node_id')) not in node_id_to_idx:
            continue
        idx = node_id_to_idx[_norm_id(sup['node_id'])]
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
    _zero_diag = sum(1 for i in range(total_dof) if abs(K_np[i,i])<1e-20)
    print(f"[K_ASSEMBLE{_tag}] total_dof={total_dof}, fixed={len(fixed_dofs)}, free={len(free_dofs)}, is_flat_y={is_flat_y}, is_flat_z={is_flat_z}, zero_diag={_zero_diag}")
    if _zero_diag == total_dof and elements_info:
        e0 = elements_info[0]
        r0 = truss_data['elements'][0]
        print(f"[K_ASSEMBLE{_tag}] WARNING: K全零！節點座標前3={nodes_coords[:3]}, Le樣本={[round(e['Le'],4) for e in elements_info[:3]]}")
        print(f"[K_ASSEMBLE{_tag}] elem[0] raw keys={list(r0.keys())}, E={r0.get('E')}, A={r0.get('A')}, I33={r0.get('I33')}, section={r0.get('section')}")
    return K_np, elements_info, nodes_coords, free_dofs


def _make_cross_samples(n_groups: int, base_vals: list, n_samples: int, seed: int = 42) -> list:
    """
    為每個斷面組獨立產生 scale 序列（Hadamard 交叉採樣）。
    回傳 shape=(n_samples, n_groups) 的 list-of-list，每格是該採樣點該組的 scale 倍率。
    base_vals: [{"E": float, "A": float, "I": float, "G": float}, ...]，長度 = n_groups
    注意：base_vals 僅保留介面一致性，實際 scale 倍率為全域基準值的倍數。
    """
    SCALES = [1.0, 5.0, 25.0, 100.0, 500.0]
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(n_samples):
        row = [SCALES[rng.integers(0, len(SCALES))] for _ in range(n_groups)]
        result.append(row)
    return result


def _build_basis_row(elem_Ls, E_s, A_s, I_s, G_s, mode,
                     elem_E_list=None, elem_A_list=None,
                     elem_I_list=None, elem_G_list=None):
    """建立單組採樣的基底列向量。
    當提供 elem_*_list 時，每根桿件使用各自的材料參數；否則使用全域 E_s/A_s/I_s/G_s。
    """
    row = []
    for k, Lk in enumerate(elem_Ls):
        Lk = max(float(Lk), 0.0)
        # 取該桿件的材料參數
        if elem_E_list is not None:
            E_k = max(float(elem_E_list[k]), 1e-30)
            A_k = max(float(elem_A_list[k]), 1e-30)
            I_k = max(float(elem_I_list[k]), 1e-30)
            G_k = max(float(elem_G_list[k]), 1e-30)
        else:
            E_k = max(E_s, 1e-30)
            A_k = max(A_s, 1e-30)
            I_k = max(I_s, 1e-30)
            G_k = max(G_s, 1e-30)
        J_k = max(2.0 * I_k, 1e-30)

        if mode == 'P':
            row.append(1.0 / (E_k * I_k))   # 彎曲基底（L 冪次折入係數）
            row.append(1.0 / (E_k * A_k))   # 軸向基底
            row.append(1.0 / (G_k * J_k))   # 扭轉基底
        else:  # 'w'
            row.append(1.0 / (E_k * I_k))   # 均佈載重基底
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

    # 每桿件符號（有 section_group_map 時各異，否則全域共用）
    elem_E_syms = sym_vars.get('elem_E_syms', None)
    elem_A_syms = sym_vars.get('elem_A_syms', None)
    elem_I_syms = sym_vars.get('elem_I_syms', None)
    elem_G_syms = sym_vars.get('elem_G_syms', None)

    # 符號基底對應表（與 _build_basis_row 的順序嚴格對齊）
    sym_bases_P = []
    sym_bases_w = []
    for k, Lk_sym in enumerate(L_syms):
        if elem_E_syms is not None:
            E_k = elem_E_syms[k]
            A_k = elem_A_syms[k]
            I_k = elem_I_syms[k]
            G_k = elem_G_syms[k]
        else:
            E_k = E_sym
            A_k = A_sym
            I_k = I_sym
            G_k = G_sym
        sym_bases_P.append(sp.S.One / (E_k * I_k))          # 彎曲：係數中含 L 冪次
        sym_bases_P.append(sp.S.One / (E_k * A_k))          # 軸向
        sym_bases_P.append(sp.S.One / (2 * G_k * I_k))      # 扭轉（J=2I）
    for k, Lk_sym in enumerate(L_syms):
        if elem_E_syms is not None:
            E_k = elem_E_syms[k]
            I_k = elem_I_syms[k]
        else:
            E_k = E_sym
            I_k = I_sym
        sym_bases_w.append(sp.S.One / (E_k * I_k))          # 均佈載重：係數含 L 冪次

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

def run_symbolic_analysis(truss_data: dict, section_group_map: dict | None = None) -> dict:
    """
    執行結構分析，輸出含 E, A, I, L_k, P, w 的全代數符號公式。
    策略：多點數值採樣 + 力學基底 lstsq 擬合 + SymPy 符號組裝。

    section_group_map: {elem_id: {"E": sp.Symbol, "A": sp.Symbol, "I33": sp.Symbol,
                                  "I22": sp.Symbol, "J": sp.Symbol, "G": sp.Symbol}}
        當提供此 map 時，每根桿件使用各自的符號而非全域 E/A/I/G。
    """
    start_time = time.time()

    # ── 符號變數 ──────────────────────────────────────────────────────────
    E_sym, A_sym, I_sym, G_sym = sp.symbols('E A I G', positive=True)
    P_sym, w_sym = sp.symbols('P w')

    node_list      = truss_data['nodes']
    n_nodes        = len(node_list)
    node_id_to_idx = {_norm_id(n['id']): i for i, n in enumerate(node_list)}
    idx_to_node_id = {i: _norm_id(n['id']) for i, n in enumerate(node_list)}
    NDOF           = 6
    total_dof      = NDOF * n_nodes

    # ── 基準採樣（用於後續載重向量與內力計算）────────────────────────────
    E_base = 200e9
    A_base = 1e-3
    I_base = 1e-5
    G_base = E_base / 2.6

    K_base, elements_info, nodes_coords, free_dofs = _assemble_K_np(
        truss_data, E_base, A_base, I_base, G_base, _tag="[SYM]"
    )

    # ── Rigid Link 前處理 ─────────────────────────────────────────────────
    rigid_links = [rl for rl in truss_data.get('rigid_links', [])
                   if _norm_id(rl.get('master','')) != _norm_id(rl.get('slave',''))]
    slave_node_ids = {_norm_id(rl['slave']) for rl in rigid_links}
    slave_node_idxs = {node_id_to_idx[sid] for sid in slave_node_ids if sid in node_id_to_idx}
    slave_dofs_set = {idx * NDOF + k for idx in slave_node_idxs for k in range(NDOF)}

    # free_dofs 排除 slave DOF（slave DOF 由 rigid link 凝縮決定）
    free_dofs = [d for d in free_dofs if d not in slave_dofs_set]

    # master_dofs：全域所有 DOF 中排除 slave DOF（含 fixed DOF，供凝縮矩陣索引用）
    all_dofs_no_slave = [d for d in range(total_dof) if d not in slave_dofs_set]
    # 凝縮後 DOF 在 all_dofs_no_slave 中的位置 → 用於從 K_cond 切出 free submatrix
    master_free_cols = [all_dofs_no_slave.index(d) for d in free_dofs]

    fixed_dofs_list = sorted(set(range(total_dof)) - set(free_dofs) - slave_dofs_set)

    # 符號 L_syms（依 elements_info 順序）
    elem_Ls  = [info['Le'] for info in elements_info]
    n_elem   = len(elem_Ls)
    L_syms   = [sp.Symbol(f'L_{k+1}') for k in range(n_elem)]

    # 建立每根桿件的符號（有 section_group_map 時使用各自符號，否則共用全域符號）
    # section_group_map 支援兩種 key 類型：
    #   整數 key（element ID）：直接以 eid 查表
    #   字串 key（section 名稱）：以桿件的 "section" 欄位查表
    _raw_elem_by_id = {e['id']: e for e in truss_data['elements']}
    elem_E_syms   = []
    elem_A_syms   = []
    elem_I_syms   = []
    elem_I22_syms = []
    elem_J_syms   = []
    elem_G_syms   = []
    for info in elements_info:
        eid = info['id']
        sym = None
        if section_group_map:
            if eid in section_group_map:
                # 整數 key 路徑（既有行為）
                sym = section_group_map[eid]
            else:
                # 字串 key 路徑：依桿件的 section 名稱查表
                sec_name = _raw_elem_by_id.get(eid, {}).get('section', '')
                if sec_name and sec_name in section_group_map:
                    sym = section_group_map[sec_name]
        if sym is not None:
            elem_E_syms.append(sym['E'])
            elem_A_syms.append(sym['A'])
            elem_I_syms.append(sym['I33'])
            elem_I22_syms.append(sym.get('I22', sym['I33']))
            elem_J_syms.append(sym.get('J', 2 * sym['I33']))
            elem_G_syms.append(sym['G'])
        else:
            elem_E_syms.append(E_sym)
            elem_A_syms.append(A_sym)
            elem_I_syms.append(I_sym)
            elem_I22_syms.append(I_sym)
            elem_J_syms.append(2 * I_sym)
            elem_G_syms.append(G_sym)

    sym_vars = {
        'E': E_sym, 'A': A_sym, 'I': I_sym,
        'G': G_sym, 'P': P_sym, 'w': w_sym,
        'L_syms': L_syms,
        'elem_E_syms': elem_E_syms,
        'elem_A_syms': elem_A_syms,
        'elem_I_syms': elem_I_syms,
        'elem_G_syms': elem_G_syms,
    }

    # ── 載重向量（僅用基準材料參數建立，含 P, w 符號）────────────────────
    F_global = sp.zeros(total_dof, 1)

    _eloads = truss_data.get('element_loads', [])
    print(f"[LOAD CHECK] element_loads 筆數={len(_eloads)}, 前3筆={_eloads[:3]}")
    for e_load in _eloads:
        info = next((e for e in elements_info if _norm_id(e['id']) == _norm_id(e_load['element_id'])), None)
        if not info:
            print(f"[LOAD MISS] element_load id={e_load['element_id']} 找不到對應桿件")
            continue
        Le_n     = info['Le']
        load_val = _to_sym(e_load.get('w', 0)) * w_sym
        f_fl = sp.Matrix([
            0, -load_val*Le_n/2, 0, 0, 0, -load_val*Le_n**2/12,
            0, -load_val*Le_n/2, 0, 0, 0,  load_val*Le_n**2/12,
        ])
        T_sym = sp.Matrix(info['T_np'].tolist())
        f_fg  = T_sym.T * f_fl
        if info['f_fixed_local_sum'] is None:
            info['f_fixed_local_sum'] = f_fl
        else:
            info['f_fixed_local_sum'] += f_fl
        dofs = _elem_dofs(info['nodes'][0], info['nodes'][1], NDOF)
        for i, dof in enumerate(dofs):
            F_global[dof, 0] -= f_fg[i]

    for p_load in truss_data.get('element_point_loads', []):
        info = next((e for e in elements_info if _norm_id(e['id']) == _norm_id(p_load['element_id'])), None)
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
        if info['f_fixed_local_sum'] is None:
            info['f_fixed_local_sum'] = f_fl
        else:
            info['f_fixed_local_sum'] += f_fl
        dofs = _elem_dofs(info['nodes'][0], info['nodes'][1], NDOF)
        for i, dof in enumerate(dofs):
            F_global[dof, 0] -= f_fg[i]

    for load in truss_data['loads']:
        if _norm_id(load.get('node_id')) not in node_id_to_idx:
            continue
        idx = node_id_to_idx[_norm_id(load['node_id'])]
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
    # 路徑 (A)（無 section_group_map）使用獨立材料採樣：E/A/I/G 分別給不同 scale，
    # 使 basis 欄（L³/EI, L²/EI, L/EA, L/GJ）線性獨立。
    # 每列 = [sE, sA, sI, sG]，由隨機對數空間採樣產生。
    rng_mat = np.random.default_rng(42)
    n_samples = max(8, n_elem * 3 + 4)
    # 4 個材料參數各自的獨立 scale（對數均勻分布在 [1, 1e4]）
    _mat_scales = np.exp(rng_mat.uniform(0, np.log(1e4), size=(n_samples, 4)))

    # F_P/F_w 只含 P/w 線性係數，不隨材料 scale 變化，迴圈外算一次
    F_P_np_free = np.array([float(F_P_sym[d, 0].subs(P_sym, 1)) for d in free_dofs])
    F_w_np_free = np.array([float(F_w_sym[d, 0].subs(w_sym, 1)) for d in free_dofs])

    # ── 判斷採樣模式 ─────────────────────────────────────────────────────────
    # section_group_map 有兩種可能的 key 類型（由呼叫端決定）：
    #   (A) 整數 key（element ID）：既有路徑，全域均勻 scale 擾動
    #   (B) 字串 key（section 名稱）：Task 2 新路徑，Hadamard 交叉採樣
    # 判斷方式：检查第一個 key 是否為 str
    _group_names = []
    if section_group_map:
        first_key = next(iter(section_group_map))
        if isinstance(first_key, str):
            # (B) 字串 key：section name → 啟用 Hadamard 交叉採樣
            _group_names = sorted(section_group_map.keys())
    _n_groups = len(_group_names)

    # 當使用 section_group_map（任意 key 類型）時，
    # 預先取出每根桿件的實際材料基準值（用於 basis 建立）
    _elem_raw = {e['id']: e for e in truss_data['elements']}
    if section_group_map:
        per_elem_E = [_to_float(_elem_raw[info['id']].get('E',   E_base), E_base) for info in elements_info]
        per_elem_A = [_to_float(_elem_raw[info['id']].get('A',   A_base), A_base) for info in elements_info]
        per_elem_I = [_to_float(_elem_raw[info['id']].get('I33', _elem_raw[info['id']].get('I', I_base)), I_base) for info in elements_info]
        per_elem_G = [_to_float(_elem_raw[info['id']].get('G',   G_base), G_base) for info in elements_info]
    else:
        per_elem_E = per_elem_A = per_elem_I = per_elem_G = None

    # (B) Hadamard 交叉採樣前置：計算 n_samples 與交叉擾動矩陣
    if _n_groups > 0:
        n_samples = max(n_samples, _n_groups * 4 + 4)
        # 擴充 _mat_scales 以對齊更新後的 n_samples
        if n_samples > _mat_scales.shape[0]:
            _extra = np.exp(rng_mat.uniform(0, np.log(1e4), size=(n_samples - _mat_scales.shape[0], 4)))
            _mat_scales = np.vstack([_mat_scales, _extra])
        # 取每組基準材料值（從屬於該組的第一根桿件）
        _group_base = []
        for gname in _group_names:
            e0 = next((e for e in truss_data['elements'] if e.get('section') == gname), {})
            _group_base.append({
                'E': _to_float(e0.get('E',   E_base), E_base),
                'A': _to_float(e0.get('A',   A_base), A_base),
                'I': _to_float(e0.get('I33', e0.get('I', I_base)), I_base),
                'G': _to_float(e0.get('G',   G_base), G_base),
            })
        _cross_scales = _make_cross_samples(_n_groups, _group_base, n_samples)

    # 重新分配採樣陣列（n_samples 可能被 _n_groups 更新）
    samples_P_full = np.zeros((n_samples, total_dof))
    samples_w_full = np.zeros((n_samples, total_dof))
    basis_P_rows   = np.zeros((n_samples, n_elem * 3))
    basis_w_rows   = np.zeros((n_samples, n_elem * 1))
    # 反力採樣（indexed by fixed_dofs_list 位置）
    _n_fixed = len(fixed_dofs_list)
    samples_react_P = np.zeros((n_samples, _n_fixed))
    samples_react_w = np.zeros((n_samples, _n_fixed))
    # 桿件端力採樣（每根桿件 12 個端力分量）
    samples_ef_P = np.zeros((n_samples, n_elem * 12))
    samples_ef_w = np.zeros((n_samples, n_elem * 12))

    print(f"-> [Step 2/4] 開始 {n_samples} 組採樣（{'Hadamard 交叉' if _n_groups > 0 else '全域均勻'}模式）...")
    for s_idx in range(n_samples):
        if _n_groups > 0:
            # (B) Hadamard 交叉採樣：每組使用獨立 scale，桿件依所屬 section 取對應 scale
            scale_by_group = {gname: _cross_scales[s_idx][gi]
                              for gi, gname in enumerate(_group_names)}
            global_scale = float(np.mean(list(scale_by_group.values())))

            # 建立依 section 獨立縮放的 truss_data 副本
            td_scaled = copy.deepcopy(truss_data)
            elem_E_scaled = []
            elem_A_scaled = []
            elem_I_scaled = []
            elem_G_scaled = []
            raw_elems = {e['id']: e for e in truss_data['elements']}
            for k, info in enumerate(elements_info):
                raw_e = raw_elems[info['id']]
                sn = raw_e.get('section', '')
                sc = scale_by_group.get(sn, global_scale)
                e_val = per_elem_E[k] * sc
                a_val = per_elem_A[k] * sc
                i_val = per_elem_I[k] * sc
                g_val = per_elem_G[k] * sc
                elem_E_scaled.append(e_val)
                elem_A_scaled.append(a_val)
                elem_I_scaled.append(i_val)
                elem_G_scaled.append(g_val)
                # 更新副本中的對應桿件（依位置索引對齊 elements_info）
                td_scaled['elements'][k]['E']   = e_val
                td_scaled['elements'][k]['A']   = a_val
                td_scaled['elements'][k]['I33'] = i_val
                td_scaled['elements'][k]['I22'] = i_val
                td_scaled['elements'][k]['G']   = g_val
            E_s = E_base * global_scale
            A_s = A_base * global_scale
            I_s = I_base * global_scale
            G_s = G_base * global_scale
            K_s, elems_s, _, free_s = _assemble_K_np(td_scaled, E_s, A_s, I_s, G_s)
        else:
            # (A) 獨立材料採樣：E/A/I/G 各自使用不同 scale，使 basis 欄線性獨立
            sE, sA, sI, sG = _mat_scales[s_idx]
            E_s = E_base * sE
            A_s = A_base * sA
            I_s = I_base * sI
            G_s = G_base * sG

            # 無論是否有 per_elem，都必須覆寫桿件材料，否則 _assemble_K_np 優先用桿件既有值
            td_scaled = copy.deepcopy(truss_data)
            _base_E = per_elem_E if per_elem_E is not None else [E_base] * n_elem
            _base_A = per_elem_A if per_elem_A is not None else [A_base] * n_elem
            _base_I = per_elem_I if per_elem_I is not None else [I_base] * n_elem
            _base_G = per_elem_G if per_elem_G is not None else [G_base] * n_elem
            for k, elem in enumerate(td_scaled['elements']):
                elem['E']   = _base_E[k] * sE
                elem['A']   = _base_A[k] * sA
                elem['I33'] = _base_I[k] * sI
                elem['I22'] = _base_I[k] * sI
                elem['G']   = _base_G[k] * sG
            K_s, elems_s, _, free_s = _assemble_K_np(td_scaled, E_s, A_s, I_s, G_s)

        # free_s（來自 _assemble_K_np）含 slave DOF，需排除後才能與凝縮矩陣對齊
        free_s_no_slave = [d for d in free_s if d not in slave_dofs_set]
        assert set(free_s_no_slave) == set(free_dofs), \
            f"採樣 {s_idx}: free_dof 集合在縮放下改變，請檢查結構輸入"

        # 若有 rigid link，先凝縮 K_s；否則直接切 submatrix
        if s_idx == 0:
            _fp_max = float(np.max(np.abs(F_P_np_free))) if F_P_np_free.size > 0 else 0.0
            _fw_max = float(np.max(np.abs(F_w_np_free))) if F_w_np_free.size > 0 else 0.0
            print(f"[SAMPLE0] free_dofs={len(free_dofs)}, master_free_cols={len(master_free_cols)}, F_P_np_free max={_fp_max:.3e}, F_w_np_free max={_fw_max:.3e}, has_P={has_P_load}, has_w={has_w_load}")
        if rigid_links:
            F_zero_s = np.zeros(total_dof)
            K_cond_s, _, _ = apply_rigid_links(K_s, F_zero_s, node_list, rigid_links)
            K_red = K_cond_s[np.ix_(master_free_cols, master_free_cols)]
        else:
            K_cond_s = K_s
            K_red = K_s[np.ix_(free_s_no_slave, free_s_no_slave)]

        F_P_s = F_P_np_free
        F_w_s = F_w_np_free

        if has_P_load or has_w_load:
            _diag = np.diag(K_red)
            _diag_max = np.max(np.abs(_diag)) if len(_diag) > 0 else 1.0
            _threshold = _diag_max * 1e-10
            _zero_rows = np.where(np.abs(_diag) < _threshold)[0]
            if len(_zero_rows) > 0:
                _zero_node_info = []
                for _zr in _zero_rows[:20]:
                    _orig_dof = all_dofs_no_slave[master_free_cols[_zr]] if _zr < len(master_free_cols) else _zr
                    _nidx = _orig_dof // 6
                    _dof_name = ['ux','uy','uz','rx','ry','rz'][_orig_dof % 6]
                    _nid = idx_to_node_id.get(_nidx, f'idx{_nidx}')
                    _zero_node_info.append(f"node={_nid} dof={_dof_name} diag={_diag[_zr]:.3e}")
                print(f"[SINGULAR] 相對零對角線 (閾值={_threshold:.2e}): {_zero_node_info}")
            # 特徵值分析找機構方向
            try:
                _eigvals = np.linalg.eigvalsh(K_red)
                _neg_or_zero = _eigvals[_eigvals < _diag_max * 1e-8]
                if len(_neg_or_zero) > 0:
                    print(f"[SINGULAR] K_red 最小特徵值前10: {_eigvals[:10]}")
                    # 找最小特徵值對應的特徵向量，反查是哪些 DOF
                    _eigvals2, _eigvecs = np.linalg.eigh(K_red)
                    for _ev_idx in range(min(3, len(_eigvals2))):
                        if abs(_eigvals2[_ev_idx]) < _diag_max * 1e-8:
                            _vec = _eigvecs[:, _ev_idx]
                            _top_cols = np.argsort(np.abs(_vec))[::-1][:5]
                            _top_dofs = []
                            for _c in _top_cols:
                                _orig = all_dofs_no_slave[master_free_cols[_c]] if _c < len(master_free_cols) else _c
                                _ni = _orig // 6
                                _dn = ['ux','uy','uz','rx','ry','rz'][_orig % 6]
                                _top_dofs.append(f"{idx_to_node_id.get(_ni, f'idx{_ni}')}.{_dn}({_vec[_c]:.2f})")
                            print(f"[SINGULAR] 機構模態 λ={_eigvals2[_ev_idx]:.3e}: {_top_dofs}")
            except Exception as _e:
                print(f"[SINGULAR] 特徵值分析失敗: {_e}")
        _n_mfc = len(master_free_cols)
        U_P_s = np.linalg.solve(K_red, F_P_s) if has_P_load else np.zeros(_n_mfc)
        U_w_s = np.linalg.solve(K_red, F_w_s) if has_w_load else np.zeros(_n_mfc)

        # 展開至全域 DOF（用 all_dofs_no_slave[master_free_cols] 對應回全域 DOF）
        U_P_full_s = np.zeros(total_dof)
        U_w_full_s = np.zeros(total_dof)
        for i, col in enumerate(master_free_cols):
            d = all_dofs_no_slave[col]
            samples_P_full[s_idx, d] = U_P_s[i]
            samples_w_full[s_idx, d] = U_w_s[i]
            U_P_full_s[d] = U_P_s[i]
            U_w_full_s[d] = U_w_s[i]

        # 採樣反力：用凝縮後 K_cond_s
        # fixed_cols_no_slave：all_dofs_no_slave 中對應固定 DOF 的位置索引
        if _n_fixed > 0:
            _free_set = set(free_dofs)
            _fixed_cols_cond = [i for i, d in enumerate(all_dofs_no_slave) if d not in _free_set and d not in slave_dofs_set]
            _fixed_global_dofs = [all_dofs_no_slave[i] for i in _fixed_cols_cond]
            if _fixed_cols_cond:
                K_fx_fr_s = K_cond_s[np.ix_(_fixed_cols_cond, master_free_cols)]
                F_P_fixed = np.array([float(F_P_sym[d, 0].subs(P_sym, 1)) for d in _fixed_global_dofs])
                F_w_fixed = np.array([float(F_w_sym[d, 0].subs(w_sym, 1)) for d in _fixed_global_dofs])
                samples_react_P[s_idx] = K_fx_fr_s @ U_P_s - F_P_fixed
                samples_react_w[s_idx] = K_fx_fr_s @ U_w_s - F_w_fixed

        # 採樣桿件端力（使用當次採樣的 elements_info）
        _elems_for_ef = elems_s if (_n_groups > 0 or per_elem_E is not None) else elements_info
        for k_idx, info_s in enumerate(_elems_for_ef):
            ni_s, nj_s = info_s['nodes']
            dofs_s = _elem_dofs(ni_s, nj_s, NDOF)
            f_fix_s = info_s.get('f_fixed_local_sum')
            fix_P_s = np.zeros(12)
            fix_w_s = np.zeros(12)
            if f_fix_s is not None:
                for kk in range(12):
                    fk = f_fix_s[kk, 0]
                    if fk != sp.S.Zero:
                        fix_P_s[kk] = float(fk.subs(P_sym, 1).subs(w_sym, 0))
                        fix_w_s[kk] = float(fk.subs(w_sym, 1).subs(P_sym, 0))
            T_s  = info_s['T_np']
            kl_s = info_s['kl_np']
            f_P_s = kl_s @ (T_s @ U_P_full_s[dofs_s]) + fix_P_s
            f_w_s = kl_s @ (T_s @ U_w_full_s[dofs_s]) + fix_w_s
            SIGN = [1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1]
            for kk in range(12):
                samples_ef_P[s_idx, k_idx * 12 + kk] = SIGN[kk] * f_P_s[kk]
                samples_ef_w[s_idx, k_idx * 12 + kk] = SIGN[kk] * f_w_s[kk]

        if _n_groups > 0:
            # Hadamard 模式：basis 使用各桿件的獨立縮放後材料參數
            basis_P_rows[s_idx] = _build_basis_row(
                elem_Ls, E_s, A_s, I_s, G_s, 'P',
                elem_E_list=elem_E_scaled,
                elem_A_list=elem_A_scaled,
                elem_I_list=elem_I_scaled,
                elem_G_list=elem_G_scaled,
            )
            basis_w_rows[s_idx] = _build_basis_row(
                elem_Ls, E_s, A_s, I_s, G_s, 'w',
                elem_E_list=elem_E_scaled,
                elem_A_list=elem_A_scaled,
                elem_I_list=elem_I_scaled,
                elem_G_list=elem_G_scaled,
            )
        elif per_elem_E is not None:
            # element-ID keyed path：basis 使用各桿件實際材料參數（×獨立 scale）
            basis_P_rows[s_idx] = _build_basis_row(
                elem_Ls, E_s, A_s, I_s, G_s, 'P',
                elem_E_list=[v * sE for v in per_elem_E],
                elem_A_list=[v * sA for v in per_elem_A],
                elem_I_list=[v * sI for v in per_elem_I],
                elem_G_list=[v * sG for v in per_elem_G],
            )
            basis_w_rows[s_idx] = _build_basis_row(
                elem_Ls, E_s, A_s, I_s, G_s, 'w',
                elem_E_list=[v * sE for v in per_elem_E],
                elem_A_list=[v * sA for v in per_elem_A],
                elem_I_list=[v * sI for v in per_elem_I],
                elem_G_list=[v * sG for v in per_elem_G],
            )
        else:
            # 路徑 (A)：傳入各桿件的採樣後材料值（_base_* × scale）
            basis_P_rows[s_idx] = _build_basis_row(
                elem_Ls, E_s, A_s, I_s, G_s, 'P',
                elem_E_list=[v * sE for v in _base_E],
                elem_A_list=[v * sA for v in _base_A],
                elem_I_list=[v * sI for v in _base_I],
                elem_G_list=[v * sG for v in _base_G],
            )
            basis_w_rows[s_idx] = _build_basis_row(
                elem_Ls, E_s, A_s, I_s, G_s, 'w',
                elem_E_list=[v * sE for v in _base_E],
                elem_A_list=[v * sA for v in _base_A],
                elem_I_list=[v * sI for v in _base_I],
                elem_G_list=[v * sG for v in _base_G],
            )

    print(f"-> [Step 2/4] 採樣完成。耗時: {time.time()-start_time:.2f}s")

    # ── Slave 節點位移（用 base 材料數值解反推）────────────────────────────
    slave_displacements = {}
    if rigid_links:
        F_base_full = np.array([float(F_P_sym[d, 0].subs(P_sym, 1)) +
                                float(F_w_sym[d, 0].subs(w_sym, 1))
                                for d in range(total_dof)], dtype=float)
        K_cond_base, F_cond_base, _ = apply_rigid_links(
            K_base, F_base_full, node_list, rigid_links)
        F_free_base = F_cond_base[master_free_cols]
        K_free_base = K_cond_base[np.ix_(master_free_cols, master_free_cols)]
        if has_P_load or has_w_load:
            try:
                U_free_base = np.linalg.solve(K_free_base, F_free_base)
            except np.linalg.LinAlgError:
                U_free_base = np.zeros(len(master_free_cols))
        else:
            U_free_base = np.zeros(len(master_free_cols))
        U_full_base = np.zeros(total_dof)
        for i, d in enumerate(free_dofs):
            U_full_base[d] = U_free_base[i]
        slave_disps = recover_slave_displacements(U_full_base, node_list, rigid_links)
        slave_displacements = {k: v.tolist() for k, v in slave_disps.items()}

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

    # ── 反力與桿件端力：用多點採樣擬合，基底為 L^1/L^0（純力，不含 E/I）──────
    # 力的量綱：P 載重 → 係數 * P；w 載重 → 係數 * w * L_k
    # 基底：[L_1, L_2, ..., L_n, 1]（涵蓋 w·L 和 P 的常數項）
    def _build_force_basis_row(elem_Ls_val):
        """力的基底列：[L_1, L_2, ..., L_n, 1]，對 P 與 w 共用。"""
        row = list(elem_Ls_val) + [1.0]
        return np.array(row, dtype=np.float64)

    def _build_force_sym_bases(L_syms_list):
        return list(L_syms_list) + [sp.S.One]

    force_basis_rows = np.array([_build_force_basis_row(elem_Ls) for _ in range(n_samples)])
    force_sym_bases  = _build_force_sym_bases(L_syms)

    def _fit_force(b_col):
        """用力基底擬合單一分量，回傳係數與殘差。"""
        if np.all(np.abs(b_col) < 1e-40):
            return np.zeros(force_basis_rows.shape[1]), 0.0
        c, _, _, _ = np.linalg.lstsq(force_basis_rows, b_col, rcond=None)
        c = c.copy()
        pred = force_basis_rows @ c
        rel_err = np.max(np.abs(pred - b_col)) / (np.max(np.abs(b_col)) + 1e-40)
        return c, rel_err

    def _make_force_expr(c_P, c_w, tol_scale=1e-6):
        """將擬合係數組裝為 SymPy 表達式 c_P*P + c_w*w。"""
        tol_P = tol_scale * (np.max(np.abs(c_P)) if np.any(c_P) else 1.0)
        tol_w = tol_scale * (np.max(np.abs(c_w)) if np.any(c_w) else 1.0)
        terms = []
        for j, (cp, cw) in enumerate(zip(c_P, c_w)):
            base = force_sym_bases[j]
            if abs(cp) > tol_P:
                terms.append(sp.nsimplify(float(cp), tolerance=1e-6, rational=True) * base * P_sym)
            if abs(cw) > tol_w:
                terms.append(sp.nsimplify(float(cw), tolerance=1e-6, rational=True) * base * w_sym)
        return sp.Add(*terms) if terms else sp.S.Zero

    # 擬合桿件端力
    element_forces = []
    for k_idx, info in enumerate(elements_info):
        ni, nj = info['nodes']
        fi_exprs = []
        for kk in range(12):
            col_P = samples_ef_P[:, k_idx * 12 + kk]
            col_w = samples_ef_w[:, k_idx * 12 + kk]
            c_P, _ = _fit_force(col_P)
            c_w, _ = _fit_force(col_w)
            fi_exprs.append(_make_force_expr(c_P, c_w))

        element_forces.append({
            "element_id": info["id"],
            "nodes": f"N{idx_to_node_id[ni]} - N{idx_to_node_id[nj]}",
            "i_end (N, V2, V3, T, M2, M3)": (
                f"({fmt(fi_exprs[0])}, {fmt(fi_exprs[1])}, {fmt(fi_exprs[2])}, "
                f"{fmt(fi_exprs[3])}, {fmt(fi_exprs[4])}, {fmt(fi_exprs[5])})"
            ),
            "j_end (N, V2, V3, T, M2, M3)": (
                f"({fmt(fi_exprs[6])}, {fmt(fi_exprs[7])}, {fmt(fi_exprs[8])}, "
                f"{fmt(fi_exprs[9])}, {fmt(fi_exprs[10])}, {fmt(fi_exprs[11])})"
            ),
            "equations": {
                "N(x)":  fmt(fi_exprs[0]),
                "V2(x)": fmt(fi_exprs[1]),
                "V3(x)": fmt(fi_exprs[2]),
                "M3(x)": fmt(fi_exprs[5]),
                "M2(x)": fmt(fi_exprs[4]),
            },
            "status": "受力桿件"
        })

    # 擬合支承反力
    Reactions_full = sp.zeros(total_dof, 1)
    for ii, dof_idx in enumerate(fixed_dofs_list):
        c_P, _ = _fit_force(samples_react_P[:, ii])
        c_w, _ = _fit_force(samples_react_w[:, ii])
        Reactions_full[dof_idx, 0] = _make_force_expr(c_P, c_w)

    support_reactions = []
    for sup in truss_data['supports']:
        node_id = _norm_id(sup['node_id'])
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

    # 收集斷面組資訊，供快取失效比對與 subs_dict 重建
    _sec_groups = sorted(section_group_map.keys()) if section_group_map else []
    _sec_sym_names = {}
    if section_group_map:
        for sn, syms in section_group_map.items():
            _sec_sym_names[sn] = {k: str(v) for k, v in syms.items()}

    result = {
        "node_displacements":   node_displacements,
        "element_forces":       element_forces,
        "support_reactions":    support_reactions,
        "section_groups":       _sec_groups,
        "section_sym_names":    _sec_sym_names,
        "slave_displacements":  slave_displacements,
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


# ==============================================================================
# 純數值分析（第二階段：直接代入實際參數求解）
# ==============================================================================

def run_numerical_analysis(truss_data: dict, force_2d: str | None = "auto") -> dict:
    """
    直接使用 truss_data 內的實際 E/A/I/G/Le 與載重數值求解。
    不做符號擬合，結果為純數值。
    """
    node_list      = truss_data['nodes']
    n_nodes        = len(node_list)
    node_id_to_idx = {_norm_id(n['id']): i for i, n in enumerate(node_list)}
    idx_to_node_id = {i: _norm_id(n['id']) for i, n in enumerate(node_list)}
    NDOF           = 6
    total_dof      = NDOF * n_nodes

    # 以實際各桿件截面參數組裝剛度矩陣（_assemble_K_np 內部已逐桿讀取 E/A/I33/I22/J/G）
    # "auto" → 沿用幾何自動偵測；"3D" → 不套用任何面外約束；"XZ"/"XY" → 強制2D
    _f2d = None if force_2d == "auto" else (None if force_2d == "3D" else force_2d)
    K_np, elements_info, nodes_coords, free_dofs = _assemble_K_np(
        truss_data, E_s=200e9, A_s=1e-3, I_s=1e-5, G_s=77e9, _tag="[NUM]", force_2d=_f2d
        # 這些 fallback 值只在桿件未設定對應欄位時生效；
        # 桿件有 E/A/I33/G 時 _assemble_K_np 會優先使用桿件值。
    )
    fixed_dofs_list = sorted(set(range(total_dof)) - set(free_dofs))

    # ── 組裝載重向量（純數值）──────────────────────────────────────────────
    F_np = np.zeros(total_dof)

    # 桿件均佈載重
    f_fixed_local = {info['id']: np.zeros(12) for info in elements_info}
    for e_load in truss_data.get('element_loads', []):
        info = next((e for e in elements_info if _norm_id(e['id']) == _norm_id(e_load['element_id'])), None)
        if not info:
            continue
        Le_n = info['Le']
        w_val = _to_float(e_load.get('w', 0))
        f_fl = np.array([
            0, -w_val*Le_n/2, 0, 0, 0, -w_val*Le_n**2/12,
            0, -w_val*Le_n/2, 0, 0, 0,  w_val*Le_n**2/12,
        ])
        T_e = info['T_np']
        f_fg = T_e.T @ f_fl
        f_fixed_local[info['id']] += f_fl
        dofs = _elem_dofs(info['nodes'][0], info['nodes'][1], NDOF)
        for i, dof in enumerate(dofs):
            F_np[dof] -= f_fg[i]

    # 桿件集中載重
    for p_load in truss_data.get('element_point_loads', []):
        info = next((e for e in elements_info if _norm_id(e['id']) == _norm_id(p_load['element_id'])), None)
        if not info:
            continue
        Le_n  = info['Le']
        p_val = _to_float(p_load.get('p', 0))
        a_val = _to_float(p_load.get('a', 0))
        b_val = Le_n - a_val
        f_fl = np.array([
            0,
            -p_val * b_val**2 * (3*a_val + b_val) / Le_n**3,
            0, 0, 0,
            -p_val * a_val * b_val**2 / Le_n**2,
            0,
            -p_val * a_val**2 * (a_val + 3*b_val) / Le_n**3,
            0, 0, 0,
            p_val * a_val**2 * b_val / Le_n**2,
        ])
        T_e = info['T_np']
        f_fg = T_e.T @ f_fl
        f_fixed_local[info['id']] += f_fl
        dofs = _elem_dofs(info['nodes'][0], info['nodes'][1], NDOF)
        for i, dof in enumerate(dofs):
            F_np[dof] -= f_fg[i]

    # 節點集中載重
    for load in truss_data['loads']:
        if _norm_id(load.get('node_id')) not in node_id_to_idx:
            continue
        idx = node_id_to_idx[_norm_id(load['node_id'])]
        F_np[NDOF*idx+0] += _to_float(load.get('fx', 0))
        F_np[NDOF*idx+1] += _to_float(load.get('fy', 0))
        F_np[NDOF*idx+2] += _to_float(load.get('fz', 0))
        F_np[NDOF*idx+3] += _to_float(load.get('mx', 0))
        F_np[NDOF*idx+4] += _to_float(load.get('my', 0))
        F_np[NDOF*idx+5] += _to_float(load.get('mz', 0))

    # ── Rigid Link 凝縮後求解 ──────────────────────────────────────────────
    _rl_num = [rl for rl in truss_data.get('rigid_links', [])
               if _norm_id(rl.get('master', '')) != _norm_id(rl.get('slave', ''))]
    if _rl_num:
        # slave DOF 排除出自由度
        _slave_ids = {_norm_id(rl['slave']) for rl in _rl_num}
        _slave_idxs = {node_id_to_idx[s] for s in _slave_ids if s in node_id_to_idx}
        _slave_dofs = {idx * NDOF + k for idx in _slave_idxs for k in range(NDOF)}
        free_dofs_no_slave = [d for d in free_dofs if d not in _slave_dofs]
        # 凝縮
        K_cond, F_cond, _slave_info = apply_rigid_links(K_np, F_np, node_list, _rl_num)
        # all_dofs_no_slave 索引
        _all_no_slave = [d for d in range(total_dof) if d not in _slave_dofs]
        _free_cols = [_all_no_slave.index(d) for d in free_dofs_no_slave]
        K_red = K_cond[np.ix_(_free_cols, _free_cols)]
        F_red = F_cond[_free_cols]
        _fr_max = float(np.max(np.abs(F_red))) if F_red.size > 0 else 0.0
        _fn_max = float(np.max(np.abs(F_np))) if F_np.size > 0 else 0.0
        print(f"[NUM] free_dofs_no_slave={len(free_dofs_no_slave)}, F_red max={_fr_max:.3e}, F_np max={_fn_max:.3e}")
        U_full = np.zeros(total_dof)
        if len(free_dofs_no_slave) > 0 and np.any(np.abs(F_red) > 1e-30):
            U_red = np.linalg.solve(K_red, F_red)
            for i, d in enumerate(free_dofs_no_slave):
                U_full[d] = U_red[i]
        # 還原 slave 節點位移
        _slave_disps = recover_slave_displacements(U_full, node_list, _rl_num)
        for _sid, _uv in _slave_disps.items():
            if _sid in node_id_to_idx:
                _si = node_id_to_idx[_sid]
                U_full[_si*NDOF:_si*NDOF+NDOF] = _uv
    else:
        K_red = K_np[np.ix_(free_dofs, free_dofs)]
        F_red = F_np[free_dofs]
        U_full = np.zeros(total_dof)
        if len(free_dofs) > 0 and np.any(np.abs(F_red) > 1e-30):
            U_red = np.linalg.solve(K_red, F_red)
            for i, d in enumerate(free_dofs):
                U_full[d] = U_red[i]

    # ── 節點位移輸出 ───────────────────────────────────────────────────────
    node_displacements = []
    for idx in range(n_nodes):
        base = NDOF * idx
        node_displacements.append({
            "node_id":  idx_to_node_id[idx],
            "ux":       float(U_full[base+0]),
            "uy":       float(U_full[base+1]),
            "uz":       float(U_full[base+2]),
            "theta_x":  float(U_full[base+3]),
            "theta_y":  float(U_full[base+4]),
            "theta_z":  float(U_full[base+5]),
        })

    # ── 桿件內力輸出 ───────────────────────────────────────────────────────
    element_forces = []
    for info in elements_info:
        ni, nj = info['nodes']
        dofs   = _elem_dofs(ni, nj, NDOF)
        u_elem = U_full[dofs]
        T_e    = info['T_np']
        kl_e   = info['kl_np']
        f_local = kl_e @ (T_e @ u_elem) + f_fixed_local[info['id']]

        # 與 run_symbolic_analysis 相同的符號方向修正
        N_i  =  f_local[0]
        V2_i =  f_local[1]
        V3_i =  f_local[2]
        T_i  =  f_local[3]
        M2_i =  f_local[4]
        M3_i = -f_local[5]
        N_j  =  f_local[6]
        V2_j = -f_local[7]
        V3_j = -f_local[8]
        T_j  =  f_local[9]
        M2_j =  f_local[10]
        M3_j = -f_local[11]

        element_forces.append({
            "element_id": info["id"],
            "nodes":      f"N{idx_to_node_id[ni]} - N{idx_to_node_id[nj]}",
            "i_end (N, V2, V3, T, M2, M3)": (
                f"({N_i:.4g}, {V2_i:.4g}, {V3_i:.4g}, "
                f"{T_i:.4g}, {M2_i:.4g}, {M3_i:.4g})"
            ),
            "j_end (N, V2, V3, T, M2, M3)": (
                f"({N_j:.4g}, {V2_j:.4g}, {V3_j:.4g}, "
                f"{T_j:.4g}, {M2_j:.4g}, {M3_j:.4g})"
            ),
            "N":  float(N_i),
            "V2": float(V2_i),
            "V3": float(V3_i),
            "M3_i": float(M3_i),
            "M3_j": float(M3_j),
            "M2_i": float(M2_i),
            "M2_j": float(M2_j),
            "Le":   float(info['Le']),
        })

    # ── 支承反力輸出 ───────────────────────────────────────────────────────
    support_reactions = []
    if _rl_num:
        # RL 凝縮後的反力：用 K_cond / F_cond，固定 DOF = all_no_slave 中排除 free_no_slave
        _fixed_no_slave = [d for d in _all_no_slave if d not in set(free_dofs_no_slave)]
        if _fixed_no_slave:
            K_fx_fr = K_cond[np.ix_(_fixed_no_slave, _free_cols)]
            R_fixed_vals = K_fx_fr @ U_full[free_dofs_no_slave] - F_cond[_fixed_no_slave]
            for ii, dof_idx in enumerate(_fixed_no_slave):
                node_idx = dof_idx // NDOF
                nid      = idx_to_node_id[node_idx]
                dof_off  = dof_idx % NDOF
                keys = ["Rx", "Ry", "Rz", "Mx", "My", "Mz"]
                entry = next((r for r in support_reactions if r["node_id"] == nid), None)
                if entry is None:
                    entry = {"node_id": nid, "Rx": 0.0, "Ry": 0.0, "Rz": 0.0,
                             "Mx": 0.0, "My": 0.0, "Mz": 0.0}
                    support_reactions.append(entry)
                entry[keys[dof_off]] = float(R_fixed_vals[ii])
    elif fixed_dofs_list:
        K_fx_fr = K_np[np.ix_(fixed_dofs_list, free_dofs)]
        R_fixed = K_fx_fr @ U_full[free_dofs] - F_np[fixed_dofs_list]
        for ii, dof_idx in enumerate(fixed_dofs_list):
            node_idx = dof_idx // NDOF
            nid      = idx_to_node_id[node_idx]
            dof_off  = dof_idx % NDOF
            keys = ["Rx", "Ry", "Rz", "Mx", "My", "Mz"]
            entry = next((r for r in support_reactions if r["node_id"] == nid), None)
            if entry is None:
                entry = {"node_id": nid, "Rx": 0.0, "Ry": 0.0, "Rz": 0.0,
                         "Mx": 0.0, "My": 0.0, "Mz": 0.0}
                support_reactions.append(entry)
            entry[keys[dof_off]] = float(R_fixed[ii])

    _active_free_dofs = free_dofs_no_slave if _rl_num else free_dofs
    print(f"[NUM_RET] _rl_num={bool(_rl_num)}, free_dofs={len(free_dofs)}, _active_free_dofs={len(list(_active_free_dofs))}")
    return {
        "node_displacements":  node_displacements,
        "element_forces":      element_forces,
        "support_reactions":   support_reactions,
        "K_red_diagonal":      K_red.diagonal().tolist(),
        "free_dofs":           list(_active_free_dofs),
        "node_id_to_idx":      {k: v for k, v in node_id_to_idx.items()},
        "nodes_coords":        [list(c) for c in nodes_coords],
        "elements_info_lite":  [{"id": e["id"], "nodes": list(e["nodes"]), "Le": float(e["Le"])}
                                 for e in elements_info],
    }
