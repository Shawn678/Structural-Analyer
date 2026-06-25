import numpy as np

NDOF = 6


def _norm_id(val) -> str:
    s = str(val).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s

def _build_T_rigid(d: np.ndarray) -> np.ndarray:
    """
    建立 6×6 剛體轉換矩陣 T，使得 u_slave = T @ u_master。
    d = slave_pos - master_pos (偏心向量)
    自由度順序：[ux, uy, uz, rx, ry, rz]
    """
    dx, dy, dz = d
    T = np.eye(6)
    T[0, 5] =  dy   # ux_slave += rz_master * dy
    T[0, 4] = -dz   # ux_slave -= ry_master * dz
    T[1, 5] = -dx   # uy_slave -= rz_master * dx
    T[1, 3] =  dz   # uy_slave += rx_master * dz
    T[2, 4] =  dx   # uz_slave += ry_master * dx
    T[2, 3] = -dy   # uz_slave -= rx_master * dy
    return T


def apply_rigid_links(
    K: np.ndarray,
    F: np.ndarray,
    nodes: list,
    rigid_links: list,
) -> tuple:
    """
    對全域剛度矩陣 K 和載重向量 F 套用 Rigid Link 靜態凝縮。
    消去所有 slave 節點的自由度，回傳縮減後的 K_red, F_red。

    回傳：
        K_red       : np.ndarray
        F_red       : np.ndarray
        slave_info  : dict, {slave_dof_start: (master_dof_start, T_6x6)}
    """
    if not rigid_links:
        return K.copy(), F.copy(), {}

    # master==slave 的 RL 無效，過濾掉
    rigid_links = [rl for rl in rigid_links if _norm_id(rl['master']) != _norm_id(rl['slave'])]
    if not rigid_links:
        return K.copy(), F.copy(), {}

    node_id_to_idx = {_norm_id(n['id']): i for i, n in enumerate(nodes)}
    n_total = K.shape[0]

    slave_dof_starts = set()
    slave_info = {}

    for rl in rigid_links:
        m_id = _norm_id(rl['master'])
        s_id = _norm_id(rl['slave'])
        if m_id not in node_id_to_idx or s_id not in node_id_to_idx:
            continue
        mi = node_id_to_idx[m_id]
        si = node_id_to_idx[s_id]

        mn = nodes[mi]
        sn = nodes[si]
        d = np.array([
            float(sn.get('x', 0)) - float(mn.get('x', 0)),
            float(sn.get('y', 0)) - float(mn.get('y', 0)),
            float(sn.get('z', 0)) - float(mn.get('z', 0)),
        ])
        T = _build_T_rigid(d)

        m_start = mi * NDOF
        s_start = si * NDOF
        slave_dof_starts.add(s_start)
        slave_info[s_start] = (m_start, T)

    if not slave_info:
        return K.copy(), F.copy(), {}

    all_dofs = list(range(n_total))
    slave_dofs_set = set()
    for s_start in slave_dof_starts:
        for k in range(NDOF):
            slave_dofs_set.add(s_start + k)
    master_dofs = [d for d in all_dofs if d not in slave_dofs_set]

    n_m = len(master_dofs)

    # 全域轉換矩陣 T_full: shape=(n_total, n_m)
    T_full = np.zeros((n_total, n_m))

    for new_col, old_row in enumerate(master_dofs):
        T_full[old_row, new_col] = 1.0

    master_dof_to_col = {dof: col for col, dof in enumerate(master_dofs)}
    for s_start, (m_start, T_6x6) in slave_info.items():
        for si_local in range(NDOF):
            s_global = s_start + si_local
            for mi_local in range(NDOF):
                m_global = m_start + mi_local
                col = master_dof_to_col.get(m_global)
                if col is not None:
                    T_full[s_global, col] += T_6x6[si_local, mi_local]

    K_red = T_full.T @ K @ T_full
    F_red = T_full.T @ F

    return K_red, F_red, slave_info


def recover_slave_displacements(
    U_master: np.ndarray,
    nodes: list,
    rigid_links: list,
) -> dict:
    """
    從已求解的全尺寸位移向量反推各 slave 節點位移。
    回傳：{slave_node_id: np.ndarray(6)}
    """
    node_id_to_idx = {_norm_id(n['id']): i for i, n in enumerate(nodes)}
    result = {}
    for rl in [r for r in rigid_links if _norm_id(r['master']) != _norm_id(r['slave'])]:
        m_id = _norm_id(rl['master'])
        s_id = _norm_id(rl['slave'])
        if m_id not in node_id_to_idx or s_id not in node_id_to_idx:
            continue
        mi = node_id_to_idx[m_id]
        si = node_id_to_idx[s_id]
        mn = nodes[mi]
        sn = nodes[si]
        d = np.array([
            float(sn.get('x', 0)) - float(mn.get('x', 0)),
            float(sn.get('y', 0)) - float(mn.get('y', 0)),
            float(sn.get('z', 0)) - float(mn.get('z', 0)),
        ])
        T = _build_T_rigid(d)
        u_m = U_master[mi * NDOF: mi * NDOF + NDOF]
        result[s_id] = T @ u_m
    return result
