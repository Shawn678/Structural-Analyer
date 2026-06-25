import math
import numpy as np

_COORD_TOL = 1e-6


def _norm_id(val) -> str:
    s = str(val).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s


def _find_existing_node(existing_nodes: list, x: float, y: float, z: float):
    for n in existing_nodes:
        if (abs(float(n.get('x', 0)) - x) < _COORD_TOL and
                abs(float(n.get('y', 0)) - y) < _COORD_TOL and
                abs(float(n.get('z', 0)) - z) < _COORD_TOL):
            return n
    return None


def generate_cable_face(params: dict, existing_nodes: list) -> dict:
    """
    依索面參數生成所有相關元素。

    params keys:
        group_name          : str
        tower_node_id       : str
        tower_node_pos      : {"x": float, "y": float, "z": float}
        tower_offset_start  : float  (負值往下)
        tower_spacing       : float  (負值往下)
        deck_x_start        : float
        deck_spacing        : float  (正往跨中, 負往橋台)
        n_cables            : int
        eccentricity_y      : float
        deck_z              : float

    回傳 {"nodes": [...], "elements": [...], "rigid_links": [...]}
    節點含 _role 欄位（deck_center / deck_ecc / tower_ecc），匯入主表格前應移除。
    """
    grp = f"gen:{params['group_name']}"
    n_cables = int(params['n_cables'])
    ecc_y = float(params['eccentricity_y'])
    tower_ecc_y = float(params.get('tower_eccentricity_y', ecc_y))
    deck_z = float(params['deck_z'])
    dx_start = float(params['deck_x_start'])
    dx_space = float(params['deck_spacing'])
    t_pos = params['tower_node_pos']
    tx = float(t_pos['x'])
    ty = float(t_pos['y'])
    tz = float(t_pos['z'])
    t_off_start = float(params['tower_offset_start'])
    t_spacing = float(params['tower_spacing'])
    tower_id = params['tower_node_id']

    new_nodes = []
    elements = []
    rigid_links = []

    all_known = list(existing_nodes)

    _added_ids = set()

    def _add_node(x, y, z, role):
        existing = _find_existing_node(all_known, x, y, z)
        if existing:
            n = {**existing, "_role": role, "group": grp}
            if n["id"] not in _added_ids:
                new_nodes.append(n)
                _added_ids.add(n["id"])
            return n, False
        uid = f"gen_{params['group_name']}_{role}_{len(new_nodes)}"
        n = {"id": uid, "x": x, "y": y, "z": z, "group": grp, "_role": role}
        new_nodes.append(n)
        _added_ids.add(uid)
        all_known.append(n)
        return n, True

    deck_center_nodes = []
    deck_ecc_nodes = []
    tower_ecc_nodes = []

    for i in range(n_cables):
        cx = dx_start + i * dx_space

        dc, _ = _add_node(cx, 0.0, deck_z, "deck_center")
        deck_center_nodes.append(dc)

        de, _ = _add_node(cx, ecc_y, deck_z, "deck_ecc")
        deck_ecc_nodes.append(de)

        tez = tz + t_off_start + i * t_spacing
        te, _ = _add_node(tx, ty + tower_ecc_y, tez, "tower_ecc")
        tower_ecc_nodes.append(te)

    rl_idx = 0
    for dc, de in zip(deck_center_nodes, deck_ecc_nodes):
        rigid_links.append({
            "id": f"gen_{params['group_name']}_rl_{rl_idx}",
            "master": dc["id"],
            "slave": de["id"],
            "group": grp,
        })
        rl_idx += 1

    for te in tower_ecc_nodes:
        rigid_links.append({
            "id": f"gen_{params['group_name']}_rl_{rl_idx}",
            "master": tower_id,
            "slave": te["id"],
            "group": grp,
        })
        rl_idx += 1

    for i, (te, de) in enumerate(zip(tower_ecc_nodes, deck_ecc_nodes)):
        elements.append({
            "id": f"gen_{params['group_name']}_cable_{i}",
            "i": te["id"],
            "j": de["id"],
            "pin_i": True,
            "pin_j": True,
            "group": grp,
            "_role": "cable",
            "section": "",
            "E": None, "G": None, "A": None,
            "I33": None, "I22": None, "J": None,
            "beta": 0.0, "dL": 0.0,
        })

    return {"nodes": new_nodes, "elements": elements, "rigid_links": rigid_links}


def split_beam_at_nodes(
    elements: list,
    beam_member: str,
    split_nodes: list,
    node_map: dict,
) -> tuple[list, int]:
    """
    將 member == beam_member 的主梁桿件，在 split_nodes 的 x 位置處拆成兩段。
    split_nodes：含 {"id", "x", "y", "z"} 的節點列表（deck_center 節點）。
    node_map：str(id) -> node dict，用來查現有節點座標。
    回傳 (新的 elements 列表, 拆分次數)。
    """
    _TOL = 1e-6
    split_count = 0
    result = []

    # 已存在的 deck_center 節點 ID 集合，用來跳過已拆分的子桿件
    _split_node_ids = {_norm_id(n["id"]) for n in split_nodes}

    for elem in elements:
        if str(elem.get("member", "")) != beam_member:
            result.append(elem)
            continue

        ni_id = _norm_id(elem.get("i", ""))
        nj_id = _norm_id(elem.get("j", ""))

        # 端點已是 deck_center 節點 → 上次拆分產生的子段，跳過
        if ni_id in _split_node_ids or nj_id in _split_node_ids:
            result.append(elem)
            continue

        ni = node_map.get(ni_id)
        nj = node_map.get(nj_id)
        if ni is None or nj is None:
            result.append(elem)
            continue

        xi, xj = float(ni.get("x", 0)), float(nj.get("x", 0))
        x_lo, x_hi = min(xi, xj), max(xi, xj)

        # 找落在此桿件範圍內（不含端點）的所有錨點，按 x 排序
        mid_nodes = [
            n for n in split_nodes
            if x_lo + _TOL < float(n.get("x", 0)) < x_hi - _TOL
        ]
        mid_nodes.sort(key=lambda n: float(n.get("x", 0)) if xi <= xj else -float(n.get("x", 0)))

        if not mid_nodes:
            result.append(elem)
            continue

        # 依序拆分：i → mid0 → mid1 → ... → j
        chain = [ni_id] + [str(n["id"]) for n in mid_nodes] + [nj_id]
        base_attrs = {k: v for k, v in elem.items() if k not in ("id", "i", "j")}
        for seg_idx in range(len(chain) - 1):
            new_elem = {
                "id": f"{elem['id']}_seg{seg_idx}",
                "i": chain[seg_idx],
                "j": chain[seg_idx + 1],
                **base_attrs,
            }
            result.append(new_elem)
        split_count += 1

    return result, split_count


def _solve_three_moment(support_xs: list, w: float) -> np.ndarray:
    """
    以三彎矩方程解均布載重 w 下連續梁各內部支承彎矩。
    support_xs：所有支承 x 座標（升序，含兩端，端點彎矩固定為 0）。
    回傳長度 n 的 array，index 對應 support_xs，端點值為 0。
    """
    xs = sorted(support_xs)
    n = len(xs)
    if n < 2:
        return np.zeros(n)
    if n == 2:
        return np.zeros(n)

    # 內部支承數量
    inner = n - 2
    if inner == 0:
        return np.zeros(n)

    # 建立方程組：對每個內部支承 i (1 ~ n-2) 寫三彎矩方程
    # M_{i-1}*L_i + 2*M_i*(L_i+L_{i+1}) + M_{i+1}*L_{i+1} = -w/4*(L_i^3 + L_{i+1}^3)
    A = np.zeros((inner, inner))
    b = np.zeros(inner)

    for k in range(inner):
        i = k + 1  # 全域 index
        Li  = xs[i]   - xs[i-1]
        Li1 = xs[i+1] - xs[i]
        A[k, k] = 2.0 * (Li + Li1)
        if k > 0:
            A[k, k-1] = Li
        if k < inner - 1:
            A[k, k+1] = Li1
        b[k] = -w / 4.0 * (Li**3 + Li1**3)

    M_inner = np.linalg.solve(A, b)
    M = np.zeros(n)
    M[1:-1] = M_inner
    return M


def _beam_moment_at(x: float, support_xs: list, M_supports: np.ndarray, w: float) -> float:
    """
    計算連續梁在位置 x 的彎矩值。
    以 x 所在跨間的端點彎矩和均布載重做插值。
    """
    xs = sorted(support_xs)
    for i in range(len(xs) - 1):
        x_l, x_r = xs[i], xs[i+1]
        if x_l <= x <= x_r + 1e-10:
            L = x_r - x_l
            xi = x - x_l
            M_l = M_supports[i]
            M_r = M_supports[i+1]
            # 簡支梁 + 端點彎矩疊加
            M_simple = w * xi * (L - xi) / 2.0
            M_end = M_l * (1 - xi / L) + M_r * (xi / L)
            return M_simple + M_end
    return 0.0


def _beam_reactions(support_xs: list, M_supports: np.ndarray, w: float) -> np.ndarray:
    """計算各支承垂直反力（向上為正）。"""
    xs = sorted(support_xs)
    n = len(xs)
    R = np.zeros(n)
    for i in range(n - 1):
        L = xs[i+1] - xs[i]
        M_l = M_supports[i]
        M_r = M_supports[i+1]
        # 靜力平衡：左端反力 = wL/2 + (M_r - M_l)/L
        R_left  = w * L / 2.0 + (M_r - M_l) / L
        R_right = w * L / 2.0 - (M_r - M_l) / L
        R[i]   += R_left
        R[i+1] += R_right
    return R


def _solve_three_moment_with_point_load(
    support_xs: list,
    w: float,
    load_x: float,
    P: float,
) -> np.ndarray:
    """
    三彎矩方程含單一集中力：在 load_x 施加向上集中力 P（N），
    回傳各支承彎矩 array。
    集中力的三彎矩右端項：
      對跨 [i, i+1] 內位置 a 的集中力 P（向下為正，向上為負），
      對左支承 i 的方程右端貢獻：-P*a*(L^2 - a^2)/(L)  （Clapeyron 標準形式）
      對右支承 i+1 的方程右端貢獻：-P*(L-a)*(2L*L - (L-a)^2 - L^2)/(L) = -P*b*(L^2-b^2)/L
    向上集中力 P_up：等效為施加向下 P = -P_up。
    """
    xs = sorted(support_xs)
    n = len(xs)
    if n < 2:
        return np.zeros(n)

    inner = n - 2
    if inner <= 0:
        return np.zeros(n)

    # 找集中力所在跨
    span_i = None
    for i in range(n - 1):
        if xs[i] - 1e-10 <= load_x <= xs[i+1] + 1e-10:
            span_i = i
            break
    if span_i is None:
        return _solve_three_moment(support_xs, w)

    L_span = xs[span_i+1] - xs[span_i]
    a = max(0.0, min(load_x - xs[span_i], L_span))
    b = L_span - a
    P_down = -P  # 向上為正 → 向下為負

    A = np.zeros((inner, inner))
    rhs = np.zeros(inner)

    for k in range(inner):
        i = k + 1
        Li  = xs[i]   - xs[i-1]
        Li1 = xs[i+1] - xs[i]
        A[k, k] = 2.0 * (Li + Li1)
        if k > 0:
            A[k, k-1] = Li
        if k < inner - 1:
            A[k, k+1] = Li1
        # 均布載重項
        rhs[k] = -w / 4.0 * (Li**3 + Li1**3)
        # 集中力貢獻（三彎矩標準公式）
        if i - 1 == span_i:
            # 集中力在左跨 [i-1, i]，對 k 方程（支承 i）的右端貢獻
            rhs[k] += -P_down * a * b * (Li + b) / Li
        if i == span_i:
            # 集中力在右跨 [i, i+1]，對 k 方程（支承 i）的右端貢獻
            rhs[k] += -P_down * a * b * (Li1 + a) / Li1

    M_inner = np.linalg.solve(A, rhs)
    M = np.zeros(n)
    M[1:-1] = M_inner
    return M


def _beam_moment_at_with_point_load(
    x_query: float,
    support_xs: list,
    M_supports: np.ndarray,
    w: float,
    load_x: float,
    P: float,
) -> float:
    """
    在含集中力的連續梁中，計算 x_query 位置的彎矩。
    M_supports 已包含集中力對支承彎矩的影響（由 _solve_three_moment_with_point_load 算出）。
    這裡只需在 x_query 所在跨再疊加集中力的「跨內簡支直接貢獻」。

    簡支梁集中力 P（向下為正）在位置 a（距左端），x_query=xi（距左端）：
      若 xi <= a：M = P * b * xi / L
      若 xi >  a：M = P * a * (L - xi) / L
    向上集中力 P_up → P_down = -P_up，最終取負。
    """
    xs = sorted(support_xs)
    base = _beam_moment_at(x_query, xs, M_supports, w)

    for i in range(len(xs) - 1):
        x_l, x_r = xs[i], xs[i+1]
        if x_l - 1e-10 <= load_x <= x_r + 1e-10:
            L = x_r - x_l
            if L < 1e-10:
                return base
            a  = max(0.0, min(load_x - x_l, L))
            b  = L - a
            xi = x_query - x_l
            if 0.0 <= xi <= L + 1e-10:
                xi = max(0.0, min(xi, L))
                if xi <= a:
                    direct = P * b * xi / L
                else:
                    direct = P * a * (L - xi) / L
            else:
                direct = 0.0
            return base + direct
    return base


def compute_three_moment(
    support_xs: list,
    anchor_xs: list,
    w: float,
) -> dict:
    """
    三彎矩方程：計算無索狀態下連續梁的彎矩分佈與各錨點影響線係數。

    Parameters
    ----------
    support_xs : 真實支承（橋台+橋墩）x 座標，升序
    anchor_xs  : 索錨點 x 座標，升序
    w          : 主梁等效線重（N/m，向下為正）

    Returns
    -------
    dict with keys:
        "M_max"      : float  無索狀態最大彎矩（N·m，正值）
        "M_max_x"    : float  最大彎矩位置 x
        "influence"  : list[float]  各錨點對 M_max 的影響線係數（N·m / N，負值表示向上力降低彎矩）
    """
    xs = sorted(support_xs)
    M_sup = _solve_three_moment(xs, w)

    x_min, x_max = xs[0], xs[-1]
    x_samples = np.linspace(x_min, x_max, 600)
    M_samples = np.array([_beam_moment_at(x, xs, M_sup, w) for x in x_samples])
    idx_max = int(np.argmax(M_samples))
    M_max   = float(M_samples[idx_max])
    M_max_x = float(x_samples[idx_max])

    # 影響線：在錨點施加 1N 向上集中力，計算 M_max_x 處彎矩變化
    M_base_at_peak = _beam_moment_at(M_max_x, xs, M_sup, w)
    influence = []
    for ax in anchor_xs:
        # 施加 1N 向上集中力（P=+1 表示向上）
        # _solve_three_moment_with_point_load 內部以向上為正處理
        M_sup_p = _solve_three_moment_with_point_load(xs, w, ax, 1.0)
        M_with  = _beam_moment_at_with_point_load(M_max_x, xs, M_sup_p, w, ax, -1.0)
        influence.append(float(M_with - M_base_at_peak))

    return {
        "M_max":     M_max,
        "M_max_x":   M_max_x,
        "influence": influence,
    }


def compute_cable_pretension_guess(
    support_xs: list,
    anchor_xs: list,
    thetas: list,
    w: float,
    M_allow: float,
) -> dict:
    """
    計算初始索力猜測值。

    Parameters
    ----------
    support_xs : 真實支承 x 座標（升序）
    anchor_xs  : 索錨點 x 座標（升序），與 thetas 對應
    thetas     : 各索傾角（弧度，與水平面夾角，正值）
    w          : 主梁等效線重（N/m）
    M_allow    : 主梁標稱彎矩容量（N·m，正值）

    Returns
    -------
    dict with keys:
        "M_max"        : float  無索狀態最大彎矩（N·m）
        "M_max_x"      : float  最大彎矩位置
        "cable_forces" : list[float]  各索初始軸力（N），與 anchor_xs 對應
        "V_uniform"    : float  每根索提供的均勻垂直力（N）
        "feasible"     : bool   M_max > M_allow（索力有意義）
    """
    result = compute_three_moment(support_xs, anchor_xs, w)
    M_max   = result["M_max"]
    M_max_x = result["M_max_x"]
    inf_list = result["influence"]

    feasible = M_max > M_allow
    if not feasible or not inf_list:
        return {
            "M_max":        M_max,
            "M_max_x":      M_max_x,
            "cable_forces": [0.0] * len(anchor_xs),
            "V_uniform":    0.0,
            "feasible":     feasible,
        }

    delta_M = M_max - M_allow
    # 只對有效降矩的錨點（influence < 0）加總，算出均勻垂直力 V
    # sum(V * inf_i for inf_i < 0) = -delta_M  →  V = delta_M / sum(|inf_i|)
    eff_sum = sum(-inf for inf in inf_list if inf < 0)
    if eff_sum < 1e-12:
        # 所有錨點位置對 M_max 點沒有有效降矩，fallback 為簡單靜力均分
        total_span = max(support_xs) - min(support_xs)
        V = delta_M / max(len(anchor_xs), 1) / (total_span / 2)
    else:
        V = delta_M / eff_sum

    cable_forces = []
    for theta in thetas:
        sin_t = math.sin(theta)
        if sin_t < 1e-6:
            cable_forces.append(0.0)
        else:
            cable_forces.append(V / sin_t)

    return {
        "M_max":        M_max,
        "M_max_x":      M_max_x,
        "cable_forces": cable_forces,
        "V_uniform":    V,
        "feasible":     feasible,
    }
