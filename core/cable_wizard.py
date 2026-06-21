_COORD_TOL = 1e-6


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
        te, _ = _add_node(tx, ty + ecc_y, tez, "tower_ecc")
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
