import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.cable_wizard import generate_cable_face


def _base_params():
    return {
        "group_name": "t1_left",
        "tower_node_id": "T1",
        "tower_node_pos": {"x": 50.0, "y": 0.0, "z": 15.0},
        "tower_offset_start": -0.5,
        "tower_spacing": -0.5,
        "deck_x_start": 30.0,
        "deck_spacing": 5.0,
        "n_cables": 3,
        "eccentricity_y": 3.0,
        "deck_z": 0.0,
    }


def test_generate_counts():
    """3 根索 → 3 個主梁節點、3 個橋面偏心節點、3 個塔側偏心節點、6 個 RL、3 根索構件。"""
    params = _base_params()
    result = generate_cable_face(params, existing_nodes=[])
    nodes = result["nodes"]
    elements = result["elements"]
    rls = result["rigid_links"]

    deck_center = [n for n in nodes if n.get("_role") == "deck_center"]
    deck_ecc    = [n for n in nodes if n.get("_role") == "deck_ecc"]
    tower_ecc   = [n for n in nodes if n.get("_role") == "tower_ecc"]
    cables      = [e for e in elements if e.get("_role") == "cable"]

    assert len(deck_center) == 3
    assert len(deck_ecc)    == 3
    assert len(tower_ecc)   == 3
    assert len(cables)      == 3
    assert len(rls)         == 6


def test_deck_center_coords():
    """橋面中心線節點 x 座標應為 30, 35, 40；y=0；z=0。"""
    params = _base_params()
    result = generate_cable_face(params, existing_nodes=[])
    deck_centers = sorted(
        [n for n in result["nodes"] if n.get("_role") == "deck_center"],
        key=lambda n: n["x"]
    )
    expected_x = [30.0, 35.0, 40.0]
    for n, ex in zip(deck_centers, expected_x):
        assert abs(n["x"] - ex) < 1e-9
        assert abs(n["y"] - 0.0) < 1e-9
        assert abs(n["z"] - 0.0) < 1e-9


def test_tower_ecc_coords():
    """塔側偏心節點應在塔頂往下 0.5/1.0/1.5m，y=3.0。"""
    params = _base_params()
    result = generate_cable_face(params, existing_nodes=[])
    tower_eccs = sorted(
        [n for n in result["nodes"] if n.get("_role") == "tower_ecc"],
        key=lambda n: n["z"], reverse=True
    )
    expected_z = [15.0 - 0.5, 15.0 - 1.0, 15.0 - 1.5]
    for n, ez in zip(tower_eccs, expected_z):
        assert abs(n["z"] - ez) < 1e-9
        assert abs(n["y"] - 3.0) < 1e-9


def test_reuse_existing_node():
    """若橋面中心線節點座標已存在，應複用其 id；回傳節點仍為 9 個（位置不少），但該節點 id=N_existing。"""
    params = _base_params()
    existing = [{"id": "N_existing", "x": 30.0, "y": 0.0, "z": 0.0, "group": ""}]
    result = generate_cable_face(params, existing_nodes=existing)
    deck_centers = [n for n in result["nodes"] if n.get("_role") == "deck_center"]
    ids = [n["id"] for n in deck_centers]
    assert "N_existing" in ids
    # 不應有重複的 id（同一節點只出現一次）
    all_ids = [n["id"] for n in result["nodes"]]
    assert len(all_ids) == len(set(all_ids))


def test_group_label():
    """所有生成節點和構件應帶正確 group 標籤。"""
    params = _base_params()
    result = generate_cable_face(params, existing_nodes=[])
    for n in result["nodes"]:
        assert n.get("group") == "gen:t1_left"
    for e in result["elements"]:
        assert e.get("group") == "gen:t1_left"
    for rl in result["rigid_links"]:
        assert rl.get("group") == "gen:t1_left"
