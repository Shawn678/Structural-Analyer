import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.rigid_link import apply_rigid_links, recover_slave_displacements, _build_T_rigid


def test_single_slave_pure_translation():
    """slave 節點只有平移偏心，拘束後 K 維度應縮減 6。"""
    n_nodes = 2
    NDOF = 6
    size = n_nodes * NDOF
    K = np.eye(size) * 1000.0
    F = np.ones(size)

    nodes = [
        {"id": "M", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "S", "x": 0.0, "y": 3.0, "z": 0.0},
    ]
    rigid_links = [{"id": "rl1", "master": "M", "slave": "S", "group": ""}]

    K_red, F_red, slave_info = apply_rigid_links(K, F, nodes, rigid_links)

    assert K_red.shape == (6, 6)
    assert F_red.shape == (6,)
    assert len(slave_info) == 1


def test_rigid_body_rotation():
    """master 轉角 rz=0.01 rad，slave 在 y=3 處，應產生 ux = 0.01*3 = 0.03。"""
    d = np.array([0.0, 3.0, 0.0])
    T = _build_T_rigid(d)
    u_master = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01])  # rz=0.01
    u_slave = T @ u_master
    assert abs(u_slave[0] - 0.03) < 1e-10
    assert abs(u_slave[1] - 0.0)  < 1e-10


def test_recover_slave_displacements():
    """驗證反推位移與 _build_T_rigid 結果一致。"""
    nodes = [
        {"id": "M", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "S", "x": 0.0, "y": 3.0, "z": 0.0},
    ]
    rigid_links = [{"id": "rl1", "master": "M", "slave": "S", "group": ""}]
    U = np.zeros(12)
    U[5] = 0.01  # M 節點 rz = 0.01
    result = recover_slave_displacements(U, nodes, rigid_links)
    assert "S" in result
    assert abs(result["S"][0] - 0.03) < 1e-10


def test_analysis_with_rigid_link():
    """
    簡單模型：主梁節點 A(0,0,0)，slave 節點 B(0,3,0) 透過 rigid link 連到 A。
    一根柱 C(0,0,5) 到 A，C 固定，A 施加水平力 fx=1000N。
    分析不應崩潰，且結果包含 slave_displacements。
    """
    from core.symbolic import run_symbolic_analysis

    truss_data = {
        "nodes": [
            {"id": "C", "x": 0.0, "y": 0.0, "z": 5.0},
            {"id": "A", "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": "B", "x": 0.0, "y": 3.0, "z": 0.0},
        ],
        "elements": [
            {"id": 1, "i": "C", "j": "A",
             "E": 200e9, "G": 77e9, "A": 0.01,
             "I33": 1e-4, "I22": 1e-4, "J": 2e-4,
             "pin_i": False, "pin_j": False},
        ],
        "supports": [
            {"node_id": "C", "ux": True, "uy": True, "uz": True,
             "rx": True, "ry": True, "rz": True},
        ],
        "loads": [
            {"node_id": "A", "fx": 1000.0},
        ],
        "element_loads": [],
        "element_point_loads": [],
        "rigid_links": [
            {"id": "rl1", "master": "A", "slave": "B", "group": ""},
        ],
    }
    result = run_symbolic_analysis(truss_data)
    assert "error" not in result, f"分析回傳錯誤: {result.get('error')}"
    assert "slave_displacements" in result
    b_disp = result["slave_displacements"].get("B", [0] * 6)
    assert len(b_disp) == 6
