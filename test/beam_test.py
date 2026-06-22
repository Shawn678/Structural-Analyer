import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from core.symbolic import run_symbolic_analysis, run_numerical_analysis

# 二跨連續梁（XZ 平面）
# 節點在 z=1（y=0）：is_flat_y=True, is_flat_z=False → XZ 平面模式
# 面外自動固定：uy, rx, rz；面內自由度：ux, uz, ry
#
# 支承：A 固端(ux+uz+ry)，B/C 滾支承(uz)
#
# element_loads 的 w 作用在 local y2，對沿 x 軸桿件 local y2 = global +z
# 故 w=-1 → 均布載重方向為 -z（向下）
# 等效節點力在 -z 方向，支承提供 +z 反力（向上）
#
# 理論（固端A(ux+uz+ry) + 滾B(uz) + 滾C(uz)，L=1m，向下均布 |w|=1 N/m）：
#   Rz1 = 13/28 ~ 0.4643, Rz2 = 8/7 ~ 1.1429, Rz3 = 11/28 ~ 0.3929
#   My1 ~ -0.0714（固端彎矩，由剛度法求得）

truss_data = {
    "nodes": [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 1.0},
        {"id": 2, "x": 1.0, "y": 0.0, "z": 1.0},
        {"id": 3, "x": 2.0, "y": 0.0, "z": 1.0}
    ],
    "elements": [
        {"id": 1, "i": 1, "j": 2, "E": 200e9, "A": 0.01, "I33": 1e-4},
        {"id": 2, "i": 2, "j": 3, "E": 200e9, "A": 0.01, "I33": 1e-4}
    ],
    "supports": [
        {"node_id": 1, "ux": True, "uz": True, "ry": True},
        {"node_id": 2, "uz": True},
        {"node_id": 3, "uz": True}
    ],
    "loads": [],
    "element_loads": [
        {"element_id": 1, "w": -1.0},
        {"element_id": 2, "w": -1.0}
    ],
    "element_point_loads": []
}

try:
    # 數值驗證
    res_num = run_numerical_analysis(truss_data)
    print("=== 數值驗證 ===")
    sr = {r['node_id']: r for r in res_num["support_reactions"]}
    print(f"Rz1={sr[1]['Rz']:.4f}  (理論 13/28 ~ 0.4643)")
    print(f"Rz2={sr[2]['Rz']:.4f}  (理論 8/7  ~ 1.1429)")
    print(f"Rz3={sr[3]['Rz']:.4f}  (理論 11/28 ~ 0.3929)")
    print(f"My1={sr[1]['My']:.4f}  (理論 ~ -0.0714)")
    print(f"SUM Rz={sum(r['Rz'] for r in sr.values()):.4f}  (理論 2.0)")
    tol = 1e-3
    assert abs(sr[1]['Rz'] - 13/28) < tol, f"Rz1 error: {sr[1]['Rz']}"
    assert abs(sr[2]['Rz'] - 8/7)   < tol, f"Rz2 error: {sr[2]['Rz']}"
    assert abs(sr[3]['Rz'] - 11/28) < tol, f"Rz3 error: {sr[3]['Rz']}"
    print("數值驗證通過")
    print()

    res = run_symbolic_analysis(truss_data)
    print("Success! Displacements (Symbolic):")
    for disp in res["node_displacements"]:
        print(f"Node {disp['node_id']}: Uz={disp['uz']}, Ry={disp['theta_y']}")

    print("\nElement Forces (Equations):")
    for force in res["element_forces"]:
        print(f"Element {force['element_id']} ({force['nodes']}):")
        print(f"  M3(x) = {force['equations']['M3(x)']}")

    print("\nSupport Reactions:")
    for react in res["support_reactions"]:
        print(f"Node {react['node_id']}: Rz={react['Rz']}, My={react['My']}")
except Exception as e:
    import traceback
    traceback.print_exc()
