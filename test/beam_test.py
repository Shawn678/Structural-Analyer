import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from core.symbolic import run_symbolic_analysis

truss_data = {
    "nodes": [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}, # A
        {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0}, # B
        {"id": 3, "x": 2.0, "y": 0.0, "z": 0.0}  # C
    ],
    "elements": [
        {"id": 1, "i": 1, "j": 2, "E": 200e9, "A": 0.01, "I33": 1e-4},
        {"id": 2, "i": 2, "j": 3, "E": 200e9, "A": 0.01, "I33": 1e-4}
    ],
    "supports": [
        {"node_id": 1, "ux": True, "uz": True, "ry": True},  # A點: 固接 (固定Ux, Uz, Ry)
        {"node_id": 2, "uz": True},                         # B點: 滾支承 (固定Uz)
        {"node_id": 3, "uz": True}                          # C點: 滾支承 (固定Uz)
    ],
    "loads": [],
    "element_loads": [
        {"element_id": 1, "w": -1.0}, # 第一跨向下均佈載重
        {"element_id": 2, "w": -1.0}  # 第二跨向下均佈載重
    ],
    "element_point_loads": []
}

try:
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