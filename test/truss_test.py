import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from core.symbolic import run_symbolic_analysis

truss_data = {
    "nodes": [
        {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": 2, "x": 3.0, "y": 0.0, "z": 0.0},
        {"id": 3, "x": 3.0, "y": 0.0, "z": 4.0}
    ],
    "elements": [
        {"id": 1, "i": 1, "j": 2, "E": 200e9, "A": 0.01, "pin_i": True, "pin_j": True},
        {"id": 2, "i": 1, "j": 3, "E": 200e9, "A": 0.01, "pin_i": True, "pin_j": True}
    ],
    "supports": [
        # 2D 桁架支承：固定移動自由度即可，旋轉自由度會被 symbolic.py 自動處理
        {"node_id": 2, "ux": True, "uy": True, "uz": True},
        {"node_id": 3, "ux": True, "uy": True, "uz": True}
    ],
    "loads": [
        {"node_id": 1, "fx": 0.0, "fy": 0.0, "fz": -2.0} # 作用在頂點的向下集中力
    ],
    "element_loads": [],
    "element_point_loads": []
}

try:
    res = run_symbolic_analysis(truss_data)
    print("Success! Displacements:")
    for disp in res["node_displacements"]:
        print(disp)
    print("Element Forces:")
    for force in res["element_forces"]:
        print(force)
except Exception as e:
    import traceback
    traceback.print_exc()
