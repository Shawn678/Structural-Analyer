import sys
import os
import json
sys.path.insert(0, os.path.abspath('.'))

from core.symbolic import run_symbolic_analysis

def test_portal_frame():
    # 門型鋼架設定：A(0,0,0) -> B(0,0,1) -> C(1,0,1) -> D(1,0,0)
    truss_data = {
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}, # A (底座左)
            {"id": 2, "x": 0.0, "y": 0.0, "z": 1.0}, # B (頂角左)
            {"id": 3, "x": 1.0, "y": 0.0, "z": 1.0}, # C (頂角右)
            {"id": 4, "x": 1.0, "y": 0.0, "z": 0.0}  # D (底座右)
        ],
        "elements": [
            {"id": 1, "i": 1, "j": 2, "E": 200e9, "A": 0.01, "I33": 1e-4}, # 左柱 AB
            {"id": 2, "i": 2, "j": 3, "E": 200e9, "A": 0.01, "I33": 1e-4}, # 橫梁 BC
            {"id": 3, "i": 4, "j": 3, "E": 200e9, "A": 0.01, "I33": 1e-4}  # 右柱 DC (由下往上)
        ],
        "supports": [
            # 固定支承 A (Node 1): 固定 Ux, Uz, Ry
            {"node_id": 1, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True},
            # 固定支承 D (Node 4): 固定 Ux, Uz, Ry
            {"node_id": 4, "ux": True, "uy": True, "uz": True, "rx": True, "ry": True, "rz": True}
        ],
        "loads": [],
        "element_loads": [
            {"element_id": 2, "w": -1.0} # 橫梁承受向下均佈載重
        ],
        "element_point_loads": []
    }

    print("開始執行門型鋼架符號分析...")
    try:
        res = run_symbolic_analysis(truss_data)
        
        print("\n[1. 節點位移 (Node Displacements)]")
        for disp in res["node_displacements"]:
            # 專注於 XZ 平面的關鍵自由度
            print(f"Node {disp['node_id']}: Ux={disp['ux']}, Uz={disp['uz']}, Ry={disp['theta_y']}")

        print("\n[2. 桿件端點內力 (Member End Forces)]")
        for force in res["element_forces"]:
            print(f"Element {force['element_id']} ({force['nodes']}):")
            print(f"  i-end: {force['i_end (N, V2, V3, T, M2, M3)']}")
            print(f"  j-end: {force['j_end (N, V2, V3, T, M2, M3)']}")

        print("\n[3. 支承反力 (Reactions)]")
        for react in res["support_reactions"]:
            print(f"Node {react['node_id']}: Rx={react['Rx']}, Rz={react['Rz']}, My={react['My']}")
            
        print("\n分析成功完成。")
        
    except Exception as e:
        print(f"\n分析失敗：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_portal_frame()