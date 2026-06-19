from core.symbolic import run_symbolic_analysis
import os
import pandas as pd
import json

def load_nodes_from_csv(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

def load_elements_from_csv(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

def load_supports_from_csv(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

def load_loads_from_csv(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

def load_element_loads_from_csv(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

def load_element_point_loads_from_csv(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

def main():
    print("=== Structural Analysis CLI Mode (Symbolic) ===")
    truss_data = {
        "nodes": [],
        "elements": [],
        "supports": [],
        "loads": [],
        "element_loads": [],
        "element_point_loads": []
    }

    # 檢查 CSV 檔案是否存在並從中載入模型
    nodes_path = 'nodes.csv'
    elements_path = 'elements.csv'

    if os.path.exists(nodes_path) and os.path.exists(elements_path):
        print(f"正在從 {nodes_path} 與 {elements_path} 載入模型...")
        truss_data["nodes"] = load_nodes_from_csv(nodes_path)
        truss_data["elements"] = load_elements_from_csv(elements_path)
    else:
        print("錯誤：找不到 CSV 檔案。請確認 nodes.csv 與 elements.csv 位於正確目錄。")
        return
    
    # 從 CSV 載入支承與載重 (如果檔案存在)
    supports_path = 'supports.csv'
    loads_path = 'loads.csv'
    e_loads_path = 'element_loads.csv'
    e_pt_loads_path = 'element_point_loads.csv'
    if os.path.exists(supports_path):
        truss_data["supports"] = load_supports_from_csv(supports_path)
    if os.path.exists(loads_path):
        truss_data["loads"] = load_loads_from_csv(loads_path)
    if os.path.exists(e_loads_path):
        truss_data["element_loads"] = load_element_loads_from_csv(e_loads_path)
    if os.path.exists(e_pt_loads_path):
        truss_data["element_point_loads"] = load_element_point_loads_from_csv(e_pt_loads_path)
    
    # 執行分析
    res = run_symbolic_analysis(truss_data)
    
    print("\n[分析結果]")
    print(json.dumps(res, indent=4, ensure_ascii=False))
    
    print("\n分析完成。")

if __name__ == "__main__":
    main()