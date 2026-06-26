"""
seismic_bridge.py — 從 FEM 結果萃取 MDOF 地震力分析所需參數。

提供四個公開函式：
  identify_pier_tops()      — 識別每個橋墩群組的頂節點與底節點
  extract_pier_masses()     — 從自重計算每墩集中質量（Mg）
  extract_pier_stiffnesses()— 從 K_red_diagonal 萃取每墩側向勁度（kN/m）
  validate_mdof_assumptions()— 驗證 MDOF 假設是否與 FEM 模型相符

另提供 compute_horizontal_force()，
  供 Tab 5 直接呼叫。
"""

import core.seismic_logic as _logic
from core.seismic_logic import Mode, SiteParams


def _norm_id(val) -> str:
    return str(val).strip()


def identify_pier_tops(pier_member_names: list[str],
                       elements_data: list[dict],
                       truss_data: dict) -> dict:
    """
    對每個橋墩 member 群組找出頂節點（連接非墩元素或 rigid_link 的端點）與底節點。

    Return:
        {pier_name: {"top_node": str, "base_nodes": [str], "element_ids": [int]}}
    """
    pier_set = set(pier_member_names)

    # 每個 member 群組包含的元素
    pier_elems: dict[str, list[dict]] = {p: [] for p in pier_set}
    non_pier_nodes: set[str] = set()

    for elem in elements_data:
        mb = (elem.get("member") or "").strip()
        ni = _norm_id(elem.get("i", ""))
        nj = _norm_id(elem.get("j", ""))
        if mb in pier_set:
            pier_elems[mb].append(elem)
        else:
            non_pier_nodes.add(ni)
            non_pier_nodes.add(nj)

    # rigid_link 節點（master 與 slave 都可能是頂節點）
    rl_nodes: set[str] = set()
    for rl in truss_data.get("rigid_links", []):
        rl_nodes.add(_norm_id(rl.get("master", "")))
        rl_nodes.add(_norm_id(rl.get("slave", "")))

    # 支承節點集合
    support_nodes: set[str] = {_norm_id(s.get("node_id", ""))
                                for s in truss_data.get("supports", [])}

    # 節點座標
    node_z: dict[str, float] = {}
    for n in truss_data.get("nodes", []):
        nid = _norm_id(n.get("id", ""))
        node_z[nid] = float(n.get("z", n.get("y", 0)))

    result = {}
    for pier_name, elems in pier_elems.items():
        if not elems:
            continue
        elem_ids = [e["id"] for e in elems]

        # 所有墩節點
        pier_nodes: set[str] = set()
        for e in elems:
            pier_nodes.add(_norm_id(e["i"]))
            pier_nodes.add(_norm_id(e["j"]))

        # 頂節點候選：連接到非墩元素 or rigid_link，且在墩節點中
        top_candidates = pier_nodes & (non_pier_nodes | rl_nodes)
        if not top_candidates:
            # fallback：Z 最高的墩節點
            top_candidates = pier_nodes

        top_node = max(top_candidates, key=lambda n: node_z.get(n, 0))

        # 底節點：有 support 且在墩節點中（排除頂節點）
        base_nodes = sorted(
            (pier_nodes & support_nodes) - {top_node},
            key=lambda n: node_z.get(n, 0)
        )

        result[pier_name] = {
            "top_node": top_node,
            "base_nodes": base_nodes,
            "element_ids": elem_ids,
        }

    return result


def extract_pier_masses(pier_tops: dict,
                        elements_data: list[dict],
                        truss_data: dict,
                        sections_data: list[dict],
                        materials_data: list[dict]) -> dict:
    """
    從 FEM 元素自重計算每墩集中質量（Mg）。

    每墩質量 = 墩柱自重 + 相鄰橋面 tributary 質量（各取一半）。

    Return:
        {pier_name: {"mass_Mg": float, "pier_mass_Mg": float,
                     "deck_mass_Mg": float, "top_node": str}}
    """
    # 建立截面 → (density, A) 的快速查詢
    mat_map  = {m["name"]: m for m in materials_data}
    sec_map  = {s["name"]: s for s in sections_data}

    def _get_density_A(elem: dict):
        sec_name = elem.get("section", "")
        sec = sec_map.get(sec_name, {})
        mat = mat_map.get(sec.get("material", ""), {})
        density = float(mat.get("density") or elem.get("density") or 0)
        A = float(elem.get("A") or sec.get("A") or 0)
        if A <= 0:
            from core.materials import compute_section_props
            props = compute_section_props(sec.get("shape", "Custom"), sec)
            A = props.get("A", 0)
        return density, A

    # 節點座標（用於計算 Le 若未存入）
    node_pos = {_norm_id(n["id"]): (float(n.get("x", 0)),
                                     float(n.get("y", 0)),
                                     float(n.get("z", 0)))
                for n in truss_data.get("nodes", [])}

    def _elem_Le(elem: dict) -> float:
        ni = _norm_id(elem.get("i", ""))
        nj = _norm_id(elem.get("j", ""))
        pi = node_pos.get(ni, (0, 0, 0))
        pj = node_pos.get(nj, (0, 0, 0))
        return ((pi[0]-pj[0])**2 + (pi[1]-pj[1])**2 + (pi[2]-pj[2])**2) ** 0.5

    # 建立 element_id → elem dict 的快速查詢
    elem_by_id = {str(e["id"]): e for e in elements_data}

    # 所有墩元素 id 集合（用於識別相鄰非墩元素）
    all_pier_elem_ids: set[str] = set()
    for info in pier_tops.values():
        all_pier_elem_ids.update(str(eid) for eid in info["element_ids"])

    result = {}
    for pier_name, info in pier_tops.items():
        top_node = info["top_node"]
        pier_elem_ids = {str(eid) for eid in info["element_ids"]}

        # 墩柱自重
        pier_mass = 0.0
        for eid in pier_elem_ids:
            elem = elem_by_id.get(eid)
            if not elem:
                continue
            density, A = _get_density_A(elem)
            Le = _elem_Le(elem)
            pier_mass += density * A * Le / 9810  # Mg

        # Tributary 橋面自重：頂節點相鄰的非墩元素
        deck_mass = 0.0
        for elem in elements_data:
            eid = str(elem.get("id", ""))
            if eid in all_pier_elem_ids:
                continue
            ni = _norm_id(elem.get("i", ""))
            nj = _norm_id(elem.get("j", ""))
            if top_node in (ni, nj):
                density, A = _get_density_A(elem)
                Le = _elem_Le(elem)
                deck_mass += density * A * (Le / 2) / 9810  # Mg，取一半

        result[pier_name] = {
            "mass_Mg":      pier_mass + deck_mass,
            "pier_mass_Mg": pier_mass,
            "deck_mass_Mg": deck_mass,
            "top_node":     top_node,
        }

    return result


def extract_pier_stiffnesses(pier_tops: dict,
                              last_result: dict,
                              bridge_axis: str = "X") -> dict:
    """
    從 K_red_diagonal 萃取每墩頂節點的側向勁度（kN/m）。

    bridge_axis: "X" → ux (DOF offset 0)；"Y" → uy (DOF offset 1)

    Return:
        {pier_name: {"stiffness_kNm": float, "top_node": str,
                     "lateral_dof": int, "warning": str|None}}
    """
    K_diag = last_result.get("K_red_diagonal", [])
    free_dofs = last_result.get("free_dofs", [])
    node_id_to_idx = last_result.get("node_id_to_idx", {})

    dof_offset = 0 if bridge_axis.upper() == "X" else 1
    free_dofs_set = {d: i for i, d in enumerate(free_dofs)}

    result = {}
    for pier_name, info in pier_tops.items():
        top_node = info["top_node"]
        node_idx = node_id_to_idx.get(top_node)
        warning = None

        if node_idx is None:
            result[pier_name] = {
                "stiffness_kNm": 0.0,
                "top_node": top_node,
                "lateral_dof": -1,
                "warning": f"節點 {top_node} 不在 FEM 節點表中",
            }
            continue

        global_dof = 6 * node_idx + dof_offset
        free_idx = free_dofs_set.get(global_dof)

        if free_idx is None:
            warning = (f"節點 {top_node} 的側向 DOF 為固定端或已被 rigid_link 消去，"
                       "無法萃取勁度")
            result[pier_name] = {
                "stiffness_kNm": 0.0,
                "top_node": top_node,
                "lateral_dof": global_dof,
                "warning": warning,
            }
            continue

        k_N_per_m = K_diag[free_idx] if free_idx < len(K_diag) else 0.0
        result[pier_name] = {
            "stiffness_kNm": k_N_per_m / 1000,
            "top_node": top_node,
            "lateral_dof": global_dof,
            "warning": warning,
        }

    return result


def validate_mdof_assumptions(pier_tops: dict,
                               truss_data: dict,
                               last_result: dict,
                               stiffness_results: dict | None = None) -> list[dict]:
    """
    驗證 FEM 模型是否符合 MDOF 假設，回傳警告/錯誤清單。

    Return:
        [{"level": "warning"|"error", "message": str}]
    """
    msgs = []

    # 1. FEM 數值分析是否已執行
    if not last_result or "K_red_diagonal" not in last_result:
        msgs.append({"level": "error",
                     "message": "請先執行 FEM 數值分析（按下分析按鈕），才能提取地震力參數。"})
        return msgs

    # 2. 各墩底節點是否全固定
    support_map: dict[str, dict] = {}
    for s in truss_data.get("supports", []):
        support_map[_norm_id(s.get("node_id", ""))] = s

    dof_keys = ["ux", "uy", "uz", "rx", "ry", "rz"]
    for pier_name, info in pier_tops.items():
        for bn in info.get("base_nodes", []):
            sup = support_map.get(bn, {})
            unfixed = [k for k in dof_keys if not sup.get(k, False)]
            if unfixed:
                msgs.append({"level": "warning",
                             "message": (f"橋墩「{pier_name}」的底節點 {bn} "
                                         f"未完全固定（{', '.join(unfixed)} 未約束）。"
                                         "MDOF 模型假設墩底為固定端，結果可能偏保守。")})

    # 3. 剛性橋面
    if not truss_data.get("rigid_links"):
        msgs.append({"level": "warning",
                     "message": ("模型中未設定 Rigid Link。MDOF 假設橋面為剛性隔板，"
                                  "建議在橋面節點間加入 Rigid Link 以符合假設。")})

    # 4. 勁度合理性
    if stiffness_results:
        for pier_name, sr in stiffness_results.items():
            if sr.get("warning"):
                msgs.append({"level": "warning", "message": sr["warning"]})
            k = sr.get("stiffness_kNm", 0)
            if 0 < k < 0.1:
                msgs.append({"level": "warning",
                             "message": f"橋墩「{pier_name}」側向勁度 {k:.4f} kN/m 偏低，請確認模型。"})
            elif k > 10_000_000:
                msgs.append({"level": "warning",
                             "message": f"橋墩「{pier_name}」側向勁度 {k:.0f} kN/m 異常高，請確認模型。"})

    return msgs


# ── 水平設計地震力（複製自 Bridge_Seismic_Analysis/app.py） ──────────────────

def compute_horizontal_force(
    t: float, mode: Mode, w: float, i: float, alpha_y: float,
    r_val: float, p: SiteParams, spec_data
) -> tuple[float, float, float, float, str]:
    """回傳 (sa, fu, cd, v_force, fu_text)，單位與 seismic_logic 一致。"""
    sa = _logic.calculate_sa(t, mode, p, spec_data)
    fu_result = _logic.calculate_fu(t, r_val, p.T0)
    fu = fu_result.value
    fu_text = fu_result.description

    if mode == Mode.L1:
        cd_raw = i * sa / alpha_y
        cd_min = i * 0.4 * p.Sas / alpha_y
    else:
        cd_raw = i * (1 / (1.2 * alpha_y)) * _logic.m_func(sa / fu)
        cd_min = i * (1 / (1.2 * alpha_y)) * _logic.m_func(0.4 * p.Sas / fu)

    cd = max(cd_raw, cd_min)
    v_force = cd * w
    return sa, fu, cd, v_force, fu_text
