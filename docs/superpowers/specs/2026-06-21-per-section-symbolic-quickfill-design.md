# Per-Section Symbolic Quick-Fill Design

**Date:** 2026-06-21  
**Status:** Approved

---

## Overview

Upgrade the symbolic analysis and quick-fill UI so that each unique section group gets independent symbols (`E_s1, A_s1, I_s1, G_s1`, `E_s2, ...`), replacing the current single global `E, A, I, G`. The quick-fill panel is replaced with a per-section editable table plus P/w multipliers.

---

## Background & Motivation

The current fast-substitution block has six global inputs (`E, A, I, G, P倍率, w倍率`). This was adequate when all elements shared one material, but after the per-element section/material system was introduced, `evaluate_real_results` ignores the global E/A/I/G inputs entirely when `use_per_elem=True` (sections provided), making those inputs non-functional. Displacement formulas also still use one global symbol, so they are only approximate under mixed sections.

Scope: initial design tool, ≤ 5 section groups.

---

## Architecture

Three layers change:

```
app.py (UI)
  └─ builds section_group_map + group_vals
       └─ symbolic.py  (run_symbolic_analysis)
            └─ per-group independent symbols, Hadamard cross-sampling
       └─ parametric_evaluator.py  (evaluate_real_results)
            └─ subs_dict keyed by per-group symbols
```

---

## 1. `symbolic.py` Changes

### 1.1 New parameter: `section_group_map`

```python
run_symbolic_analysis(truss_data, section_group_map=None)
```

`section_group_map` maps each unique section name to a dict of SymPy symbols:

```python
{
  "H300": {"E": Symbol("E_s1"), "A": Symbol("A_s1"),
            "I33": Symbol("I_s1"), "I22": Symbol("I_s1"), "G": Symbol("G_s1")},
  "I200": {"E": Symbol("E_s2"), ...},
}
```

The caller (app.py) builds this from `st.session_state["sections"]` before calling the solver. The number of groups `G = len(section_group_map)`.

Elements with no section assigned fall back to global `E, A, I, G` symbols (existing behaviour).

### 1.2 Symbol assignment per element

In `run_symbolic_analysis`, after `elements_info` is built, each element's `elem_E_syms[k]`, `elem_A_syms[k]`, etc. are assigned by looking up the element's `section` field in `section_group_map`. Elements without a named section use the global symbols as before.

### 1.3 Cross-sampling strategy (Hadamard)

Current: all elements scale by the same `scale` factor.  
New: each group gets an independent scale drawn from a fixed grid, producing `n_samples` rows where groups vary independently.

```
SCALES = [1.0, 5.0, 25.0, 100.0, 500.0]
n_samples = max(len(SCALES), G * 4 + 4)   # e.g. G=5 → 24 samples
```

For each sample row `s`:
- Each group `g` draws `scale_g = SCALES[hash_combo[s][g] % len(SCALES)]`
- Elements in group `g` get `E_k = base_E * scale_g`, etc.
- Elements with no section use a shared global scale (mean of group scales, or fixed 1.0)

The scale combinations are generated deterministically using `np.random.default_rng(seed=42)` to keep results reproducible. A grid of at least `G+1` distinct scale values per dimension is required; with `G ≤ 5` and `len(SCALES)=5`, the grid is always sufficient.

### 1.4 Basis and symbolic expression

`_build_basis_row` already accepts `elem_E_list`, `elem_A_list`, etc. — no change needed there.

`_fit_and_symbolize` already reads `elem_E_syms` from `sym_vars` — no change needed there.

The resulting displacement formula will contain terms like:
```
uz = c1·P·L_1³/(E_s1·I_s1) + c2·P·L_2³/(E_s2·I_s2) + ...
```

### 1.5 Cache compatibility

The cache stores `raw_result` (symbolic formulas as strings) and `fingerprint` (geometry hash). Two new fields are stored:

- `section_groups`: sorted list of section group names, e.g. `["H300", "I200"]`. Used for invalidation comparison.
- `section_sym_names`: `{sec_name: {"E": "E_s1", "A": "A_s1", "I33": "I_s1", "G": "G_s1"}, ...}` — the string names of the SymPy symbols used during solve. On cache load, `evaluate_real_results` reconstructs the live SymPy symbol objects via `sp.Symbol(name)` before building `subs_dict`.

If the set of unique section names in the current elements differs from `cache["section_groups"]`, the fingerprint check fails and full re-analysis is required.

---

## 2. `parametric_evaluator.py` Changes

### 2.1 `real_params` format upgrade

Old: `{"E": 200e9, "A": 0.01, "I": 1e-4, "G": 77e9, "P": 1.0, "w": 0.0}`

New (preferred):
```python
{
  "groups": {
    "H300": {"E": 210e9, "A": 0.015, "I": 2.4e-4, "G": 80e9},
    "I200": {"E": 206e9, "A": 0.010, "I": 1.2e-4, "G": 77e9},
  },
  "P": 1.0,
  "w": 0.0,
}
```

Backward-compatible: if `"groups"` key is absent, fall back to old flat format (global E/A/I/G applied to all elements). This preserves existing `run_btn` path which still passes flat params.

### 2.2 `subs_dict` construction

```python
if "groups" in real_params:
    for sec_name, vals in real_params["groups"].items():
        sym = section_group_map[sec_name]  # fetched from cache
        subs_dict[sym["E"]]   = float(vals["E"])
        subs_dict[sym["A"]]   = float(vals["A"])
        subs_dict[sym["I33"]] = float(vals["I"])
        subs_dict[sym["G"]]   = float(vals["G"])
else:
    # legacy path
    subs_dict[E_s] = float(real_params.get("E", 200e9))
    ...
```

`section_group_map` must be stored in the symbolic cache so `evaluate_real_results` can reconstruct it without re-running the solver.

### 2.3 Numerical path (element forces)

The numerical re-solve path (`td_num`) already uses `expand_truss_data` to fill per-element E/A/I/G from sections. When the user overrides values in the quick-fill table, `expand_truss_data` is bypassed: instead, each element's properties are patched directly from `real_params["groups"]` keyed by the element's section name before calling `run_numerical_analysis`. Load multipliers P and w are applied as before.

---

## 3. `app.py` UI Changes

### 3.1 Remove old inputs

Remove the six `st.number_input` widgets (`pe_E`, `pe_A`, `pe_I`, `pe_G`, `pe_P`, `pe_w`) and the two-column layout inside the `⚡ 快速代入參數` expander.

### 3.2 New per-section table

```
⚡ 快速代入參數（expander）
┌─────────────────────────────────────────────────────┐
│ 斷面名稱 | E (Pa)   | A (m²)  | I33 (m⁴)  | G (Pa) │
│ H300    | 200e9   | 0.015   | 2.4e-4    | 77e9   │  ← editable
│ I200    | 206e9   | 0.010   | 1.2e-4    | 79e9   │  ← editable
├─────────────────────────────────────────────────────┤
│  P 倍率 [1.0]          w 倍率 [0.0]                  │
│  [⚡ 代入參數（快速）]  ← disabled if cache invalid   │
└─────────────────────────────────────────────────────┘
```

Default values are read from `st.session_state["sections"]` + `materials` at render time (same logic as `_sec_val`). The table is a `st.data_editor` with `num_rows="fixed"` (rows = unique sections; user cannot add/delete rows here). Column `斷面名稱` is disabled (read-only); E/A/I33/G are editable numbers.

Session state key: `"quickfill_overrides"` — a dict `{sec_name: {E, A, I33, G}}` that persists overrides between reruns.

Reset rules:
- On first render (key absent): initialise from `sections`+`materials`.
- When `sorted(section names)` differs from `st.session_state.get("quickfill_sec_key")`: reinitialise and update `quickfill_sec_key`. This covers section add/delete/rename.
- User edits in the table write back to `quickfill_overrides` immediately via `st.data_editor` return value.

### 3.3 Cache invalidation

Cache validity check is extended: in addition to geometry fingerprint, compare `sorted(unique section names in elements)` against `cache["section_groups"]`. If mismatch, `cache_valid = False` and the quick-fill button is disabled with tooltip "斷面組已變更，請重新執行符號解".

### 3.4 Button handler

```python
if fast_btn:
    group_vals = {row["section"]: {"E": row["E"], "A": row["A"],
                                    "I": row["I33"], "G": row["G"]}
                  for _, row in qf_df.iterrows()}
    real_params = {"groups": group_vals, "P": pe_P, "w": pe_w}
    res_eval = evaluate_real_results(
        truss_data, real_params,
        symbolic_cache=st.session_state["sym_cache"],
        ...
    )
```

### 3.5 `run_btn` path (symbolic solve)

When the user clicks 執行分析（符號解）, `app.py` builds `section_group_map` from current sections and passes it to `run_symbolic_analysis`. The resulting cache stores `section_group_map` (serialized as `{sec_name: {"E": "E_s1", ...}}` string form) so it can be reconstructed for subsequent fast substitutions.

---

## 4. Error Handling

| Scenario | Behaviour |
|---|---|
| Section not found in `section_group_map` during subs | Use global fallback E/A/I/G; log warning |
| `n_groups > 5` | Show `st.warning` in quick-fill panel; still allow analysis but note accuracy may degrade |
| `lstsq` relative error > 1e-3 after cross-sampling | Mark formula as `[近似]` in output; existing `any_invalid` flag already handles this |
| Cache `section_groups` mismatch | Disable fast-fill button, show tooltip to re-run symbolic solve |

---

## 5. Out of Scope

- Changing the section table in the quick-fill panel does NOT add/remove sections from `st.session_state["sections"]`; it only overrides values for the fast-substitution path.
- No support for per-element symbol overrides (only per-section-group).
- No UI for adding sections inside the quick-fill expander.
