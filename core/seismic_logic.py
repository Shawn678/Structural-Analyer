"""
橋梁耐震分析 — 核心邏輯模組
================================
依據「公路橋梁耐震設計規範」實作工址參數擷取、譜加速度計算、
韌性折減係數、垂直地震力及載重組合等核心計算。

本模組提供:
    - Mode: 分析等級列舉 (L1, LEVEL_475, LEVEL_2500)
    - get_site_params(): 擷取工址參數
    - SiteParams: 工址參數資料容器
    - get_ss_for_mode(): 取得 Ss 係數
    - calculate_sa(): 計算譜加速度
    - calculate_fu(): 計算韌性折減係數
    - m_func(): m 修正函數
    - calculate_vertical_seismic_force(): 垂直地震力
    - calculate_load_combinations(): 載重組合
    - calculate_mdof_periods(): 多自由度振動週期分析
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
#  常數／設定
# ──────────────────────────────────────────────

class Mode(Enum):
    """分析等級列舉"""
    L1 = "L1"               # 等級一（簡易分析）
    LEVEL_475 = 475         # 475 年回歸期
    LEVEL_2500 = 2500       # 2500 年回歸期

    @property
    def column_name(self) -> str:
        """回歸期對應的 CSV 欄位名稱"""
        if self == Mode.L1:
            return "475年"
        return "475年" if self.value == 475 else "2500年"


# 設計譜分段之門檻係數
P_START_FACTOR = 0.2       # 啟動週期 = P_START_FACTOR * T0
TRANSITION_LOWER = 0.2     # 極短週期 / 能量等值分界 = 0.2 * T0
TRANSITION_UPPER = 0.6     # 能量等值 / 過渡區分界 = 0.6 * T0

# 地震力放大係數
ALPHA_V = 2.0 / 3.0        # 垂直地震力係數 αv（規範 2.6 節）
SS_DIVISOR = 3.25          # L1 等級 Ss → Sas 之除數

# 載重組合係數
LONGITUDINAL_FACTOR = 1.0  # 縱向係數
TRANSVERSE_FACTOR = 0.3    # 橫向係數
VERTICAL_FACTOR = 0.3      # 垂直係數

# CSV 欄位索引名稱（共用常數，減少 Magic String 散落）
KEY_SS = "一般區域震區短週期譜加速度係數(Ss)"
KEY_S1 = "一般區域震區一秒週期譜加速度係數(S1)"
KEY_SAS = "工址短週期譜加速度係數(Sas)"
KEY_SA1 = "工址一秒週期譜加速度係數(Sa1)"
KEY_T0 = "短週期與中週期分界之轉換週期T0(秒)"
KEY_TL = "中週期與長週期分界之轉換週期(秒)"


# ──────────────────────────────────────────────
#  資料容器
# ──────────────────────────────────────────────

@dataclass
class SiteParams:
    """工址譜加速度參數 (Sas, Sa1, T0, TL)"""
    Sas: float
    Sa1: float
    T0: float
    TL: float


@dataclass
class FuResult:
    """韌性折減係數 Fu 計算結果"""
    value: float
    description: str


@dataclass
class VerticalSeismicForceResult:
    """垂直地震力 Vv 計算結果"""
    alpha_v: float
    coeff_sup: float
    coeff_sub: float
    Vv_sup: float
    Vv_sub: float
    Vv_total: float


@dataclass
class LoadCombination:
    """單一載重組合"""
    name: str
    desc: str
    formula: str
    V_L: float
    V_T: float
    V_v: float
    total: float


@dataclass
class LoadCombinationResult:
    """載重組合計算結果"""
    combinations: list[LoadCombination]
    max_combo: LoadCombination


@dataclass
class ModalResult:
    """單一模態分析結果"""
    mode_number: int      # 模態編號 (1, 2, ..., n)
    omega: float          # 角頻率 ω (rad/s)
    period: float         # 自然週期 T (秒)
    phi: np.ndarray       # 正規化振型向量（最大值 = 1）
    gamma: float          # 模態參與係數 Γ
    m_eff: float          # 有效模態質量 (Mg)
    m_eff_ratio: float    # 有效模態質量比 (%)


@dataclass
class MdofResult:
    """多自由度週期分析結果"""
    n_dof: int                      # 自由度數
    masses: np.ndarray              # 質量陣列
    stiffnesses: np.ndarray         # 勁度陣列
    M: np.ndarray                   # 質量矩陣
    K: np.ndarray                   # 勁度矩陣
    modes: list[ModalResult]        # 各模態結果
    total_mass: float               # 總質量 (Mg)


# ──────────────────────────────────────────────
#  輔助函數
# ──────────────────────────────────────────────

def _validate_df_params(df_params: pd.DataFrame, keys: list[str]) -> None:
    """檢查 df_params 是否包含必要的索引與欄位"""
    missing = [k for k in keys if k not in df_params.index]
    if missing:
        raise KeyError(f"df_params 缺少必要索引: {missing}")


# ──────────────────────────────────────────────
#  多自由度週期分析
# ──────────────────────────────────────────────

def calculate_mdof_periods(
    masses: list[float],
    stiffnesses: list[float],
) -> MdofResult:
    """計算多自由度（剪力構架）系統之自然振動週期與模態

    依據結構動力學理論，求解廣義特徵值問題:

        K · φ = ω² · M · φ

    其中：
        M = diag(m₁, m₂, ..., mₙ)  質量矩陣
        K = [[k₁+k₂, -k₂,   0,  ...],
             [ -k₂,  k₂+k₃, -k₃, ...],
             [   0,   -k₃,   k₃, ...]]  三對角勁度矩陣

    參數:
        masses:      各自由度集中質量 [m₁, m₂, ..., mₙ] (Mg)
        stiffnesses: 各橋墩側向勁度 [k₁, k₂, ..., kₙ] (kN/m)

    回傳:
        MdofResult 含各模態週期、振型、有效模態質量

    異常:
        ValueError: 陣列長度不一致或無效參數

    參考:
        公路橋梁耐震設計規範 §2.3
    """
    n = len(masses)
    if n < 1:
        raise ValueError("自由度數必須大於 0")
    if len(stiffnesses) != n:
        raise ValueError(f"質量({n})與勁度({len(stiffnesses)})陣列長度不一致")
    if any(m <= 0 for m in masses):
        raise ValueError("質量必須為正值")
    if any(k <= 0 for k in stiffnesses):
        raise ValueError("勁度必須為正值")

    # ── 組立質量矩陣 M（對角矩陣）──
    M = np.diag(masses)

    # ── 組立勁度矩陣 K（三對角矩陣）──
    K = np.zeros((n, n))
    if n == 1:
        # 單自由度特例
        K[0, 0] = stiffnesses[0]
    else:
        for i in range(n):
            if i == 0:
                K[i, i] = stiffnesses[i] + stiffnesses[i + 1]
                K[i, i + 1] = -stiffnesses[i + 1]
            elif i == n - 1:
                K[i, i] = stiffnesses[i]
                K[i, i - 1] = -stiffnesses[i]
            else:
                K[i, i] = stiffnesses[i] + stiffnesses[i + 1]
                K[i, i - 1] = -stiffnesses[i]
                K[i, i + 1] = -stiffnesses[i + 1]

    # ── 解廣義特徵值問題 K·φ = λ·M·φ (λ = ω²) ──
    # 轉換為標準特徵值問題：M⁻¹·K·φ = λ·φ
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(M, K))

    # 排序：最小的 eigenvalue = 最長週期（第一模態）
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    omega = np.sqrt(np.abs(eigenvalues))
    periods = 2 * np.pi / omega
    total_mass = np.sum(masses)
    one = np.ones(n)

    modes: list[ModalResult] = []
    for i in range(n):
        phi = eigenvectors[:, i]

        # 正規化：讓最大絕對值 = 1（形狀不變）
        phi_norm = phi / np.max(np.abs(phi))

        # 模態參與係數 Γ = (φᵀ·M·1) / (φᵀ·M·φ)
        numerator = phi_norm @ M @ one
        denominator = phi_norm @ M @ phi_norm
        gamma = numerator / denominator

        # 有效模態質量 M_eff = (φᵀ·M·1)² / (φᵀ·M·φ)
        m_eff = numerator**2 / denominator
        m_eff_ratio = m_eff / total_mass * 100

        modes.append(ModalResult(
            mode_number=i + 1,
            omega=omega[i],
            period=periods[i],
            phi=phi_norm,
            gamma=gamma,
            m_eff=m_eff,
            m_eff_ratio=m_eff_ratio,
        ))

    return MdofResult(
        n_dof=n,
        masses=np.array(masses),
        stiffnesses=np.array(stiffnesses),
        M=M,
        K=K,
        modes=modes,
        total_mass=total_mass,
    )


# ──────────────────────────────────────────────
#  核心 API（既有功能，維持不變）
# ──────────────────────────────────────────────

def get_site_params(mode: Mode, df_params: pd.DataFrame) -> SiteParams:
    """從 CSV 參數資料框中提取工址參數

    參數:
        mode: 分析等級 (Mode.L1 / Mode.LEVEL_475 / Mode.LEVEL_2500)
        df_params: DataFrame（索引為參數名稱，欄位 '475年' / '2500年'）

    回傳:
        SiteParams 資料容器

    異常:
        KeyError: 缺少必要索引或欄位時拋出
    """
    if mode == Mode.L1:
        # 等級一：依據 475 年參數衍生 (Ss/3.25, S1/3.25)
        required = [KEY_SS, KEY_S1, KEY_TL]
        _validate_df_params(df_params, required)
        ss = float(df_params.loc[KEY_SS, "475年"])
        s1 = float(df_params.loc[KEY_S1, "475年"])
        tl = float(df_params.loc[KEY_TL, "475年"])
        site_sas = ss / SS_DIVISOR
        site_sa1 = s1 / SS_DIVISOR
        t0 = site_sa1 / site_sas if site_sas != 0.0 else 0.0
        return SiteParams(Sas=site_sas, Sa1=site_sa1, T0=t0, TL=tl)

    # 475 / 2500 年：直接讀取工址參數
    col = mode.column_name
    required = [KEY_SAS, KEY_SA1, KEY_T0, KEY_TL]
    _validate_df_params(df_params, required)
    return SiteParams(
        Sas=float(df_params.loc[KEY_SAS, col]),
        Sa1=float(df_params.loc[KEY_SA1, col]),
        T0=float(df_params.loc[KEY_T0, col]),
        TL=float(df_params.loc[KEY_TL, col]),
    )


def get_ss_for_mode(mode: Mode, df_params: pd.DataFrame) -> float:
    """取得工址短週期譜加速度係數 Ss，用於垂直地震力計算

    若為 L1 則取自「一般區域震區」475 年 Ss（未經 Fa 放大）；
    若為 475 / 2500 則取自「工址短週期譜加速度係數」Sas（經 Fa 放大）。

    參數:
        mode: 分析等級
        df_params: 工址參數 DataFrame

    回傳:
        Ss / Sas (float)
    """
    if mode == Mode.L1:
        _validate_df_params(df_params, [KEY_SS])
        return float(df_params.loc[KEY_SS, "475年"])
    _validate_df_params(df_params, [KEY_SAS])
    return float(df_params.loc[KEY_SAS, mode.column_name])


def calculate_sa(
    t: float,
    mode: Mode,
    site_params: SiteParams,
    spectrum_data: Optional[dict[str, np.ndarray]] = None,
) -> float:
    """計算或插值譜加速度 Sa

    等級一 (L1) 使用設計譜公式；
    等級二／三 (475 / 2500) 使用 CSV 譜資料進行線性插值。

    參數:
        t: 週期 (秒)
        mode: 分析等級
        site_params: 工址參數
        spectrum_data: CSV 譜資料 (dict 含 'T' 與 'Sa' 陣列)

    回傳:
        譜加速度 Sa (g)
    """
    if mode == Mode.L1:
        p_start = P_START_FACTOR * site_params.T0

        if t < p_start:
            return site_params.Sas * (0.4 + 0.6 * t / p_start)
        if t <= site_params.T0:
            return site_params.Sas
        if t <= site_params.TL:
            return site_params.Sa1 / t
        return (site_params.Sa1 * site_params.TL) / (t ** 2)

    # 475 / 2500 年使用 CSV 資料進行線性插值
    if spectrum_data is not None:
        return float(np.interp(t, spectrum_data["T"], spectrum_data["Sa"]))

    raise ValueError(
        f"mode={mode} 需要提供 spectrum_data 才能計算譜加速度"
    )


def calculate_fu(t: float, r: float, t0: float) -> FuResult:
    """計算韌性折減係數 Fu（含極短週期修正）

    參數:
        t:  自然週期 (秒)
        r:  韌性容量 R
        t0: 短週期與中週期分界轉換週期 (秒)

    回傳:
        FuResult (value, description)
    """
    if r <= 1.0:
        return FuResult(1.0, "彈性設計 (R=1)")
    if t >= t0:
        return FuResult(r, "T ≥ T₀ (位移等值)")

    sqrt_2r_1 = np.sqrt(2.0 * r - 1.0)

    if t >= TRANSITION_UPPER * t0:
        # 過渡區內插
        val = sqrt_2r_1 + (r - sqrt_2r_1) * (t - TRANSITION_UPPER * t0) / (
            (1.0 - TRANSITION_UPPER) * t0
        )
        return FuResult(val, "0.6T₀ ≤ T < T₀ (過渡區內插)")

    if t >= TRANSITION_LOWER * t0:
        return FuResult(sqrt_2r_1, "0.2T₀ ≤ T < 0.6T₀ (能量等值)")

    # 極短週期修正 (T < 0.2 T₀)
    val = 1.0 + (sqrt_2r_1 - 1.0) * (t / (TRANSITION_LOWER * t0))
    return FuResult(val, "T < 0.2T₀ (極短週期修正)")


def m_func(x: float) -> float:
    """計算 m 修正函數（線性分段函數）"""
    if x <= 0.3:
        return x
    if x <= 0.8:
        return 0.52 * x + 0.144
    return 0.7 * x


def calculate_vertical_seismic_force(
    w_sup: float,
    w_sub: float,
    i: float,
    alpha_y: float,
    ss: float,
) -> VerticalSeismicForceResult:
    """計算總垂直地震力 Vv（依規範 2.6 節 (2-25) 式）

    Vv = I * (αv / αy) * Ss * W_sup
       + I * (αv / αy) * (0.4 * Ss) * W_sub

    參數:
        w_sup:   上部結構靜載重 (kN)
        w_sub:   下部結構靜載重 (kN)
        i:       用途係數 I
        alpha_y: 起始降伏地震力放大倍數 αy
        ss:      工址短週期譜加速度係數 Ss

    回傳:
        VerticalSeismicForceResult
    """
    coeff = ALPHA_V / alpha_y  # αv / αy 僅計算一次

    coeff_sup = i * coeff * ss
    vv_sup = coeff_sup * w_sup

    coeff_sub = i * coeff * (0.4 * ss)
    vv_sub = coeff_sub * w_sub

    return VerticalSeismicForceResult(
        alpha_v=ALPHA_V,
        coeff_sup=coeff_sup,
        coeff_sub=coeff_sub,
        Vv_sup=vv_sup,
        Vv_sub=vv_sub,
        Vv_total=vv_sup + vv_sub,
    )


def calculate_load_combinations(
    v_l: float,
    v_t: float,
    v_v: float,
) -> LoadCombinationResult:
    """計算地震力載重組合（依規範 2.7 節）

    載重組合:
        1: 100% 縱向 + 30% 橫向 + 30% 垂直
        2:  30% 縱向 + 100% 橫向 + 30% 垂直
        3:  30% 縱向 +  30% 橫向 + 100% 垂直

    參數:
        v_l: 縱向水平設計地震力 (kN)
        v_t: 橫向水平設計地震力 (kN)
        v_v: 垂直設計地震力 (kN)

    回傳:
        LoadCombinationResult（含所有組合與最大控制組合）
    """
    combos = [
        LoadCombination(
            name="載重組合一",
            desc="100%縱向 + 30%橫向 + 30%垂直",
            formula="1.0·V_L + 0.3·V_T + 0.3·V_v",
            V_L=LONGITUDINAL_FACTOR * v_l,
            V_T=TRANSVERSE_FACTOR * v_t,
            V_v=VERTICAL_FACTOR * v_v,
            total=LONGITUDINAL_FACTOR * v_l
                   + TRANSVERSE_FACTOR * v_t
                   + VERTICAL_FACTOR * v_v,
        ),
        LoadCombination(
            name="載重組合二",
            desc="30%縱向 + 100%橫向 + 30%垂直",
            formula="0.3·V_L + 1.0·V_T + 0.3·V_v",
            V_L=TRANSVERSE_FACTOR * v_l,        # 30% 縱向
            V_T=LONGITUDINAL_FACTOR * v_t,       # 100% 橫向
            V_v=VERTICAL_FACTOR * v_v,
            total=TRANSVERSE_FACTOR * v_l
                   + LONGITUDINAL_FACTOR * v_t
                   + VERTICAL_FACTOR * v_v,
        ),
        LoadCombination(
            name="載重組合三",
            desc="30%縱向 + 30%橫向 + 100%垂直",
            formula="0.3·V_L + 0.3·V_T + 1.0·V_v",
            V_L=TRANSVERSE_FACTOR * v_l,
            V_T=TRANSVERSE_FACTOR * v_t,
            V_v=LONGITUDINAL_FACTOR * v_v,       # 100% 垂直
            total=TRANSVERSE_FACTOR * v_l
                   + TRANSVERSE_FACTOR * v_t
                   + LONGITUDINAL_FACTOR * v_v,
        ),
    ]

    max_combo = max(combos, key=lambda c: c.total)

    return LoadCombinationResult(combinations=combos, max_combo=max_combo)