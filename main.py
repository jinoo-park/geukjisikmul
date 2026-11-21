#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
송도고 극지식물 개체별 생육 차이 분석 대시보드
─────────────────────────────────────────────
CSV 인코딩(utf-8-sig / cp949) 자동 처리, 학교별 다른 열 이름 자동 대응,
한글 폰트 깨짐 방지, Streamlit Cloud 에러 ZERO 를 목표로 작성했습니다.

통계 용어 설명: 고등학생이 이해할 수 있도록 각 지표 옆에 짧은 설명을 붙였습니다.
"""

import asyncio
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 0. 한글 폰트 설정 (matplotlib + Plotly 공용)
#    Streamlit Cloud 리눅스 환경에서 NanumGothic 을 찾지 못하면
#    먼저 설치 시도하고, 그래도 없으면 기본 폰트로 fallback 합니다.
# ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_korean_font() -> None:
    """한글 폰트를 찾아서 matplotlib 에 등록한다. 실패해도 조용히 넘어간다."""
    candidates = [
        "NanumGothic",
        "nanumgothic",
        "NanumBarunGothic",
        "NanumSquare",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]
    found = False
    for name in candidates:
        try:
            fp = fm.findfont(name, fallback_to_default=False)
            if fp and Path(fp).exists():
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                found = True
                break
        except Exception:
            continue
    if not found:
        try:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "nanumfont", "-q"],
                capture_output=True, timeout=60,
            )
            plt.rcParams["font.family"] = "NanumGothic"
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            plt.rcParams["font.family"] = "DejaVu Sans"

setup_korean_font()

PLOTLY_KR = "NanumGothic, sans-serif"

# ─────────────────────────────────────────────────────────────
# 1. 인코딩 자동 감지 로드 함수
#    utf-8-sig → cp949 → euc-kr → latin-1(최종 fallback) 순으로 시도
# ─────────────────────────────────────────────────────────────
def load_csv_auto(path: str | Path) -> pd.DataFrame:
    """인코딩을 순차적으로 시도해서 CSV 를 읽어온다."""
    path = Path(path)
    for enc in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            if df.empty:
                continue
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return pd.read_csv(path, encoding="latin-1")

# ─────────────────────────────────────────────────────────────
# 2. 열 이름 자동 정규화 (학교마다 다른 열 이름 대응)
# ─────────────────────────────────────────────────────────────
def normalize_env_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    환경 데이터 열 이름을 표준 이름으로 매핑한다.
    지원 형식 A: time, temperature, humidity, ph, ec, water_temp, co2
    지원 형식 B: categories, 온도, 습도, EC, PH  등
    """
    def find_col(*keywords):
        for c in df.columns:
            cl = str(c).lower().strip()
            for kw in keywords:
                if kw in cl:
                    return c
        return None

    time_candidates = ["time", "날짜", "일시", "timestamp", "date", "시간"]
    time_col = None
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in time_candidates or "time" in cl or "날짜" in cl or "일시" in cl:
            time_col = c
            break
    if time_col is None:
        for c in df.columns:
            if len(df) > 0 and "2026" in str(df[c].iloc[0]):
                time_col = c
                break

    temp_col  = find_col("temp", "온도")
    hum_col   = find_col("humid", "습도")
    ec_col    = find_col("ec")
    ph_col    = find_col("ph")
    wt_col    = find_col("water_temp", "수질", "수온")
    co2_col   = find_col("co2", "co")

    out = pd.DataFrame()
    if time_col:
        out["time"] = df[time_col]
    if temp_col:
        out["temperature"] = pd.to_numeric(df[temp_col], errors="coerce")
    if hum_col:
        out["humidity"] = pd.to_numeric(df[hum_col], errors="coerce")
    if ec_col:
        out["ec"] = pd.to_numeric(df[ec_col], errors="coerce")
    if ph_col:
        out["ph"] = pd.to_numeric(df[ph_col], errors="coerce")
    if wt_col:
        out["water_temp"] = pd.to_numeric(df[wt_col], errors="coerce")
    if co2_col:
        out["co2"] = pd.to_numeric(df[co2_col], errors="coerce")

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], errors="coerce", format="mixed")

    return out

def normalize_growth_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    생육 결과 데이터 열 이름을 표준 이름으로 매핑한다.
    열 이름에 단위·띄어쓰기가 붙어 있어도 키워드 기반으로 찾는다.
    """
    def find_col(*keywords):
        for c in df.columns:
            cl = str(c).lower().strip()
            for kw in keywords:
                if kw in cl:
                    return c
        return None

    id_col        = find_col("개체번호", "번호", "id", "개체")
    leaf_col      = find_col("잎", "leaf", "엽")
    shoot_len_col = find_col("지상부 길이", "지상부길이", "shoot", "지상")
    root_len_col  = find_col("지하부길이", "지하부 길이", "root", "지하")
    fresh_col     = find_col("생중량", "fresh")
    shoot_mass    = find_col("지상부질량", "지상부 질량", "shoot_mass")
    root_mass     = find_col("지하부질량", "지하부 질량", "root_mass")

    out = pd.DataFrame()
    if id_col:
        out["개체번호"] = df[id_col]
    if leaf_col:
        out["잎 수"] = pd.to_numeric(df[leaf_col], errors="coerce")
    if shoot_len_col:
        out["지상부 길이"] = pd.to_numeric(df[shoot_len_col], errors="coerce")
    if root_len_col:
        out["지하부 길이"] = pd.to_numeric(df[root_len_col], errors="coerce")
    if fresh_col:
        out["생중량"] = pd.to_numeric(df[fresh_col], errors="coerce")
    if shoot_mass:
        out["지상부질량"] = pd.to_numeric(df[shoot_mass], errors="coerce")
    if root_mass:
        out["지하부질량"] = pd.to_numeric(df[root_mass], errors="coerce")

    return out

# ─────────────────────────────────────────────────────────────
# 3. 사이드바: 데이터 로드
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="극지식물 생육 분석 대시보드",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🌿 극지식물 생육 대시보드")
st.sidebar.markdown("""
**송도고 극지식물 프로젝트**

CSV 2개(환경 + 생육)를 업로드하면  
개체별 생육 차이를 한눈에 분석할 수 있습니다.
""")

env_file = st.sidebar.file_uploader(
    "📊 환경 데이터 CSV 업로드",
    type=["csv"],
    help="형식 A: time, temperature, humidity, ph, ec, water_temp, co2  |  형식 B: categories, 온도, 습도, EC, PH 등 학교마다 달라도 자동 처리합니다.",
)
growth_file = st.sidebar.file_uploader(
    "🌱 생육 결과 데이터 CSV 업로드",
    type=["csv"],
    help="개체번호, 잎 수(장), 지상부 길이(mm) 등 열 이름에 단위가 붙어 있어도 자동 인식합니다.",
)

BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
SAMPLE_ENV     = BASE_DIR / "송도고등학교_환경데이터.csv"
SAMPLE_GROWTH  = BASE_DIR / "송도고등학교_생육결과데이터.csv"

def resolve_path(uploaded, sample_path: Path) -> Path | None:
    """업로드 파일이 있으면 임시 저장 후 경로 반환, 없으면 샘플 경로 반환."""
    if uploaded is not None:
        tmp = Path("/tmp") / uploaded.name
        tmp.write_bytes(uploaded.getvalue())
        return tmp
    if sample_path.exists():
        return sample_path
    return None

env_path     = resolve_path(env_file,     SAMPLE_ENV)
growth_path  = resolve_path(growth_file,  SAMPLE_GROWTH)

if env_path is None and growth_path is None:
    st.error("⚠️ 환경 데이터와 생육 결과 데이터를 모두 업로드하거나, 같은 폴더에 CSV 파일을 넣어주세요.")
    st.stop()

env_raw     = load_csv_auto(env_path)     if env_path     else pd.DataFrame()
growth_raw  = load_csv_auto(growth_path)  if growth_path  else pd.DataFrame()

env_df     = normalize_env_cols(env_raw)     if not env_raw.empty     else pd.DataFrame()
growth_df  = normalize_growth_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()

if env_df.empty:
    st.warning("⚠️ 환경 데이터에서 인식할 수 있는 열이 없습니다. CSV 열 이름을 확인해주세요.")
if growth_df.empty:
    st.warning("⚠️ 생육 데이터에서 인식할 수 있는 열이 없습니다. CSV 열 이름을 확인해주세요.")

if env_df.empty and growth_df.empty:
    st.stop()

# ─────────────────────────────────────────────────────────────
# 4. 메인 화면
# ─────────────────────────────────────────────────────────────
st.title("🌿 극지식물 개체별 생육 차이 분석 대시보드")
st.caption("CSV 인코딩(utf-8-sig/cp949)·열 이름 차이·한글 폰트 깨짐을 자동 처리합니다.  |  통계 용어는 고등학생 눈높이로 설명합니다.")

if "selected_id" not in st.session_state:
    st.session_state.selected_id = int(growth_df["개체번호"].iloc[0]) if not growth_df.empty else 1

# ─────────────────────────────────────────────────────────────
# 4-1. 환경 시계열 탭
# ─────────────────────────────────────────────────────────────
tab_env = st.tabs(["📈 환경 데이터 시계열", "📊 생육 통계 요약", "🔗 환경-생육 비교", "📋 개체별 상세 비교"])

with tab_env[0]:
    st.subheader("📈 환경 데이터 시계열")
    st.markdown("""
**환경 데이터란?** 식물이 자라는 동안의 온도·습도·토양 상태 등을 시간에 따라 기록한 데이터입니다.  
아래 그래프는 센서가 측정한 값이 있는 시간만 표시됩니다(측정하지 못한 빈 기간은 자동으로 건너뛰어요).
""")

    env_cols_show = [c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"] if c in env_df.columns]
    if not env_cols_show:
        st.warning("⚠️ 환경 데이터에서 표시할 수치 열이 없습니다.")
    else:
        selected_env = st.multiselect(
            "표시할 환경 지표 선택",
            env_cols_show,
            default=env_cols_show[:2] if len(env_cols_show) >= 2 else env_cols_show,
            key="env_indicators",
        )

        if selected_env:
            t_min = env_df["time"].min()
            t_max = env_df["time"].max()
            col1, col2 = st.columns(2)
            with col1:
                start = st.date_input("시작 날짜", value=t_min.date() if pd.notna(t_min) else None)
            with col2:
                end   = st.date_input("끝 날짜",   value=t_max.date() if pd.notna(t_max) else None)

            mask = pd.Series(True, index=env_df.index)
            if start and pd.notna(t_min):
                mask &= env_df["time"].dt.date >= start
            if end and pd.notna(t_max):
                mask &= env_df["time"].dt.date <= end

            plot_df = env_df[mask].dropna(subset=["time"] + selected_env).sort_values("time")

            if plot_df.empty:
                st.info("⚠️ 선택한 기간에 표시할 데이터가 없습니다.")
            else:
                fig = make_subplots(
                    rows=len(selected_env),
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=selected_env,
                )
                colors = px.colors.qualitative.Set2

                for i, col in enumerate(selected_env):
                    y = plot_df[col].astype(float)
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df["time"],
                            y=y,
                            mode="lines",
                            name=col,
                            line=dict(color=colors[i % len(colors)], width=1.5),
                            fill="tozeroy",
                            fillcolor=f"rgba({i*40+100}, {i*20+100}, 200, 0.15)",
                        ),
                        row=i + 1,
                        col=1,
                    )

                fig.update_layout(
                    height=max(200, 150 * len(selected_env)),
                    template="plotly_white",
                    margin=dict(l=40, r=20, t=40, b=20),
                    font=dict(family=PLOTLY_KR, size=12),
                    hovermode="x unified",
                )
                fig.update_xaxes(tickformat="%m/%d %H:%M", row=len(selected_env), col=1)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 환경 지표 요약")
            summary_rows = []
            for col in selected_env:
                s = env_df[col].dropna()
                if s.empty:
                    continue
                summary_rows.append({
                    "지표": col,
                    "측정 횟수": len(s),
                    "평균": f"{s.mean():.2f}",
                    "표준편차": f"{s.std():.2f}",
                    "최저": f"{s.min():.2f}",
                    "최고": f"{s.max():.2f}",
                })
            if summary_rows:
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("📄 환경 데이터 원본 미리보기 (처음 5행)"):
        st.dataframe(env_df.dropna(how="all").head(5), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 4-2. 생육 통계 요약 탭
# ─────────────────────────────────────────────────────────────
with tab_env[1]:
    st.subheader("📊 생육 통계 요약")
    st.markdown("""
**개체별 생육 데이터란?** 각각 독립적으로 자란 극지식물 한 개체마다  
잎 수·길이·무게 등을 측정한 결과입니다.

아래 통계량은 **전체 개체**를 하나로 묶어 계산한 값입니다.
""")

    growth_metrics = [c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                   "생중량", "지상부질량", "지하부질량"] if c in growth_df.columns]

    if not growth_metrics:
        st.warning("⚠️ 생육 데이터에서 표시할 열이 없습니다.")
    else:
        stat_rows = []
        for m in growth_metrics:
            s = growth_df[m].dropna()
            if s.empty:
                continue
            stat_rows.append({
                "생육 지표": m,
                "개체 수": len(s),
                "평균": f"{s.mean():.2f}",
                "표준편차": f"{s.std():.2f}",
                "최소값": f"{s.min():.2f}",
                "중앙값": f"{s.median():.2f}",
                "최대값": f"{s.max():.2f}",
                "변동계수(CV)": f"{s.std() / s.mean():.2f}",
                "사분위범위(IQR)": f"{s.quantile(0.75) - s.quantile(0.25):.2f}",
            })
        stat_df = pd.DataFrame(stat_rows)
        st.dataframe(stat_df, use_container_width=True, hide_index=True)
        st.caption("※ 변동계수(CV) = 표준편차 ÷ 평균. 값이 클수록 개체 간 차이가 상대적으로 큼.  |  사분위범위(IQR) = 상위 25%~75% 구간 길이.")

        st.markdown("#### 생육 지표 분포 (히스토그램)")
        st.markdown("""
**히스토그램이란?** 측정된 값들을 일정한 구간(막대)으로 나눠 각 구간에 몇 개체가 속하는지 막대 그래프로 보여준 것입니다.  
막대가 한쪽에 몰려 있으면 개체들이 비슷한 값을 가진다는 뜻이고, 넓게 퍼져 있으면 개체 간 차이가 크다는 뜻입니다.
""")
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            hist_metric = st.selectbox("분포 볼 지표 선택", growth_metrics, index=0, key="hist_metric")
        with dist_col2:
            bins = st.slider("구간(막대) 개수", 5, 30, 15, key="hist_bins")

        fig_hist = px.histogram(
            growth_df,
            x=hist_metric,
            nbins=bins,
            title=f"{hist_metric} 분포 (n={len(growth_df)})",
            labels={hist_metric: hist_metric, "count": "개체 수"},
            color_discrete_sequence=["#4C72B0"],
        )
        fig_hist.update_layout(
            template="plotly_white",
            font=dict(family=PLOTLY_KR, size=12),
            xaxis_title=hist_metric,
            yaxis_title="개체 수",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 생육 지표 박스플롯 (상자 수염 그림)")
        st.markdown("""
**박스플롯이란?**  
- 상자 가운데 선 = 중앙값  
- 상자 위아래 테두리 = 1사분위(25%)·3사분위(75%)  
- 상자 밖으로 뻗은 선(수염) = 일반적인 범위  
- 점으로 찍힌 값 = 이상치(평균에서 특히 멀리 떨어진 개체)
""")
        bp_metric = st.selectbox("박스플롯 볼 지표 선택", growth_metrics, index=1 if len(growth_metrics) > 1 else 0, key="bp_metric")

        fig_bp = px.box(
            growth_df,
            y=bp_metric,
            points="outliers",
            title=f"{bp_metric} 박스플롯",
            color_discrete_sequence=["#55A868"],
        )
        fig_bp.update_layout(
            template="plotly_white",
            font=dict(family=PLOTLY_KR, size=12),
            yaxis_title=bp_metric,
            showlegend=False,
        )
        st.plotly_chart(fig_bp, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 4-3. 환경-생육 비교 탭
# ─────────────────────────────────────────────────────────────
with tab_env[2]:
    st.subheader("🔗 환경-생육 비교")
    st.markdown("""
**상관관계란?** 두 지표가 함께 변하는 경향을 숫자로 나타낸 것(범위: −1 ~ +1)입니다.

- **+1에 가까움**: 한 값이 커질 때 다른 값도 커지는 경향 (양의 상관)  
- **−1에 가까움**: 한 값이 커질 때 다른 값은 작아지는 경향 (음의 상관)  
- **0에 가까움**: 두 값 사이에 뚜렷한 함께 변하는 경향이 없음

※ 상관관계가 높다고 해서 반드시 '원인-결과' 관계는 아닙니다. (상관 ≠ 인과)
""")

    env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"] if c in env_df.columns]
    growth_num_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부질량", "지하부질량"] if c in growth_df.columns]

    if not env_num_cols or not growth_num_cols:
        st.warning("⚠️ 비교 분석을 하려면 환경 데이터(수치 열)와 생육 데이터(지표 열)가 모두 필요합니다.")
    else:
        col_e = st.selectbox("환경 지표 (X축)", env_num_cols, key="corr_env_metric")
        col_g = st.selectbox("생육 지표 (Y축)", growth_num_cols, key="corr_growth_metric")

        fig_compare = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"{col_e} 분포 (환경)", f"{col_g} 분포 (개체별 생육)"),
        )

        fig_compare.add_trace(
            go.Histogram(x=env_df[col_e].dropna(), nbinsx=30,
                         marker_color="#4C72B0", name="환경"),
            row=1, col=1,
        )
        fig_compare.add_trace(
            go.Histogram(x=growth_df[col_g].dropna(), nbinsx=30,
                         marker_color="#55A868", name="생육"),
            row=1, col=2,
        )
        fig_compare.update_layout(
            template="plotly_white",
            font=dict(family=PLOTLY_KR, size=12),
            showlegend=False,
            height=360,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        env_daily = env_df.set_index("time").resample("D").mean(numeric_only=True)
        st.markdown("#### 환경 일별 평균 요약")
        if not env_daily.empty:
            env_summary_tbl = pd.DataFrame({
                "일": env_daily.index.date,
                col_e: env_daily[col_e].round(2),
            }).dropna()
            st.dataframe(env_summary_tbl, use_container_width=True, hide_index=True)

        env_mean_val = env_df[col_e].dropna().mean()
        st.info(f"""
**현재 선택된 지표**  
환경: `{col_e}` (전체 평균 ≈ {env_mean_val:.2f})  
생육: `{col_g}` (평균 ≈ {growth_df[col_g].mean():.2f}, 표준편차 ≈ {growth_df[col_g].std():.2f})

두 데이터는 서로 다른 개체·시점에서 측정되었으므로 직접적인 상관계수 계산은 제한됩니다.  
구체적인 상관 분석이 필요하면 **개체별 측정 일시** 정보가 포함된 데이터가 필요합니다.
""")

# ─────────────────────────────────────────────────────────────
# 4-4. 개체별 상세 비교 탭
# ─────────────────────────────────────────────────────────────
with tab_env[3]:
    st.subheader("📋 개체별 상세 비교")
    st.markdown("""
**개체번호를 선택하면** 해당 개체의 생육 값을 전체 분포와 비교해 볼 수 있습니다.  
산점도에서 ★(선택 개체)과 ●(전체 개체)를 함께 표시해 상대적 위치를 확인합니다.
""")

    if growth_df.empty:
        st.warning("⚠️ 생육 데이터가 없어 개체 비교를 할 수 없습니다.")
    else:
        id_col_display = "개체번호" if "개체번호" in growth_df.columns else growth_df.columns[0]

        id_min = int(growth_df[id_col_display].min())
        id_max = int(growth_df[id_col_display].max())
        selected_id = st.slider(
            "선택 개체번호",
            min_value=id_min,
            max_value=id_max,
            value=st.session_state.selected_id,
            step=1,
            key="selected_id_slider",
        )
        st.session_state.selected_id = selected_id

        sel_row = growth_df[growth_df[id_col_display] == selected_id]
        if sel_row.empty:
            st.warning(f"⚠️ 개체번호 {selected_id}에 해당하는 데이터가 없습니다.")
        else:
            sel_row = sel_row.iloc[0]

            cols = st.columns(3)
            metric_pairs = [
                ("잎 수",         "잎 수"),
                ("지상부 길이",   "지상부 길이"),
                ("지하부 길이",   "지하부 길이"),
                ("생중량",        "생중량"),
                ("지상부질량",    "지상부질량"),
                ("지하부질량",    "지하부질량"),
            ]
            for i, (label, col) in enumerate(metric_pairs):
                if col in growth_df.columns:
                    total_vals = growth_df[col].dropna()
                    val = sel_row[col]
                    pst = (total_vals < val).sum() / len(total_vals) * 100
                    with cols[i % 3]:
                        st.metric(
                            label=label,
                            value=f"{val:.1f}",
                            delta=f"상위 {100-pst:.0f}% ({pst:.0f}백분위)",
                        )

            # 산점도 옵션 (walrus operator 없이 명시적 정의)
            scatter_options = [
                ("잎 수", "잎 수", "지상부 길이", "지상부 길이"),
                ("잎 수", "잎 수", "생중량", "생중량"),
                ("지상부 길이", "지상부 길이", "지상부질량", "지상부질량"),
                ("지하부 길이", "지하부 길이", "지하부질량", "지하부질량"),
            ]
            scatter_labels = [f"{a} vs {b}" for a, _, b, _ in scatter_options]
            scatter_choice = st.selectbox(
                "산점도 쌍 선택",
                scatter_labels,
                index=0,
                key="scatter_pair",
            )
            idx = scatter_labels.index(scatter_choice)
            x_label, x_col, y_label, y_col = scatter_options[idx]

            if x_col in growth_df.columns and y_col in growth_df.columns:
                fig_scatter = go.Figure()

                # 크기용 데이터: 생중량이 있으면 사용, 없으면 상수열
                size_vals = growth_df["생중량"] if "생중량" in growth_df.columns else pd.Series([8.0]*len(growth_df))

                fig_scatter.add_trace(
                    go.Scatter(
                        x=growth_df[x_col],
                        y=growth_df[y_col],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=size_vals,
                            colorscale="Blues",
                            showscale=True,
                            colorbar=dict(title="생중량(g)", xanchor="right"),
                            opacity=0.5,
                            line=dict(width=0.5, color="gray"),
                        ),
                        name="전체 개체",
                        text=growth_df[id_col_display].astype(str),
                        hovertemplate="%{text}번<br>" + x_label + ": %{x}<br>" + y_label + ": %{y}<extra></extra>",
                    )
                )

                fig_scatter.add_trace(
                    go.Scatter(
                        x=[sel_row[x_col]],
                        y=[sel_row[y_col]],
                        mode="markers",
                        marker=dict(
                            size=18,
                            color="crimson",
                            line=dict(width=2, color="white"),
                            symbol="star",
                        ),
                        name=f"선택 개체 ({selected_id}번)",
                        hovertemplate=f"★ {selected_id}번<br>" + x_label + ": %{x}<br>" + y_label + ": %{y}<extra></extra>",
                    )
                )

                fig_scatter.update_layout(
                    template="plotly_white",
                    font=dict(family=PLOTLY_KR, size=12),
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    height=480,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("#### 개체번호 순 생육 지표 비교")
            bar_metric = st.selectbox(
                "막대 그래프로 볼 지표",
                growth_metrics,
                index=3 if len(growth_metrics) > 3 else 0,
                key="bar_metric",
            )
            if bar_metric in growth_df.columns:
                bar_df = growth_df[[id_col_display, bar_metric]].sort_values(id_col_display)
                # color 인자에 DataFrame 열을 전달 (문자열 컬럼명이 아님)
                is_selected = bar_df[id_col_display].astype(str) == str(selected_id)
                fig_bar = px.bar(
                    bar_df,
                    x=id_col_display,
                    y=bar_metric,
                    color=is_selected,
                    color_discrete_map={True: "crimson", False: "#88BFE8"},
                    title=f"개체번호별 {bar_metric} (선택 개체: crimson)",
                )
                fig_bar.update_layout(
                    template="plotly_white",
                    font=dict(family=PLOTLY_KR, size=11),
                    xaxis_title="개체번호",
                    yaxis_title=bar_metric,
                    height=360,
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 5. 페이지 하단: 데이터 요약 & 통계 용어 정리
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption(f"""
**데이터 정보**  
- 환경 데이터: {env_df.shape[0]:,}행 × {env_df.shape[1]}열  |  생육 데이터: {growth_df.shape[0]:,}개체 × {growth_df.shape[1]}열  
- 측정 기간: {env_df["time"].min().date()} ~ {env_df["time"].max().date()} (환경 데이터 기준)  
- 결측치(센서 미측정 구간)는 그래프에서 자동으로 제외되어 표시됩니다.
""")

st.caption("""
**통계 용어 간단 정리**  
| 용어 | 뜻 |
|---|---|
| 평균 | 모든 값을 더한 뒤 개수로 나눈 값. 데이터의 '대표값' 중 하나 |
| 표준편차 | 데이터가 평균에서 평균적으로 얼마나 떨어져 있는지. 클수록 퍼짐이 큼 |
| 중앙값 | 크기순으로 나열했을 때 가운데 값. 극단값의 영향을 덜 받음 |
| 사분위범위(IQR) | 상위 25%~75% 구간 길이. 중간 절반 데이터의 퍼진 범위 |
| 변동계수(CV) | 표준편차 ÷ 평균. 단위가 다른 데이터끼리 상대적 퍼짐을 비교 |
| 상관관계(r) | 두 지표가 함께 변하는 정도(−1~+1). 0이면 함께 변하는 경향이 거의 없음 |
| 히스토그램 | 값을 구간별 막대 그래프로 나타낸 것. 분포를 한눈에 봄 |
| 박스플롯 | 중앙값·quartile·이상치를 한눈에 보여주는 상자 모양 그래프 |
""")
