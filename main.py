import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

st.set_page_config(page_title="극지식물 개체별 생육 차이 분석", page_icon="🌱", layout="wide")

# 한글 폰트 (깨짐 방지)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"
COLORS = {"상위 25%": "#d9480f", "중간 50%": "#adb5bd", "하위 25%": "#1c7ed6", "전체": "#868e96"}
UNIT = {"생중량": "g", "잎 수": "장", "지상부 길이": "mm", "지하부 길이": "mm"}
GROWTH_COLS = ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
ENV_LABEL = {"temperature": "온도(℃)", "humidity": "습도(%)", "ph": "pH", "ec": "EC(mS/cm)"}


# ---------------------------------------------------------------- 데이터 읽기
def read_csv(data):
    # 바이트를 한 번만 꺼내 두고 인코딩을 바꿔 가며 읽기
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


def id_text(v):
    # 개체번호를 56.0이 아니라 원래 값(56, A1)으로 표시
    try:
        f = float(v)
        if f == int(f):
            return str(int(f))
    except Exception:
        pass
    return str(v)


@st.cache_data(show_spinner=False)
def load_df(data):
    df = std_cols(read_csv(data))
    if "개체번호" in df.columns:
        df["개체번호"] = df["개체번호"].map(id_text)
    return df


def has_num(df, col, n=2):
    return df is not None and col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].dropna().shape[0] >= n


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def style(fig, title=None, height=380):
    fig.update_layout(font=dict(family=FONT), title=title, height=height, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- 사이드바 (파일은 한 번만 읽기)
st.sidebar.header("📂 데이터 올리기")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")

env_df, growth_df = None, None
with st.spinner("데이터를 불러오는 중..."):
    if env_file is not None:
        env_df = load_df(env_file.getvalue())
    if growth_file is not None:
        growth_df = load_df(growth_file.getvalue())
if env_df is not None and env_df.empty:
    st.sidebar.info("환경 CSV를 읽지 못했어요. 파일을 확인해 주세요.")
    env_df = None
if growth_df is not None and growth_df.empty:
    st.sidebar.info("생육 CSV를 읽지 못했어요. 파일을 확인해 주세요.")
    growth_df = None

school = "우리 학교"
for f in (env_file, growth_file):
    if f is not None:
        name = re.split(r"[_\-\s.]", f.name)[0].strip()
        if name:
            school = name
            break

avail = [c for c in GROWTH_COLS if has_num(growth_df, c)]
selected = st.sidebar.multiselect("생육 측정 항목 선택", avail, default=avail)
if not selected:
    selected = avail

st.title("🌱 같은 스마트팜, 다른 생육 — 극지식물 개체별 생육 차이 분석")
if env_file is not None or growth_file is not None:
    st.caption(f"📍 {school} 데이터")
else:
    st.info("왼쪽 사이드바에서 환경 CSV와 생육 CSV를 올려 주세요. 한 파일만 올려도 해당 탭은 볼 수 있어요.")


def add_groups(df):
    # 생중량 상위 25% / 하위 25% 구분 (같은 값은 더 큰 쪽 그룹에 포함)
    g = df.copy()
    n = len(g)
    g["개체"] = g["개체번호"].astype(str) if "개체번호" in g.columns else [str(i + 1) for i in range(n)]
    g["그룹"] = "전체"
    if has_num(g, "생중량", 4):
        q75 = g["생중량"].quantile(0.75)
        q25 = g["생중량"].quantile(0.25)
        g.loc[g["생중량"] <= q25, "그룹"] = "하위 25%"
        g.loc[g["생중량"] >= q75, "그룹"] = "상위 25%"
        g.loc[(g["생중량"] > q25) & (g["생중량"] < q75), "그룹"] = "중간 50%"
    return g


def line_chart(daily, col, title, color):
    d = daily.dropna(subset=[col])
    if len(d) < 2:
        st.info(f"{title}를 그릴 값이 부족해요.")
        return
    fig = px.line(d, x="date", y=col, color_discrete_sequence=[color])
    fig.update_layout(xaxis_title="날짜", yaxis_title=ENV_LABEL.get(col, col))
    show(style(fig, title))


def bar_chart(g, col):
    d = g.dropna(subset=[col]).sort_values(col, ascending=False)
    if d.empty:
        st.info(f"{col} 값이 없어 막대 그래프를 건너뛰어요.")
        return
    unit = UNIT.get(col, "")
    fig = px.bar(d, x="개체", y=col, color="그룹", color_discrete_map=COLORS,
                 category_orders={"개체": list(d["개체"])})
    fig.add_hline(y=float(d[col].mean()), line_dash="dash", line_color="#343a40",
                  annotation_text=f"평균 {d[col].mean():.2f}")
    fig.update_xaxes(type="category", tickangle=-90, tickfont=dict(size=9), title="개체번호")
    fig.update_yaxes(title=f"{col} ({unit})")
    show(style(fig, f"개체별 {col} 비교 (큰 값부터, 주황=상위 25% · 파랑=하위 25%)"))


def scatter_chart(g, x, y="생중량"):
    if not (has_num(g, x) and has_num(g, y)):
        st.info(f"{x} 또는 {y} 열이 없어 산점도를 건너뛰어요.")
        return
    d = g[[x, y, "개체", "그룹"]].dropna(subset=[x, y])
    if len(d) < 2 or d[x].nunique() < 2:
        st.info(f"{x} 값이 2개 이상 달라야 상관을 계산할 수 있어요.")
        return
    r = float(np.corrcoef(d[x], d[y])[0, 1])
    slope, inter = np.polyfit(d[x], d[y], 1)
    fig = px.scatter(d, x=x, y=y, color="그룹", color_discrete_map=COLORS, hover_name="개체")
    xs = np.linspace(d[x].min(), d[x].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + inter, mode="lines", name="추세선",
                             line=dict(color="#212529", dash="dash")))
    fig.update_layout(xaxis_title=f"{x} ({UNIT.get(x, '')})", yaxis_title=f"{y} ({UNIT.get(y, '')})")
    show(style(fig, f"{x} vs {y} (r = {r:.2f})"))
    a = abs(r)
    word = "강한" if a >= 0.7 else "중간 정도의" if a >= 0.4 else "약한" if a >= 0.2 else "거의 없는"
    sign = "양(+)" if r > 0 else "음(-)"
    st.caption(f"상관계수 r = {r:.2f} → {sign}의 {word} 상관 (상관은 '함께 변한다'는 뜻이지 '원인'이라는 뜻은 아니에요).")


tab1, tab2, tab3, tab4 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과", "🏁 결론"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    try:
        st.subheader("연구 배경 및 목적")
        st.write("극지식물은 낮은 온도에서도 자랄 수 있어 저온 스마트팜 재배에 활용될 가능성이 있어요. "
                 "스마트팜은 온도·습도·EC·pH를 조절해 같은 환경을 만들어 주지만, 같은 환경에서 키워도 "
                 "개체마다 자라는 정도가 다를 수 있어요. 이 대시보드는 **같은 스마트팜 안에서 개체별 생육 차이가 "
                 "얼마나 큰지**, 그리고 **어떤 생육 지표가 생중량과 가장 관련 있는지** 데이터로 확인합니다.")
        if env_df is None and growth_df is None:
            st.info("CSV를 올리면 데이터 요약과 주요 지표가 나타나요.")
        else:
            rows = []
            if env_df is not None:
                env_items = [ENV_LABEL[c] for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
                rows.append({"구분": "환경 데이터 행 수", "내용": f"{len(env_df):,}행"})
                if "time" in env_df.columns and env_df["time"].notna().any():
                    t0, t1 = env_df["time"].min(), env_df["time"].max()
                    rows.append({"구분": "측정 기간", "내용": f"{t0:%Y-%m-%d} ~ {t1:%Y-%m-%d} ({(t1 - t0).days + 1}일)"})
                rows.append({"구분": "환경 측정 항목", "내용": ", ".join(env_items) if env_items else "인식된 항목 없음"})
            if growth_df is not None:
                rows.append({"구분": "개체 수", "내용": f"{len(growth_df)}개"})
                rows.append({"구분": "생육 측정 항목", "내용": ", ".join(avail) if avail else "인식된 항목 없음"})
            st.subheader("데이터 요약")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            st.subheader("주요 지표")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 개체수", f"{len(growth_df)}개" if growth_df is not None else "-")
            t_ok = has_num(env_df, "temperature", 1)
            h_ok = has_num(env_df, "humidity", 1)
            w_ok = has_num(growth_df, "생중량", 1)
            c2.metric("평균 온도", f"{env_df['temperature'].dropna().mean():.1f} ℃" if t_ok else "-")
            c3.metric("평균 습도", f"{env_df['humidity'].dropna().mean():.1f} %" if h_ok else "-")
            c4.metric("평균 생중량", f"{growth_df['생중량'].dropna().mean():.2f} g" if w_ok else "-")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ---------------------------------------------------------------- Tab 2
with tab2:
    try:
        if env_df is None:
            st.info("사이드바에서 환경 CSV를 올리면 환경 분석이 나타나요.")
        else:
            cols = [c for c in ["temperature", "humidity", "ph", "ec"] if has_num(env_df, c, 1)]
            if not cols:
                st.info("온도·습도·pH·EC 열을 찾지 못했어요. 열 이름을 확인해 주세요.")
            else:
                st.subheader("환경 요약 (평균·최소·최대)")
                rows = []
                for c in cols:
                    s = env_df[c].dropna()
                    rows.append({"항목": ENV_LABEL[c], "평균": round(s.mean(), 2), "최소": round(s.min(), 2),
                                 "최대": round(s.max(), 2)})
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

                st.subheader("날짜별 변화 (일별 평균)")
                if "time" not in env_df.columns or env_df["time"].notna().sum() < 2:
                    st.info("시간 열을 읽지 못해 날짜별 그래프를 건너뛰어요.")
                else:
                    d = env_df.dropna(subset=["time"]).copy()
                    d["date"] = d["time"].dt.normalize()
                    daily = d.groupby("date")[cols].mean().reset_index()
                    left, right = st.columns(2)
                    with left:
                        if "temperature" in cols:
                            line_chart(daily, "temperature", "🌡️ 온도 변화", "#e03131")
                        else:
                            st.info("온도 열이 없어 건너뛰어요.")
                    with right:
                        if "humidity" in cols:
                            line_chart(daily, "humidity", "💧 습도 변화", "#1c7ed6")
                        else:
                            st.info("습도 열이 없어 건너뛰어요.")
                    if "ph" in cols and "ec" in cols:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        e = daily.dropna(subset=["ec"])
                        p = daily.dropna(subset=["ph"])
                        fig.add_trace(go.Scatter(x=e["date"], y=e["ec"], name="EC(mS/cm)", mode="lines",
                                                 line=dict(color="#7048e8")), secondary_y=False)
                        fig.add_trace(go.Scatter(x=p["date"], y=p["ph"], name="pH", mode="lines",
                                                 line=dict(color="#2f9e44")), secondary_y=True)
                        fig.update_yaxes(title_text="EC (mS/cm)", secondary_y=False)
                        fig.update_yaxes(title_text="pH", secondary_y=True)
                        show(style(fig, "🧪 pH · EC 변화"))
                    elif "ph" in cols:
                        line_chart(daily, "ph", "🧪 pH 변화", "#2f9e44")
                    elif "ec" in cols:
                        line_chart(daily, "ec", "🧪 EC 변화", "#7048e8")
                    else:
                        st.info("pH·EC 열이 없는 파일이라 이 그래프는 건너뛰어요.")
            with st.expander("환경 데이터 원본 보기 · 다운로드"):
                st.dataframe(env_df, width="stretch")
                st.download_button("환경 데이터 CSV 다운로드", to_csv_bytes(env_df), "환경데이터.csv", "text/csv",
                                   key="dl_env")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ---------------------------------------------------------------- Tab 3
with tab3:
    try:
        if growth_df is None:
            st.info("사이드바에서 생육 CSV를 올리면 생육 분석이 나타나요.")
        elif not avail:
            st.info("생중량·잎 수·지상부 길이·지하부 길이 열을 찾지 못했어요. 열 이름을 확인해 주세요.")
        else:
            g = add_groups(growth_df)
            with st.expander("📘 통계 용어 쉽게 보기"):
                st.markdown("- **평균**: 모든 값을 더해 개수로 나눈 값\n"
                            "- **표준편차(SD)**: 값들이 평균에서 평균적으로 얼마나 떨어져 있는지 (클수록 들쭉날쭉)\n"
                            "- **CV(변동계수)**: 표준편차 ÷ 평균 × 100(%). 단위가 달라도 흩어진 정도를 비교할 수 있어요\n"
                            "- **상관계수 r**: -1~+1. +1에 가까울수록 함께 커지는 관계가 강해요\n"
                            "- **p값**: 두 그룹이 사실 같다고 가정할 때 이런 차이가 나올 확률. 0.05보다 작으면 차이가 뚜렷하다고 봐요")

            st.subheader("🥇 핵심 결과")
            if has_num(g, "생중량", 1):
                w = g.dropna(subset=["생중량"])
                top, low = w.loc[w["생중량"].idxmax()], w.loc[w["생중량"].idxmin()]
                c1, c2, c3 = st.columns(3)
                c1.metric("평균 생중량", f"{w['생중량'].mean():.2f} g")
                c2.metric(f"최대 개체 ({top['개체']}번)", f"{top['생중량']:.2f} g")
                c3.metric(f"최소 개체 ({low['개체']}번)", f"{low['생중량']:.2f} g")
            else:
                st.info("생중량 열이 없어 핵심 결과 카드를 건너뛰어요.")

            st.subheader("1️⃣ 개체별 생육 비교")
            order = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"] if c in selected]
            for c in order:
                bar_chart(g, c)

            rows = []
            for c in order:
                s = g[c].dropna()
                if len(s) < 2:
                    continue
                m, sd = s.mean(), s.std()
                rows.append({"항목": f"{c}({UNIT[c]})", "평균": round(m, 2), "표준편차": round(sd, 2),
                             "CV(%)": round(sd / m * 100, 1) if m else None, "최소": round(s.min(), 2),
                             "최대": round(s.max(), 2),
                             "최대/최소": round(s.max() / s.min(), 1) if s.min() > 0 else None})
            if rows:
                st.markdown("**개체 간 차이 요약** (CV가 클수록 같은 환경에서도 개체 차이가 크다는 뜻)")
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            st.subheader("2️⃣ 생중량 분포")
            if has_num(g, "생중량"):
                kind = st.radio("그래프 종류", ["박스플롯", "바이올린 플롯"], horizontal=True, key="kind")
                y = g["생중량"].dropna()
                if kind == "박스플롯":
                    fig = go.Figure(go.Box(y=y, name="생중량", boxpoints="all", jitter=0.4, boxmean="sd",
                                           fillcolor="rgba(47,158,68,0.25)", line=dict(color="#2f9e44"),
                                           marker=dict(color="#2f9e44")))
                else:
                    fig = go.Figure(go.Violin(y=y, name="생중량", box_visible=True, meanline_visible=True,
                                              points="all", fillcolor="rgba(47,158,68,0.25)",
                                              line=dict(color="#2f9e44")))
                fig.update_layout(yaxis_title="생중량 (g)")
                show(style(fig, "개체별 생중량 분포 (점선=평균 ± 표준편차)" if kind == "박스플롯" else "개체별 생중량 분포"))
            else:
                st.info("생중량 열이 없어 분포 그래프를 건너뛰어요.")

            st.subheader("3️⃣ 상관관계 분석")
            sc1, sc2 = st.columns(2)
            with sc1:
                scatter_chart(g, "잎 수")
            with sc2:
                scatter_chart(g, "지상부 길이")

            st.subheader("4️⃣ 생중량 상위 25% vs 하위 25%")
            if has_num(g, "생중량", 8):
                hi, lo = g[g["그룹"] == "상위 25%"], g[g["그룹"] == "하위 25%"]
                rows = []
                for c in avail:
                    a, b = hi[c].dropna(), lo[c].dropna()
                    if len(a) < 2 or len(b) < 2:
                        continue
                    p = stats.ttest_ind(a, b, equal_var=False).pvalue
                    rows.append({"항목": f"{c}({UNIT[c]})", "상위 25% 평균": round(a.mean(), 2),
                                 "하위 25% 평균": round(b.mean(), 2), "차이": round(a.mean() - b.mean(), 2),
                                 "비율(배)": round(a.mean() / b.mean(), 2) if b.mean() else None,
                                 "p값": "<0.001" if p < 0.001 else f"{p:.3f}"})
                st.caption(f"상위 {len(hi)}개체 vs 하위 {len(lo)}개체 (Welch t검정). 비율이 클수록 두 그룹을 크게 구분하는 항목이에요.")
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                else:
                    st.info("비교할 값이 부족해요.")
            else:
                st.info("생중량 값이 8개 이상 있어야 상위·하위 그룹을 비교할 수 있어요.")

            with st.expander("생육 데이터 원본 보기 · 다운로드"):
                st.dataframe(growth_df, width="stretch")
                st.download_button("생육 데이터 CSV 다운로드", to_csv_bytes(growth_df), "생육결과데이터.csv", "text/csv",
                                   key="dl_growth")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ---------------------------------------------------------------- Tab 4 (결론)
# CSV를 올리지 않아도 항상 보이는 탭입니다. 숫자는 탐구 결과를 그대로 적은 고정 텍스트예요.
with tab4:
    st.header("🏁 탐구 결론")
    st.success("**같은 스마트팜, 다른 생육** — 동일한 환경에서 재배했는데도 81개체의 생육 차이는 매우 컸고, "
               "생중량과 가장 관련이 큰 지표는 잎 수였다.")

    st.subheader("📌 한눈에 보기")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("분석 개체 수", "81개")
    m2.metric("평균 생중량", "약 2.48 g", "SD 약 2.21 g", delta_color="off")
    m3.metric("생중량 최대/최소", "약 250배")
    m4.metric("생중량 CV", "약 89%")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("잎 수 ↔ 생중량", "r = 0.8857")
    k2.metric("지하부 길이 ↔ 생중량", "r = 0.6055")
    k3.metric("지상부 길이 ↔ 생중량", "r = 0.3521")
    k4.metric("지상부 ↔ 지하부 길이", "r = 0.6946")

    st.divider()

    with st.container(border=True):
        st.subheader("1️⃣ 개체 간 생육 차이는 매우 컸다")
        st.write("이번 분석에서 가장 먼저 확인되는 사실은 동일한 스마트팜 환경에서 재배했음에도 81개체의 생육 차이가 "
                 "매우 컸다는 점이다. 생중량의 평균은 약 2.48 g이고 표준편차는 약 2.21 g로, 평균 대비 변동성이 크게 "
                 "나타났다. 생중량 최대·최소 차이는 약 250배에 이르렀으며, CV 역시 약 89% 수준으로 높아, 같은 방식·환경에서 "
                 "키워도 개체별로 결과가 크게 갈릴 수 있음을 보여준다. 지상부 길이는 평균 약 39.4 mm, 지하부 길이는 평균 "
                 "약 84.0 mm로, 개체 간 차이가 지하부에서 더 두드러졌다.")

    with st.container(border=True):
        st.subheader("2️⃣ 생중량과 가장 관련이 큰 항목은 잎 수였다")
        st.write("생중량과 가장 관련이 큰 항목은 잎 수였다. 잎 수와 생중량의 상관계수는 r=0.8857로 가장 높아, 잎 수가 "
                 "많을수록 생중량도 커지는 경향이 뚜렷하게 나타났다. 실제로 생중량 상위·하위 21개체를 비교했을 때도 잎 수 "
                 "평균 차이가 57.7개와 17.0개 수준으로 크게 벌어졌고, 이는 다른 항목보다 더 두드러졌다. 지상부 길이와 "
                 "생중량의 상관은 r=0.3521, 지하부 길이와 생중량의 상관은 r=0.6055로 나타나, 생중량을 설명할 때는 잎 수가 "
                 "가장 강한 지표였고 지하부 길이도 일정한 설명력을 가졌다.")

    with st.container(border=True):
        st.subheader("3️⃣ 지상부와 지하부는 함께 자라지만 같은 방식은 아니다")
        st.write("지상부 길이와 지하부 길이는 서로 양의 상관을 보였으나, 상관이 매우 강한 수준은 아니었다(r=0.6946, "
                 "R²≈0.4824). 이는 지상부 길이만으로 지하부 길이를 정확히 예측하기 어렵고, 개체마다 지상부와 지하부의 증가 "
                 "양상이 서로 다를 수 있음을 시사한다. 생중량 예측에서는 잎 수만 사용한 모델도 비교적 높은 설명력을 보였지만, "
                 "잎 수와 지상부 길이를 함께 쓰면 R²=0.8155로 더 나아졌고, 실제 생중량 범위 내에서 비교적 합리적인 예측을 "
                 "제공했다. 다만 생중량의 변동 폭이 큰 집단에서는 예측이 더 어려워질 수 있다는 점도 확인된다.")

    st.subheader("💡 핵심 인사이트")
    i1, i2 = st.columns(2)
    with i1:
        st.info("**첫째, 같은 환경이라도 개체 차이는 크다.**\n\n"
                "동일한 스마트팜 재배 조건이라도 개체별 생육 차이는 상당히 클 수 있으며, 생중량은 평균값만으로 이해하기보다 "
                "분포와 분산을 함께 봐야 한다.")
        st.info("**셋째, 지상부와 지하부는 똑같이 자라지 않는다.**\n\n"
                "지상부와 지하부는 함께 자라지만 완전히 같은 방식으로 증가하지는 않으므로, 한쪽 지표만으로 전체 생육을 "
                "단정하기는 어렵다.")
    with i2:
        st.info("**둘째, 잎 수가 생중량을 가장 잘 설명한다.**\n\n"
                "생중량과 가장 강하게 연관된 형태는 잎 수였고, 이는 생중량 추정에 유용한 기초 지표가 될 수 있다.")
        st.info("**넷째, 환경 해석은 신중해야 한다.**\n\n"
                "환경 요인에 대한 해석은 데이터 시기의 특성과 이상치를 함께 고려해야 한다.")

    with st.container(border=True):
        st.subheader("🔭 향후 과제")
        st.write("향후 과제로는 생중량 차이의 원인을 더 구체적으로 확인하는 작업이 필요하다. 생중량 차이가 재배 후기 "
                 "EC·pH 변화와 관련될 가능성이 있으나, 생육 측정 시점과 환경 변화의 대응 관계가 불확실하므로 단정할 수는 "
                 "없다. 다음에는 생육 측정 시점을 명확히 기록하고, 개체별 위치 정보나 광량·초기 상태처럼 이번 데이터에 없던 "
                 "정보를 함께 수집하면 원인과 결과의 관계를 더 구체적으로 검토할 수 있다. 또한 생중량 상위·하위 집단의 "
                 "차이를 단순히 수치 비교에 그치지 않고, 차이의 배경을 설명할 수 있는 추가 변수를 확보하는 방향으로 "
                 "확장하면 분석의 해석력이 높아질 것이다.")
        st.markdown("- 📅 생육 **측정일** 기록\n"
                    "- 📍 개체별 **위치** 정보 수집\n"
                    "- 💡 **광량**·초기 상태 등 추가 변수 확보")

    st.caption("※ 상관은 '함께 변한다'는 뜻이지 '원인'이라는 뜻은 아니에요. 이 결과는 송도고등학교 스마트팜의 "
               "해당 시기 데이터에 한정됩니다.")
