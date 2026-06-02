"""
EnergyLens — redesigned Streamlit app
158.755 Data Science | Group 9

Pages
1. Global Dashboard
2. Country Explorer
3. Forecasting (Prophet + BiLSTM)
4. Cluster Analysis

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
DATA_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

KEY_COLS = {
    "Renewable share": "renewables_share_energy",
    "Fossil share": "fossil_share_energy",
    "Solar share": "solar_share_energy",
    "Wind share": "wind_share_energy",
    "Hydro share": "hydro_share_energy",
    "Nuclear share": "nuclear_share_energy",
    "Fossil energy per capita": "fossil_energy_per_capita",
    "Energy intensity": "energy_per_gdp",
}

DEEP_DIVE_METRICS = {
    "Renewable share": "renewables_share_energy",
    "Fossil share": "fossil_share_energy",
    "Hydro share": "hydro_share_energy",
    "Solar share": "solar_share_energy",
    "Wind share": "wind_share_energy",
    "Fossil energy per capita": "fossil_energy_per_capita",
}

CLUSTER_FEATURES = [
    "renewables_share_energy",
    "fossil_share_energy",
    "fossil_energy_per_capita",
    "energy_per_gdp",
    "solar_share_energy",
    "wind_share_energy",
    "hydro_share_energy",
    "nuclear_share_energy",
]

STAGE_ORDER = ["Leading", "Emerging", "Developing", "Lagging"]
STAGE_COLORS = {
    "Leading": "#2EAD6B",
    "Emerging": "#F59E0B",
    "Developing": "#3B82F6",
    "Lagging": "#EF4444",
}

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EnergyLens",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2.0rem;}
    .main-title {font-size: 2.35rem; font-weight: 800; margin-bottom: 0.15rem;}
    .subtitle {color:#5b6770; font-size:1.02rem; margin-bottom: 1.1rem;}
    div[data-testid="stMetric"] {
        background-color: #F7F9FB;
        border: 1px solid #E7EDF3;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.03);
    }
    .insight-box {
        background-color: #F7F9FB;
        border-left: 5px solid #2EAD6B;
        padding: 14px 16px;
        border-radius: 10px;
        margin: 8px 0 18px 0;
        color: #263238;
    }
    .warning-box {
        background-color: #FFF8EB;
        border-left: 5px solid #F59E0B;
        padding: 14px 16px;
        border-radius: 10px;
        margin: 8px 0 18px 0;
        color: #263238;
    }
    .small-note {color:#6c757d; font-size:0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class DataContext:
    df: pd.DataFrame
    latest_year: int
    plot_year: int


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def classify_stage(share: float | int | None) -> str | float:
    if pd.isna(share):
        return np.nan
    if share >= 50:
        return "Leading"
    if share >= 25:
        return "Emerging"
    if share >= 10:
        return "Developing"
    return "Lagging"


def _safe_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def clean_label(col: str) -> str:
    return (
        col.replace("_share_energy", " share")
        .replace("_energy", "")
        .replace("_per_capita", " per capita")
        .replace("_per_gdp", " per GDP")
        .replace("_", " ")
        .title()
    )


@st.cache_data(show_spinner="Loading OWID energy data ...", ttl=24 * 60 * 60)
def load_data(uploaded_bytes: bytes | None = None) -> DataContext:
    if uploaded_bytes:
        raw = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        raw = pd.read_csv(DATA_URL)

    required = {"country", "year", "iso_code"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = raw[
        raw["iso_code"].notna()
        & (raw["iso_code"].astype(str).str.len() == 3)
        & (~raw["iso_code"].astype(str).str.startswith("OWID"))
        & (raw["year"] >= 1990)
        & (raw["year"] <= 2024)
    ].copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    if "renewables_share_energy" in df.columns:
        df["transition_stage"] = df["renewables_share_energy"].apply(classify_stage)
        df["renew_5yr_avg"] = df.groupby("country")["renewables_share_energy"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df["renew_momentum"] = df.groupby("country")["renewables_share_energy"].transform(lambda x: x.diff(3))

    if {"gdp", "population"}.issubset(df.columns):
        df["gdp_per_capita"] = np.where(df["population"] > 0, df["gdp"] / df["population"], np.nan)
        df["log_gdp_per_capita"] = np.log1p(df["gdp_per_capita"].clip(lower=0))

    for col in ["gdp", "population", "primary_energy_consumption"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    latest_year = int(df["year"].max())
    if "renewables_share_energy" in df.columns:
        counts = df.groupby("year")["renewables_share_energy"].count()
        good_years = counts[counts >= 50]
        plot_year = int(good_years.index.max()) if len(good_years) else latest_year
    else:
        plot_year = latest_year

    return DataContext(df=df, latest_year=latest_year, plot_year=plot_year)


def get_series(df: pd.DataFrame, country: str, metric: str) -> pd.DataFrame:
    if country == "Global average":
        out = df.groupby("year", as_index=False)[metric].mean(numeric_only=True)
    else:
        out = df[df["country"] == country][["year", metric]].copy()
    out = out.dropna().sort_values("year")
    return out.rename(columns={metric: "value"})


def latest_country_row(df: pd.DataFrame, country: str, preferred_year: int | None = None) -> pd.Series | None:
    sub = df[df["country"] == country].copy()
    if sub.empty:
        return None
    if preferred_year is not None:
        sub_year = sub[sub["year"] == preferred_year]
        if not sub_year.empty:
            return sub_year.iloc[-1]
    return sub.sort_values("year").iloc[-1]


def latest_year_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    sub = df[df["year"] == year].copy()
    if sub.empty:
        return df[df["year"] == df["year"].max()].copy()
    return sub


# -----------------------------------------------------------------------------
# Global Dashboard figures
# -----------------------------------------------------------------------------
def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_kpi_cards(df: pd.DataFrame, plot_year: int) -> None:
    latest = latest_year_slice(df, plot_year)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries covered", f"{df['country'].nunique():,}")
    c2.metric("Year range", f"{int(df['year'].min())}–{int(df['year'].max())}")

    if "renewables_share_energy" in latest.columns:
        renew = latest["renewables_share_energy"].mean()
        c3.metric(f"Avg renewable share ({plot_year})", f"{renew:.1f}%")
        top = latest.dropna(subset=["renewables_share_energy"]).nlargest(1, "renewables_share_energy")
        if not top.empty:
            c4.metric("Top renewable country", top.iloc[0]["country"], f"{top.iloc[0]['renewables_share_energy']:.1f}%")
        else:
            c4.metric("Top renewable country", "N/A")
    else:
        c3.metric("Avg renewable share", "N/A")
        c4.metric("Top renewable country", "N/A")


def figure_animated_choropleth(df: pd.DataFrame, metric: str, start_year: int, end_year: int) -> go.Figure:
    sub = df[(df["year"] >= start_year) & (df["year"] <= end_year)].dropna(subset=[metric, "iso_code"]).copy()
    sub["year"] = sub["year"].astype(int).astype(str)
    color_scale = "RdYlGn" if any(x in metric for x in ["renew", "solar", "wind", "hydro", "nuclear"]) else "RdYlGn_r"
    fig = px.choropleth(
        sub,
        locations="iso_code",
        color=metric,
        hover_name="country",
        animation_frame="year",
        projection="natural earth",
        color_continuous_scale=color_scale,
        template="plotly_white",
        title=f"Animated Global Map — {clean_label(metric)}",
    )
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def figure_global_trends(df: pd.DataFrame) -> go.Figure:
    available = {label: col for label, col in KEY_COLS.items() if col in df.columns}
    trend = df.groupby("year", as_index=False)[list(available.values())].mean(numeric_only=True)
    items = list(available.items())[:8]

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[x[0] for x in items],
        horizontal_spacing=0.06,
        vertical_spacing=0.16,
    )
    positions = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]

    for (label, col), (r, c) in zip(items, positions):
        s = trend[["year", col]].dropna()
        if s.empty:
            continue
        first, latest = s[col].iloc[0], s[col].iloc[-1]
        delta = latest - first
        fig.add_trace(
            go.Scatter(x=s["year"], y=s[col], mode="lines", name=label, showlegend=False),
            row=r,
            col=c,
        )
        fig.add_annotation(
            text=f"Δ {delta:+.1f}",
            x=s["year"].iloc[-1],
            y=latest,
            xref=f"x{'' if (r, c) == (1, 1) else positions.index((r, c)) + 1}",
            yref=f"y{'' if (r, c) == (1, 1) else positions.index((r, c)) + 1}",
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.75)",
        )

    fig.update_layout(height=610, template="plotly_white", title="Global Energy Mix and Efficiency Trends")
    fig.update_xaxes(title_text="Year")
    return fig


def figure_stage_distribution(df: pd.DataFrame) -> go.Figure:
    if "transition_stage" not in df.columns:
        return go.Figure()
    stage_time = (
        df.dropna(subset=["transition_stage"])
        .groupby(["year", "transition_stage"])
        .size()
        .reset_index(name="countries")
    )
    fig = px.area(
        stage_time,
        x="year",
        y="countries",
        color="transition_stage",
        category_orders={"transition_stage": STAGE_ORDER},
        color_discrete_map=STAGE_COLORS,
        template="plotly_white",
        title="Number of Countries by Renewable Transition Stage",
    )
    fig.update_layout(height=430)
    return fig


# -----------------------------------------------------------------------------
# Country figures
# -----------------------------------------------------------------------------
def figure_country_deep_dive(df: pd.DataFrame, country: str) -> go.Figure:
    metrics = {label: col for label, col in DEEP_DIVE_METRICS.items() if col in df.columns}
    sub = df[df["country"] == country].copy()
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=list(metrics.keys()),
        horizontal_spacing=0.07,
        vertical_spacing=0.18,
    )
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    for (label, col), (r, c) in zip(metrics.items(), positions):
        s = sub[["year", col]].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s["year"], y=s[col], mode="lines", showlegend=False, name=label), row=r, col=c)
        fig.add_trace(go.Scatter(x=[s["year"].iloc[-1]], y=[s[col].iloc[-1]], mode="markers", showlegend=False), row=r, col=c)
        fig.add_annotation(
            x=s["year"].iloc[-1],
            y=s[col].iloc[-1],
            text=f"{s[col].iloc[-1]:.1f}",
            showarrow=False,
            font=dict(size=10),
            row=r,
            col=c,
        )
    fig.update_layout(height=600, template="plotly_white", title=f"Six-panel Energy Profile — {country}")
    fig.update_xaxes(title_text="Year")
    return fig


def figure_benchmark_tabs(df: pd.DataFrame, countries: list[str], metric: str, title: str) -> go.Figure:
    sub = df[df["country"].isin(countries)][["country", "year", metric]].dropna().copy()
    fig = px.line(
        sub,
        x="year",
        y=metric,
        color="country",
        markers=False,
        template="plotly_white",
        title=title,
    )
    fig.update_layout(height=480, yaxis_title=clean_label(metric))
    return fig


def figure_gdp_renewable_bubble(df: pd.DataFrame, year: int, selected_country: str) -> go.Figure:
    needed = ["country", "iso_code", "year", "renewables_share_energy", "fossil_share_energy"]
    if "gdp_per_capita" in df.columns:
        x_col = "gdp_per_capita"
    elif "gdp" in df.columns:
        x_col = "gdp"
    else:
        x_col = "energy_per_gdp" if "energy_per_gdp" in df.columns else None
    if x_col is None:
        return go.Figure()

    size_col = "population" if "population" in df.columns else None
    cols = needed + [x_col] + ([size_col] if size_col else [])
    sub = df[df["year"] == year][_safe_cols(df, cols)].copy()

    # Forward/backward fallback by country if the selected year has missing GDP values.
    if x_col in df.columns:
        fallback = df.sort_values(["country", "year"]).copy()
        fallback[x_col] = fallback.groupby("country")[x_col].ffill().bfill()
        if size_col:
            fallback[size_col] = fallback.groupby("country")[size_col].ffill().bfill()
        sub = fallback[fallback["year"] == year][_safe_cols(fallback, cols)].copy()

    sub = sub.dropna(subset=[x_col, "renewables_share_energy"])
    if sub.empty:
        return go.Figure()

    fig = px.scatter(
        sub,
        x=x_col,
        y="renewables_share_energy",
        size=size_col,
        color="fossil_share_energy" if "fossil_share_energy" in sub.columns else None,
        hover_name="country",
        log_x=True if x_col in ["gdp", "gdp_per_capita"] else False,
        template="plotly_white",
        title=f"Development Level vs Renewable Share ({year})",
        labels={x_col: clean_label(x_col), "renewables_share_energy": "Renewable share (%)"},
    )
    target = sub[sub["country"] == selected_country]
    if not target.empty:
        fig.add_trace(
            go.Scatter(
                x=target[x_col],
                y=target["renewables_share_energy"],
                mode="markers+text",
                text=[selected_country],
                textposition="top center",
                marker=dict(size=18, symbol="star", color="black"),
                name=f"Selected: {selected_country}",
            )
        )
    fig.update_layout(height=530)
    return fig


# -----------------------------------------------------------------------------
# Forecasting
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_prophet_forecast(series: pd.DataFrame, periods: int, changepoint_prior_scale: float) -> pd.DataFrame:
    from prophet import Prophet

    train = series.copy()
    train["ds"] = pd.to_datetime(train["year"].astype(int).astype(str) + "-01-01")
    train["y"] = train["value"]

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.90,
    )
    model.fit(train[["ds", "y"]])
    future = model.make_future_dataframe(periods=periods, freq="YS")
    fc = model.predict(future)
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    out["year"] = out["ds"].dt.year
    return out


def prophet_plot(series: pd.DataFrame, forecast: pd.DataFrame, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["year"], y=series["value"], mode="markers+lines", name="Observed"))
    hist = forecast[forecast["year"] <= int(series["year"].max())]
    future = forecast[forecast["year"] > int(series["year"].max())]
    fig.add_trace(go.Scatter(x=hist["year"], y=hist["yhat"], mode="lines", name="Prophet fit"))
    fig.add_trace(go.Scatter(x=future["year"], y=future["yhat"], mode="lines", name="Prophet forecast", line=dict(dash="dash")))
    if not future.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([future["year"], future["year"][::-1]]),
                y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
                fill="toself",
                line=dict(width=0),
                hoverinfo="skip",
                name="90% interval",
            )
        )
    fig.add_vline(x=int(series["year"].max()), line_dash="dot")
    fig.update_layout(template="plotly_white", height=500, title=f"Prophet Forecast — {label}", xaxis_title="Year")
    return fig


def make_sequences(arr: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback : i])
        y.append(arr[i])
    return np.array(X), np.array(y)


@st.cache_data(show_spinner=False)
def run_lstm_forecast(series: pd.DataFrame, lookback: int, epochs: int, horizon: int) -> dict:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    values = series[["value"]].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = make_sequences(scaled, lookback)
    if len(X) < 8:
        raise ValueError("Not enough annual observations for LSTM. Try a longer series.")

    test_size = min(5, max(2, len(X) // 5))
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = Sequential(
        [
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, 1)),
            Dropout(0.20),
            Bidirectional(LSTM(32)),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=8,
        callbacks=callbacks,
        verbose=0,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).flatten()
    actual = scaler.inverse_transform(y_test).flatten()
    test_years = series["year"].iloc[-test_size:].astype(int).values

    seq = scaled[-lookback:].reshape(1, lookback, 1)
    future_scaled = []
    for _ in range(horizon):
        p = model.predict(seq, verbose=0)[0, 0]
        future_scaled.append(p)
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = p
    future_values = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    future_years = np.arange(int(series["year"].max()) + 1, int(series["year"].max()) + horizon + 1)

    metrics = {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)) if len(actual) > 1 else np.nan,
    }
    return {
        "history": {k: list(map(float, v)) for k, v in history.history.items()},
        "test": pd.DataFrame({"year": test_years, "actual": actual, "predicted": pred}),
        "future": pd.DataFrame({"year": future_years, "forecast": future_values}),
        "metrics": metrics,
    }


def lstm_plot(series: pd.DataFrame, lstm_result: dict, label: str) -> go.Figure:
    future = lstm_result["future"]
    test = lstm_result["test"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["year"], y=series["value"], mode="lines+markers", name="Historical"))
    fig.add_trace(go.Scatter(x=test["year"], y=test["actual"], mode="markers", name="Held-out actual"))
    fig.add_trace(go.Scatter(x=test["year"], y=test["predicted"], mode="markers", name="Held-out predicted"))
    fig.add_trace(go.Scatter(x=future["year"], y=future["forecast"], mode="lines+markers", name="BiLSTM forecast", line=dict(dash="dash")))
    fig.add_vline(x=int(series["year"].max()), line_dash="dot")
    fig.update_layout(template="plotly_white", height=500, title=f"BiLSTM Forecast — {label}", xaxis_title="Year")
    return fig


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
def cluster_label(row: pd.Series) -> str:
    renew = row.get("renewables_share_energy", 0)
    hydro = row.get("hydro_share_energy", 0)
    wind = row.get("wind_share_energy", 0)
    solar = row.get("solar_share_energy", 0)
    fossil = row.get("fossil_share_energy", 100)
    nuclear = row.get("nuclear_share_energy", 0)
    if renew >= 60 and hydro >= max(wind + solar, 15):
        return "Hydro-led leaders"
    if renew >= 35 and wind + solar >= hydro:
        return "Wind/solar transition"
    if nuclear >= 15:
        return "Nuclear-mixed economies"
    if fossil >= 85:
        return "Fossil-dependent"
    if renew >= 30:
        return "Emerging renewables"
    return "Mixed/early transition"


@st.cache_data(show_spinner=False)
def run_clusters(df: pd.DataFrame, year: int, features: tuple[str, ...], k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    features = [f for f in features if f in df.columns]
    data = df[df["year"] == year][["country", "iso_code"] + features].copy()
    features = [f for f in features if data[f].notna().sum() > 10]
    if len(features) < 2:
        raise ValueError("At least two usable numeric features are needed for clustering.")

    for col in features:
        data[col] = data[col].fillna(df[col].median())
    data = data.dropna(subset=features).reset_index(drop=True)
    if len(data) <= k:
        raise ValueError("The selected year has too few countries for this k value.")

    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])

    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    data["cluster"] = labels

    profile = data.groupby("cluster")[features].mean()
    label_lookup = {i: cluster_label(profile.loc[i]) for i in profile.index}
    data["cluster_label"] = data["cluster"].map(lambda i: f"C{i + 1}: {label_lookup[i]}")

    pca = PCA(n_components=2, random_state=SEED)
    pcs = pca.fit_transform(X)
    data["PCA1"] = pcs[:, 0]
    data["PCA2"] = pcs[:, 1]

    if len(data) >= 8:
        perplexity = min(30, max(2, len(data) // 3))
        perplexity = min(perplexity, len(data) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED, init="pca", learning_rate="auto")
        ts = tsne.fit_transform(X)
        data["TSNE1"] = ts[:, 0]
        data["TSNE2"] = ts[:, 1]
    else:
        data["TSNE1"] = np.nan
        data["TSNE2"] = np.nan

    diagnostics = []
    upper_k = min(10, len(data) - 1)
    for kk in range(2, upper_k + 1):
        kk_labels = KMeans(n_clusters=kk, random_state=SEED, n_init=10).fit_predict(X)
        diagnostics.append({"k": kk, "silhouette": silhouette_score(X, kk_labels)})
    diagnostics_df = pd.DataFrame(diagnostics)
    return data, profile, diagnostics_df, X, features


def figure_pca_tsne_dual(data: pd.DataFrame) -> go.Figure:
    clusters = sorted(data["cluster_label"].dropna().unique())
    fig = make_subplots(rows=1, cols=2, subplot_titles=["PCA view", "t-SNE view"], horizontal_spacing=0.08)

    for cluster in clusters:
        sub = data[data["cluster_label"] == cluster]
        fig.add_trace(
            go.Scatter(
                x=sub["PCA1"], y=sub["PCA2"], mode="markers", name=cluster,
                text=sub["country"], hovertemplate="%{text}<br>PCA1=%{x:.2f}<br>PCA2=%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )
        if sub["TSNE1"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=sub["TSNE1"], y=sub["TSNE2"], mode="markers", name=cluster,
                    text=sub["country"], showlegend=False,
                    hovertemplate="%{text}<br>t-SNE1=%{x:.2f}<br>t-SNE2=%{y:.2f}<extra></extra>",
                ),
                row=1, col=2,
            )

    nz = data[data["country"] == "New Zealand"]
    if not nz.empty:
        fig.add_trace(
            go.Scatter(x=nz["PCA1"], y=nz["PCA2"], mode="markers+text", text=["★ NZ"], textposition="top center", marker=dict(size=18, symbol="star", color="black"), name="New Zealand"),
            row=1, col=1,
        )
        if nz["TSNE1"].notna().any():
            fig.add_trace(
                go.Scatter(x=nz["TSNE1"], y=nz["TSNE2"], mode="markers+text", text=["★ NZ"], textposition="top center", marker=dict(size=18, symbol="star", color="black"), showlegend=False),
                row=1, col=2,
            )

    fig.update_layout(template="plotly_white", height=570, title="Country Energy Clusters — PCA vs t-SNE", hovermode="closest")
    fig.update_xaxes(title_text="PCA1", row=1, col=1)
    fig.update_yaxes(title_text="PCA2", row=1, col=1)
    fig.update_xaxes(title_text="t-SNE1", row=1, col=2)
    fig.update_yaxes(title_text="t-SNE2", row=1, col=2)
    return fig


def figure_cluster_heatmap(profile: pd.DataFrame) -> go.Figure:
    z = StandardScaler().fit_transform(profile.values)
    x = [clean_label(c) for c in profile.columns]
    y = [f"C{int(i) + 1}" for i in profile.index]
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorbar=dict(title="Std. value")))
    fig.update_layout(template="plotly_white", height=430, title="Cluster Profile Heatmap")
    return fig


def figure_cluster_radar(data: pd.DataFrame, features: list[str], country: str) -> go.Figure:
    if country not in set(data["country"]):
        country = data["country"].iloc[0]
    row = data[data["country"] == country].iloc[0]
    cluster_data = data[data["cluster"] == row["cluster"]]

    mins = data[features].min()
    maxs = data[features].max()
    denom = (maxs - mins).replace(0, np.nan)
    country_values = ((row[features] - mins) / denom).fillna(0).astype(float).values
    cluster_values = ((cluster_data[features].mean() - mins) / denom).fillna(0).astype(float).values
    labels = [clean_label(f) for f in features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cluster_values, theta=labels, fill="toself", name="Cluster average"))
    fig.add_trace(go.Scatterpolar(r=country_values, theta=labels, fill="toself", name=country))
    fig.update_layout(
        template="plotly_white",
        height=470,
        title=f"Country vs Cluster Average — {country}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    return fig


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def page_global_dashboard(ctx: DataContext) -> None:
    df, plot_year = ctx.df, ctx.plot_year
    render_page_header(
        "EnergyLens: Global Dashboard",
        "A high-level view of global energy transition patterns, from broad renewable growth to persistent fossil dependency.",
    )
    render_kpi_cards(df, plot_year)

    st.markdown(
        '<div class="insight-box"><b>How to read this page:</b> start with the map to see where transition is happening, then use the trend panels and ranking tables to identify global patterns and outliers.</div>',
        unsafe_allow_html=True,
    )

    metric_options = {k: v for k, v in KEY_COLS.items() if v in df.columns}
    years = sorted(df["year"].dropna().astype(int).unique())
    c1, c2 = st.columns([1.2, 2.0])
    with c1:
        map_metric_label = st.selectbox("Animated map metric", list(metric_options.keys()), index=0)
    with c2:
        default_start = max(min(years), plot_year - 15)
        year_range = st.slider("Animation year range", min(years), max(years), (default_start, plot_year))

    st.plotly_chart(
        figure_animated_choropleth(df, metric_options[map_metric_label], year_range[0], year_range[1]),
        use_container_width=True,
    )

    tab1, tab2, tab3 = st.tabs(["Global trend dashboard", "Transition stages", "Country rankings"])
    with tab1:
        st.plotly_chart(figure_global_trends(df), use_container_width=True)
    with tab2:
        st.plotly_chart(figure_stage_distribution(df), use_container_width=True)
    with tab3:
        latest = latest_year_slice(df, plot_year)
        if "renewables_share_energy" in latest.columns:
            left, right = st.columns(2)
            with left:
                st.subheader(f"Top renewable countries ({plot_year})")
                cols = _safe_cols(latest, ["country", "renewables_share_energy", "fossil_share_energy", "solar_share_energy", "wind_share_energy", "hydro_share_energy"])
                top = latest.dropna(subset=["renewables_share_energy"]).nlargest(10, "renewables_share_energy")[cols]
                st.dataframe(top.round(2), use_container_width=True, hide_index=True)
            with right:
                st.subheader(f"High fossil dependency ({plot_year})")
                fossil_col = "fossil_share_energy" if "fossil_share_energy" in latest.columns else "fossil_energy_per_capita"
                cols = _safe_cols(latest, ["country", fossil_col, "renewables_share_energy", "fossil_energy_per_capita", "energy_per_gdp"])
                bottom = latest.dropna(subset=[fossil_col]).nlargest(10, fossil_col)[cols]
                st.dataframe(bottom.round(2), use_container_width=True, hide_index=True)


def page_country_explorer(ctx: DataContext) -> None:
    df, plot_year = ctx.df, ctx.plot_year
    render_page_header(
        "Country Explorer: National Energy Transition Profile",
        "Select a country to inspect its energy mix, benchmark it against reference countries, and place it in the global development-energy space.",
    )

    countries = sorted(df["country"].dropna().unique())
    selected_country = st.selectbox("Focus country", countries, index=countries.index("New Zealand") if "New Zealand" in countries else 0)
    row = latest_country_row(df, selected_country, plot_year)

    if row is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transition stage", row.get("transition_stage", "N/A"))
        renew = row.get("renewables_share_energy", np.nan)
        fossil = row.get("fossil_share_energy", np.nan)
        c2.metric("Renewable share", f"{renew:.1f}%" if pd.notna(renew) else "N/A")
        c3.metric("Fossil share", f"{fossil:.1f}%" if pd.notna(fossil) else "N/A")
        c4.metric("Latest profile year", int(row["year"]))

    st.plotly_chart(figure_country_deep_dive(df, selected_country), use_container_width=True)

    st.subheader("Benchmark comparison")
    default_bench = [c for c in ["New Zealand", "Norway", "Germany", "Australia", "China", "United States"] if c in countries]
    benchmark_countries = st.multiselect("Benchmark countries", countries, default=default_bench)
    metric_tabs = st.tabs(["Renewable", "Fossil", "Solar", "Wind"])
    metric_map = [
        ("Renewable share", "renewables_share_energy"),
        ("Fossil share", "fossil_share_energy"),
        ("Solar share", "solar_share_energy"),
        ("Wind share", "wind_share_energy"),
    ]
    for tab, (label, metric) in zip(metric_tabs, metric_map):
        with tab:
            if metric in df.columns and benchmark_countries:
                st.plotly_chart(
                    figure_benchmark_tabs(df, benchmark_countries, metric, f"{label} benchmark comparison"),
                    use_container_width=True,
                )

    st.subheader("Global position: development level vs renewable share")
    st.plotly_chart(figure_gdp_renewable_bubble(df, plot_year, selected_country), use_container_width=True)


def page_forecasting(ctx: DataContext) -> None:
    df = ctx.df
    render_page_header(
        "Forecasting: Renewable Transition Outlook",
        "Compare a trend-based Prophet forecast with an exploratory BiLSTM sequence model on annual energy data.",
    )

    st.markdown(
        '<div class="warning-box"><b>Model interpretation note:</b> annual country-level energy series are short. Prophet is used for stable long-term trend extrapolation, while BiLSTM is kept as an exploratory nonlinear sequence model.</div>',
        unsafe_allow_html=True,
    )

    model_role = pd.DataFrame(
        {
            "Model": ["Prophet", "BiLSTM"],
            "Role in the app": ["Main trend-based forecast", "Exploratory sequence forecast"],
            "Best use": ["Annual long-term renewable/fossil trend", "Testing nonlinear temporal patterns"],
            "Main risk": ["Can extrapolate past trend too smoothly", "Short annual series can overfit"],
        }
    )
    st.dataframe(model_role, use_container_width=True, hide_index=True)

    countries = ["Global average"] + sorted(df["country"].dropna().unique())
    preferred = ["Global average", "New Zealand", "Norway", "Germany", "Australia"]
    available_preferred = [c for c in preferred if c in countries]
    metric_options = {
        "Renewable share (%)": "renewables_share_energy",
        "Fossil share (%)": "fossil_share_energy",
        "Fossil energy per capita": "fossil_energy_per_capita",
    }
    metric_options = {k: v for k, v in metric_options.items() if v in df.columns}

    c1, c2, c3 = st.columns([1.3, 1.1, 0.8])
    with c1:
        country = st.selectbox("Forecast series", countries, index=countries.index("Global average"))
    with c2:
        metric_label = st.selectbox("Target metric", list(metric_options.keys()))
    with c3:
        horizon = st.slider("Forecast horizon", 5, 20, 10)

    if country not in available_preferred and country != "Global average":
        st.caption("Tip: the original benchmark set used New Zealand, Norway, Germany and Australia, but any country with enough data can be explored here.")

    metric = metric_options[metric_label]
    series = get_series(df, country, metric)
    if len(series) < 12:
        st.error("This series has too few annual observations for reliable forecasting. Choose another country or metric.")
        return

    st.markdown(f"**Selected series:** {country} · {metric_label} · {len(series)} annual observations")
    prophet_tab, lstm_tab = st.tabs(["Prophet forecast", "BiLSTM forecast"])

    with prophet_tab:
        default_cps = 0.05 if country == "Global average" else 0.15
        cps = st.slider("Prophet changepoint prior scale", 0.01, 0.50, default_cps, step=0.01)
        try:
            with st.spinner("Fitting Prophet model ..."):
                forecast = run_prophet_forecast(series, periods=horizon, changepoint_prior_scale=cps)
            st.plotly_chart(prophet_plot(series, forecast, f"{country} · {metric_label}"), use_container_width=True)

            future = forecast[forecast["year"] > int(series["year"].max())].copy()
            start_val = series["value"].iloc[-1]
            end_val = future["yhat"].iloc[-1] if not future.empty else np.nan
            delta = end_val - start_val if pd.notna(end_val) else np.nan
            st.markdown(
                f'<div class="insight-box"><b>Forecast reading:</b> from the latest observed value ({start_val:.2f}), Prophet projects {end_val:.2f} by {int(future["year"].max()) if not future.empty else "N/A"}, a change of {delta:+.2f}.</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(future[["year", "yhat", "yhat_lower", "yhat_upper", "trend"]].round(3), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error("Prophet could not run. Check that `prophet` is installed in requirements.txt.")
            st.exception(exc)

    with lstm_tab:
        c1, c2 = st.columns(2)
        with c1:
            lookback = st.slider("LSTM lookback window", 3, 8, 5)
        with c2:
            epochs = st.slider("Training epochs", 20, 200, 80, step=10)

        st.markdown(
            '<p class="small-note">BiLSTM trains only after you click the button. This prevents Streamlit Cloud from becoming slow when the page first loads.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Train / refresh BiLSTM forecast", type="primary"):
            try:
                with st.spinner("Training BiLSTM model ..."):
                    result = run_lstm_forecast(series, lookback=lookback, epochs=epochs, horizon=horizon)
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{result['metrics']['MAE']:.3f}")
                m2.metric("RMSE", f"{result['metrics']['RMSE']:.3f}")
                r2 = result["metrics"]["R2"]
                m3.metric("R²", f"{r2:.3f}" if pd.notna(r2) else "N/A")
                st.plotly_chart(lstm_plot(series, result, f"{country} · {metric_label}"), use_container_width=True)
                st.dataframe(result["future"].round(3), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error("BiLSTM could not run. On Streamlit Cloud, use Python 3.11 and include `tensorflow-cpu` in requirements.txt.")
                st.exception(exc)


def page_cluster_analysis(ctx: DataContext) -> None:
    df, plot_year = ctx.df, ctx.plot_year
    render_page_header(
        "Cluster Analysis: Country Energy Transition Archetypes",
        "K-Means groups countries with similar energy structures, while PCA and t-SNE show complementary views of cluster separation.",
    )

    years = sorted(df["year"].dropna().astype(int).unique())
    available_features = [f for f in CLUSTER_FEATURES if f in df.columns]

    c1, c2, c3 = st.columns([0.8, 0.8, 2.4])
    with c1:
        year = st.slider("Cluster year", min(years), max(years), plot_year)
    with c2:
        k = st.slider("Number of clusters", 2, 10, 7)
    with c3:
        features = st.multiselect("Clustering features", available_features, default=available_features)

    if len(features) < 2:
        st.warning("Please select at least two features.")
        return

    try:
        data, profile, diagnostics, X, features = run_clusters(df, year, tuple(features), k)
    except Exception as exc:
        st.error("Cluster analysis failed for the selected year/features.")
        st.exception(exc)
        return

    sil_selected = silhouette_score(X, data["cluster"]) if len(set(data["cluster"])) > 1 else np.nan
    m1, m2, m3 = st.columns(3)
    m1.metric("Countries clustered", f"{len(data):,}")
    m2.metric("Features used", f"{len(features)}")
    m3.metric("Silhouette score", f"{sil_selected:.3f}" if pd.notna(sil_selected) else "N/A")

    st.markdown(
        '<div class="insight-box"><b>Interpretation:</b> PCA preserves broad variance structure; t-SNE emphasises local neighbourhood similarity. Showing both reduces the risk of over-reading one projection.</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(figure_pca_tsne_dual(data), use_container_width=True)

    d1, d2 = st.columns([1.2, 1.0])
    with d1:
        if not diagnostics.empty:
            fig_diag = px.line(diagnostics, x="k", y="silhouette", markers=True, template="plotly_white", title="Silhouette diagnostic by k")
            fig_diag.add_vline(x=k, line_dash="dot")
            fig_diag.update_layout(height=360)
            st.plotly_chart(fig_diag, use_container_width=True)
    with d2:
        st.subheader("Cluster size")
        cluster_count = data["cluster_label"].value_counts().rename_axis("Cluster").reset_index(name="Countries")
        st.dataframe(cluster_count, use_container_width=True, hide_index=True)

    st.subheader("Geographic distribution of clusters")
    map_fig = px.choropleth(
        data,
        locations="iso_code",
        color="cluster_label",
        hover_name="country",
        projection="natural earth",
        template="plotly_white",
        title=f"Cluster Map ({year})",
    )
    map_fig.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(map_fig, use_container_width=True)

    heat_tab, radar_tab, table_tab = st.tabs(["Cluster heatmap", "Country vs cluster radar", "Download table"])
    with heat_tab:
        st.plotly_chart(figure_cluster_heatmap(profile), use_container_width=True)
        renamed = profile.rename(columns={c: clean_label(c) for c in profile.columns})
        st.dataframe(renamed.round(2), use_container_width=True)
    with radar_tab:
        countries_in_cluster = sorted(data["country"].unique())
        default_index = countries_in_cluster.index("New Zealand") if "New Zealand" in countries_in_cluster else 0
        radar_country = st.selectbox("Select country for radar comparison", countries_in_cluster, index=default_index)
        st.plotly_chart(figure_cluster_radar(data, features, radar_country), use_container_width=True)
    with table_tab:
        display_cols = ["country", "iso_code", "cluster_label"] + features
        table = data[display_cols].sort_values(["cluster_label", "country"]).round(3)
        st.dataframe(table, use_container_width=True, hide_index=True)
        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download cluster assignment CSV",
            data=csv,
            file_name=f"energylens_cluster_assignments_{year}_k{k}.csv",
            mime="text/csv",
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    st.sidebar.title("⚡ EnergyLens")
    st.sidebar.caption("Global Energy Transition Intelligence Platform")

    uploaded = st.sidebar.file_uploader("Optional: upload local OWID CSV", type=["csv"])
    try:
        ctx = load_data(uploaded.getvalue() if uploaded else None)
    except Exception as exc:
        st.error("Could not load the dataset. Upload a valid OWID energy CSV or check internet access.")
        st.exception(exc)
        return

    page = st.sidebar.radio(
        "Navigation",
        ["Global Dashboard", "Country Explorer", "Forecasting (LSTM + Prophet)", "Cluster Analysis"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
        **Dataset:** OWID Energy Data  
        **Filtered years:** 1990–{ctx.latest_year}  
        **Default analysis year:** {ctx.plot_year}
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Designed for Assignment 4 Streamlit deployment.")

    if page == "Global Dashboard":
        page_global_dashboard(ctx)
    elif page == "Country Explorer":
        page_country_explorer(ctx)
    elif page == "Forecasting (LSTM + Prophet)":
        page_forecasting(ctx)
    elif page == "Cluster Analysis":
        page_cluster_analysis(ctx)


if __name__ == "__main__":
    main()
