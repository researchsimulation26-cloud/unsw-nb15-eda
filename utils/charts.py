import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .constants import ATTACK_COLORS, PLOTLY_TEMPLATE, COLORS


def apply_template(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


def create_kpi_row(cols, metrics):
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label=label, value=value, delta=delta)


def donut_chart(labels, values, title="", colors=None):
    if colors is None:
        colors = ["#3fb950", "#f85149"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors, line=dict(color="#161b22", width=2)),
                textinfo="label+percent",
                textfont=dict(size=14, color="#e6edf3"),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%: %{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#e6edf3"), x=0.5),
        **PLOTLY_TEMPLATE["layout"],
        showlegend=True,
        legend=dict(font=dict(color="#e6edf3")),
    )
    return fig


def horizontal_bar(
    df, y, x, title="", color="#00d4ff", sort_values=True, height=500
):
    if sort_values:
        df = df.sort_values(x, ascending=True)
    fig = px.bar(
        df,
        y=y,
        x=x,
        title=title,
        orientation="h",
        color_discrete_sequence=[color],
        text_auto=".2s",
    )
    fig.update_traces(textposition="outside", marker_line=dict(width=0))
    fig.update_layout(
        height=height,
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=10, r=40, t=40, b=10),
        **PLOTLY_TEMPLATE["layout"],
    )
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


def colored_bar(df, y, x, color_col, title="", height=500, color_map=None):
    if color_map is None:
        color_map = ATTACK_COLORS
    fig = px.bar(
        df,
        y=y,
        x=x,
        title=title,
        orientation="h",
        color=color_col,
        color_discrete_map=color_map,
        text_auto=".2s",
    )
    fig.update_traces(textposition="outside", marker_line=dict(width=0))
    fig.update_layout(
        height=height,
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=10, r=40, t=40, b=10),
        legend=dict(font=dict(color="#e6edf3"), orientation="h", y=-0.2),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def stacked_bar(
    df, x, y, color, title="", height=500, color_map=None, barmode="stack"
):
    if color_map is None:
        color_map = ATTACK_COLORS
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        color=color,
        color_discrete_map=color_map,
        barmode=barmode,
    )
    fig.update_layout(
        height=height,
        xaxis_title="",
        yaxis_title="Count",
        margin=dict(l=10, r=20, t=40, b=80),
        legend=dict(font=dict(color="#e6edf3"), orientation="h", y=-0.3),
        **PLOTLY_TEMPLATE["layout"],
    )
    fig.update_xaxes(tickangle=45)
    return fig


def histogram(
    df,
    x,
    title="",
    color=None,
    color_map=None,
    log_x=False,
    log_y=False,
    barmode="overlay",
    opacity=0.65,
    height=400,
):
    if color_map is None:
        color_map = ATTACK_COLORS
    fig = px.histogram(
        df,
        x=x,
        title=title,
        color=color,
        color_discrete_map=color_map,
        barmode=barmode,
        opacity=opacity,
        log_x=log_x,
        log_y=log_y,
        marginal="box",
    )
    fig.update_layout(
        height=height,
        xaxis_title=x,
        yaxis_title="Count",
        margin=dict(l=10, r=20, t=40, b=10),
        legend=dict(font=dict(color="#e6edf3")),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def violin_plot(
    df, x, y, title="", color_map=None, log_y=False, height=500
):
    if color_map is None:
        color_map = {"Normal": "#3fb950", "Attack": "#f85149", 0: "#3fb950", 1: "#f85149"}
    fig = px.violin(
        df,
        x=x,
        y=y,
        title=title,
        color=x,
        color_discrete_map=color_map,
        box=True,
        points=False,
        log_y=log_y,
    )
    fig.update_layout(
        height=height,
        xaxis_title="",
        yaxis_title=y,
        margin=dict(l=10, r=20, t=40, b=10),
        legend=dict(font=dict(color="#e6edf3")),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def box_plot(
    df, x, y, title="", color_map=None, log_y=False, height=500
):
    if color_map is None:
        color_map = ATTACK_COLORS
    fig = px.box(
        df,
        x=x,
        y=y,
        title=title,
        color=x,
        color_discrete_map=color_map,
        log_y=log_y,
        points=False,
    )
    fig.update_layout(
        height=height,
        xaxis_title="",
        yaxis_title=y,
        margin=dict(l=10, r=20, t=40, b=80),
        legend=dict(font=dict(color="#e6edf3")),
        **PLOTLY_TEMPLATE["layout"],
    )
    fig.update_xaxes(tickangle=45)
    return fig


def line_chart(df, x, y, title="", color=None, color_map=None, height=400):
    if color_map is None:
        color_map = ATTACK_COLORS
    fig = px.line(
        df,
        x=x,
        y=y,
        title=title,
        color=color,
        color_discrete_map=color_map,
        markers=False,
    )
    fig.update_layout(
        height=height,
        xaxis_title="",
        yaxis_title="Count",
        margin=dict(l=10, r=20, t=40, b=10),
        legend=dict(font=dict(color="#e6edf3")),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def heatmap(
    df, title="", colorscale="RdBu_r", height=600, zmin=None, zmax=None
):
    mask = df.isna().values if df.isna().any().any() else None
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values if mask is None else np.ma.masked_where(mask, df.values),
            x=df.columns.tolist(),
            y=df.index.tolist(),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            text=np.round(df.values, 2) if mask is None else np.round(df.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9, color="#e6edf3"),
            hovertemplate="<b>%{y}</b> | <b>%{x}</b><br>Value: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e6edf3"), x=0.5),
        height=height,
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=10, r=20, t=50, b=100),
        **PLOTLY_TEMPLATE["layout"],
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def radar_chart(categories, values, title="", color="#00d4ff", fill=True):
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself" if fill else None,
            name=title,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=4),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e6edf3"), x=0.5),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="#21262d",
                tickfont=dict(color="#8b949e", size=10),
            ),
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(gridcolor="#21262d", tickfont=dict(color="#e6edf3", size=10)),
        ),
        height=450,
        margin=dict(l=40, r=40, t=40, b=40),
        **PLOTLY_TEMPLATE["layout"],
    )
    return fig


def style_metric_cards():
    return st.markdown(
        """
        <style>
        div[data-testid="metric-container"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        div[data-testid="metric-container"] label {
            color: #8b949e;
            font-size: 0.85rem;
        }
        div[data-testid="metric-container"] div[data-testid="metric-value"] {
            color: #e6edf3;
            font-weight: 700;
        }
        div[data-testid="metric-container"] div[data-testid="metric-delta"] {
            font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def download_csv_button(df, filename, label="Download CSV"):
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"📥 {label}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )
