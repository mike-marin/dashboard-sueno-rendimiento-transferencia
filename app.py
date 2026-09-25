# -*- coding: utf-8 -*-
"""
Dashboard - Actividad de Transferencia: Sueño, Estrés y Rendimiento Académico
Programación para Ciencia de Datos II - Fundación Universitaria Compensar

Identidad Visual: Neuro-Circadian Analytics & Deep Slate
Autor: Michael Marín Herrera (mmarinh@ucompensar.edu.co)
Docente: William Eduardo Clavijo Bohorquez
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sueno_rendimiento.csv")

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "sueno_rendimiento.csv")

df = pd.read_csv(DATA_PATH)

# Homologación de nombres de columnas
col_map = {
    'horas_sueno_promedio': 'horas_sueno',
    'horas_estudio_semanales': 'horas_estudio',
    'nivel_estres_1_10': 'nivel_estres',
    'puntaje_rendimiento': 'puntaje'
}
df = df.rename(columns=col_map)

FEATURES = ['horas_sueno', 'horas_estudio', 'nivel_estres']
TARGET = 'puntaje'

mediana_puntaje = float(df[TARGET].median())
mediana_sueno = float(df['horas_sueno'].median())
df['en_riesgo'] = (df[TARGET] <= mediana_puntaje).astype(int)

# Detección de atípico clínico (|z| > 2.5 -> ID 9)
z_scores = np.abs(stats.zscore(df[TARGET]))
outlier_idx = df.index[z_scores > 2.5].tolist()

# ---------------------------------------------------------------------------
# 2. MODELOS BASE DE REFERENCIA
# ---------------------------------------------------------------------------

X_full = df[FEATURES].to_numpy()
y_full = df[TARGET].to_numpy()

X_simple = df[['horas_sueno']].to_numpy()
modelo_simple = LinearRegression().fit(X_simple, y_full)
r_pearson = float(np.corrcoef(df['horas_sueno'], df[TARGET])[0, 1])
r2_simple = float(r2_score(y_full, modelo_simple.predict(X_simple)))

modelo_multiple_ref = LinearRegression().fit(X_full, y_full)
r2_multiple = float(r2_score(y_full, modelo_multiple_ref.predict(X_full)))

modelo_logit_ref = LogisticRegression(C=1.0, random_state=42).fit(X_full, df['en_riesgo'])

# ---------------------------------------------------------------------------
# 3. PALETA CIRCADIAN & HELPERS VISUALES
# ---------------------------------------------------------------------------

THEME = {
    "midnight": "#0F172A",
    "deep_space": "#1E1B4B",
    "cyan": "#06B6D4",
    "teal": "#0D9488",
    "violet": "#6366F1",
    "coral": "#F43F5E",
    "amber": "#F59E0B",
    "emerald": "#10B981",
    "text": "#0F172A",
    "muted": "#64748B",
    "card_bg": "#FFFFFF",
    "grid": "#F1F5F9"
}

def neuro_kpi(titulo, valor, subtitulo, icon_class, icon_bg, icon_color, badge_text=None, badge_bg=None, badge_color=None):
    return html.Div([
        html.Div([
            html.Div([
                html.I(className=f"{icon_class}", style={"color": icon_color})
            ], className="kpi-icon-pill", style={"backgroundColor": icon_bg}),
            html.Span(badge_text, className="kpi-badge-indicator", style={"backgroundColor": badge_bg or "#F1F5F9", "color": badge_color or "#475569"}) if badge_text else None
        ], className="kpi-header-row"),
        html.Div([
            html.Div(valor, className="kpi-number-display"),
            html.P(titulo, className="kpi-label-text"),
            html.Div(subtitulo, className="kpi-footer-subtext")
        ])
    ], className="neuro-kpi-card")

def chart_styler(fig, titulo=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Outfit, -apple-system, sans-serif", size=12, color=THEME["text"]),
        margin=dict(l=45, r=25, t=55 if titulo else 20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01, font=dict(size=11, color=THEME["muted"])),
        hoverlabel=dict(bgcolor=THEME["midnight"], font_size=12, font_family="Outfit", font_color="white"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["grid"], zerolinecolor="#CBD5E1")
    fig.update_yaxes(showgrid=True, gridcolor=THEME["grid"], zerolinecolor="#CBD5E1")
    if titulo:
        fig.update_layout(title=dict(text=f"<b>{titulo}</b>", x=0.01, y=0.98, font=dict(size=14, color=THEME["midnight"], family="Space Grotesk, sans-serif")))
    return fig

# ---------------------------------------------------------------------------
# 4. INICIALIZACIÓN DE LA APP (COMPATIBLE CON BINDER & JUPYTERHUB)
# ---------------------------------------------------------------------------

jupyterhub_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "")
if jupyterhub_prefix:
    requests_prefix = f"{jupyterhub_prefix.rstrip('/')}/proxy/8050/"
else:
    requests_prefix = "/"

app = dash.Dash(
    __name__,
    requests_pathname_prefix=requests_prefix,
    routes_pathname_prefix="/",
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Neuro-Analytics | Sueño y Rendimiento Académico",
)
server = app.server

# ---------------------------------------------------------------------------
# 5. HEADER / HERO SECTION
# ---------------------------------------------------------------------------

hero_header = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.I(className="fa-solid fa-dna me-1"),
                html.Span("Estudio Neurocognitivo & Bienestar Universitario")
            ], className="hero-tag"),
            html.H1("Impacto del Sueño y Estrés en el Rendimiento Académico", className="hero-title"),
            html.P("Plataforma analítica e interactiva para la modelación predictiva, contraste de hipótesis y diagnóstico de riesgo estudiantil.", className="hero-subtitle")
        ], md=8),
        dbc.Col([
            html.Div([
                html.Div("ACTIVIDAD DE TRANSFERENCIA", className="small fw-bold text-uppercase", style={"color": "#67E8F9", "letterSpacing": "0.06em"}),
                html.Div("Michael Marín Herrera", className="fs-6 fw-bold text-white mt-1"),
                html.Small("mmarinh@ucompensar.edu.co", className="text-white-50 font-monospace d-block"),
                html.Small("Docente: William E. Clavijo Bohorquez", className="text-white-50 mt-1 d-block")
            ], className="hero-author-badge")
        ], md=4, className="d-flex justify-content-md-end align-items-center mt-3 mt-md-0")
    ])
], className="neuro-hero")

# ---------------------------------------------------------------------------
# 5.1 PESTAÑA 1: CONTEXTO & RESUMEN
# ---------------------------------------------------------------------------

tab_contexto = html.Div([
    dbc.Row([
        dbc.Col(neuro_kpi(
            "Muestra Analizada", f"{len(df)} Estudiantes", "Población universitaria",
            "fa-solid fa-users", "#EEF2FF", THEME["violet"], "Dataset 2026", "#E0E7FF", "#4338CA"
        ), md=3),
        dbc.Col(neuro_kpi(
            "Media de Sueño", f"{df['horas_sueno'].mean():.2f} hrs/día", f"Mediana: {mediana_sueno:.2f} hrs",
            "fa-solid fa-moon", "#ECFEFF", THEME["cyan"], "Hábito", "#CFFAFE", "#0E7490"
        ), md=3),
        dbc.Col(neuro_kpi(
            "Correlación r", f"r = {r_pearson:+.2f}", "Asociación directa positiva",
            "fa-solid fa-bolt", "#F0FDF4", THEME["emerald"], "Fuerza Media", "#DCFCE7", "#15803D"
        ), md=3),
        dbc.Col(neuro_kpi(
            "Caso Atípico", f"{len(outlier_idx)} Estudiante", "ID 9 (22 pts, |z| > 2.5)",
            "fa-solid fa-triangle-exclamation", "#FFF1F2", THEME["coral"], "Alerta Outlier", "#FFE4E6", "#BE123C"
        ), md=3),
    ], className="g-3 mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-brain me-2", style={"color": THEME["teal"]}),
                html.Span("Fundamentación del Problema", className="panel-header-badge")
            ], className="mb-3"),
            html.P("El descanso nocturno y el manejo de estresores representan factores determinantes en la consolidación de la memoria y la capacidad de resolución analítica en estudiantes de educación superior.", className="text-secondary"),
            html.P("Este entorno integra el ciclo metodológico formal de Ciencia de Datos: Exploración de distribuciones y atípicos, Contraste de Hipótesis poblacional (Welch t-test y Mann-Whitney U), Regresión Múltiple/Ridge y Clasificación Logística con Simulador de Riesgo en tiempo real.", className="text-secondary"),
            html.Hr(className="my-3 text-muted"),
            html.H6("Variables Estudiadas", className="fw-bold mb-2 text-dark"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("🌙 horas_sueno: ", className="fw-bold text-dark"),
                        html.Span("Promedio diario de descanso nocturno (horas/día).", className="text-muted small")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("📚 horas_estudio: ", className="fw-bold text-dark"),
                        html.Span("Tiempo semanal de preparación académica individual.", className="text-muted small")
                    ], className="mb-2"),
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.Span("⚡ nivel_estres: ", className="fw-bold text-dark"),
                        html.Span("Índice de sobrecarga autopercibida (escala 1 a 10).", className="text-muted small")
                    ], className="mb-2"),
                    html.Div([
                        html.Span("🎯 puntaje: ", className="fw-bold text-dark"),
                        html.Span("Calificación estandarizada final (0 a 100 puntos).", className="text-muted small")
                    ], className="mb-2"),
                ], md=6),
            ])
        ])), md=7),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-compass me-2", style={"color": THEME["cyan"]}),
                html.Span("Módulos de la Plataforma", className="panel-header-badge")
            ], className="mb-3"),
            html.Div([
                html.Div([
                    html.Span("1", className="badge bg-dark rounded-circle me-2"),
                    html.B("Exploración (EDA): "), "Filtro dinámico de sueño y visualización bivariada."
                ], className="mb-2 text-secondary"),
                html.Div([
                    html.Span("2", className="badge bg-dark rounded-circle me-2"),
                    html.B("Contraste de Hipótesis: "), "Prueba de impacto de alto descanso con Welch t y MW-U."
                ], className="mb-2 text-secondary"),
                html.Div([
                    html.Span("3", className="badge bg-dark rounded-circle me-2"),
                    html.B("Regresión & Ridge: "), "Ajuste multivariado, evaluación R² y penalización L2."
                ], className="mb-2 text-secondary"),
                html.Div([
                    html.Span("4", className="badge bg-dark rounded-circle me-2"),
                    html.B("Simulador & Diagnóstico: "), "Calculadora de notas y probabilidad de alerta académica."
                ], className="mb-3 text-secondary"),
            ]),
            html.Div([
                html.I(className="fa-solid fa-wand-magic-sparkles me-2", style={"color": THEME["amber"]}),
                html.Small("Interactúa con los controles para simular escenarios y contrastar hipótesis en vivo.", className="text-muted")
            ], className="p-2 rounded bg-light border d-flex align-items-center")
        ])), md=5),
    ], className="g-3"),
])

# ---------------------------------------------------------------------------
# 5.2 PESTAÑA 2: EXPLORACIÓN DE DATOS (EDA)
# ---------------------------------------------------------------------------

min_s, max_s = float(df['horas_sueno'].min()), float(df['horas_sueno'].max())

tab_exploracion = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Filtro Interactivo de Horas de Sueño:", className="fw-bold mb-0 me-2 small"),
                    html.Span(id="badge-rango-sueno", className="badge px-3 py-1 rounded-pill", style={"backgroundColor": THEME["midnight"], "color": "#67E8F9"})
                ], className="d-flex align-items-center mb-2"),
                dcc.RangeSlider(
                    id="slider-rango-sueno",
                    min=np.floor(min_s), max=np.ceil(max_s), step=0.5,
                    value=[np.floor(min_s), np.ceil(max_s)],
                    marks={i: f"{i} hrs" for i in range(int(np.floor(min_s)), int(np.ceil(max_s)) + 1)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], md=7),
            dbc.Col([
                html.Label("Tratamiento del Caso Atípico", className="fw-bold mb-2 small"),
                dcc.Checklist(
                    id="chk-outliers-eda",
                    options=[{"label": " Destacar Estudiante ID 9 (Puntaje = 22 pts)", "value": "resaltar"}],
                    value=["resaltar"],
                    className="text-secondary small mt-1",
                ),
            ], md=5),
        ], className="g-3 align-items-center")
    ], className="control-drawer mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-histograma-eda"))), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.Label("Segmentar color por:", className="fw-bold me-2 mb-0 small"),
                dcc.Dropdown(
                    id="dd-color-eda",
                    options=[
                        {"label": "⚡ Nivel de Estrés (1 al 10)", "value": "nivel_estres"},
                        {"label": "📚 Horas de Estudio Semanales", "value": "horas_estudio"},
                    ],
                    value="nivel_estres",
                    clearable=False,
                    style={"width": "230px", "fontSize": "0.85rem"}
                )
            ], className="d-flex align-items-center justify-content-between mb-2"),
            dcc.Graph(id="graf-dispersion-eda")
        ])), md=6),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="resumen-estadistico-eda")), md=12),
    ], className="g-3 mt-1"),
])

# ---------------------------------------------------------------------------
# 5.3 PESTAÑA 3: CONTRASTE DE HIPÓTESIS
# ---------------------------------------------------------------------------

tab_hipotesis = html.Div([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-scale-balanced me-2", style={"color": THEME["violet"]}),
                html.Span("Formulación de Hipótesis", className="panel-header-badge")
            ], className="mb-3"),
            html.Div([
                html.P([html.Span("H₀: ", className="fw-bold text-danger"), "μ_alto ≤ μ_bajo (Dormir más no aumenta significativamente la media de calificaciones)"], className="small mb-2"),
                html.P([html.Span("H₁: ", className="fw-bold text-success"), "μ_alto > μ_bajo (Dormir más incrementa sustancialmente el rendimiento académico)"], className="small mb-0"),
            ], className="p-3 bg-light rounded border mb-3"),
            
            html.Label("Percentil de cohorte de sueño:", className="fw-bold mb-1 small"),
            dcc.Slider(
                id="slider-percentil-hip",
                min=20, max=80, step=5, value=50,
                marks={25: "P25", 50: "Mediana", 75: "P75"},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div([
                html.Label("Nivel de significancia (α):", className="fw-bold mb-1 mt-3 small"),
                dcc.RadioItems(
                    id="radio-alpha-hip",
                    options=[
                        {"label": " α = 0.01 (99% conf.)", "value": 0.01},
                        {"label": " α = 0.05 (95% conf.)", "value": 0.05},
                        {"label": " α = 0.10 (90% conf.)", "value": 0.10},
                    ],
                    value=0.05,
                    className="text-secondary small d-flex justify-content-between",
                ),
            ]),
            html.Div([
                html.Label("Manejo de outlier (ID 9):", className="fw-bold mb-1 mt-3 small"),
                dcc.RadioItems(
                    id="radio-outlier-hip",
                    options=[
                        {"label": " Incluir atípico (N=70)", "value": "con"},
                        {"label": " Excluir atípico (Prueba Robusta N=69)", "value": "sin"},
                    ],
                    value="con",
                    className="text-secondary small",
                ),
            ], className="pt-2 border-top mt-3"),
        ])), md=4),

        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-boxplot-hip"))), md=8),
    ], className="g-3 mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="resultado-hipotesis-box")), md=12),
    ], className="g-3"),
])

# ---------------------------------------------------------------------------
# 5.4 PESTAÑA 4: REGRESIÓN & MODELADO
# ---------------------------------------------------------------------------

tab_regresion = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Algoritmo de Regresión:", className="fw-bold mb-1 small"),
                dcc.Dropdown(
                    id="dd-modelo-reg",
                    options=[
                        {"label": "Regresión Lineal Simple (Solo Horas de Sueño)", "value": "simple"},
                        {"label": "Regresión Lineal Múltiple (Sueño + Estudio + Estrés)", "value": "multiple"},
                        {"label": "Regresión Regularizada Ridge (Penalización L2)", "value": "ridge"},
                    ],
                    value="multiple",
                    clearable=False,
                ),
            ], md=5),
            dbc.Col([
                html.Label("Partición de Prueba (Test Split):", className="fw-bold mb-1 small"),
                dcc.Slider(
                    id="slider-split-reg",
                    min=0.1, max=0.4, step=0.05, value=0.2,
                    marks={0.1: "10%", 0.2: "20%", 0.3: "30%", 0.4: "40%"},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], md=4),
            dbc.Col([
                html.Label("Filtro de Caso Atípico:", className="fw-bold mb-1 small"),
                dcc.RadioItems(
                    id="radio-outlier-reg",
                    options=[
                        {"label": " Con outlier (N=70)", "value": "con"},
                        {"label": " Sin outlier (N=69)", "value": "sin"},
                    ],
                    value="con",
                    className="text-secondary small pt-1",
                ),
            ], md=3),
        ], className="g-3 align-items-center")
    ], className="control-drawer mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-regresion-pred"))), md=7),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-regresion-coefs"))), md=5),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="metricas-regresion-card")), md=12),
    ], className="g-3 mt-1"),
])

# ---------------------------------------------------------------------------
# 5.5 PESTAÑA 5: SIMULADOR INDIVIDUAL & DIAGNÓSTICO
# ---------------------------------------------------------------------------

tab_simulador = html.Div([
    dbc.Row([
        # Calculadora interactiva
        dbc.Col(html.Div([
            html.Div([
                html.I(className="fa-solid fa-sliders me-2", style={"color": THEME["teal"]}),
                html.Span("Simulador de Rendimiento Individual", className="panel-header-badge")
            ], className="mb-3"),
            html.P("Ajuste los hábitos individuales de un estudiante para obtener la estimación en vivo de su calificación y el nivel de riesgo académico:", className="text-secondary small mb-3"),

            html.Label("🌙 Horas diarias de sueño:", className="fw-bold mt-1 small"),
            dcc.Slider(id="sim-sueno", min=3.0, max=10.0, step=0.25, value=7.25,
                       marks={3: "3h", 5: "5h", 7: "7h", 8: "8h", 10: "10h"},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label("📚 Horas de estudio autónomo por semana:", className="fw-bold mt-3 small"),
            dcc.Slider(id="sim-estudio", min=0.0, max=30.0, step=1.0, value=14.0,
                       marks={0: "0h", 10: "10h", 20: "20h", 30: "30h"},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label("⚡ Nivel de estrés académico (1 al 10):", className="fw-bold mt-3 small"),
            dcc.Slider(id="sim-estres", min=1, max=10, step=1, value=5,
                       marks={1: "1 (Bajo)", 5: "5 (Medio)", 10: "10 (Severo)"},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Hr(className="my-3"),
            html.Div(id="resultado-simulador-card")
        ], className="simulator-box h-100"), md=6),

        # Modelo Logístico y Curva Sigmoide
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-chart-line me-2", style={"color": THEME["cyan"]}),
                html.Span("Curva de Probabilidad Logística", className="panel-header-badge")
            ], className="mb-2"),
            html.Small(f"Clasificación binaria de riesgo definida para calificaciones ≤ {mediana_puntaje:.1f} pts (mediana poblacional).", className="text-muted d-block mb-2"),
            dcc.Graph(id="graf-curva-sigmoide", style={"height": "320px"}),
            html.Div(id="metricas-logistica-card", className="mt-2")
        ])), md=6),
    ], className="g-3"),
])

# ---------------------------------------------------------------------------
# 6. LAYOUT PRINCIPAL CON NAVEGACIÓN CAPSULE
# ---------------------------------------------------------------------------

app.layout = html.Div([
    dbc.Container([
        hero_header,
        dbc.Tabs([
            dbc.Tab(tab_contexto, label="Contexto & Resumen", tab_id="tab-contexto", label_class_name="capsule-item"),
            dbc.Tab(tab_exploracion, label="Exploración EDA", tab_id="tab-eda", label_class_name="capsule-item"),
            dbc.Tab(tab_hipotesis, label="Contraste de Hipótesis", tab_id="tab-hipotesis", label_class_name="capsule-item"),
            dbc.Tab(tab_regresion, label="Regresión & Modelado", tab_id="tab-regresion", label_class_name="capsule-item"),
            dbc.Tab(tab_simulador, label="Simulador & Diagnóstico", tab_id="tab-simulador", label_class_name="capsule-item"),
        ], id="tabs-navegacion", active_tab="tab-contexto", className="custom-capsule-tabs"),
        
        html.Footer([
            html.Div([
                html.Span("Actividad de Transferencia — Programación para Ciencia de Datos II · Fundación Universitaria Compensar", className="small text-muted"),
                html.Span("Michael Marín Herrera (2026)", className="small fw-bold text-dark")
            ], className="d-flex justify-content-between align-items-center py-4 border-top mt-5")
        ])
    ], fluid=False, style={"maxWidth": "1240px"})
], style={"backgroundColor": "#F1F5F9", "minHeight": "100vh", "paddingTop": "1.5rem"})

# ---------------------------------------------------------------------------
# 7. CALLBACKS INTERACTIVOS
# ---------------------------------------------------------------------------

# Callback EDA
@app.callback(
    [
        Output("badge-rango-sueno", "children"),
        Output("graf-histograma-eda", "figure"),
        Output("graf-dispersion-eda", "figure"),
        Output("resumen-estadistico-eda", "children"),
    ],
    [
        Input("slider-rango-sueno", "value"),
        Input("chk-outliers-eda", "value"),
        Input("dd-color-eda", "value"),
    ]
)
def actualizar_eda(rango_sueno, chk_outliers, color_var):
    dff = df[(df['horas_sueno'] >= rango_sueno[0]) & (df['horas_sueno'] <= rango_sueno[1])].copy()
    badge_txt = f"{rango_sueno[0]:.1f}h – {rango_sueno[1]:.1f}h ({len(dff)} estudiantes)"

    # Histograma
    fig_hist = px.histogram(
        dff, x="puntaje", nbins=14,
        title="Distribución de Rendimiento Académico",
        color_discrete_sequence=[THEME["teal"]],
        labels={"puntaje": "Puntaje en Examen (0-100 pts)"}
    )
    media_val = dff["puntaje"].mean() if len(dff) > 0 else 0
    fig_hist.add_vline(
        x=media_val, line_dash="dash", line_color=THEME["coral"],
        annotation_text=f"Media: {media_val:.1f} pts", annotation_position="top left"
    )
    chart_styler(fig_hist)

    # Dispersión
    color_label = "Nivel de Estrés (1-10)" if color_var == "nivel_estres" else "Horas de Estudio Semanales"
    colorscale = "Thermal" if color_var == "nivel_estres" else "Teal"

    fig_disp = px.scatter(
        dff, x="horas_sueno", y="puntaje",
        color=color_var,
        color_continuous_scale=colorscale,
        hover_data=["estudiante_id", "horas_sueno", "horas_estudio", "nivel_estres", "puntaje"],
        labels={"horas_sueno": "Horas de Sueño Diarias", "puntaje": "Puntaje Académico", color_var: color_label},
        title=f"Sueño vs. Rendimiento (Color: {color_label})"
    )
    fig_disp.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color="white")))

    # Resaltar atípico
    if "resaltar" in chk_outliers and 8 in dff.index:
        row_atp = dff.loc[8]
        fig_disp.add_trace(go.Scatter(
            x=[row_atp['horas_sueno']], y=[row_atp['puntaje']],
            mode="markers+text",
            marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=THEME["coral"], width=3)),
            text=["⚠️ Atípico (ID 9)"],
            textposition="top right",
            name="Caso Atípico (ID 9)"
        ))

    # Línea de tendencia
    if len(dff) > 2:
        m, b = np.polyfit(dff['horas_sueno'], dff['puntaje'], 1)
        x_line = np.linspace(dff['horas_sueno'].min(), dff['horas_sueno'].max(), 50)
        fig_disp.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines",
            line=dict(color=THEME["violet"], width=2.5, dash="dot"),
            name=f"Tendencia (Pendiente: {m:+.2f})"
        ))
    chart_styler(fig_disp)

    # Tabla descriptiva
    r_corr = np.corrcoef(dff['horas_sueno'], dff['puntaje'])[0, 1] if len(dff) > 1 else 0
    resumen = dbc.Row([
        dbc.Col([
            html.H6("Resumen Estadístico Descriptivo (Muestra Filtrada)", className="fw-bold mb-2 small text-uppercase"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Variable"), html.Th("Media ± Desv."), html.Th("Mín"), html.Th("Mediana"), html.Th("Máx")
                ])),
                html.Tbody([
                    html.Tr([html.Td("Horas de Sueño"), html.Td(f"{dff['horas_sueno'].mean():.2f} ± {dff['horas_sueno'].std():.2f}"), html.Td(f"{dff['horas_sueno'].min():.2f}"), html.Td(f"{dff['horas_sueno'].median():.2f}"), html.Td(f"{dff['horas_sueno'].max():.2f}")]),
                    html.Tr([html.Td("Horas de Estudio"), html.Td(f"{dff['horas_estudio'].mean():.2f} ± {dff['horas_estudio'].std():.2f}"), html.Td(f"{dff['horas_estudio'].min():.2f}"), html.Td(f"{dff['horas_estudio'].median():.2f}"), html.Td(f"{dff['horas_estudio'].max():.2f}")]),
                    html.Tr([html.Td("Nivel de Estrés"), html.Td(f"{dff['nivel_estres'].mean():.2f} ± {dff['nivel_estres'].std():.2f}"), html.Td(f"{dff['nivel_estres'].min():.1f}"), html.Td(f"{dff['nivel_estres'].median():.1f}"), html.Td(f"{dff['nivel_estres'].max():.1f}")]),
                    html.Tr([html.Td("Puntaje Rendimiento"), html.Td(f"{dff['puntaje'].mean():.2f} ± {dff['puntaje'].std():.2f}"), html.Td(f"{dff['puntaje'].min():.1f}"), html.Td(f"{dff['puntaje'].median():.1f}"), html.Td(f"{dff['puntaje'].max():.1f}")]),
                ])
            ], className="table table-sm table-neuro")
        ], md=8),
        dbc.Col([
            html.Div([
                html.Div("Correlación de Pearson:", className="small text-muted text-uppercase fw-bold"),
                html.Div(f"r = {r_corr:+.3f}", className="fw-bold fs-3", style={"color": THEME["teal"]}),
                html.Small("Existe asociación directa y estadísticamente consistente entre el hábito de sueño y la calificación.", className="text-secondary d-block mt-1")
            ], className="p-3 bg-light rounded border h-100 d-flex flex-column justify-content-center")
        ], md=4)
    ], className="g-3")

    return badge_txt, fig_hist, fig_disp, resumen


# Callback Hipótesis
@app.callback(
    [
        Output("graf-boxplot-hip", "figure"),
        Output("resultado-hipotesis-box", "children"),
    ],
    [
        Input("slider-percentil-hip", "value"),
        Input("radio-alpha-hip", "value"),
        Input("radio-outlier-hip", "value"),
    ]
)
def actualizar_hipotesis(percentil, alpha, modo_outlier):
    data_hip = df.copy()
    if modo_outlier == "sin" and len(outlier_idx) > 0:
        data_hip = data_hip.drop(index=outlier_idx)

    corte = float(np.percentile(data_hip['horas_sueno'], percentil))
    data_hip['Grupo'] = np.where(data_hip['horas_sueno'] >= corte, f"Alto Sueño (≥ {corte:.2f}h)", f"Bajo Sueño (< {corte:.2f}h)")

    grupo_alto = data_hip[data_hip['horas_sueno'] >= corte]['puntaje']
    grupo_bajo = data_hip[data_hip['horas_sueno'] < corte]['puntaje']

    t_stat, p_dos_colas = stats.ttest_ind(grupo_alto, grupo_bajo, equal_var=False)
    p_val_welch = p_dos_colas / 2 if t_stat > 0 else 1.0 - (p_dos_colas / 2)
    u_stat, p_val_mwu = stats.mannwhitneyu(grupo_alto, grupo_bajo, alternative='greater')

    # Boxplot
    fig_box = px.box(
        data_hip, x="Grupo", y="puntaje", color="Grupo",
        points="all",
        color_discrete_map={
            f"Alto Sueño (≥ {corte:.2f}h)": THEME["teal"],
            f"Bajo Sueño (< {corte:.2f}h)": THEME["amber"]
        },
        title=f"Distribución de Calificaciones por Cohorte de Sueño (Corte P{percentil}: {corte:.2f} hrs)",
        labels={"puntaje": "Calificación Obtenida (0-100 pts)"}
    )
    chart_styler(fig_box)

    rechaza = (p_val_welch < alpha)
    resultado_ui = dbc.Row([
        dbc.Col([
            html.Div([
                html.Span(f"Prueba t de Welch (Unilateral): t = {t_stat:.4f} | p-valor = {p_val_welch:.5f}", className="fw-bold d-block"),
                html.Span(f"Prueba U de Mann-Whitney (No Paramétrica): U = {u_stat:.1f} | p-valor = {p_val_mwu:.5f}", className="small text-muted d-block mt-1"),
                html.Small(f"Nivel fijado: α = {alpha} | n(Alto)={len(grupo_alto)} (Media={grupo_alto.mean():.2f}), n(Bajo)={len(grupo_bajo)} (Media={grupo_bajo.mean():.2f})", className="text-secondary d-block mt-1")
            ])
        ], md=8),
        dbc.Col([
            html.Div([
                html.Span("DECISIÓN ESTADÍSTICA", className="badge bg-dark mb-1"),
                html.Div(
                    "Rechazar H₀ a favor de H₁" if rechaza else "No se rechaza H₀",
                    className="fw-bold fs-6",
                    style={"color": THEME["emerald"] if rechaza else THEME["coral"]}
                ),
                html.Small(
                    "Evidencia concluyente: El mayor descanso incrementa significativamente el rendimiento." if rechaza else "No existe evidencia estadística suficiente para el nivel α seleccionado.",
                    className="text-muted d-block"
                )
            ], className="text-md-end")
        ], md=4)
    ], className="align-items-center p-2")

    return fig_box, resultado_ui


# Callback Regresión
@app.callback(
    [
        Output("graf-regresion-pred", "figure"),
        Output("graf-regresion-coefs", "figure"),
        Output("metricas-regresion-card", "children"),
    ],
    [
        Input("dd-modelo-reg", "value"),
        Input("slider-split-reg", "value"),
        Input("radio-outlier-reg", "value"),
    ]
)
def actualizar_regresion(tipo_modelo, test_size, modo_outlier):
    data_reg = df.copy()
    if modo_outlier == "sin" and len(outlier_idx) > 0:
        data_reg = data_reg.drop(index=outlier_idx)

    cols = ['horas_sueno'] if tipo_modelo == "simple" else ['horas_sueno', 'horas_estudio', 'nivel_estres']
    X = data_reg[cols].to_numpy()
    y = data_reg[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if tipo_modelo == "ridge":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        coefs = model.coef_
        intercept = model.intercept_
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        coefs = model.coef_
        intercept = model.intercept_

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Dispersión Real vs Predicho
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(size=12, color=THEME["cyan"], opacity=0.85, line=dict(color=THEME["midnight"], width=1.5)),
        name="Estudiantes en Test"
    ))
    min_val, max_val = min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2
    fig_pred.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color=THEME["coral"], dash="dash", width=2),
        name="Ajuste Ideal (y = x)"
    ))
    fig_pred.update_layout(
        title="<b>Valores Reales vs. Predichos (Conjunto de Evaluación)</b>",
        xaxis_title="Calificación Real", yaxis_title="Calificación Predicha"
    )
    chart_styler(fig_pred)

    # Coeficientes
    coef_df = pd.DataFrame({"Variable": cols, "Impacto": coefs})
    fig_coefs = px.bar(
        coef_df, x="Variable", y="Impacto",
        color="Impacto",
        color_continuous_scale="Darkmint",
        title="<b>Magnitud de Coeficientes de Regresión</b>",
        text_auto=".2f"
    )
    chart_styler(fig_coefs)

    card_ui = dbc.Row([
        dbc.Col(neuro_kpi("R² Bondad de Ajuste", f"{r2:.3f}", "Varianza explicada en test", "fa-solid fa-chart-pie", "#ECFEFF", THEME["cyan"]), md=3),
        dbc.Col(neuro_kpi("Error RMSE", f"{rmse:.2f} pts", "Desviación estándar residual", "fa-solid fa-calculator", "#FFFBEB", THEME["amber"]), md=3),
        dbc.Col(neuro_kpi("Error MAE", f"{mae:.2f} pts", "Error absoluto medio", "fa-solid fa-bullseye", "#F0FDF4", THEME["emerald"]), md=3),
        dbc.Col(neuro_kpi("Intercepto b₀", f"{intercept:.2f}", f"Modelo: {tipo_modelo.upper()}", "fa-solid fa-sliders", "#EEF2FF", THEME["violet"]), md=3),
    ], className="g-3")

    return fig_pred, fig_coefs, card_ui


# Callback Simulador
@app.callback(
    [
        Output("resultado-simulador-card", "children"),
        Output("graf-curva-sigmoide", "figure"),
        Output("metricas-logistica-card", "children"),
    ],
    [
        Input("sim-sueno", "value"),
        Input("sim-estudio", "value"),
        Input("sim-estres", "value"),
    ]
)
def actualizar_simulador(sueno, estudio, estres):
    X_input = np.array([[sueno, estudio, estres]])
    puntaje_pred = float(np.clip(modelo_multiple_ref.predict(X_input)[0], 0, 100))
    prob_riesgo = float(modelo_logit_ref.predict_proba(X_input)[0, 1])

    es_riesgo = (prob_riesgo >= 0.50)

    sim_ui = html.Div([
        html.Div([
            html.Div("CALIFICACIÓN ESTIMADA PROYECTADA", className="small fw-bold text-white-50 text-uppercase"),
            html.Div(f"{puntaje_pred:.1f} pts", className="score-huge-value"),
            html.Div(f"Probabilidad de Riesgo Académico: {prob_riesgo:.1%}", className="small fw-semibold text-white")
        ], className="score-display-card mb-3"),

        html.Div([
            html.Div(
                "⚠️ ALERTA: Perfil Estudiantil en Riesgo de Bajo Rendimiento" if es_riesgo else "✅ ÓPTIMO: Perfil Protector de Alto Rendimiento",
                className="fw-bold mb-1"
            ),
            html.Small(
                f"La combinación de sueño insuficiente ({sueno:.1f}h) y estrés ({estres}/10) proyecta una probabilidad del {prob_riesgo:.1%} de quedar por debajo de la mediana ({mediana_puntaje:.1f} pts)."
                if es_riesgo else
                f"El equilibrio entre horas de descanso ({sueno:.1f}h) y dedicación de estudio ({estudio:.1f}h) favorece un desempeño académico superior.",
                className="d-block"
            )
        ], className="risk-alert-critical" if es_riesgo else "risk-alert-optimal")
    ])

    # Curva sigmoide
    rango_sueno_curva = np.linspace(3, 10, 80)
    X_curva = np.column_stack([rango_sueno_curva, np.full(80, estudio), np.full(80, estres)])
    probs_curva = modelo_logit_ref.predict_proba(X_curva)[:, 1]

    fig_sigmoide = go.Figure()
    fig_sigmoide.add_trace(go.Scatter(
        x=rango_sueno_curva, y=probs_curva,
        mode="lines",
        line=dict(color=THEME["violet"], width=3),
        name="P(Riesgo) vs. Sueño"
    ))
    fig_sigmoide.add_trace(go.Scatter(
        x=[sueno], y=[prob_riesgo],
        mode="markers",
        marker=dict(size=14, color=THEME["coral"] if es_riesgo else THEME["emerald"], line=dict(color="white", width=2)),
        name="Estudiante Simulado"
    ))
    fig_sigmoide.add_hline(y=0.5, line_dash="dot", line_color=THEME["muted"], annotation_text="Corte 50%")
    fig_sigmoide.update_layout(
        title="<b>Curva Sigmoide de Riesgo vs. Horas de Sueño</b>",
        xaxis_title="Horas de Sueño Diarias",
        yaxis_title="Probabilidad de Riesgo P(Y=1)",
        yaxis=dict(range=[-0.05, 1.05])
    )
    chart_styler(fig_sigmoide)

    preds_pob = modelo_logit_ref.predict(X_full)
    acc = accuracy_score(df['en_riesgo'], preds_pob)
    prec = precision_score(df['en_riesgo'], preds_pob, zero_division=0)
    rec = recall_score(df['en_riesgo'], preds_pob, zero_division=0)

    metricas_log_ui = html.Div([
        html.Span(f"Exactitud Global (Accuracy): {acc:.1%} | Precisión: {prec:.2f} | Sensibilidad (Recall): {rec:.2f}", className="small fw-semibold text-muted d-block text-center")
    ])

    return sim_ui, fig_sigmoide, metricas_log_ui


# ---------------------------------------------------------------------------
# 8. EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
