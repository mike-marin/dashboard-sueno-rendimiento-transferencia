# -*- coding: utf-8 -*-
"""
Dashboard - Actividad de Transferencia: Sueño, Estrés y Rendimiento Académico
Programación para Ciencia de Datos II - Fundación Universitaria Compensar

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
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

# Estandarización de nombres si difieren
column_map = {
    'horas_sueno_promedio': 'horas_sueno',
    'horas_estudio_semanales': 'horas_estudio',
    'nivel_estres_1_10': 'nivel_estres',
    'puntaje_rendimiento': 'puntaje'
}
for col, new_col in column_map.items():
    if col in df.columns:
        df = df.rename(columns={col: new_col})

FEATURES = ['horas_sueno', 'horas_estudio', 'nivel_estres']
TARGET = 'puntaje'

mediana_puntaje = float(df[TARGET].median())
mediana_sueno = float(df['horas_sueno'].median())
df['en_riesgo'] = (df[TARGET] <= mediana_puntaje).astype(int)

# Identificar outlier clínico (ID 9: puntaje 22 pts, |z| > 2.5)
z_scores = np.abs(stats.zscore(df[TARGET]))
outlier_idx = df.index[z_scores > 2.5].tolist()

# Paleta de colores: Emerald Clinical & Slate Minimalist
PALETA = {
    "primario": "#059669",     # Verde Esmeralda Institucional
    "primario_light": "#ECFDF5", # Esmeralda suave
    "secundario": "#0EA5E9",   # Azul Turquesa / Sky
    "acento": "#6366F1",       # Índigo
    "exito": "#10B981",        # Esmeralda éxito
    "peligro": "#EF4444",      # Rojo riesgo
    "warning": "#F59E0B",      # Ámbar
    "fondo": "#F8FAFC",        # Slate 50
    "texto": "#0F172A",        # Slate 900
    "muted": "#64748B",        # Slate 500
    "border": "#E2E8F0"
}

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
# 3. FUNCIONES AUXILIARES DE DISEÑO
# ---------------------------------------------------------------------------

def kpi_card(titulo, valor, subtitulo, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(titulo, className="kpi-titulo"),
            html.Div(valor, className="kpi-valor", style={"color": color}),
            html.Div(subtitulo, className="kpi-subtitulo"),
        ]),
        className="kpi-card shadow-sm",
        style={"--kpi-color": color}
    )

def figura_base(fig, titulo=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Plus Jakarta Sans, -apple-system, sans-serif", size=12, color=PALETA["texto"]),
        margin=dict(l=50, r=30, t=65 if titulo else 25, b=45),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01, font=dict(size=11, color=PALETA["muted"])),
        hoverlabel=dict(bgcolor=PALETA["texto"], font_size=12, font_family="Plus Jakarta Sans", font_color="white"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zerolinecolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zerolinecolor="#E2E8F0")
    if titulo:
        fig.update_layout(title=dict(text=f"<b>{titulo}</b>", x=0.01, y=0.98, font=dict(size=14, color=PALETA["texto"])))
    return fig

# ---------------------------------------------------------------------------
# 4. INICIALIZACIÓN DE LA APP (COMPATIBLE CON BINDER & LOCAL)
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
        "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Dashboard | Sueño & Rendimiento Académico",
)
server = app.server

# ---------------------------------------------------------------------------
# 5. LAYOUT — ENCABEZADO
# ---------------------------------------------------------------------------

encabezado = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.Div([
                html.I(className="fa-solid fa-brain me-2 fs-4", style={"color": PALETA["primario"]}),
                html.H4("Sueño, Estrés y Rendimiento Académico", className="text-white d-inline-block fw-bold align-middle mb-0"),
                html.Span("TRANSFERENCIA UCOMPENSAR", className="brand-badge ms-3 align-middle")
            ], className="d-flex align-items-center mb-1"),
            html.Small("Programación para Ciencia de Datos II · Autor: Michael Marín Herrera · Docente: William Eduardo Clavijo Bohorquez",
                       className="text-white-50 font-monospace")
        ])
    ], fluid=True),
    className="app-header mb-4",
    dark=True,
)

# ---------------------------------------------------------------------------
# 5.1 PESTAÑA — CONTEXTO & RESUMEN
# ---------------------------------------------------------------------------

tab_contexto = dbc.Container([
    dbc.Row([
        dbc.Col(kpi_card("Muestra Estudiantil", f"{len(df)} estudiantes", "Dataset universitario", PALETA["primario"]), md=3),
        dbc.Col(kpi_card("Sueño Promedio", f"{df['horas_sueno'].mean():.2f} hrs", f"Mediana: {mediana_sueno:.2f} hrs", PALETA["secundario"]), md=3),
        dbc.Col(kpi_card("Correlación Sueño–Nota", f"r = {r_pearson:+.2f}", "Asociación directa moderada", PALETA["exito"]), md=3),
        dbc.Col(kpi_card("Caso Atípico Detectado", f"{len(outlier_idx)} caso", "ID 9 (Puntaje 22 pts, |z|>2.5)", PALETA["peligro"]), md=3),
    ], className="g-3 mb-4"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-bullseye me-2", style={"color": PALETA["primario"]}),
                html.Span("Planteamiento del Estudio", className="card-title-modern")
            ], className="d-flex align-items-center mb-3"),
            html.P("El departamento de Bienestar Universitario analiza cómo los hábitos del sueño, el tiempo de estudio independiente y la carga de estrés psicológico impactan el desempeño académico de los estudiantes de educación superior.", className="text-secondary"),
            html.P("A través de este dashboard interactivo, se evalúan tres metodologías rigurosas de ciencia de datos: Análisis Exploratorio con detección de atípicos, Contraste de Hipótesis formal (Welch t-test y Mann-Whitney U), y Modelado Predictivo (Regresión Múltiple/Ridge y Clasificación Logística de Riesgo).", className="text-secondary"),
            html.Hr(className="my-3 text-muted"),
            html.H6("Variables del Estudio", className="fw-bold mb-2 text-dark"),
            html.Ul([
                html.Li([html.B("horas_sueno: ", className="text-dark"), "Horas promedio de sueño diario por estudiante (hrs/día)."]),
                html.Li([html.B("horas_estudio: ", className="text-dark"), "Horas dedicadas al estudio independiente por semana."]),
                html.Li([html.B("nivel_estres: ", className="text-dark"), "Escala autopercibida de estrés académico (1 a 10)."]),
                html.Li([html.B("puntaje: ", className="text-dark"), "Calificación obtenida en el examen estandarizado (0 a 100 pts)."]),
            ], className="text-secondary mb-0 ps-3"),
        ])), md=7),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-compass me-2", style={"color": PALETA["primario"]}),
                html.Span("Estructura de la Plataforma", className="card-title-modern")
            ], className="d-flex align-items-center mb-3"),
            html.Ol([
                html.Li([html.B("Exploración: "), "Distribución de notas, correlaciones dinámicas y filtro de casos atípicos."]),
                html.Li([html.B("Contraste de hipótesis: "), "Comparación estadística formal entre estudiantes de alto vs. bajo sueño."]),
                html.Li([html.B("Regresión Múltiple: "), "Modelos lineales, regularización Ridge y evaluación de $R^2$ y RMSE."]),
                html.Li([html.B("Diagnóstico & Simulador: "), "Calculadora en tiempo real y modelo logístico de alerta temprana."]),
            ], className="text-secondary ps-3 mb-3"),
            html.Div([
                html.I(className="fa-solid fa-lightbulb me-2 text-warning"),
                html.Small("Ajuste los controles interactivos en cada pestaña para recalcular modelos y contrastes en vivo.", className="text-muted")
            ], className="p-2 rounded bg-light border d-flex align-items-center"),
        ])), md=5),
    ], className="g-3"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.2 PESTAÑA — EXPLORACIÓN DE DATOS (EDA)
# ---------------------------------------------------------------------------

min_s, max_s = float(df['horas_sueno'].min()), float(df['horas_sueno'].max())

tab_exploracion = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Filtro por Rango de Horas de Sueño:", className="fw-semibold mb-0 me-2"),
                html.Span(id="badge-rango-sueno", className="badge px-3 py-1 rounded-pill shadow-sm", style={"backgroundColor": PALETA["primario"], "color": "#FFFFFF", "fontSize": "0.85rem", "fontWeight": "600"})
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
            html.Label("Tratamiento de Valores Atípicos", className="fw-semibold mb-2"),
            dcc.Checklist(
                id="chk-outliers-eda",
                options=[{"label": " Resaltar estudiante atípico (ID 9, Puntaje=22 pts)", "value": "resaltar"}],
                value=["resaltar"],
                className="mt-1 text-secondary",
            ),
        ], md=5),
    ], className="mb-4 g-3 p-3 bg-white rounded-3 border"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-histograma-eda"))), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.Label("Colorear dispersión por:", className="fw-semibold me-2 mb-0 small"),
                dcc.Dropdown(
                    id="dd-color-eda",
                    options=[
                        {"label": "Nivel de Estrés (1 a 10)", "value": "nivel_estres"},
                        {"label": "Horas de Estudio Semanales", "value": "horas_estudio"},
                    ],
                    value="nivel_estres",
                    clearable=False,
                    style={"width": "220px", "fontSize": "0.85rem"}
                )
            ], className="d-flex align-items-center justify-content-between mb-2"),
            dcc.Graph(id="graf-dispersion-eda")
        ])), md=6),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="resumen-estadistico-eda")), md=12),
    ], className="g-3 mt-1"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.3 PESTAÑA — CONTRASTE DE HIPÓTESIS
# ---------------------------------------------------------------------------

tab_hipotesis = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Formulación de Hipótesis", className="card-title-modern mb-3"),
            html.Div([
                html.P([html.Span("H₀: ", className="fw-bold text-danger"), "μ_alto ≤ μ_bajo (Dormir más no incrementa el puntaje promedio)"]),
                html.P([html.Span("H₁: ", className="fw-bold text-success"), "μ_alto > μ_bajo (Dormir más incrementa sustancialmente el puntaje)"]),
            ], className="p-3 bg-light rounded border mb-3"),
            html.Label("Corte de grupo por percentil de sueño:", className="fw-semibold mb-1 small"),
            dcc.Slider(
                id="slider-percentil-hip",
                min=20, max=80, step=5, value=50,
                marks={25: "P25", 50: "Mediana (P50)", 75: "P75"},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div([
                html.Label("Nivel de significancia (α):", className="fw-semibold mb-1 mt-3 small"),
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
                html.Label("Filtro de caso atípico (ID 9):", className="fw-semibold mb-1 mt-3 small"),
                dcc.RadioItems(
                    id="radio-outlier-hip",
                    options=[
                        {"label": " Incluir outlier (Dataset completo)", "value": "con"},
                        {"label": " Excluir outlier (Prueba de robustez)", "value": "sin"},
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
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.4 PESTAÑA — REGRESIÓN & MODELADO
# ---------------------------------------------------------------------------

tab_regresion = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Selección de Modelo de Regresión", className="fw-semibold mb-2"),
            dcc.Dropdown(
                id="dd-modelo-reg",
                options=[
                    {"label": "Regresión Lineal Simple (Solo Horas de Sueño)", "value": "simple"},
                    {"label": "Regresión Lineal Múltiple (Sueño + Estudio + Estrés)", "value": "multiple"},
                    {"label": "Regresión Regularizada Ridge (L2 Penalization)", "value": "ridge"},
                ],
                value="multiple",
                clearable=False,
            ),
        ], md=5),
        dbc.Col([
            html.Label("Tamaño del Conjunto de Prueba (Test Split):", className="fw-semibold mb-2"),
            dcc.Slider(
                id="slider-split-reg",
                min=0.1, max=0.4, step=0.05, value=0.2,
                marks={0.1: "10%", 0.2: "20%", 0.3: "30%", 0.4: "40%"},
                tooltip={"placement": "bottom", "always_visible": False}
            ),
        ], md=4),
        dbc.Col([
            html.Label("Tratamiento de Outlier:", className="fw-semibold mb-2"),
            dcc.RadioItems(
                id="radio-outlier-reg",
                options=[
                    {"label": " Con outlier (N=70)", "value": "con"},
                    {"label": " Sin outlier (N=69)", "value": "sin"},
                ],
                value="con",
                className="text-secondary pt-1",
            ),
        ], md=3),
    ], className="mb-4 g-3 p-3 bg-white rounded-3 border"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-regresion-pred"))), md=7),
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graf-regresion-coefs"))), md=5),
    ], className="g-3"),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(id="metricas-regresion-card")), md=12),
    ], className="g-3 mt-1"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 5.5 PESTAÑA — DIAGNÓSTICO & SIMULADOR INDIVIDUAL
# ---------------------------------------------------------------------------

tab_simulador = dbc.Container([
    dbc.Row([
        # Simulador interactivo individual
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-sliders me-2", style={"color": PALETA["primario"]}),
                html.Span("Simulador de Rendimiento y Riesgo Individual", className="card-title-modern")
            ], className="d-flex align-items-center mb-3"),
            html.P("Ingrese el perfil de hábitos de un estudiante para predecir su puntaje esperado y clasificar su riesgo de bajo desempeño:", className="text-secondary small"),

            html.Label("Horas de sueño diarias:", className="fw-semibold mt-2 small"),
            dcc.Slider(id="sim-sueno", min=3.0, max=10.0, step=0.25, value=7.0,
                       marks={3: "3h", 5: "5h", 7: "7h", 8: "8h", 10: "10h"},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label("Horas de estudio semanales:", className="fw-semibold mt-3 small"),
            dcc.Slider(id="sim-estudio", min=0.0, max=30.0, step=1.0, value=12.0,
                       marks={0: "0h", 10: "10h", 20: "20h", 30: "30h"},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label("Nivel de estrés autopercibido (1 al 10):", className="fw-semibold mt-3 small"),
            dcc.Slider(id="sim-estres", min=1, max=10, step=1, value=5,
                       marks={1: "1 (Bajo)", 5: "5 (Medio)", 10: "10 (Alto)"},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Hr(className="my-3"),
            html.Div(id="resultado-simulador-card")
        ])), md=6),

        # Modelo Logístico y Curva Sigmoide
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="fa-solid fa-shield-halved me-2", style={"color": PALETA["secundario"]}),
                html.Span("Modelo Logístico de Riesgo Académico", className="card-title-modern")
            ], className="d-flex align-items-center mb-3"),
            html.P(f"El riesgo se define para puntajes por debajo de la mediana poblacional ({mediana_puntaje:.1f} pts).", className="text-secondary small mb-2"),
            dcc.Graph(id="graf-curva-sigmoide", style={"height": "340px"}),
            html.Div(id="metricas-logistica-card", className="mt-2")
        ])), md=6),
    ], className="g-3"),
], fluid=True, className="py-2")

# ---------------------------------------------------------------------------
# 6. LAYOUT PRINCIPAL CON PESTAÑAS
# ---------------------------------------------------------------------------

app.layout = html.Div([
    encabezado,
    dbc.Container([
        dbc.Tabs([
            dbc.Tab(tab_contexto, label="Contexto & Resumen", tab_id="tab-contexto", label_class_name="fw-semibold"),
            dbc.Tab(tab_exploracion, label="Exploración de Datos (EDA)", tab_id="tab-eda", label_class_name="fw-semibold"),
            dbc.Tab(tab_hipotesis, label="Contraste de Hipótesis", tab_id="tab-hipotesis", label_class_name="fw-semibold"),
            dbc.Tab(tab_regresion, label="Regresión & Modelado", tab_id="tab-regresion", label_class_name="fw-semibold"),
            dbc.Tab(tab_simulador, label="Simulador & Diagnóstico", tab_id="tab-simulador", label_class_name="fw-semibold"),
        ], id="tabs-navegacion", active_tab="tab-contexto", className="mb-3 custom-tabs"),
    ], fluid=True, className="px-4"),
    html.Footer(
        dbc.Container([
            html.Div([
                html.Span("Actividad de Transferencia — Programación para Ciencia de Datos II · Fundación Universitaria Compensar (2026)", className="small text-muted"),
                html.Span("Autor: Michael Marín Herrera", className="small text-muted fw-bold")
            ], className="d-flex justify-content-between align-items-center py-3 border-top mt-4")
        ], fluid=True, className="px-4")
    )
], style={"backgroundColor": PALETA["fondo"], "minHeight": "100vh"})

# ---------------------------------------------------------------------------
# 7. CALLBACKS INTERACTIVOS
# ---------------------------------------------------------------------------

# Callback EDA: Slider Badge & Gráficos
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
        title="Distribución del Puntaje de Rendimiento Académico",
        color_discrete_sequence=[PALETA["primario"]],
        labels={"puntaje": "Puntaje en Examen (0-100 pts)"}
    )
    media_val = dff["puntaje"].mean() if len(dff) > 0 else 0
    fig_hist.add_vline(
        x=media_val, line_dash="dash", line_color=PALETA["peligro"],
        annotation_text=f"Media: {media_val:.1f} pts", annotation_position="top left"
    )
    figura_base(fig_hist)

    # Dispersión
    color_label = "Nivel de Estrés (1-10)" if color_var == "nivel_estres" else "Horas de Estudio Semanales"
    colorscale = "Reds" if color_var == "nivel_estres" else "Teal"

    fig_disp = px.scatter(
        dff, x="horas_sueno", y="puntaje",
        color=color_var,
        color_continuous_scale=colorscale,
        hover_data=["estudiante_id", "horas_sueno", "horas_estudio", "nivel_estres", "puntaje"],
        labels={"horas_sueno": "Horas de Sueño Diarias", "puntaje": "Puntaje Académico", color_var: color_label},
        title=f"Sueño vs. Rendimiento (Color: {color_label})"
    )
    fig_disp.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="white")))

    # Resaltar atípico si aplica
    if "resaltar" in chk_outliers and 8 in dff.index:
        row_atp = dff.loc[8] # ID 9
        fig_disp.add_trace(go.Scatter(
            x=[row_atp['horas_sueno']], y=[row_atp['puntaje']],
            mode="markers+text",
            marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=PALETA["peligro"], width=3)),
            text=["⚠️ Estudiante Atípico (ID 9)"],
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
            line=dict(color=PALETA["acento"], width=2.5, dash="dot"),
            name=f"Tendencia (Pendiente: {m:+.2f})"
        ))
    figura_base(fig_disp)

    # Resumen descriptivo
    r_corr = np.corrcoef(dff['horas_sueno'], dff['puntaje'])[0, 1] if len(dff) > 1 else 0
    resumen = dbc.Row([
        dbc.Col([
            html.H6("Resumen Descriptivo del Segmento Seleccionado", className="fw-bold mb-2"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Variable"), html.Th("Media ± Desv."), html.Th("Mínimo"), html.Th("Mediana"), html.Th("Máximo")
                ])),
                html.Tbody([
                    html.Tr([html.Td("Horas de Sueño"), html.Td(f"{dff['horas_sueno'].mean():.2f} ± {dff['horas_sueno'].std():.2f}"), html.Td(f"{dff['horas_sueno'].min():.2f}"), html.Td(f"{dff['horas_sueno'].median():.2f}"), html.Td(f"{dff['horas_sueno'].max():.2f}")]),
                    html.Tr([html.Td("Horas de Estudio"), html.Td(f"{dff['horas_estudio'].mean():.2f} ± {dff['horas_estudio'].std():.2f}"), html.Td(f"{dff['horas_estudio'].min():.2f}"), html.Td(f"{dff['horas_estudio'].median():.2f}"), html.Td(f"{dff['horas_estudio'].max():.2f}")]),
                    html.Tr([html.Td("Nivel de Estrés"), html.Td(f"{dff['nivel_estres'].mean():.2f} ± {dff['nivel_estres'].std():.2f}"), html.Td(f"{dff['nivel_estres'].min():.1f}"), html.Td(f"{dff['nivel_estres'].median():.1f}"), html.Td(f"{dff['nivel_estres'].max():.1f}")]),
                    html.Tr([html.Td("Puntaje Rendimiento"), html.Td(f"{dff['puntaje'].mean():.2f} ± {dff['puntaje'].std():.2f}"), html.Td(f"{dff['puntaje'].min():.1f}"), html.Td(f"{dff['puntaje'].median():.1f}"), html.Td(f"{dff['puntaje'].max():.1f}")]),
                ])
            ], className="table table-sm table-modern")
        ], md=8),
        dbc.Col([
            html.Div([
                html.Div("Correlación de Pearson en muestra:", className="small text-muted"),
                html.Div(f"r = {r_corr:+.3f}", className="fw-bold fs-4", style={"color": PALETA["primario"]}),
                html.Small("Asociación positiva: mayor sueño se relaciona con mejor calificación.", className="text-secondary d-block mt-1")
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
    data_hip['Grupo'] = np.where(data_hip['horas_sueno'] >= corte, f"Alto Sueño (≥ {corte:.2f} hrs)", f"Bajo Sueño (< {corte:.2f} hrs)")

    grupo_alto = data_hip[data_hip['horas_sueno'] >= corte]['puntaje']
    grupo_bajo = data_hip[data_hip['horas_sueno'] < corte]['puntaje']

    # Pruebas estadísticas
    t_stat, p_dos_colas = stats.ttest_ind(grupo_alto, grupo_bajo, equal_var=False)
    p_val_welch = p_dos_colas / 2 if t_stat > 0 else 1.0 - (p_dos_colas / 2)

    u_stat, p_val_mwu = stats.mannwhitneyu(grupo_alto, grupo_bajo, alternative='greater')

    # Boxplot
    fig_box = px.box(
        data_hip, x="Grupo", y="puntaje", color="Grupo",
        points="all",
        color_discrete_map={
            f"Alto Sueño (≥ {corte:.2f} hrs)": PALETA["primario"],
            f"Bajo Sueño (< {corte:.2f} hrs)": PALETA["warning"]
        },
        title=f"Comparación de Puntajes por Cohorte de Sueño (Corte P{percentil}: {corte:.2f} hrs)",
        labels={"puntaje": "Puntaje Obtenido"}
    )
    figura_base(fig_box)

    # Conclusión
    rechaza = (p_val_welch < alpha)
    resultado_ui = dbc.Row([
        dbc.Col([
            html.Div([
                html.Span(f"Prueba t de Welch (Unilateral): t = {t_stat:.4f} | p-valor = {p_val_welch:.5f}", className="fw-bold d-block"),
                html.Span(f"Prueba U de Mann-Whitney (No paramétrica): U = {u_stat:.1f} | p-valor = {p_val_mwu:.5f}", className="small text-muted d-block mt-1"),
                html.Small(f"Nivel de significancia fijado: α = {alpha} | Muestra Alto: n={len(grupo_alto)} (Media={grupo_alto.mean():.2f}), Bajo: n={len(grupo_bajo)} (Media={grupo_bajo.mean():.2f})", className="text-secondary d-block mt-1")
            ])
        ], md=8),
        dbc.Col([
            html.Div([
                html.Span("DECISIÓN ESTADÍSTICA", className="badge bg-dark mb-1"),
                html.Div(
                    "Rechazar H₀ a favor de H₁" if rechaza else "No se rechaza H₀",
                    className="fw-bold fs-6",
                    style={"color": PALETA["exito"] if rechaza else PALETA["peligro"]}
                ),
                html.Small(
                    "Evidencia significativa: Dormir adecuadamente incrementa el rendimiento." if rechaza else "No hay evidencia suficiente al nivel α seleccionado.",
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

    if tipo_modelo == "simple":
        cols = ['horas_sueno']
    else:
        cols = ['horas_sueno', 'horas_estudio', 'nivel_estres']

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

    # Gráfica Real vs Predicho
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(size=11, color=PALETA["primario"], opacity=0.8, line=dict(color="white", width=1.5)),
        name="Estudiantes (Test)"
    ))
    min_val, max_val = min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2
    fig_pred.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color=PALETA["peligro"], dash="dash", width=2),
        name="Predicción Perfecta (y = x)"
    ))
    fig_pred.update_layout(
        title="<b>Valores Reales vs. Predichos en Conjunto de Prueba</b>",
        xaxis_title="Puntaje Real", yaxis_title="Puntaje Predicho"
    )
    figura_base(fig_pred)

    # Gráfica Coeficientes
    coef_df = pd.DataFrame({"Variable": cols, "Coeficiente": coefs})
    fig_coefs = px.bar(
        coef_df, x="Variable", y="Coeficiente",
        color="Coeficiente",
        color_continuous_scale="Tealgrn",
        title="<b>Magnitud de Coeficientes de Impacto</b>",
        text_auto=".2f"
    )
    figura_base(fig_coefs)

    # Métricas UI
    card_ui = dbc.Row([
        dbc.Col(kpi_card("R² de Ajuste (Test)", f"{r2:.3f}", "Proporción de varianza explicada", PALETA["primario"]), md=3),
        dbc.Col(kpi_card("Error RMSE", f"{rmse:.2f} pts", "Desviación estándar de residuos", PALETA["warning"]), md=3),
        dbc.Col(kpi_card("Error MAE", f"{mae:.2f} pts", "Error medio absoluto", PALETA["secundario"]), md=3),
        dbc.Col(kpi_card("Intercepto (b₀)", f"{intercept:.2f}", f"{len(cols)} variables en modelo", PALETA["acento"]), md=3),
    ], className="g-3")

    return fig_pred, fig_coefs, card_ui


# Callback Simulador & Clasificación de Riesgo
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
    # Predicción lineal múltiple
    X_input = np.array([[sueno, estudio, estres]])
    puntaje_pred = float(modelo_multiple_ref.predict(X_input)[0])
    puntaje_pred = float(np.clip(puntaje_pred, 0, 100))

    # Predicción logística
    prob_riesgo = float(modelo_logit_ref.predict_proba(X_input)[0, 1])

    # UI del simulador
    es_riesgo = (prob_riesgo >= 0.50)
    color_diagnostico = PALETA["peligro"] if es_riesgo else PALETA["exito"]

    sim_ui = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div("Puntaje Estimado", className="small text-muted text-uppercase fw-bold"),
                html.Div(f"{puntaje_pred:.1f} / 100 pts", className="fs-3 fw-bold", style={"color": PALETA["primario"]}),
            ], md=6),
            dbc.Col([
                html.Div("Probabilidad de Riesgo", className="small text-muted text-uppercase fw-bold"),
                html.Div(f"{prob_riesgo:.1%}", className="fs-3 fw-bold", style={"color": color_diagnostico}),
            ], md=6),
        ], className="text-center p-3 bg-white rounded border mb-3"),
        html.Div([
            html.Div(f"Diagnóstico: {'ALERTA - Alto Riesgo Académico' if es_riesgo else 'ÓPTIMO - Desempeño Favorable'}", className="fw-bold mb-1"),
            html.Small(
                f"El estudiante tiene una probabilidad del {prob_riesgo:.1%} de ubicarse por debajo de la mediana ({mediana_puntaje:.1f} pts). Se sugiere reforzar hábitos de sueño y control de estrés."
                if es_riesgo else
                f"El perfil combina sueño y estudio adecuados con estrés manejable. Puntaje proyectado satisfactorio.",
                className="d-block"
            )
        ], className="risk-card-high" if es_riesgo else "risk-card-low")
    ])

    # Curva sigmoide (variando horas de sueño con estudio y estrés fijos)
    rango_sueno_curva = np.linspace(3, 10, 80)
    X_curva = np.column_stack([rango_sueno_curva, np.full(80, estudio), np.full(80, estres)])
    probs_curva = modelo_logit_ref.predict_proba(X_curva)[:, 1]

    fig_sigmoide = go.Figure()
    fig_sigmoide.add_trace(go.Scatter(
        x=rango_sueno_curva, y=probs_curva,
        mode="lines",
        line=dict(color=PALETA["primario"], width=3),
        name="Curva de Probabilidad de Riesgo"
    ))
    fig_sigmoide.add_trace(go.Scatter(
        x=[sueno], y=[prob_riesgo],
        mode="markers",
        marker=dict(size=14, color=color_diagnostico, line=dict(color="white", width=2)),
        name="Estudiante Simulado"
    ))
    fig_sigmoide.add_hline(y=0.5, line_dash="dot", line_color=PALETA["muted"], annotation_text="Umbral de Riesgo (50%)")
    fig_sigmoide.update_layout(
        title="<b>Curva Sigmoide: Probabilidad de Riesgo vs. Horas de Sueño</b>",
        xaxis_title="Horas de Sueño Diarias",
        yaxis_title="Probabilidad P(Riesgo)",
        yaxis=dict(range=[-0.05, 1.05])
    )
    figura_base(fig_sigmoide)

    # Métricas del modelo de clasificación
    preds_pob = modelo_logit_ref.predict(X_full)
    acc = accuracy_score(df['en_riesgo'], preds_pob)
    prec = precision_score(df['en_riesgo'], preds_pob, zero_division=0)
    rec = recall_score(df['en_riesgo'], preds_pob, zero_division=0)

    metricas_log_ui = html.Div([
        html.Span(f"Accuracy Global: {acc:.1%} | Precision: {prec:.2f} | Recall: {rec:.2f}", className="small fw-semibold text-muted d-block text-center")
    ])

    return sim_ui, fig_sigmoide, metricas_log_ui


# ---------------------------------------------------------------------------
# 8. EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
