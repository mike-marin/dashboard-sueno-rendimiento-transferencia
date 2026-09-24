"""
Dashboard interactivo — Sueño y rendimiento académico
Actividad de Transferencia | Programación para Ciencia de Datos II | UCompensar
Autor: Michael Marín Herrera (mmarinh@ucompensar.edu.co)
Docente: William Eduardo Clavijo Bohorquez
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y TEMA VISUAL
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sueño & Rendimiento Académico | Bienestar Universitario",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados para identidad visual única (Emerald & Slate Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #064E3B 0%, #047857 50%, #059669 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.25);
    }
    .main-header h1 {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 0.95rem;
        color: #D1FAE5 !important;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        background-color: #ECFDF5;
        color: #047857;
        display: inline-block;
        margin-top: 4px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #047857 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# Paleta de visualización para Plotly
COLOR_PRIMARY = "#059669"      # Emerald
COLOR_SECONDARY = "#0EA5E9"    # Cyan / Sky
COLOR_ACCENT = "#F59E0B"       # Amber
COLOR_DANGER = "#EF4444"       # Red
COLOR_PURPLE = "#8B5CF6"       # Violet
COLOR_BG_PLOT = "rgba(0,0,0,0)"

def apply_chart_theme(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1E293B", size=12),
        margin=dict(l=40, r=30, t=55 if title else 25, b=40),
        plot_bgcolor=COLOR_BG_PLOT,
        paper_bgcolor=COLOR_BG_PLOT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zerolinecolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zerolinecolor="#E2E8F0")
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.01, y=0.98, font=dict(size=14, color="#0F172A")))
    return fig

# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS ROBUSTA
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    posibles_rutas = [
        os.path.join(base_dir, "data", "sueno_rendimiento.csv"),
        os.path.join(base_dir, "sueno_rendimiento.csv"),
        "data/sueno_rendimiento.csv",
        "sueno_rendimiento.csv"
    ]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            return pd.read_csv(ruta)
    raise FileNotFoundError("No se encontro el archivo sueno_rendimiento.csv en ninguna de las rutas esperadas.")

df_raw = load_data()

# ---------------------------------------------------------------------------
# 2. ENCABEZADO Y PANEL LATERAL (FILTROS GLOBALES)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🧠 Dashboard de Bienestar: Sueño y Rendimiento Académico</h1>
    <p>Herramienta interactiva de analítica y modelado para el seguimiento del bienestar estudiantil — UCompensar</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Filtros Globales")
    st.caption("Los filtros aplicados aquí modifican los cálculos y gráficos de todas las pestañas.")
    
    min_s = float(df_raw["horas_sueno_promedio"].min())
    max_s = float(df_raw["horas_sueno_promedio"].max())
    
    rango_sueno = st.slider(
        "Rango de horas de sueño",
        min_value=min_s,
        max_value=max_s,
        value=(min_s, max_s),
        step=0.1,
        help="Filtra la muestra por el intervalo de horas de descanso."
    )
    
    incluir_atipico = st.checkbox(
        "Incluir estudiante atípico (ID 9)",
        value=True,
        help="El estudiante ID=9 reporta 7.77h de sueño pero un puntaje bajo (22 pts). Permite evaluar el impacto de outliers."
    )
    
    st.divider()
    st.markdown("##### 👤 Información del Proyecto")
    st.markdown("""
    - **Autor:** Michael Marín Herrera
    - **Docente:** William Eduardo Clavijo
    - **Programa:** Ing. Ciencia de Datos
    - **Asignatura:** Programación II
    """)

# Aplicar filtros
df_f = df_raw[
    (df_raw["horas_sueno_promedio"] >= rango_sueno[0]) &
    (df_raw["horas_sueno_promedio"] <= rango_sueno[1])
].copy()

if not incluir_atipico:
    df_f = df_f[df_f["estudiante_id"] != 9]

# ---------------------------------------------------------------------------
# PESTAÑAS PRINCIPALES
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Resumen Exploratorio",
    "🔬 2. Contraste de Hipótesis",
    "📈 3. Regresión Lineal Múltiple",
    "🎯 4. Regresión Logística y Riesgo"
])

# ---------------------------------------------------------------------------
# TAB 1: RESUMEN EXPLORATORIO
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("#### 📌 Panorama General de la Muestra")
    
    # Tarjetas de Métricas KPI
    k1, k2, k3, k4 = st.columns(4)
    corr_val = df_f["horas_sueno_promedio"].corr(df_f["puntaje_rendimiento"])
    
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Muestra Activa</div>
            <div class="metric-value">{len(df_f)} <span style="font-size:0.9rem; color:#64748B;">/ {len(df_raw)}</span></div>
            <div class="metric-badge">Estudiantes</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Sueño Promedio</div>
            <div class="metric-value">{df_f['horas_sueno_promedio'].mean():.2f} <span style="font-size:0.9rem; color:#64748B;">h</span></div>
            <div class="metric-badge">Mediana: {df_f['horas_sueno_promedio'].median():.2f}h</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Puntaje Promedio</div>
            <div class="metric-value">{df_f['puntaje_rendimiento'].mean():.1f} <span style="font-size:0.9rem; color:#64748B;">pts</span></div>
            <div class="metric-badge">Desv. Est: {df_f['puntaje_rendimiento'].std():.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Correlación (r)</div>
            <div class="metric-value">{corr_val:+.3f}</div>
            <div class="metric-badge">{'Moderada Positiva' if corr_val > 0.3 else 'Débil'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráficos exploratorios
    col_hist, col_scat = st.columns([1, 1.2])
    
    with col_hist:
        fig_hist = px.histogram(
            df_f, x="puntaje_rendimiento", nbins=14,
            title="Distribución del Puntaje de Rendimiento Académico",
            labels={"puntaje_rendimiento": "Puntaje Obtenido (0-100)"},
            color_discrete_sequence=[COLOR_PRIMARY],
            opacity=0.85
        )
        fig_hist.add_vline(
            x=df_f["puntaje_rendimiento"].mean(),
            line_dash="dash", line_color=COLOR_ACCENT,
            annotation_text=f"Media: {df_f['puntaje_rendimiento'].mean():.1f}",
            annotation_position="top right"
        )
        apply_chart_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_scat:
        color_choice = st.selectbox(
            "Segmentar dispersograma por variable complementaria:",
            options=["horas_estudio_semanales", "nivel_estres_1_10"],
            format_func=lambda x: "📚 Horas de Estudio Semanales" if x == "horas_estudio_semanales" else "⚡ Nivel de Estrés (1-10)"
        )
        
        fig_scatter = px.scatter(
            df_f, x="horas_sueno_promedio", y="puntaje_rendimiento",
            color=color_choice,
            color_continuous_scale="Viridis" if color_choice == "horas_estudio_semanales" else "Plasma",
            hover_data=["estudiante_id", "horas_estudio_semanales", "nivel_estres_1_10"],
            labels={
                "horas_sueno_promedio": "Horas de Sueño Promedio",
                "puntaje_rendimiento": "Puntaje de Rendimiento",
                "horas_estudio_semanales": "Estudio (h/sem)",
                "nivel_estres_1_10": "Estrés (1-10)"
            },
            title="Relación: Sueño vs. Puntaje Académico"
        )
        apply_chart_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("📋 Ver Tabla de Estadísticas Descriptivas y Datos"):
        st.dataframe(df_f.describe().T.style.format("{:.2f}"), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: CONTRASTE DE HIPÓTESIS
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("#### 🔬 Prueba de Hipótesis: Impacto del Descanso en el Rendimiento")
    st.info("""
    **Hipótesis de Investigación (Unilateral Superior):**
    - $H_0: \mu_{\\text{alto sueño}} \le \mu_{\\text{bajo sueño}}$ (El mayor descanso no incrementa el rendimiento promedio)
    - $H_1: \mu_{\\text{alto sueño}} > \mu_{\\text{bajo sueño}}$ (El mayor descanso se asocia a un puntaje significativamente superior)
    """)

    c_ctrl, c_res = st.columns([1, 1.4])
    
    with c_ctrl:
        st.markdown("##### 🎛️ Parámetros de la Prueba")
        percentil = st.slider(
            "Percentil de corte para definir grupos:",
            min_value=10, max_value=90, value=50, step=5,
            help="El percentil 50 corresponde a la mediana (corte balanceado)."
        )
        alfa = st.slider(
            "Nivel de significancia (α):",
            min_value=0.01, max_value=0.10, value=0.05, step=0.01
        )
        
        corte_val = float(np.percentile(df_f["horas_sueno_promedio"], percentil))
        st.metric("Punto de corte en horas de sueño", f"{corte_val:.2f} h")
        
        grupo_alto = df_f.loc[df_f["horas_sueno_promedio"] > corte_val, "puntaje_rendimiento"]
        grupo_bajo = df_f.loc[df_f["horas_sueno_promedio"] <= corte_val, "puntaje_rendimiento"]
        
        st.caption(f"Tamaño de grupos: **Alto sueño** (n={len(grupo_alto)}) | **Bajo sueño** (n={len(grupo_bajo)})")

    with c_res:
        if len(grupo_alto) > 2 and len(grupo_bajo) > 2:
            t_stat, p_val_t = stats.ttest_ind(grupo_alto, grupo_bajo, equal_var=False, alternative="greater")
            u_stat, p_val_u = stats.mannwhitneyu(grupo_alto, grupo_bajo, alternative="greater")
            
            rechaza_h0 = (p_val_t < alfa)
            
            st.markdown("##### 📊 Resultados Estadísticos")
            r1, r2, r3 = st.columns(3)
            r1.metric("Estadístico t (Welch)", f"{t_stat:.3f}")
            r2.metric("p-valor (t-test)", f"{p_val_t:.4e}")
            r3.metric("p-valor (Mann-Whitney)", f"{p_val_u:.4e}")
            
            if rechaza_h0:
                st.success(f"✅ **Decisión:** Se rechaza $H_0$ a un nivel de significancia $\\alpha={alfa:.2f}$. Hay evidencia estadística suficiente de que los estudiantes con mayor descanso obtienen mejores calificaciones.")
            else:
                st.warning(f"⚠️ **Decisión:** No se rechaza $H_0$ a $\\alpha={alfa:.2f}$. No hay evidencia concluyente con el corte seleccionado.")
            
            # Boxplot comparativo
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=grupo_bajo, name=f"Bajo Sueño (≤{corte_val:.2f}h)", marker_color=COLOR_DANGER, boxmean=True))
            fig_box.add_trace(go.Box(y=grupo_alto, name=f"Alto Sueño (>{corte_val:.2f}h)", marker_color=COLOR_PRIMARY, boxmean=True))
            fig_box.update_layout(title="Distribución del Rendimiento por Nivel de Sueño", yaxis_title="Puntaje (0-100)")
            apply_chart_theme(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.error("Uno de los grupos contiene muy pocas observaciones para realizar la prueba estadística.")

# ---------------------------------------------------------------------------
# TAB 3: REGRESIÓN LINEAL MÚLTIPLE
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("#### 📈 Modelo Predictivo: Regresión Lineal Múltiple")
    
    col_reg_cfg, col_reg_res = st.columns([1, 1.4])
    
    with col_reg_cfg:
        st.markdown("##### 🎛️ Configuración del Modelo")
        features_sel = st.multiselect(
            "Variables predictoras a incluir:",
            options=["horas_sueno_promedio", "horas_estudio_semanales", "nivel_estres_1_10"],
            default=["horas_sueno_promedio", "horas_estudio_semanales", "nivel_estres_1_10"],
            format_func=lambda x: {
                "horas_sueno_promedio": "Horas de Sueño",
                "horas_estudio_semanales": "Horas de Estudio",
                "nivel_estres_1_10": "Nivel de Estrés"
            }[x]
        )
        test_prop = st.slider("Proporción del conjunto de prueba (Test Size):", 0.1, 0.4, 0.3, step=0.05)

    if len(features_sel) >= 1 and len(df_f) > 10:
        X = df_f[features_sel]
        y = df_f["puntaje_rendimiento"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=42)
        modelo_lin = LinearRegression().fit(X_train, y_train)
        y_pred = modelo_lin.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        with col_reg_res:
            st.markdown("##### 🏆 Desempeño en Test")
            m_r1, m_r2 = st.columns(2)
            m_r1.metric("Coeficiente R² (Test)", f"{r2:.3f}")
            m_r2.metric("Error RMSE", f"{rmse:.2f} pts")
            
            coef_df = pd.DataFrame({
                "Variable": [f.replace("_", " ").title() for f in features_sel],
                "Coeficiente": np.round(modelo_lin.coef_, 3)
            })
            st.dataframe(coef_df.set_index("Variable"), use_container_width=True)
            st.caption(f"**Intercepto ($\beta_0$):** {modelo_lin.intercept_:.3f}")

        # Gráfico Real vs Predicho
        fig_pred = px.scatter(
            x=y_test, y=y_pred,
            labels={"x": "Puntaje Real (Observado)", "y": "Puntaje Predicho por el Modelo"},
            title="Bondad de Ajuste: Puntaje Real vs. Predicho (Test Set)",
            color_discrete_sequence=[COLOR_PRIMARY]
        )
        lims = [min(y_test.min(), y_pred.min()) - 3, max(y_test.max(), y_pred.max()) + 3]
        fig_pred.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Ajuste Ideal (y = x)", line=dict(dash="dash", color=COLOR_DANGER)))
        apply_chart_theme(fig_pred)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Simulador en vivo
        st.markdown("---")
        st.markdown("##### 🎯 Simulador de Rendimiento para un Estudiante Hipotético")
        sim_cols = st.columns(len(features_sel))
        input_data = {}
        for idx, feat in enumerate(features_sel):
            with sim_cols[idx]:
                if feat == "horas_sueno_promedio":
                    input_data[feat] = st.slider("Horas de Sueño", 3.0, 10.0, 6.5, 0.1, key="sim_s")
                elif feat == "horas_estudio_semanales":
                    input_data[feat] = st.slider("Horas de Estudio/sem", 2.0, 30.0, 14.0, 0.5, key="sim_e")
                elif feat == "nivel_estres_1_10":
                    input_data[feat] = st.slider("Nivel de Estrés (1-10)", 1.0, 10.0, 5.0, 0.5, key="sim_str")
        
        sim_df = pd.DataFrame([[input_data[f] for f in features_sel]], columns=features_sel)
        pred_val = float(modelo_lin.predict(sim_df)[0])
        pred_val_clamped = max(0.0, min(100.0, pred_val))
        
        st.success(f"🎓 **Puntaje Académico Estimado:** `{pred_val_clamped:.1f} / 100 puntos`")
    else:
        st.warning("Seleccione al menos una variable independiente para entrenar el modelo.")

# ---------------------------------------------------------------------------
# TAB 4: REGRESIÓN LOGÍSTICA
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("#### 🎯 Clasificación: Detección Temprana de Riesgo Académico")
    
    mediana_ref = float(df_f["puntaje_rendimiento"].median())
    df_f["riesgo_bajo_rendimiento"] = (df_f["puntaje_rendimiento"] <= mediana_ref).astype(int)
    
    col_log_cfg, col_log_res = st.columns([1, 1.4])
    
    with col_log_cfg:
        st.markdown("##### 🎛️ Hiperparámetros de Clasificación")
        umbral_prob = st.slider("Umbral de Decisión ($p_{\\text{corte}}$):", 0.10, 0.90, 0.45, step=0.05)
        c_reg = st.select_slider("Parámetro de Regularización $C$:", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        st.caption("Corte de riesgo: Puntaje ≤ mediana ({:.1f} pts). Predictor principal: Horas de sueño.".format(mediana_ref))

    X_log = df_f[["horas_sueno_promedio"]]
    y_log = df_f["riesgo_bajo_rendimiento"]

    if len(df_f) > 10 and y_log.nunique() == 2:
        X_tr, X_te, y_tr, y_te = train_test_split(X_log, y_log, test_size=0.3, random_state=42, stratify=y_log)
        modelo_log = LogisticRegression(C=c_reg, solver="lbfgs").fit(X_tr, y_tr)
        
        probs_te = modelo_log.predict_proba(X_te)[:, 1]
        y_preds_bin = (probs_te >= umbral_prob).astype(int)
        
        acc = accuracy_score(y_te, y_preds_bin)
        rec = recall_score(y_te, y_preds_bin, zero_division=0)
        prec = precision_score(y_te, y_preds_bin, zero_division=0)
        cm = confusion_matrix(y_te, y_preds_bin)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        espec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        with col_log_res:
            st.markdown("##### 📊 Métricas de Evaluación de Riesgo")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Exactitud (Accuracy)", f"{acc:.2%}")
            m2.metric("Sensibilidad (Recall)", f"{rec:.2%}")
            m3.metric("Especificidad", f"{espec:.2%}")
            m4.metric("Precisión", f"{prec:.2%}")

        col_sig, col_mat = st.columns(2)
        
        with col_sig:
            xs = np.linspace(df_f["horas_sueno_promedio"].min() - 0.5, df_f["horas_sueno_promedio"].max() + 0.5, 250)
            probs_curva = modelo_log.predict_proba(pd.DataFrame(xs, columns=["horas_sueno_promedio"]))[:, 1]
            
            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=xs, y=probs_curva, mode="lines", name="Probabilidad de Riesgo", line=dict(color=COLOR_DANGER, width=3)))
            fig_sig.add_hline(y=umbral_prob, line_dash="dash", line_color="#475569", annotation_text=f"Umbral = {umbral_prob:.2f}")
            fig_sig.update_layout(title="Curva Sigmoide: Riesgo vs. Horas de Sueño", xaxis_title="Horas de Sueño", yaxis_title="P(Riesgo)")
            apply_chart_theme(fig_sig)
            st.plotly_chart(fig_sig, use_container_width=True)

        with col_mat:
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Teal",
                x=["Pred: Sin Riesgo", "Pred: En Riesgo"],
                y=["Real: Sin Riesgo", "Real: En Riesgo"],
                title=f"Matriz de Confusión (Umbral = {umbral_prob:.2f})"
            )
            apply_chart_theme(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🩺 Calculadora de Riesgo Estudiantil Individual")
        s_ind = st.slider("Horas de sueño promedio del estudiante evaluado:", 3.0, 10.0, 6.2, 0.1, key="slider_ind_log")
        prob_riesgo_ind = float(modelo_log.predict_proba(pd.DataFrame([[s_ind]], columns=["horas_sueno_promedio"]))[0, 1])
        
        es_riesgo = prob_riesgo_ind >= umbral_prob
        
        if es_riesgo:
            st.error(f"⚠️ **Diagnóstico:** El estudiante presenta una **Probabilidad de Riesgo de {prob_riesgo_ind:.1%}** (Mayor al umbral {umbral_prob:.0%}) → **Alerta: Remitir a Consejería / Bienestar.**")
        else:
            st.success(f"✅ **Diagnóstico:** El estudiante presenta una **Probabilidad de Riesgo de {prob_riesgo_ind:.1%}** (Menor al umbral {umbral_prob:.0%}) → **Estado Favorable / Sin Riesgo Inminente.**")
    else:
        st.warning("Ajuste los filtros: se requieren observaciones en ambas categorías para entrenar la regresión logística.")

# ---------------------------------------------------------------------------
# PIE DE PÁGINA
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Fundación Universitaria Compensar | Ingeniería en Ciencia de Datos | Programación para Ciencia de Datos II — 2026")
