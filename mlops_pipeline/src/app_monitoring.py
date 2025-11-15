"""
Dashboard de Monitoreo de Data Drift
Universidad Católica Luis Amigó

Aplicación Streamlit para visualizar:
- Métricas de data drift
- Distribuciones de variables
- Alertas y recomendaciones
- Predicciones del modelo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Monitoreo Data Drift - MLOps",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colores Universidad Católica Luis Amigó
COLORS = {
    'primary': '#005F9E',      # Azul institucional
    'secondary': '#FF8C00',    # Naranja
    'success': '#28A745',      # Verde
    'warning': '#FFC107',      # Amarillo
    'danger': '#DC3545',       # Rojo
    'info': '#17A2B8',         # Azul claro
    'dark': '#343A40',         # Gris oscuro
    'light': '#F8F9FA'         # Gris claro
}

# CSS personalizado con colores institucionales mejorados
st.markdown("""
    <style>
    /* ========== ESTILOS GLOBALES ========== */
    .main {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
    }
    
    /* ========== TIPOGRAFÍA ========== */
    h1 {
        color: #005F9E !important;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 10px 0;
    }
    h2 {
        color: #004A7C !important;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif !important;
        font-weight: 600 !important;
        padding: 8px 0;
        border-bottom: 3px solid #FF8C00;
        display: inline-block;
    }
    h3 {
        color: #FF8C00 !important;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif !important;
        font-weight: 600 !important;
    }
    h4 {
        color: #343A40 !important;
        font-weight: 600 !important;
    }
    
    /* ========== TEXTO Y PÁRRAFOS ========== */
    p, span, div {
        color: #343A40 !important;
    }
    
    .stMarkdown {
        color: #343A40 !important;
    }
    
    /* ========== TARJETAS DE MÉTRICAS ========== */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #005F9E;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .metric-card h4 {
        margin: 0 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .metric-card h2 {
        margin: 15px 0 !important;
        font-size: 42px !important;
        font-weight: 700 !important;
        border: none !important;
    }
    .metric-card p {
        margin: 0 !important;
        font-size: 13px !important;
        opacity: 0.8;
    }
    
    /* ========== ALERTAS MEJORADAS ========== */
    .alert-critical {
        background: linear-gradient(135deg, #FFE5E5 0%, #FFD5D5 100%);
        color: #721C24 !important;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #DC3545;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.2);
        font-weight: 500;
    }
    .alert-critical strong {
        color: #DC3545 !important;
        font-weight: 700 !important;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #FFF9E5 0%, #FFF3D5 100%);
        color: #856404 !important;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #FFC107;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2);
        font-weight: 500;
    }
    .alert-warning strong {
        color: #FF8C00 !important;
        font-weight: 700 !important;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #E5F9E5 0%, #D5F4D5 100%);
        color: #155724 !important;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #28A745;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
        font-weight: 500;
    }
    .alert-success strong {
        color: #28A745 !important;
        font-weight: 700 !important;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #E5F3FF 0%, #D5EBFF 100%);
        color: #004085 !important;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #17A2B8;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(23, 162, 184, 0.2);
        font-weight: 500;
    }
    
    /* ========== BOTONES MEJORADOS ========== */
    .stButton>button {
        background: linear-gradient(135deg, #005F9E 0%, #004A7C 100%);
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,95,158,0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 14px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF8C00 0%, #E67E00 100%);
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,140,0,0.4);
    }
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Botones primarios (seleccionados) */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #FF8C00 0%, #E67E00 100%);
        box-shadow: 0 4px 15px rgba(255,140,0,0.4);
    }
    
    /* ========== SIDEBAR MEJORADO ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%);
        border-right: 3px solid #005F9E;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }
    [data-testid="stSidebar"] h3 {
        color: #005F9E !important;
        border-bottom: 2px solid #FF8C00;
        padding-bottom: 8px;
    }
    
    /* ========== DATAFRAMES Y TABLAS ========== */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .dataframe th {
        background: linear-gradient(135deg, #005F9E 0%, #004A7C 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
        text-align: left !important;
    }
    .dataframe td {
        background-color: white !important;
        color: #343A40 !important;
        padding: 10px !important;
        border-bottom: 1px solid #DEE2E6 !important;
    }
    .dataframe tr:hover {
        background-color: #F8F9FA !important;
    }
    
    /* ========== PESTAÑAS (TABS) ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        color: #343A40 !important;
        font-weight: 600;
        padding: 12px 24px;
        border: 2px solid #DEE2E6;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #005F9E 0%, #004A7C 100%);
        color: white !important;
        border-color: #005F9E;
    }
    
    /* ========== SELECTBOX Y MULTISELECT ========== */
    .stSelectbox label, .stMultiSelect label {
        color: #343A40 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* ========== MENSAJES DE STREAMLIT ========== */
    .stAlert {
        border-radius: 10px;
        border-width: 2px;
        font-weight: 500;
    }
    
    /* Info messages */
    .stAlert[data-baseweb="notification"] > div:first-child {
        background-color: #E5F3FF;
        color: #004085 !important;
        border-left: 5px solid #17A2B8;
    }
    
    /* ========== MÉTRICAS DE STREAMLIT ========== */
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        color: #005F9E !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #343A40 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background-color: white !important;
        border-radius: 8px;
        color: #005F9E !important;
        font-weight: 600 !important;
    }
    
    /* ========== DIVISORES ========== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #005F9E 50%, transparent 100%);
        margin: 30px 0;
    }
    
    /* ========== SCROLLBAR PERSONALIZADO ========== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #F8F9FA;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #005F9E 0%, #004A7C 100%);
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FF8C00 0%, #E67E00 100%);
    }
    
    /* ========== ANIMACIONES ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main > div {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_logo():
    """Carga el logo de la universidad"""
    logo_path = Path('assets/logo_universidad.png')
    if logo_path.exists():
        return str(logo_path)
    return None


@st.cache_data
def load_drift_results():
    """Carga los resultados de drift más recientes"""
    monitoring_dir = Path('outputs/monitoring')
    
    if not monitoring_dir.exists():
        st.warning(f"📂 Directorio de monitoreo no encontrado: {monitoring_dir.absolute()}")
        return None
    
    # Buscar el archivo más reciente
    drift_files = list(monitoring_dir.glob('drift_results_*.csv'))
    
    if not drift_files:
        st.info(f"📊 No se encontraron archivos de drift en: {monitoring_dir.absolute()}")
        return None
    
    latest_file = max(drift_files, key=lambda x: x.stat().st_mtime)
    
    return pd.read_csv(latest_file)


@st.cache_data
def load_alerts():
    """Carga las alertas más recientes"""
    monitoring_dir = Path('outputs/monitoring')
    
    if not monitoring_dir.exists():
        return None
    
    # Buscar el archivo más reciente
    alert_files = list(monitoring_dir.glob('alerts_*.json'))
    
    if not alert_files:
        return None
    
    latest_file = max(alert_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    """Carga las predicciones más recientes"""
    monitoring_dir = Path('outputs/monitoring')
    
    if not monitoring_dir.exists():
        return None
    
    # Buscar el archivo más reciente
    pred_files = list(monitoring_dir.glob('predictions_*.csv'))
    
    if not pred_files:
        return None
    
    latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
    
    return pd.read_csv(latest_file)


@st.cache_data
def load_summary():
    """Carga el resumen del último monitoreo"""
    summary_file = Path('outputs/monitoring/latest_summary.json')
    
    if not summary_file.exists():
        return None
    
    with open(summary_file, 'r') as f:
        return json.load(f)


@st.cache_data
def load_eda_images():
    """Carga las rutas de las imágenes del EDA"""
    output_dir = Path('outputs')
    
    if not output_dir.exists():
        return []
    
    # Buscar todas las imágenes PNG
    image_files = list(output_dir.glob('eda_*.png'))
    
    return sorted([str(f) for f in image_files])


def create_drift_gauge(value, threshold_medium=0.1, threshold_high=0.2, title="Drift Score"):
    """Crea un gauge para visualizar el nivel de drift"""
    
    if value < threshold_medium:
        color = COLORS['success']
        status = "Bajo"
    elif value < threshold_high:
        color = COLORS['warning']
        status = "Medio"
    else:
        color = COLORS['danger']
        status = "Alto"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title}<br><span style='font-size:0.8em;color:{color}'>{status}</span>"},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold_medium], 'color': "rgba(40, 167, 69, 0.2)"},
                {'range': [threshold_medium, threshold_high], 'color': "rgba(255, 193, 7, 0.2)"},
                {'range': [threshold_high, 1], 'color': "rgba(220, 53, 69, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_high
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'size': 12}
    )
    
    return fig


def create_distribution_comparison(df_ref, df_prod, column):
    """Crea gráfico comparativo de distribuciones"""
    
    fig = go.Figure()
    
    # Histograma de referencia
    fig.add_trace(go.Histogram(
        x=df_ref[column],
        name='Referencia (Entrenamiento)',
        opacity=0.7,
        marker_color=COLORS['primary'],
        nbinsx=50
    ))
    
    # Histograma de producción
    fig.add_trace(go.Histogram(
        x=df_prod[column],
        name='Producción (Actual)',
        opacity=0.7,
        marker_color=COLORS['secondary'],
        nbinsx=50
    ))
    
    fig.update_layout(
        title=f'Distribución: {column}',
        xaxis_title=column,
        yaxis_title='Frecuencia',
        barmode='overlay',
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_drift_heatmap(drift_df):
    """Crea un heatmap de drift por variable"""
    
    # Preparar datos
    numeric_drift = drift_df[drift_df['tipo'] == 'numérica'].copy()
    
    if len(numeric_drift) == 0:
        return None
    
    # Crear matriz de métricas
    variables = numeric_drift['variable'].tolist()
    
    metrics_matrix = numeric_drift[['ks_statistic', 'psi', 'js_divergence']].values
    
    fig = go.Figure(data=go.Heatmap(
        z=metrics_matrix.T,
        x=variables,
        y=['KS Statistic', 'PSI', 'JS Divergence'],
        colorscale=[
            [0, COLORS['success']],
            [0.5, COLORS['warning']],
            [1, COLORS['danger']]
        ],
        text=np.round(metrics_matrix.T, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Mapa de Calor: Métricas de Drift por Variable',
        height=400,
        xaxis_title='Variables',
        yaxis_title='Métricas'
    )
    
    return fig


def create_severity_pie(drift_df):
    """Crea gráfico de pastel de severidad"""
    
    severity_counts = drift_df['severity'].value_counts()
    
    colors_map = {
        'low': COLORS['success'],
        'medium': COLORS['warning'],
        'high': COLORS['danger']
    }
    
    colors = [colors_map.get(sev, COLORS['info']) for sev in severity_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=['Bajo', 'Medio', 'Alto'],
        values=severity_counts.values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title='Distribución de Severidad del Drift',
        height=350,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def create_predictions_chart(predictions_df):
    """Crea gráfico de predicciones"""
    
    # Distribución de probabilidades
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribución de Probabilidades', 'Predicciones'),
        specs=[[{'type': 'histogram'}, {'type': 'pie'}]]
    )
    
    # Histograma de probabilidades
    fig.add_trace(
        go.Histogram(
            x=predictions_df['prediction_proba'],
            nbinsx=50,
            marker_color=COLORS['primary'],
            name='Probabilidad'
        ),
        row=1, col=1
    )
    
    # Pie chart de predicciones
    pred_counts = predictions_df['prediction'].value_counts()
    
    fig.add_trace(
        go.Pie(
            labels=['No Fraude', 'Fraude'],
            values=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
            marker=dict(colors=[COLORS['success'], COLORS['danger']]),
            hole=0.3
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True
    )
    
    return fig


def main():
    """Función principal de la aplicación"""
    
    # Header mejorado con banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #005F9E 0%, #004A7C 100%); 
                padding: 30px 20px; 
                border-radius: 15px; 
                margin-bottom: 30px;
                box-shadow: 0 8px 25px rgba(0,95,158,0.3);'>
        <div style='text-align: center;'>
            <h1 style='color: white !important; 
                       font-size: 42px; 
                       margin: 0; 
                       text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
                       font-weight: 800;'>
                🔍 Sistema de Monitoreo de Data Drift
            </h1>
            <h3 style='color: #FFD700 !important; 
                       margin: 15px 0; 
                       font-size: 24px;
                       font-weight: 600;'>
                Pipeline MLOps - Detección de Fraude en Transacciones
            </h3>
            <div style='background: rgba(255,255,255,0.1); 
                        padding: 12px 24px; 
                        border-radius: 25px; 
                        display: inline-block;
                        margin-top: 10px;'>
                <p style='color: white !important; 
                          font-weight: 700; 
                          margin: 0;
                          font-size: 16px;
                          letter-spacing: 0.5px;'>
                    👨‍💻 DANIEL ALEJANDRO RINCON VALENCIA | 🎓 Universidad Católica Luis Amigó
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar mejorado
    with st.sidebar:
        # Logo con efecto
        st.markdown("""
        <div style='background: white; 
                    padding: 15px; 
                    border-radius: 12px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    margin-bottom: 20px;'>
        """, unsafe_allow_html=True)
        
        st.image("https://www.ucatolicaluisamigo.edu.co/wp-content/uploads/2023/02/logo-universidad-catolica-luis-amigo.png", 
                 use_column_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style='color: #005F9E !important; 
                   font-weight: 700; 
                   text-align: center;
                   border-bottom: 3px solid #FF8C00;
                   padding-bottom: 10px;
                   margin-bottom: 20px;'>
            📊 Navegación
        </h3>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='color: #343A40 !important; font-weight: 600; margin-bottom: 15px;'>Selecciona una sección:</p>", unsafe_allow_html=True)
        
        # Usar session_state para mantener la página seleccionada
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Resumen General"
        
        # Botones de navegación
        if st.button("🏠 Resumen General", use_container_width=True, 
                    type="primary" if st.session_state.current_page == "Resumen General" else "secondary"):
            st.session_state.current_page = "Resumen General"
            st.rerun()
            
        if st.button("📈 Métricas de Drift", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Métricas de Drift" else "secondary"):
            st.session_state.current_page = "Métricas de Drift"
            st.rerun()
            
        if st.button("🚨 Alertas y Recomendaciones", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Alertas y Recomendaciones" else "secondary"):
            st.session_state.current_page = "Alertas y Recomendaciones"
            st.rerun()
            
        if st.button("🎯 Predicciones del Modelo", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Predicciones del Modelo" else "secondary"):
            st.session_state.current_page = "Predicciones del Modelo"
            st.rerun()
            
        if st.button("📊 Gráficos EDA", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Gráficos EDA" else "secondary"):
            st.session_state.current_page = "Gráficos EDA"
            st.rerun()
            
        if st.button("📋 Tabla de Datos", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Tabla de Datos" else "secondary"):
            st.session_state.current_page = "Tabla de Datos"
            st.rerun()
        
        if st.button("🏆 Comparación de Modelos", use_container_width=True,
                    type="primary" if st.session_state.current_page == "Comparación de Modelos" else "secondary"):
            st.session_state.current_page = "Comparación de Modelos"
            st.rerun()
        
        page = st.session_state.current_page
        
        st.markdown("---")
        
        # Botón para refrescar datos
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #E5F3FF 0%, #D5EBFF 100%); 
                    padding: 20px; 
                    border-radius: 12px;
                    border-left: 5px solid #005F9E;
                    box-shadow: 0 4px 12px rgba(0,95,158,0.2);'>
            <h3 style='color: #005F9E !important; 
                       margin-top: 0;
                       font-size: 18px;
                       font-weight: 700;'>
                ℹ️ Guía de Métricas
            </h3>
            <div style='margin: 15px 0;'>
                <p style='color: #343A40 !important; 
                          font-weight: 600; 
                          margin: 8px 0;
                          font-size: 14px;'>
                    <span style='color: #28A745; font-size: 18px;'>🟢</span> 
                    <strong>Bajo:</strong> &lt; 0.1
                </p>
                <p style='color: #343A40 !important; 
                          font-weight: 600; 
                          margin: 8px 0;
                          font-size: 14px;'>
                    <span style='color: #FFC107; font-size: 18px;'>🟡</span> 
                    <strong>Medio:</strong> 0.1 - 0.2
                </p>
                <p style='color: #343A40 !important; 
                          font-weight: 600; 
                          margin: 8px 0;
                          font-size: 14px;'>
                    <span style='color: #DC3545; font-size: 18px;'>🔴</span> 
                    <strong>Alto:</strong> &gt; 0.2
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        summary = load_summary()
        if summary:
            timestamp = summary.get('timestamp', 'N/A')
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #E5F9E5 0%, #D5F4D5 100%); 
                        padding: 15px; 
                        border-radius: 10px;
                        border-left: 5px solid #28A745;
                        text-align: center;'>
                <p style='color: #155724 !important; 
                          font-weight: 700; 
                          margin: 0;
                          font-size: 13px;'>
                    📅 ÚLTIMA ACTUALIZACIÓN
                </p>
                <p style='color: #28A745 !important; 
                          font-weight: 600; 
                          margin: 8px 0 0 0;
                          font-size: 14px;'>
                    {timestamp}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No hay datos disponibles")
    
    # Cargar datos
    drift_df = load_drift_results()
    alerts = load_alerts()
    predictions_df = load_predictions()
    summary = load_summary()
    
    # Verificar si hay datos
    if drift_df is None:
        st.error("⚠️ No se encontraron resultados de monitoreo. Por favor, ejecuta primero `model_monitoring.py`")
        st.info("Para generar los datos de monitoreo, ejecuta:\n\n```bash\ncd mlops_pipeline/src\npython model_monitoring.py\n```")
        return
    
    # PÁGINA: Resumen General
    if page == "Resumen General":
        st.header("📊 Resumen General del Monitoreo")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vars = summary.get('total_variables', 0) if summary else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: {COLORS["primary"]}; margin: 0; font-weight: 700;'>TOTAL VARIABLES</h4>
                <h2 style='color: {COLORS["primary"]}; margin: 15px 0; font-size: 48px; font-weight: 700;'>{total_vars}</h2>
                <p style='margin: 0; color: #343A40; font-weight: 600; font-size: 14px;'>📊 Monitoreadas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            drift_detected = summary.get('drift_detected', 0) if summary else 0
            drift_color = COLORS["danger"] if drift_detected > 0 else COLORS["success"]
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: {COLORS["warning"]}; margin: 0; font-weight: 700;'>VARIABLES CON DRIFT</h4>
                <h2 style='color: {drift_color}; margin: 15px 0; font-size: 48px; font-weight: 700;'>{drift_detected}</h2>
                <p style='margin: 0; color: #343A40; font-weight: 600; font-size: 14px;'>⚡ Detectadas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_severity = summary.get('high_severity', 0) if summary else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: {COLORS["danger"]}; margin: 0; font-weight: 700;'>SEVERIDAD ALTA</h4>
                <h2 style='color: {COLORS["danger"]}; margin: 15px 0; font-size: 48px; font-weight: 700;'>{high_severity}</h2>
                <p style='margin: 0; color: #343A40; font-weight: 600; font-size: 14px;'>🚨 Críticas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            fraud_rate = summary.get('predictions', {}).get('fraud_rate', 0) if summary else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: {COLORS["info"]}; margin: 0; font-weight: 700;'>TASA DE FRAUDE</h4>
                <h2 style='color: {COLORS["secondary"]}; margin: 15px 0; font-size: 48px; font-weight: 700;'>{fraud_rate:.2f}%</h2>
                <p style='margin: 0; color: #343A40; font-weight: 600; font-size: 14px;'>🎯 Detectada</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Gráficos de resumen
        col1, col2 = st.columns(2)
        
        with col1:
            fig_severity = create_severity_pie(drift_df)
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            fig_heatmap = create_drift_heatmap(drift_df)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Top variables con drift
        st.subheader("🔝 Top 10 Variables con Mayor Drift")
        
        numeric_drift = drift_df[drift_df['tipo'] == 'numérica'].copy()
        if len(numeric_drift) > 0:
            top_drift = numeric_drift.nlargest(10, 'psi')[['variable', 'psi', 'ks_statistic', 'js_divergence', 'severity']]
            
            # Colorear por severidad
            def color_severity(val):
                if val == 'high':
                    return f'background-color: {COLORS["danger"]}; color: white;'
                elif val == 'medium':
                    return f'background-color: {COLORS["warning"]};'
                else:
                    return f'background-color: {COLORS["success"]}; color: white;'
            
            styled_df = top_drift.style.applymap(color_severity, subset=['severity'])
            st.dataframe(styled_df, use_container_width=True)
    
    # PÁGINA: Métricas de Drift
    elif page == "Métricas de Drift":
        st.header("📈 Análisis Detallado de Métricas de Drift")
        
        # Filtros
        col1, col2 = st.columns(2)
        
        with col1:
            tipo_filter = st.multiselect(
                "Filtrar por tipo de variable:",
                options=drift_df['tipo'].unique().tolist(),
                default=drift_df['tipo'].unique().tolist()
            )
        
        with col2:
            severity_filter = st.multiselect(
                "Filtrar por severidad:",
                options=['low', 'medium', 'high'],
                default=['low', 'medium', 'high']
            )
        
        # Aplicar filtros
        filtered_df = drift_df[
            (drift_df['tipo'].isin(tipo_filter)) &
            (drift_df['severity'].isin(severity_filter))
        ]
        
        st.info(f"📊 Mostrando {len(filtered_df)} variables de {len(drift_df)} totales")
        
        # Seleccionar variable para análisis detallado
        if len(filtered_df) > 0:
            selected_var = st.selectbox(
                "Selecciona una variable para análisis detallado:",
                options=filtered_df['variable'].tolist()
            )
            
            var_data = filtered_df[filtered_df['variable'] == selected_var].iloc[0]
            
            # Mostrar métricas en gauges
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'ks_statistic' in var_data:
                    fig_ks = create_drift_gauge(var_data['ks_statistic'], 0.1, 0.2, "KS Statistic")
                    st.plotly_chart(fig_ks, use_container_width=True)
            
            with col2:
                if 'psi' in var_data:
                    fig_psi = create_drift_gauge(var_data['psi'], 0.1, 0.2, "PSI")
                    st.plotly_chart(fig_psi, use_container_width=True)
            
            with col3:
                if 'js_divergence' in var_data:
                    fig_js = create_drift_gauge(var_data['js_divergence'], 0.1, 0.2, "JS Divergence")
                    st.plotly_chart(fig_js, use_container_width=True)
            
            # Mostrar estadísticas comparativas
            if var_data['tipo'] == 'numérica':
                st.subheader("📊 Estadísticas Comparativas")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Datos de Referencia (Entrenamiento)**
                    - Media: {var_data.get('ref_mean', 'N/A'):.4f}
                    - Desv. Est.: {var_data.get('ref_std', 'N/A'):.4f}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Datos de Producción (Actual)**
                    - Media: {var_data.get('prod_mean', 'N/A'):.4f}
                    - Desv. Est.: {var_data.get('prod_std', 'N/A'):.4f}
                    - Cambio: {var_data.get('mean_change_%', 0):.2f}%
                    """)
        
        # Tabla completa de métricas
        st.subheader("📋 Tabla Completa de Métricas")
        st.dataframe(filtered_df, use_container_width=True)
    
    # PÁGINA: Alertas
    elif page == "Alertas y Recomendaciones":
        st.header("🚨 Alertas y Recomendaciones")
        
        if alerts:
            for alert in alerts:
                level = alert.get('level', 'INFO')
                message = alert.get('message', '')
                recommendation = alert.get('recommendation', '')
                
                if level == 'CRÍTICO':
                    st.markdown(f"""
                    <div class='alert-critical'>
                        <h3 style='color: #DC3545 !important; margin-top: 0; font-size: 20px;'>🚨 {message}</h3>
                        <p style='color: #721C24 !important; margin-bottom: 0; font-size: 15px;'><strong style='color: #DC3545 !important;'>Recomendación:</strong> {recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar detalles
                    if 'details' in alert:
                        with st.expander("🔍 Ver detalles"):
                            st.json(alert['details'])
                
                elif level == 'ADVERTENCIA':
                    st.markdown(f"""
                    <div class='alert-warning'>
                        <h3 style='color: #FF8C00 !important; margin-top: 0; font-size: 20px;'>⚠️ {message}</h3>
                        <p style='color: #856404 !important; margin-bottom: 0; font-size: 15px;'><strong style='color: #FF8C00 !important;'>Recomendación:</strong> {recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'variables' in alert:
                        with st.expander("📋 Variables afectadas"):
                            st.write(", ".join(alert['variables']))
                
                else:
                    st.markdown(f"""
                    <div class='alert-success'>
                        <h3 style='color: #28A745 !important; margin-top: 0; font-size: 20px;'>✅ {message}</h3>
                        <p style='color: #155724 !important; margin-bottom: 0; font-size: 15px;'><strong style='color: #28A745 !important;'>Recomendación:</strong> {recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("✨ No hay alertas disponibles - El sistema está operando normalmente")
    
    # PÁGINA: Predicciones
    elif page == "Predicciones del Modelo":
        st.header("🎯 Predicciones del Modelo")
        
        if predictions_df is not None:
            # Estadísticas de predicciones
            total_pred = len(predictions_df)
            fraud_pred = predictions_df['prediction'].sum()
            fraud_rate = (fraud_pred / total_pred * 100) if total_pred > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predicciones", f"{total_pred:,}")
            
            with col2:
                st.metric("Fraudes Detectados", f"{fraud_pred:,}")
            
            with col3:
                st.metric("Tasa de Fraude", f"{fraud_rate:.2f}%")
            
            # Gráficos de predicciones
            st.subheader("📊 Análisis de Predicciones")
            fig_pred = create_predictions_chart(predictions_df)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Distribución de probabilidades por rango
            st.subheader("📈 Distribución de Probabilidades de Fraude")
            
            bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-100%']
            
            predictions_df['prob_range'] = pd.cut(predictions_df['prediction_proba'], bins=bins, labels=labels)
            prob_counts = predictions_df['prob_range'].value_counts().sort_index()
            
            fig_bars = go.Figure(data=[
                go.Bar(
                    x=prob_counts.index,
                    y=prob_counts.values,
                    marker_color=COLORS['primary'],
                    text=prob_counts.values,
                    textposition='auto'
                )
            ])
            
            fig_bars.update_layout(
                title='Distribución de Transacciones por Rango de Probabilidad',
                xaxis_title='Rango de Probabilidad de Fraude',
                yaxis_title='Número de Transacciones',
                height=400
            )
            
            st.plotly_chart(fig_bars, use_container_width=True)
            
            # Tabla de predicciones
            st.subheader("📋 Tabla de Predicciones (Muestra)")
            
            # Mostrar solo una muestra
            sample_size = st.slider("Tamaño de muestra:", 10, 100, 50)
            st.dataframe(predictions_df.head(sample_size), use_container_width=True)
            
            # Descargar predicciones
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Predicciones Completas (CSV)",
                data=csv,
                file_name=f"predicciones_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No hay predicciones disponibles")
    
    # PÁGINA: Gráficos EDA
    elif page == "Gráficos EDA":
        st.header("📊 Gráficos del Análisis Exploratorio de Datos")
        
        eda_images = load_eda_images()
        
        if eda_images:
            st.info(f"📁 Mostrando {len(eda_images)} gráficos del análisis exploratorio")
            
            # Crear tabs para diferentes categorías
            tabs = st.tabs([
                "📈 Distribuciones",
                "📦 Boxplots", 
                "🔍 Correlaciones",
                "💰 Análisis de Fraude",
                "⏰ Análisis Temporal",
                "🔗 Multivariable"
            ])
            
            # Distribuciones
            with tabs[0]:
                dist_images = [img for img in eda_images if 'distribucion' in img or 'categoricas' in img]
                for img_path in dist_images:
                    try:
                        image = Image.open(img_path)
                        st.image(image, use_column_width=True, caption=Path(img_path).name)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error cargando imagen {img_path}: {e}")
            
            # Boxplots
            with tabs[1]:
                box_images = [img for img in eda_images if 'boxplot' in img]
                for img_path in box_images:
                    try:
                        image = Image.open(img_path)
                        st.image(image, use_column_width=True, caption=Path(img_path).name)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error cargando imagen {img_path}: {e}")
            
            # Correlaciones
            with tabs[2]:
                corr_images = [img for img in eda_images if 'correlacion' in img]
                for img_path in corr_images:
                    try:
                        image = Image.open(img_path)
                        st.image(image, use_column_width=True, caption=Path(img_path).name)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error cargando imagen {img_path}: {e}")
            
            # Fraude
            with tabs[3]:
                fraud_images = [img for img in eda_images if 'fraude' in img and 'temporal' not in img]
                for img_path in fraud_images:
                    try:
                        image = Image.open(img_path)
                        st.image(image, use_column_width=True, caption=Path(img_path).name)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error cargando imagen {img_path}: {e}")
            
            # Temporal
            with tabs[4]:
                temp_images = [img for img in eda_images if 'temporal' in img]
                for img_path in temp_images:
                    try:
                        image = Image.open(img_path)
                        st.image(image, use_column_width=True, caption=Path(img_path).name)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error cargando imagen {img_path}: {e}")
            
            # Multivariable
            with tabs[5]:
                multi_images = [img for img in eda_images if 'pairplot' in img or 'multivariable' in img]
                for img_path in multi_images:
                    try:
                        image = Image.open(img_path)
                        st.image(image, use_column_width=True, caption=Path(img_path).name)
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Error cargando imagen {img_path}: {e}")
        else:
            st.warning("No se encontraron gráficos del EDA. Ejecuta primero el notebook `Comprension_eda_completo.ipynb`")
    
    # PÁGINA: Tabla de Datos
    elif page == "Tabla de Datos":
        st.header("📋 Tabla Completa de Resultados")
        
        # Combinar drift results con predicciones
        if predictions_df is not None and drift_df is not None:
            st.subheader("🔍 Explorador de Datos")
            
            # Selector de tabla
            table_option = st.selectbox(
                "Selecciona la tabla a visualizar:",
                ["Resultados de Drift", "Predicciones", "Alertas"]
            )
            
            if table_option == "Resultados de Drift":
                st.dataframe(drift_df, use_container_width=True, height=600)
                
                # Descargar
                csv = drift_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Resultados de Drift (CSV)",
                    data=csv,
                    file_name=f"drift_results_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            elif table_option == "Predicciones":
                st.dataframe(predictions_df, use_container_width=True, height=600)
                
                # Descargar
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Predicciones (CSV)",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            elif table_option == "Alertas":
                if alerts:
                    st.json(alerts)
                else:
                    st.warning("No hay alertas disponibles")
        else:
            st.warning("No hay datos disponibles")
    
    # PÁGINA: Comparación de Modelos
    elif page == "Comparación de Modelos":
        st.header("🏆 Comparación de Modelos Entrenados")
        
        # Cargar resultados de comparación
        comparison_path = "outputs/model_comparison.csv"
        results_path = "outputs/all_models_results.json"
        metadata_path = "models/best_model_metadata.json"
        
        if Path(comparison_path).exists():
            # Cargar datos
            df_comparison = pd.read_csv(comparison_path)
            
            # Banner del mejor modelo
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    best_metadata = json.load(f)
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #28A745 0%, #20C997 100%); 
                            padding: 30px; 
                            border-radius: 15px;
                            box-shadow: 0 8px 20px rgba(40,167,69,0.3);
                            margin-bottom: 30px;
                            text-align: center;'>
                    <h2 style='color: white !important; 
                               margin: 0; 
                               font-size: 32px;
                               text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
                        🥇 Mejor Modelo: {best_metadata['model_name']}
                    </h2>
                    <p style='color: white !important; 
                              font-size: 18px; 
                              margin: 10px 0 0 0;
                              opacity: 0.95;'>
                        ROC-AUC: {best_metadata['metrics']['roc_auc']:.4f} | 
                        F1-Score: {best_metadata['metrics']['f1_score']:.4f} | 
                        Precision: {best_metadata['metrics']['precision']:.4f} | 
                        Recall: {best_metadata['metrics']['recall']:.4f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Métricas en columnas
            st.subheader("📊 Comparación de Métricas Principales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Ordenar por ROC-AUC
            df_sorted = df_comparison.sort_values('roc_auc', ascending=False)
            
            with col1:
                st.metric(
                    label="🎯 Mejor ROC-AUC",
                    value=f"{df_sorted.iloc[0]['roc_auc']:.4f}",
                    delta=f"{df_sorted.iloc[0]['Model']}"
                )
            
            with col2:
                best_f1 = df_sorted.sort_values('f1_score', ascending=False).iloc[0]
                st.metric(
                    label="📈 Mejor F1-Score",
                    value=f"{best_f1['f1_score']:.4f}",
                    delta=f"{best_f1['Model']}"
                )
            
            with col3:
                best_precision = df_sorted.sort_values('precision', ascending=False).iloc[0]
                st.metric(
                    label="🎯 Mejor Precision",
                    value=f"{best_precision['precision']:.4f}",
                    delta=f"{best_precision['Model']}"
                )
            
            with col4:
                best_recall = df_sorted.sort_values('recall', ascending=False).iloc[0]
                st.metric(
                    label="🔍 Mejor Recall",
                    value=f"{best_recall['recall']:.4f}",
                    delta=f"{best_recall['Model']}"
                )
            
            st.markdown("---")
            
            # Tabla comparativa
            st.subheader("📋 Tabla Comparativa de Todos los Modelos")
            
            # Formatear tabla para mejor visualización
            df_display = df_comparison.copy()
            df_display = df_display.sort_values('roc_auc', ascending=False)
            
            # Resaltar mejor modelo
            def highlight_best(s):
                is_best = s == s.max()
                return ['background-color: #D4EDDA; font-weight: bold' if v else '' for v in is_best]
            
            st.dataframe(
                df_display.style.apply(highlight_best, subset=['roc_auc', 'f1_score', 'precision', 'recall']),
                use_container_width=True,
                height=300
            )
            
            st.markdown("---")
            
            # Gráficos comparativos
            st.subheader("📊 Visualización de Métricas")
            
            tab1, tab2, tab3 = st.tabs(["📊 Comparación General", "⏱️ Tiempo de Entrenamiento", "🎯 Detalle por Métrica"])
            
            with tab1:
                # Gráfico de barras para todas las métricas
                metrics_to_plot = ['roc_auc', 'pr_auc', 'f1_score', 'precision', 'recall', 'accuracy']
                
                fig = go.Figure()
                
                for metric in metrics_to_plot:
                    fig.add_trace(go.Bar(
                        name=metric.upper().replace('_', '-'),
                        x=df_comparison['Model'],
                        y=df_comparison[metric],
                        text=df_comparison[metric].round(4),
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title="Comparación de Todas las Métricas",
                    xaxis_title="Modelo",
                    yaxis_title="Score",
                    barmode='group',
                    height=500,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Gráfico de tiempo de entrenamiento
                fig_time = px.bar(
                    df_comparison.sort_values('Training_Time'),
                    x='Model',
                    y='Training_Time',
                    title='Tiempo de Entrenamiento por Modelo',
                    labels={'Training_Time': 'Tiempo (segundos)', 'Model': 'Modelo'},
                    color='Training_Time',
                    color_continuous_scale='Viridis',
                    text='Training_Time'
                )
                
                fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
                fig_time.update_layout(height=500, template='plotly_white')
                
                st.plotly_chart(fig_time, use_container_width=True)
                
                # Análisis de eficiencia
                st.markdown("### ⚡ Análisis de Eficiencia")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fastest = df_comparison.loc[df_comparison['Training_Time'].idxmin()]
                    st.success(f"""
                    **Modelo Más Rápido:** {fastest['Model']}  
                    **Tiempo:** {fastest['Training_Time']:.2f}s  
                    **ROC-AUC:** {fastest['roc_auc']:.4f}
                    """)
                
                with col2:
                    # Mejor balance entre velocidad y performance
                    df_comparison['efficiency_score'] = df_comparison['roc_auc'] / (df_comparison['Training_Time'] / 60)
                    most_efficient = df_comparison.loc[df_comparison['efficiency_score'].idxmax()]
                    st.info(f"""
                    **Modelo Más Eficiente:** {most_efficient['Model']}  
                    **ROC-AUC:** {most_efficient['roc_auc']:.4f}  
                    **Tiempo:** {most_efficient['Training_Time']:.2f}s  
                    **Ratio Eficiencia:** {most_efficient['efficiency_score']:.2f}
                    """)
            
            with tab3:
                # Gráficos individuales para cada métrica
                col1, col2 = st.columns(2)
                
                with col1:
                    # ROC-AUC
                    fig_roc = px.bar(
                        df_comparison.sort_values('roc_auc', ascending=False),
                        x='Model',
                        y='roc_auc',
                        title='ROC-AUC Score',
                        color='roc_auc',
                        color_continuous_scale='Blues',
                        text='roc_auc'
                    )
                    fig_roc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_roc.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                    # Precision
                    fig_prec = px.bar(
                        df_comparison.sort_values('precision', ascending=False),
                        x='Model',
                        y='precision',
                        title='Precision',
                        color='precision',
                        color_continuous_scale='Greens',
                        text='precision'
                    )
                    fig_prec.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_prec.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_prec, use_container_width=True)
                
                with col2:
                    # F1-Score
                    fig_f1 = px.bar(
                        df_comparison.sort_values('f1_score', ascending=False),
                        x='Model',
                        y='f1_score',
                        title='F1-Score',
                        color='f1_score',
                        color_continuous_scale='Oranges',
                        text='f1_score'
                    )
                    fig_f1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_f1.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_f1, use_container_width=True)
                    
                    # Recall
                    fig_rec = px.bar(
                        df_comparison.sort_values('recall', ascending=False),
                        x='Model',
                        y='recall',
                        title='Recall',
                        color='recall',
                        color_continuous_scale='Purples',
                        text='recall'
                    )
                    fig_rec.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                    fig_rec.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_rec, use_container_width=True)
            
            st.markdown("---")
            
            # Matriz de confusión del mejor modelo
            if Path(results_path).exists():
                with open(results_path, 'r') as f:
                    all_results = json.load(f)
                
                st.subheader("🎯 Matriz de Confusión del Mejor Modelo")
                
                best_model_name = df_sorted.iloc[0]['Model']
                if best_model_name in all_results:
                    cm = np.array(all_results[best_model_name]['confusion_matrix'])
                    
                    # Crear heatmap de matriz de confusión
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=['Predicho: No Fraude', 'Predicho: Fraude'],
                        y=['Real: No Fraude', 'Real: Fraude'],
                        colorscale='Blues',
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 20},
                        hoverongaps=False
                    ))
                    
                    fig_cm.update_layout(
                        title=f'Matriz de Confusión - {best_model_name}',
                        height=500,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Interpretación
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("✅ Verdaderos Negativos", f"{cm[0,0]:,}")
                    with col2:
                        st.metric("❌ Falsos Positivos", f"{cm[0,1]:,}")
                    with col3:
                        st.metric("❌ Falsos Negativos", f"{cm[1,0]:,}")
                    with col4:
                        st.metric("✅ Verdaderos Positivos", f"{cm[1,1]:,}")
            
            # Descarga de resultados
            st.markdown("---")
            st.subheader("📥 Descargar Resultados")
            
            csv = df_comparison.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Tabla de Comparación (CSV)",
                data=csv,
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("""
            ⚠️ No se encontraron resultados de comparación de modelos.  
            
            Para generar estos resultados, ejecuta:
            ```bash
            python run_mlops.py
            ```
            
            Esto entrenará 5 modelos diferentes y generará la comparación completa.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>
            🎓 <strong>Universidad Católica Luis Amigó</strong> | 
            🔍 <strong>Pipeline MLOps - Detección de Fraude</strong> | 
            📅 {date}
        </p>
    </div>
    """.format(date=datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
