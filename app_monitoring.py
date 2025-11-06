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

# CSS personalizado con colores institucionales
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
    }
    .stApp {
        background-color: #F8F9FA;
    }
    h1 {
        color: #005F9E;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
    }
    h2 {
        color: #005F9E;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #FF8C00;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #005F9E;
    }
    .alert-critical {
        background-color: #FFE5E5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #DC3545;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #FFF9E5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin: 10px 0;
    }
    .alert-success {
        background-color: #E5F9E5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28A745;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #005F9E;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF8C00;
        color: white;
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
    
    # Header con logo
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <h1>🔍 Sistema de Monitoreo de Data Drift</h1>
            <h3 style='color: #FF8C00;'>Pipeline MLOps - Detección de Fraude en Transacciones</h3>
            <p style='color: #005F9E; font-weight: bold;'>DANIEL ALEJANDRO RINCON VALENCIA - Universidad Católica Luis Amigó</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.ucatolicaluisamigo.edu.co/wp-content/uploads/2023/02/logo-universidad-catolica-luis-amigo.png", 
                 use_container_width=True)
        
        st.markdown("### 📊 Navegación")
        
        st.write("Selecciona una sección:")
        
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
        
        page = st.session_state.current_page
        
        st.markdown("---")
        
        # Botón para refrescar datos
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        ### ℹ️ Información
        
        **Métricas de Drift:**
        - 🟢 Bajo: < 0.1
        - 🟡 Medio: 0.1 - 0.2
        - 🔴 Alto: > 0.2
        
        **Última actualización:**
        """)
        
        summary = load_summary()
        if summary:
            timestamp = summary.get('timestamp', 'N/A')
            st.info(f"📅 {timestamp}")
        else:
            st.warning("No hay datos disponibles")
    
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
                <h4 style='color: {COLORS["primary"]}; margin: 0;'>Total Variables</h4>
                <h2 style='color: {COLORS["dark"]}; margin: 10px 0;'>{total_vars}</h2>
                <p style='margin: 0; color: #666;'>Monitoreadas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            drift_detected = summary.get('drift_detected', 0) if summary else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: {COLORS["warning"]}; margin: 0;'>Variables con Drift</h4>
                <h2 style='color: {COLORS["dark"]}; margin: 10px 0;'>{drift_detected}</h2>
                <p style='margin: 0; color: #666;'>Detectadas</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_severity = summary.get('high_severity', 0) if summary else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: {COLORS["danger"]}; margin: 0;'>Severidad Alta</h4>
                <h2 style='color: {COLORS["danger"]}; margin: 10px 0;'>{high_severity}</h2>
                <p style='margin: 0; color: #666;'>⚠️ Crítico</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            fraud_rate = summary.get('predictions', {}).get('fraud_rate', 0) if summary else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: {COLORS["info"]}; margin: 0;'>Tasa de Fraude</h4>
                <h2 style='color: {COLORS["dark"]}; margin: 10px 0;'>{fraud_rate:.2f}%</h2>
                <p style='margin: 0; color: #666;'>Detectada</p>
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
                        <h3>🚨 {message}</h3>
                        <p><strong>Recomendación:</strong> {recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar detalles
                    if 'details' in alert:
                        with st.expander("Ver detalles"):
                            st.json(alert['details'])
                
                elif level == 'ADVERTENCIA':
                    st.markdown(f"""
                    <div class='alert-warning'>
                        <h3>⚠️ {message}</h3>
                        <p><strong>Recomendación:</strong> {recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'variables' in alert:
                        with st.expander("Variables afectadas"):
                            st.write(", ".join(alert['variables']))
                
                else:
                    st.markdown(f"""
                    <div class='alert-success'>
                        <h3>ℹ️ {message}</h3>
                        <p><strong>Recomendación:</strong> {recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No hay alertas disponibles")
    
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
                    st.image(img_path, use_container_width=True)
                    st.markdown("---")
            
            # Boxplots
            with tabs[1]:
                box_images = [img for img in eda_images if 'boxplot' in img]
                for img_path in box_images:
                    st.image(img_path, use_container_width=True)
                    st.markdown("---")
            
            # Correlaciones
            with tabs[2]:
                corr_images = [img for img in eda_images if 'correlacion' in img]
                for img_path in corr_images:
                    st.image(img_path, use_container_width=True)
                    st.markdown("---")
            
            # Fraude
            with tabs[3]:
                fraud_images = [img for img in eda_images if 'fraude' in img and 'temporal' not in img]
                for img_path in fraud_images:
                    st.image(img_path, use_container_width=True)
                    st.markdown("---")
            
            # Temporal
            with tabs[4]:
                temp_images = [img for img in eda_images if 'temporal' in img]
                for img_path in temp_images:
                    st.image(img_path, use_container_width=True)
                    st.markdown("---")
            
            # Multivariable
            with tabs[5]:
                multi_images = [img for img in eda_images if 'pairplot' in img or 'multivariable' in img]
                for img_path in multi_images:
                    st.image(img_path, use_container_width=True)
                    st.markdown("---")
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
