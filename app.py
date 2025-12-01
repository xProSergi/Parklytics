import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ====================================================
# 1. PAGE CONFIGURATION
# ====================================================
st.set_page_config(
    page_title="ParkBeat - Predicción de Tiempos de Espera",
    page_icon="img/logoParklytics.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/xProSergi/Parklytics'
    }
)

# ====================================================
# 2. CUSTOM CSS STYLING
# ====================================================
def load_css():
    """Load custom CSS for the application."""
    st.markdown("""
    <style>
        /* Base Styles */
        :root {
            --primary: #4361ee;
            --primary-light: #4cc9f0;
            --secondary: #7209b7;
            --accent: #f72585;
            --success: #4cc9f0;
            --warning: #f8961e;
            --danger: #ef233c;
            --text: #2b2d42;
            --text-light: #8d99ae;
            --bg: #ffffff;
            --card-bg: #f8f9fa;
            --card-hover: #e9ecef;
            --card-border: #e2e8f0;
            --sidebar-bg: #f8f9fa;
            --disclaimer-bg: #e9f5ff;
            --disclaimer-text: #1e40af;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 1rem;
            --transition: all 0.2s ease-in-out;
        }

        /* Dark Mode */
        @media (prefers-color-scheme: dark) {
            :root {
                --primary: #4cc9f0;
                --bg: #0f172a;
                --card: #1e293b;
                --text: #f8fafc;
                --border: #334155;
                --muted: #94a3b8;
                --shadow: rgba(0, 0, 0, 0.3);
            }
            
            .stApp {
                background: var(--bg) !important;
                color: var(--text) !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: var(--card) !important;
                color: var(--text) !important;
            }
            
            .stTabs [aria-selected="true"] {
                background: var(--primary) !important;
                color: white !important;
            }
        }

        /* Base Styles */
        html, body, .stApp {
            background: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
            transition: background-color 0.3s, color 0.3s;
        }

        /* Main Container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            font-weight: 700;
            margin-top: 0;
            line-height: 1.2;
        }

        h1 { font-size: 2.5rem; }
        h2 { font-size: 2rem; }
        h3 { font-size: 1.5rem; }
        h4 { font-size: 1.25rem; }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: var(--transition) !important;
            box-shadow: var(--shadow-sm) !important;
            width: 100%;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-md) !important;
            background: linear-gradient(135deg, var(--primary-light), var(--primary)) !important;
        }

        /* Form Elements */
        .stSelectbox > div > div,
        .stDateInput > div > div > input,
        .stTimeInput > div > div > input,
        .stSlider > div > div > div > div > div > div {
            background-color: var(--card) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            transition: var(--transition);
            box-shadow: var(--shadow-sm) !important;
        }

        .stSelectbox > div > div:hover,
        .stDateInput > div > div > input:hover,
        .stTimeInput > div > div > input:hover {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 1px var(--primary) !important;
        }

        .stSelectbox > div > div > div,
        .stDateInput > div > div > input::placeholder,
        .stTimeInput > div > div > input::placeholder {
            color: var(--text) !important;
            opacity: 0.8;
        }

        /* Cards */
        .card {
            background: var(--card);
            border-radius: var(--radius-md);
            padding: 1.5rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border);
            transition: var(--transition);
            height: 100%;
        }

        .card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-md);
        }

        .card-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Prediction Box */
        .prediction-box {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            padding: 2.5rem 2rem;
            border-radius: var(--radius-lg);
            color: white;
            text-align: center;
            box-shadow: var(--shadow-lg);
            margin: 1.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: var(--transition);
        }

        .prediction-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.25);
        }

        .prediction-value {
            font-size: 4.5rem;
            font-weight: 800;
            line-height: 1;
            margin: 0.5rem 0;
            background: -webkit-linear-gradient(white, #e0e0e0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .prediction-label {
            font-size: 1.25rem;
            opacity: 0.9;
            margin-bottom: 1rem;
        }

        /* Sidebar */
        .css-1d391kg {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--card-border);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 1.5rem;
            align-items: center;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--primary);
            color: white !important;
            border-radius: var(--radius-sm);
        }

        /* Footer */
        .footer {
            text-align: center;
            color: var(--text-light);
            padding: 2rem 1rem;
            margin-top: 4rem;
            border-top: 1px solid var(--card-border);
            font-size: 0.9rem;
            background: var(--card-bg);
            border-radius: 0 0 var(--radius-lg) var(--radius-lg);
        }

        /* Responsive Adjustments */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }
            
            h1 { font-size: 2rem; }
            h2 { font-size: 1.75rem; }
            h3 { font-size: 1.5rem; }
            
            .prediction-value {
                font-size: 3.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# ====================================================
# 3. HELPER FUNCTIONS
# ====================================================
def generate_time_options():
    """Generate time options from 12:00 to 23:45 in 15-minute intervals."""
    times = []
    current_time = datetime.strptime("12:00", "%H:%M")
    end_time = datetime.strptime("23:59", "%H:%M")
    
    while current_time <= end_time:
        times.append(current_time.strftime("%H:%M"))
        current_time += timedelta(minutes=15)
    
    # Add midnight
    times.append("00:00")
    return times

def get_day_type(fecha):
    """Get day type (weekday/weekend) and day name in Spanish."""
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dia_num = fecha.weekday()
    es_fin_semana = dia_num >= 5
    tipo_dia = "Fin de semana" if es_fin_semana else "Día laborable"
    return {
        'nombre': dias_semana[dia_num],
        'tipo': tipo_dia,
        'es_fin_semana': es_fin_semana
    }

def get_time_of_day(hora_str):
    """Get time of day category based on hour."""
    hora = int(hora_str.split(':')[0])
    if 10 <= hora < 12:
        return "Apertura", "🟢"
    elif (12 <= hora <= 16) or (hora == 0 and hora_str == "00:00"):
        return "Hora Pico", "🔴"
    else:
        return "Hora Valle", "🟡"

def get_weather_icon(code):
    """Get weather icon and description based on weather code."""
    weather_info = {
        1: ("☀️", "Soleado", "Excelente", "#f59e0b"),
        2: ("⛅", "Parcialmente nublado", "Bueno", "#60a5fa"),
        3: ("☁️", "Nublado", "Normal", "#94a3b8"),
        4: ("🌧️", "Lluvia ligera", "Malo", "#60a5fa"),
        5: ("⛈️", "Tormenta", "Muy malo", "#7c3aed")
    }
    return weather_info.get(code, ("❓", "Desconocido", "", "#94a3b8"))

def create_prediction_card(minutes, attraction):
    """Create a prediction card with appropriate styling."""
    if minutes < 15:
        color_primary = "#10b981"  # Verde
        color_secondary = "#34d399"
        emoji = "🟢"
        nivel = "Bajo"
        mensaje = "¡Excelente momento para disfrutar!"
    elif minutes < 30:
        color_primary = "#f59e0b"  # Amarillo
        color_secondary = "#fbbf24"
        emoji = "🟡"
        nivel = "Moderado"
        mensaje = "Buen momento para visitar la atracción"
    elif minutes < 60:
        color_primary = "#f97316"  # Naranja
        color_secondary = "#fb923c"
        emoji = "🟠"
        nivel = "Alto"
        mensaje = "Tiempo de espera elevado, considera volver más tarde"
    else:
        color_primary = "#ef4444"  # Rojo
        color_secondary = "#f87171"
        emoji = "🔴"
        nivel = "Muy Alto"
        mensaje = "Tiempo de espera muy elevado, mejor elegir otra atracción"
    
    return f"""
    <div class="prediction-box" style="background: linear-gradient(135deg, {color_primary}, {color_secondary});">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div style="font-size: 1.25rem; background: rgba(255,255,255,0.2); padding: 0.5rem 0.75rem; border-radius: 50px; display: inline-flex; align-items: center; gap: 0.5rem;">
                <span>{emoji}</span>
                <span style="font-size: 0.9rem; font-weight: 600;">{nivel}</span>
            </div>
            <div style="font-size: 0.9rem; background: rgba(0,0,0,0.1); padding: 0.5rem 0.75rem; border-radius: 50px;">
                {attraction}
            </div>
        </div>
        
        <div class="prediction-value">
            {minutes:.0f}
            <span style="font-size: 1.5rem; font-weight: 500; opacity: 0.8;">min</span>
        </div>
        
        <div style="font-size: 1.1rem; margin: 0.5rem 0 1rem 0; font-weight: 500;">
            {mensaje}
        </div>
    </div>
    """

# ====================================================
# 4. MAIN APP LAYOUT
# ====================================================
def main():
    # Load custom CSS
    load_css()
    
    # Sample data (replace with your actual data)
    atracciones = [
        "Batman: The Ride", "Superman: La Atracción de Acero", 
         "Stunt Fall"
    ]
    zonas = ["Gotham City", "Metrópolis", "Mediterránea", "Cine Exprés", "DC Super Heros"]
    
    # Main container for controls
    st.markdown("### ⚙️ Configuración de la predicción")
    st.markdown("Ajusta los parámetros para obtener una predicción precisa.")
    
    # Create columns for better organization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Attraction selection
        atraccion_seleccionada = st.selectbox(
            "🎯 Atracción",
            options=atracciones,
            index=0,
            help="Selecciona la atracción que deseas consultar"
        )
        
        # Auto-detect zone
        zona_auto = get_zone_for_attraction(atraccion_seleccionada)
        if zona_auto:
            st.info(f"📍 Zona: **{zona_auto}**")
            
        # Date selection
        fecha_seleccionada = st.date_input(
            "📅 Fecha de visita",
            value=date.today(),
            min_value=date.today(),
            format="DD/MM/YYYY"
        )
        
        # Day info
        dia_semana_es = {
            "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
            "Thursday": "Jueves", "Friday": "Viernes", 
            "Saturday": "Sábado", "Sunday": "Domingo"
        }
        dia_nombre = fecha_seleccionada.strftime("%A")
        es_fin_semana = fecha_seleccionada.weekday() >= 5
        st.info(f"📆 {dia_semana_es.get(dia_nombre, dia_nombre)} — "
                f"{'Fin de semana' if es_fin_semana else 'Día laborable'}")
    
    with col2:
        # Time selection
        hora_seleccionada = st.time_input(
            "🕒 Hora de visita",
            value=time(14, 0),  # Default to 2 PM
            step=timedelta(minutes=15)
        )
        
        # Time of day classification with better styling
        hora_int = hora_seleccionada.hour
        if 12 <= hora_int <= 16:
            tipo_hora = "🔴 Hora Pico"
            time_style = "background: rgba(239, 68, 68, 0.1); padding: 8px; border-radius: 8px;"
        elif 10 <= hora_int < 12:
            tipo_hora = "🟡 Hora Media"
            time_style = "background: rgba(245, 158, 11, 0.1); padding: 8px; border-radius: 8px;"
        else:
            tipo_hora = "🟢 Hora Tranquila"
            time_style = "background: rgba(16, 185, 129, 0.1); padding: 8px; border-radius: 8px;"
            
        st.markdown(f"<div style='{time_style}'>{tipo_hora}</div>", unsafe_allow_html=True)
        
        # Weather settings
        st.markdown("#### 🌤️ Condiciones meteorológicas")
        temperatura, humedad, sensacion_termica, codigo_clima = render_weather_controls()
    
    # Prediction button - full width
    predecir = st.button(
        "🚀 Calcular tiempo de espera",
        type="primary",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Render hero section
    render_hero()

    # Disclaimer with improved dark mode support
    st.markdown("""
    <div style="background: var(--card); border-left: 4px solid var(--warning); 
                padding: 1rem; border-radius: 8px; margin: 1rem 0 2rem 0;
                box-shadow: 0 2px 8px var(--shadow);">
        <strong>⚠️ Aviso:</strong> Esta aplicación es independiente y educativa. 
        No está afiliada a Parque Warner.
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 class="main-header">Afluencia Tiempos</h1>
            <div style="font-size: 2.5rem; font-weight: 700; color: var(--primary); margin: -0.5rem 0 0.5rem 0;">
                Parque Warner Madrid
            </div>
            <p class="sub-header">Predicción inteligente de tiempos de espera en atracciones</p>
        </div>
        """,
        unsafe_append_html=True
    )
    
    # Mostrar predicción si se ha hecho clic en el botón
    if predecir:
        # Simular predicción (reemplazar con tu lógica de predicción real)
        import random
        minutos_predichos = random.uniform(5, 120)
        
        # Mostrar tarjeta de predicción
        st.markdown(
            create_prediction_card(minutos_predichos, atraccion_seleccionada),
            unsafe_append_html=True
        )
        
        # Sección de métricas
        st.markdown("<h3 style='font-size: 1.5rem; margin: 2rem 0 1rem 0;'>📊 Métricas de Predicción</h3>", 
                   unsafe_append_html=True)
        
        # Métricas simuladas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="card">
                    <div class="card-header">📊 Predicción Base</div>
                    <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0; color: var(--primary);">
                        {:.1f} min
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-light);">
                        Modelo base sin ajustes
                    </div>
                </div>
            """.format(minutos_predichos * 0.9), unsafe_append_html=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <div class="card-header">📈 P75 Histórico</div>
                    <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0; color: var(--secondary);">
                        {:.1f} min
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-light);">
                        Percentil 75 histórico
                    </div>
                </div>
            """.format(minutos_predichos * 1.1), unsafe_append_html=True)
        
        with col3:
            st.markdown("""
                <div class="card">
                    <div class="card-header">📉 Mediana Histórica</div>
                    <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0; color: #f59e0b;">
                        {:.1f} min
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-light);">
                        Mediana de tiempos históricos
                    </div>
                </div>
            """.format(minutos_predichos * 0.8), unsafe_append_html=True)
        
        with col4:
            st.markdown("""
                <div class="card">
                    <div class="card-header">🎯 Nivel de Confianza</div>
                    <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0; color: #10b981;">
                        Alto
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-light);">
                        Precisión estimada: 85-90%
                    </div>
                </div>
            """, unsafe_append_html=True)
        
        # Gráfico de comparación
        st.markdown("<h3 style='font-size: 1.5rem; margin: 2rem 0 1rem 0;'>📈 Comparación de Predicciones</h3>", 
                   unsafe_append_html=True)
        
        # Datos para el gráfico
        datos_grafico = {
            'Tipo': ['Predicción Actual', 'Promedio Histórico', 'Máximo Histórico', 'Mínimo Histórico'],
            'Minutos': [minutos_predichos, minutos_predichos * 0.8, minutos_predichos * 1.3, minutos_predichos * 0.6],
            'Color': ['#4361ee', '#4cc9f0', '#f72585', '#10b981']
        }
        
        fig = px.bar(
            datos_grafico, 
            x='Tipo', 
            y='Minutos',
            color='Tipo',
            color_discrete_map=dict(zip(datos_grafico['Tipo'], datos_grafico['Color'])),
            text=[f"{x:.1f} min" for x in datos_grafico['Minutos']],
            height=400
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis_title="Minutos de Espera",
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor='var(--card-border)'),
            xaxis=dict(tickfont=dict(size=12)),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        fig.update_traces(
            textposition='outside',
            textfont_size=12,
            textfont_color='var(--text)',
            marker_line_color='rgba(0,0,0,0)',
            width=0.6
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recomendaciones
        st.markdown("<h3 style='font-size: 1.5rem; margin: 2rem 0 1rem 0;'>💡 Recomendaciones</h3>", 
                   unsafe_append_html=True)
        
        recomendaciones = []
        
        if minutos_predichos < 15:
            recomendaciones.append("✅ **Excelente momento** para visitar esta atracción. Los tiempos de espera son muy bajos.")
            recomendaciones.append("🎢 Aprovecha para montar varias veces seguidas si lo deseas.")
        elif minutos_predichos < 30:
            recomendaciones.append("👍 **Buen momento** para visitar esta atracción. Los tiempos de espera son razonables.")
            recomendaciones.append("⏱️ Considera usar el acceso rápido si está disponible.")
        elif minutos_predichos < 60:
            recomendaciones.append("⚠️ **Tiempo de espera elevado**. Considera volver más tarde si no quieres esperar tanto.")
            recomendaciones.append("🔄 Intenta visitar la atracción durante las primeras o últimas horas del parque.")
        else:
            recomendaciones.append("⛔ **Tiempo de espera muy alto**. Podrías considerar otra atracción por ahora.")
            recomendaciones.append("📱 Revisa la aplicación oficial del parque para tiempos de espera en tiempo real.")
        
        if dia_semana_es.get(dia_nombre, dia_nombre) in ["Sábado", "Domingo"]:
            recomendaciones.append("📅 **Fin de semana**: Los tiempos de espera suelen ser más altos. Considera visitar en día laborable si es posible.")
        
        if tipo_hora == "Hora Pico":
            recomendaciones.append("⏰ **Hora pico**: Los tiempos de espera son mayores. Intenta visitar la atracción por la mañana temprano o al final del día.")
        
        # Mostrar recomendaciones en tarjetas
        for i, rec in enumerate(recomendaciones):
            st.markdown(f"""
                <div class="card" style="margin-bottom: 0.75rem;">
                    <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
                        <div style="font-size: 1.25rem;">{rec.split(' ')[0]}</div>
                        <div style="flex: 1;">
                            {' '.join(rec.split(' ')[1:])}
                        </div>
                    </div>
                </div>
            """, unsafe_append_html=True)
    else:
        # Mensaje inicial
        st.markdown("""
            <div class="card" style="text-align: center; padding: 3rem 2rem; margin: 2rem 0;">
                <div style="font-size: 1.5rem; margin-bottom: 1rem;">👋 ¡Bienvenido a ParkBeat!</div>
                <p style="color: var(--text-light); margin-bottom: 1.5rem;">
                    Configura los parámetros en el panel lateral y haz clic en 
                    <strong>"Predecir Tiempo de Espera"</strong> para obtener una estimación 
                    del tiempo de espera en las atracciones del Parque Warner Madrid.
                </p>
                <div style="font-size: 2rem;">⬅️</div>
            </div>
        """, unsafe_append_html=True)
        
        # Sección de información
        st.markdown("<h3 style='font-size: 1.5rem; margin: 2rem 0 1rem 0;'>ℹ️ Sobre ParkBeat</h3>", 
                   unsafe_append_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="card">
                    <div class="card-header">🎯 Características Principales</div>
                    <ul style="margin: 0.5rem 0 0 1rem; padding-left: 1rem; color: var(--text-light);">
                        <li>Predicción basada en Machine Learning (XGBoost)</li>
                        <li>Considera día de semana, mes y hora</li>
                        <li>Incluye condiciones climáticas</li>
                        <li>Usa históricos granulares por atracción</li>
                        <li>Detecta eventos especiales</li>
                    </ul>
                </div>
            """, unsafe_append_html=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <div class="card-header">📊 Factores Considerados</div>
                    <ul style="margin: 0.5rem 0 0 1rem; padding-left: 1rem; color: var(--text-light);">
                        <li><strong>📅 Temporal</strong>: Día, mes, hora, día de la semana</li>
                        <li><strong>🎢 Atracción</strong>: Popularidad, tipo, ubicación</li>
                        <li><strong>🌤️ Clima</strong>: Temperatura, humedad, condiciones</li>
                        <li><strong>📈 Históricos</strong>: Patrones de comportamiento</li>
                        <li><strong>🎉 Eventos</strong>: Fines de semana, festivos</li>
                    </ul>
                </div>
            """, unsafe_append_html=True)
    
    # Footer with improved styling
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--muted); padding: 1.5rem 0; margin-top: 3rem;">
        🎢 <strong>ParkBeat</strong> — Predicción de tiempos de espera en tiempo real<br>
        <small>Desarrollado con ❤️ por Sergio López | v2.1</small>
    </div>
                    <span>📊 Estadísticas</span>
                </a>
                <a href="#" style="color: var(--primary); text-decoration: none; display: flex; align-items: center; gap: 0.5rem;">
                    <span>📧 Contacto</span>
                </a>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-light);">
                🎢 <strong>ParkBeat</strong> - Sistema de Predicción de Afluencia | 
                <span style="opacity: 0.8;">Hecho con ❤️ por Sergio López</span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-light); opacity: 0.7;">
                © 2023 ParkBeat. No afiliado a Parque Warner Madrid. Los datos son estimaciones basadas en modelos predictivos.
            </div>
        </div>
    """, unsafe_append_html=True)

if __name__ == "__main__":
    main()
