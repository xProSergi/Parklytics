import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import plotly.graph_objects as go
from predict import load_model_artifacts, predict_wait_time
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# Configuración de página
# -----------------------
st.set_page_config(
    page_title="Parklytics — Predicción Parque Warner",
    page_icon="img/logoParklytics.png",
    layout="wide"
)

# -----------------------
# CSS MODERNO + HERO
# -----------------------
st.markdown("""
<style>

:root{
    --primary: #2b6ef6;
    --accent: #6c63ff;
    --muted: #6b7280;
    --bg: #ffffff;
    --text: #111827;
    --card: #f8fafc;
    --border: #e6e9ee;
}

html, body {
    background: var(--bg) !important;
    color: var(--text);
    font-family: Inter, system-ui;
}

/* Hero */
.hero-container {
    position: relative;
    width: 100%;
    height: 340px;
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 2rem;
}

.hero-img {
    width: 100%;
    height: 340px;
    object-fit: cover;
    filter: brightness(0.55);
}

.hero-title-container {
    position: absolute;
    top: 0;
    left:0;
    width: 100%;
    height: 100%;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    color:white;
    text-align:center;
    padding: 0 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    text-shadow: 0 4px 18px rgba(0,0,0,0.45);
}

.hero-sub {
    margin-top: 0.6rem;
    font-size: 1.15rem;
    text-shadow: 0 4px 18px rgba(0,0,0,0.35);
}

/* Prediction box */
.prediction-box {
    border-radius: 14px;
    padding: 1.3rem;
    color: #fff;
    box-shadow: 0 12px 30px rgba(16,24,40,0.12);
    text-align: center;
}
.prediction-value { font-size: 3.3rem; font-weight: 800; }
.prediction-label { font-size: 1.1rem; opacity: 0.95; margin-top: 0.25rem; }

/* info cards */
.info-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: .7rem;
}

/* Sidebar buttons */
.stButton>button {
    background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    color: white !important;
    border: none !important;
    padding: 0.65rem 1.2rem !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

.disclaimer {
    background: linear-gradient(180deg, #fff8e6, #fffdf6);
    color: #5c3d00;
    padding: 1rem;
    border-radius: 12px;
    border-left: 4px solid #ffc107;
}

/* Footer */
.footer {
    color: var(--muted);
    text-align: center;
    padding: 1rem 0;
    margin-top: 1.5rem;
    border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# HERO (imagen full-width)
# -----------------------
st.markdown("""
<div class="hero-container">
    <img src="img/fotoBatman.jpg" class="hero-img"/>
    <div class="hero-title-container">
        <div class="hero-title">Parklytics</div>
        <div class="hero-sub">Predicción inteligente de tiempos de espera — Parque Warner</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Aviso:</strong> Esta aplicación es independiente y educativa. No está afiliada a Parque Warner.
</div>
""", unsafe_allow_html=True)


# -----------------------
# CARGO MODELO
# -----------------------
artifacts = load_model_artifacts()
if not artifacts or "error" in artifacts:
    st.error("❌ Error cargando modelo.")
    st.stop()

df = artifacts["df_processed"]

# -----------------------
# Helpers cacheados
# -----------------------
@st.cache_data
def get_attractions():
    return sorted(df["atraccion"].dropna().unique().tolist())

@st.cache_data
def get_zones():
    return sorted(df["zona"].dropna().unique().tolist())

def get_zone_for_attraction(atr):
    row = df[df["atraccion"] == atr]
    return row["zona"].iloc[0] if not row.empty else ""

atracciones = get_attractions()
zonas = get_zones()

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption("Ajusta los parámetros para obtener una predicción optimizada.")
    st.markdown("---")

    # Atracción
    atraccion_seleccionada = st.selectbox(
        "🎯 Atracción",
        options=atracciones,
        index=0
    )

    zona_auto = get_zone_for_attraction(atraccion_seleccionada)

    st.markdown("---")

    # Fecha
    fecha_seleccionada = st.date_input(
        "📅 Fecha",
        value=date.today(),
        min_value=date.today()
    )

    # Hora (12:00 → 00:00)
    def generate_time_options():
        t = datetime.strptime("12:00", "%H:%M")
        end = datetime.strptime("23:59", "%H:%M")
        times = []
        while t <= end:
            times.append(t.strftime("%H:%M"))
            t += timedelta(minutes=15)
        times.append("00:00")
        return times

    hora_str = st.selectbox(
        "🕐 Hora (12:00 – 00:00)",
        generate_time_options(),
        index=0
    )
    hora_seleccionada = datetime.strptime(hora_str, "%H:%M").time()

    # Clima
    st.markdown("---")
    st.subheader("🌤️ Clima")

    temperatura = st.slider("Temperatura (°C)", -5, 45, 22)
    humedad = st.slider("Humedad (%)", 0, 100, 60)
    sensacion = st.slider("Sensación térmica (°C)", -10, 50, temperatura)

    codigo_clima = st.selectbox(
        "Condición",
        options=[1,2,3,4,5],
        format_func=lambda x: {
            1: "☀️ Soleado",
            2: "⛅ Parcial",
            3: "☁️ Nublado",
            4: "🌧️ Lluvia ligera",
            5: "⛈️ Fuerte/tormenta"
        }[x]
    )

    st.markdown("---")
    predecir = st.button("🚀 Predecir Tiempo de Espera")

# -----------------------
# PREDICCIÓN
# -----------------------
if predecir:
    hora_final = f"{hora_seleccionada.hour:02d}:{hora_seleccionada.minute:02d}:00"
    fecha_final = fecha_seleccionada.strftime("%Y-%m-%d")

    entrada = {
        "atraccion": atraccion_seleccionada,
        "zona": zona_auto,
        "fecha": fecha_final,
        "hora": hora_final,
        "temperatura": temperatura,
        "humedad": humedad,
        "sensacion_termica": sensacion,
        "codigo_clima": codigo_clima
    }

    with st.spinner("🔮 Calculando predicción..."):
        resultado = predict_wait_time(entrada, artifacts)

    minutos = resultado.get("minutos_predichos", 0)

    # Estilo según nivel
    if minutos < 15:
        grad = "linear-gradient(135deg, #16a085, #2ecc71)"
        emoji, nivel = "🟢", "Bajo"
    elif minutos < 30:
        grad = "linear-gradient(135deg, #f6d365, #fda085)"
        emoji, nivel = "🟡", "Moderado"
    elif minutos < 60:
        grad = "linear-gradient(135deg, #f7971e, #ffd200)"
        emoji, nivel = "🟠", "Alto"
    else:
        grad = "linear-gradient(135deg, #ff416c, #ff4b2b)"
        emoji, nivel = "🔴", "Muy Alto"

    # RESULTADO CENTRAL
    colA, colB, colC = st.columns([1,2,1])
    with colB:
        st.markdown(f"""
        <div class="prediction-box" style="background:{grad}">
            <div class="prediction-label">{emoji} Tiempo de Espera Predicho</div>
            <div class="prediction-value">{minutos:.1f} min</div>
            <div class="prediction-label">{nivel} — {atraccion_seleccionada}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # MÉTRICAS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicción Base", f"{resultado['prediccion_base']:.1f} min")
    m2.metric("P75 Histórico", f"{resultado['p75_historico']:.1f} min")
    m3.metric("Mediana", f"{resultado['median_historico']:.1f} min")
    m4.metric("Especificidad", resultado['especificidad_historico'])

    st.markdown("---")

    # INFO Y CONTEXTO
    ic1, ic2 = st.columns(2)
    with ic1:
        st.subheader("ℹ️ Información")
        st.markdown(f"""
        <div class='info-box'><strong>Hora:</strong> {hora_str}</div>
        <div class='info-box'><strong>Día semana:</strong> {resultado['dia_semana']}</div>
        <div class='info-box'><strong>Día mes:</strong> {resultado['dia_mes']}</div>
        <div class='info-box'><strong>Muestra histórica:</strong> {resultado['count_historico']} registros</div>
        """, unsafe_allow_html=True)

    with ic2:
        st.subheader("🔍 Contexto")
        for label, key in [
            ("Fin de semana", "es_fin_de_semana"),
            ("Evento Batman octubre", "es_batman_octubre"),
            ("Es puente", "es_puente"),
            ("Hora apertura", "es_hora_apertura"),
            ("Hora pico", "es_hora_pico"),
            ("Hora valle", "es_hora_valle")
        ]:
            val = "Sí" if resultado.get(key) else "No"
            color = "#16a085" if val == "Sí" else "#6b7280"
            st.markdown(
                f"<div class='info-box'><strong>{label}:</strong> "
                f"<span style='color:{color}'>{val}</span></div>",
                unsafe_allow_html=True
            )

    # GRÁFICO
    st.markdown("---")
    st.subheader("📊 Comparación de predicciones")
    vals = {
        "Final": minutos,
        "Modelo Base": resultado['prediccion_base'],
        "P75": resultado['p75_historico'],
        "Mediana": resultado['median_historico'],
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(vals.keys()),
        y=list(vals.values()),
        text=[f"{v:.1f} min" for v in vals.values()],
        textposition="auto"
    ))

    fig.update_layout(
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # RECOMENDACIONES
    st.markdown("---")
    st.subheader("💡 Recomendaciones")

    recs = []
    if minutos < 15:
        recs.append("✅ **Excelente momento** para subir ahora.")
    elif minutos < 30:
        recs.append("👍 **Tiempo moderado**, buen momento.")
    elif minutos < 60:
        recs.append("⚠️ **Alto**, planifica o usa acceso rápido.")
    else:
        recs.append("🚫 **Muy alto**, cambia de hora o atracción.")

    if resultado['es_hora_pico']:
        recs.append("⏰ Hora pico detectada: evita 11:00–16:00.")
    if resultado['es_batman_octubre']:
        recs.append("🎃 Octubre incrementa afluencia en Batman.")

    for r in recs:
        st.markdown(f"<div class='info-box'>{r}</div>", unsafe_allow_html=True)

# -----------------------
# FOOTER
# -----------------------
st.markdown("""
<div class="footer">
    🎢 Parklytics — Predicción de afluencias | Hecho por Sergio López
</div>
""", unsafe_allow_html=True)
