# 🌍 Proyecto ParkLytics – Visión General

## 🎯 Visión
ParkLytics es una plataforma de analítica inteligente diseñada para **monitorizar, analizar y predecir la afluencia y los tiempos de espera** en parques temáticos.  
El objetivo es ayudar a la **dirección de Parques Reunidos / Parque Warner Madrid** a optimizar la gestión de recursos, mejorar la experiencia del visitante y tomar decisiones basadas en datos en tiempo real y predictivos.

---

## 💡 Contexto
En la actualidad, los parques temáticos cuentan con información sobre ventas de entradas y accesos, pero no disponen de un **modelo analítico predictivo y explicativo** que:
- Combine datos de afluencia real, tiempos de espera y clima.
- Prediga picos de ocupación con antelación.
- Permita entender **por qué** se producen esos patrones.
- Ofrezca simulaciones de escenarios para la planificación de personal, mantenimiento o marketing.

ParkLytics aborda este vacío con una solución de **Big Data y Machine Learning explicable (XAI)**, fácil de integrar con sistemas ya existentes.

---

## ⚙️ Funcionalidades principales (visión a medio plazo)
1. **Ingesta de datos** desde fuentes públicas (APIs y datasets históricos):
   - API de tiempos de espera: [Queue-Times](https://queue-times.com/parks/298/queue_times.json)
   - API meteorológica (Open-Meteo u otra similar)
   - Datos propios del parque (ficticios o anonimizados para pruebas)

2. **Procesamiento en Spark**:
   - Limpieza, estructuración y consolidación de datos masivos.
   - Generación de métricas agregadas (afluencia media, ocupación, tiempos medios, etc.)

3. **Análisis y visualización de patrones**:
   - Detección de picos de afluencia por hora/día/atracción.
   - Correlación clima ↔ afluencia.

4. **Modelos predictivos y explicables (ML)**:
   - Predicción de afluencia futura a 7 días vista.
   - Modelos con interpretación (SHAP / Feature importance).

5. **Optimización económica 💰**:
   - Análisis de costes frente a afluencia y meteorología.
   - Recomendaciones de apertura, personal o mantenimiento.

6. **Simulador de estrategias (nivel experto)**:
   - Simulación “¿Qué pasaría si…?” (What-if scenarios).
   - Ejemplo: ¿qué ocurre si se reduce el aforo un 10 % o si llueve un fin de semana?

---

## 🧩 Objetivo general
Desarrollar una plataforma modular de analítica predictiva para parques temáticos, capaz de:
- Integrar datos históricos y en tiempo real.
- Predecir afluencia y tiempos de espera.
- Explicar los factores que influyen en esos patrones.
- Optimizar la toma de decisiones operativas y económicas.

---

## 🧱 Alcance inicial (Fase 0–2)
- Fase 0: Preparación del entorno, definición del contexto y KPIs.
- Fase 1: Estructura del proyecto, entorno PySpark y pipeline inicial.
- Fase 2: Ingesta de datos desde Queue-Times y API meteorológica + limpieza.

---

## 🔍 Impacto esperado
- **Visión estratégica:** mejora de la previsión operativa y planificación de personal.
- **Experiencia del visitante:** reducción de tiempos de espera.
- **Sostenibilidad económica:** decisiones basadas en coste y clima.
- **Innovación:** uso de analítica avanzada en un entorno real de ocio y turismo.

---

## 🏁 Entregables finales
- Plataforma local de análisis en PySpark.
- Conjunto de notebooks de análisis exploratorio (EDA).
- Modelos predictivos con interpretación (SHAP / Feature importance).
- Dashboard o notebook interactivo para simulación y toma de decisiones.

---

## 👨‍💻 Tecnologías base
| Categoría | Tecnología |
|------------|-------------|
| Lenguaje | Python 3.11 |
| Framework Big Data | Apache Spark (PySpark) |
| Almacenamiento local | CSV / Parquet |
| APIs | Queue-Times, Open-Meteo |
| ML / XAI | scikit-learn, shap |
| Visualización | matplotlib, seaborn, Plotly |
| Control de versiones | Git |
| Documentación | Markdown / Jupyter |

---

## 📅 Duración estimada
| Fase | Duración estimada |
|------|--------------------|
| Fase 0 – Preparación y contexto | 2 días |
| Fase 1 – Setup estructural | 2–3 días |
| Fase 2 – Ingesta y limpieza de datos | 4–6 días |
| Fase 4 – ML predictivo y simulador | 7–10 días |
| **Total estimado:** | **≈ 15–20 días hábiles** |

---

## 🚀 Resultado final esperado
Un prototipo funcional de analítica predictiva para parques temáticos, capaz de:
- Predecir la afluencia con precisión razonable.
- Mostrar los factores que más la afectan.
- Simular decisiones operativas.
- Servir como base para una propuesta real a Parques Reunidos.

