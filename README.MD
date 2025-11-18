# 🌟 **PARKLYTICS — Analítica, IA y Machine Learning aplicados al Parque Warner Madrid**
### *Una plataforma completa de Data Engineering, Data Science y MLOps diseñada desde cero por un solo desarrollador.*

---

## 🧠 **1. Introducción — La historia detrás del proyecto**

Soy un gran aficionado a las montañas rusas y parques temáticos y, a finales de 2024, mientras visitaba el **Parque Warner Madrid**, observé algo que millones de personas experimentan cada día:

- Colas impredecibles  
- Tiempos de espera confusos  
- Falta de información en tiempo real  

Como data scientist en formación, me hice una pregunta muy simple:

> **¿Y si pudiéramos predecir las colas antes de que existan?  
> ¿Y si la experiencia completa de un parque pudiera ser optimizada usando datos?**

Ese fue el nacimiento de **Parklytics**, un proyecto personal que ha evolucionado hasta convertirse en un sistema avanzado con:

✔ Recolección automática de datos reales del parque  
✔ Data engineering y limpieza avanzada  
✔ Feature Engineering completo  
✔ Modelos de ML y XGBoost  
✔ Predicción exacta de tiempos de espera  
✔ Dashboard y API (en proceso)  

Este README cuenta el viaje técnico, las decisiones, los retos y cómo estoy construyendo la plataforma **de principio a fin**.

---

## 🎢 **2. ¿Qué es Parklytics?**

**Parklytics** es un sistema de analítica y predicción desarrollado para entender el comportamiento y flujo de los visitantes del Parque Warner Madrid.

### 🔍 ¿Qué resuelve?

- Predicción de tiempos de espera por atracción  
- Detección de patrones según clima, día, hora y temporada  
- Dashboards avanzados para análisis  

### 🧩 Tecnologías clave

- **Python** (Pandas, NumPy, scikit-learn, XGBoost)  
- **APIs** (Queue-Times y OpenMeteo)  
- **Machine Learning** (Regresión predictiva)  
- **EDA y Feature Engineering**  
- **MLOps básico** (model, scaler, pipeline)  
- **Git & GitHub**

---

## 🔥 **3. Recolección de datos reales (APIs)**

Para capturar datos del Parque Warner utilicé la API de **Queue-Times** y los datos climáticos de **OpenMeteo**, realizando requests automáticas cada pocos minutos:

- Temperatura  
- Humedad  
- Sensación térmica  
- Código del clima  
- Tiempo de espera de cada atracción  
- Fecha y hora exacta  
- Estado (abierta/cerrada)  
- Zona temática  

Gracias a esto, construí un dataset propio totalmente original.  
No descargado, no simulado: **datos reales en producción**.

---

## 🧼 **4. Data Engineering & Feature Engineering**

La transformación de datos incluyó:

✔ Manejo de valores nulos  
✔ Limpieza de formatos  
✔ Conversión de fechas a features útiles  
✔ `hora_num`, `dia_semana_num`, `fin_de_semana`, `temporada`  
✔ **Ordinal encoding** para *alta, media, baja*  
✔ **One-hot encoding** para zonas y atracciones  
✔ **Escalado estandarizado**  

El resultado final es un dataset robusto, escalable y totalmente listo para modelos de Machine Learning.

---

## 🤖 **5. Machine Learning: Predicción Exacta de Tiempos de Espera**

El modelo principal utilizado es:

### ⭐ **XGBoost Regressor (mejor rendimiento global)**

Entrenado con:

- División 80/20  
- StandardScaler  
- Feature set robusto  
- **1430 registros reales tras limpieza** (creciendo cada día)

El modelo predice el **tiempo de espera real en minutos** para cualquier atracción, teniendo en cuenta múltiples factores.

---

## 📈 **6. EDA — Insights reales del parque**

Durante el análisis exploratorio surgieron insights interesantes:

- Las horas con mayor afluencia real  
- Cómo afecta el clima al tiempo de espera  
- Qué atracciones tienen mayor variabilidad  
- Zonas que acumulan más cola  
- Patrones de comportamiento según día/hora  

---

## 🌐 **7. Resultados globales del proyecto**

✔ Modelo predictivo real funcionando  
✔ Dataset profesional creado desde cero  
✔ Pipeline de ML completo  
✔ Comportamiento de visitantes analizado  
✔ Patrón de gustos y hábitos detectados  

---

## 👨‍💻 **8. Sobre mí**

Soy **Sergio López**, Data Scientist en formación y desarrollador de software.  
Este proyecto ha sido construido completamente por mí, aplicando todo lo que sé sobre:

- Python  
- Data Engineering  
- Machine Learning  
- Análisis de datos  
- Modelado predictivo  
- Git  
- Pipelines y buenas prácticas  

---

## 🚀 **9. Próximos pasos**

- Dockerización  
- API con FastAPI  
- Dashboard interactivo para usuarios (Streamlit o Dash)  
- Monitorización de modelo  
- Nuevos datasets y features avanzadas  

---

## 🏁 **Gracias por leer Parklytics**

Si te ha gustado, una estrella ⭐ en el repositorio me ayuda muchísimo.

### 📩 Contacto  
**Email:** slopezmar3@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/sergio-lopez-dev/
