🚀 Roadmap completo — Parklytics Warner Madrid (versión pública y realizable)
🎯 Objetivo general

Desarrollar un sistema de analítica predictiva y simulación para Parque Warner Madrid, utilizando datos reales de:

Tiempos de espera (Queue-Times API)

Clima (Open-Meteo API)

Variables temporales (día, mes, festivos, etc.)

El sistema predecirá afluencia y esperas futuras, identificará patrones y permitirá simular escenarios “¿qué pasaría si…?” con Machine Learning explicable.

🧩 FASE 0 – Preparación del entorno (⏱ 1-2 días)
🎯 Objetivo

Configurar un entorno profesional sin rutas de Users ni Documents, completamente limpio y controlado.

🔧 Instalaciones necesarias
Herramienta	Versión recomendada	Descripción
Python	3.10 – 3.12	PySpark no soporta 3.14 aún
Java JDK	8 o 11	Requerido para Spark
Apache Spark	3.5.1 o 4.0.1 (Hadoop 3)	Motor de procesamiento
winutils.exe	✔️ Ya instalado	Necesario en Windows
VSCode o PyCharm	Última	Editor
venv	-	Entorno virtual
Librerías	pandas, pyspark, matplotlib, seaborn, requests, scikit-learn, xgboost, shap, dash	Requeridas
📁 Estructura del proyecto

Todo dentro de C:\Parklytics:

C:\Parklytics\
│
├── data\
│   ├── raw\
│   ├── processed\
│   └── models\
│
├── notebooks\
├── src\
│   ├── ingestion\
│   ├── processing\
│   ├── analytics\
│   ├── simulation\
│   ├── visualization\
│   └── utils\
│
└── requirements.txt

🌦️ FASE 1 – Ingesta y exploración de datos (⏱ 3-5 días)
🎯 Objetivo

Obtener y analizar datos reales de:

https://queue-times.com/parks/298/queue_times.json

API meteorológica (Open-Meteo)

Fechas (semana, mes, festivos, etc.)

🧰 Tecnologías

requests, pandas, json, pyspark.sql, matplotlib

🧩 Tareas

Crear scripts:

queue_data_ingestor.py

weather_data_ingestor.py

Fusión de datasets (por fecha/hora).

Limpieza: valores nulos, formatos, duplicados.

EDA (Exploratory Data Analysis):

Promedio de esperas por día.

Comparación clima ↔️ espera.

Estacionalidad (findes, verano, etc.).

📈 Resultado

Dataset consolidado, limpio y exportado a data/processed/parklytics_data.csv.

🧠 FASE 2 – Modelado predictivo + Machine Learning explicable (⏱ 7-10 días)
🎯 Objetivo

Predecir:

Afluencia esperada (estimada a partir de número de colas activas / esperas)

Tiempos de espera futuros (por atracción o global)

🧰 Tecnologías

scikit-learn, xgboost, pyspark.ml, SHAP

🧩 Tareas

Feature Engineering:

Día de la semana, temperatura, humedad, precipitación, mes, festivo.

Modelos base:

Regresión lineal, Random Forest, XGBoost.

Evaluación:

Métricas MAE, RMSE, R².

Explicabilidad:

Usar SHAP para mostrar qué variables más afectan la espera.

Crear gráficos interpretables.

Guardado de modelos (.joblib en data/models).

📈 Resultado

Modelos listos para usar, interpretables, con informes visuales de rendimiento.

🧩 FASE 3 – Simulador de estrategias (nivel experto) (⏱ 7-10 días)
🎯 Objetivo

Crear un simulador interactivo donde se pueda modificar:

Día / clima / tipo de evento / afluencia esperada
y observar las predicciones de espera y comportamiento del parque.

🧰 Tecnologías

Dash (o Streamlit), Plotly, pandas, joblib

🧩 Tareas

Crear interfaz:

Inputs: temperatura, lluvia, día, tipo (laboral/festivo).

Output: predicción de espera media / afluencia.

Mostrar gráficos dinámicos.

Añadir explicabilidad visual (por qué se predijo eso).

Opción “simulación comparativa”:

“¿Qué pasa si llueve?”

“¿Qué pasa si cambio de sábado a martes?”

📈 Resultado

App local o web que permita probar escenarios y ver su impacto en tiempo real.

📊 FASE 4 – Visualización y presentación (⏱ 5 días)
🎯 Objetivo

Consolidar toda la información en un dashboard final y preparar el proyecto para portfolio o demo.

🧰 Tecnologías

Dash, Plotly, matplotlib, pandas

🧩 Tareas

Crear dashboard con:

📊 Métricas principales (media de espera, clima, predicciones)

🌦️ Clima y afluencia

🤖 Resultados de modelos

🧩 Simulador

Preparar versión presentable.

Documentar:

README completo

Capturas y vídeo corto para LinkedIn

📄 FASE 5 – Documentación final (⏱ 2-3 días)
🎯 Objetivo

Cerrar el proyecto con una presentación profesional y divulgativa.

🧩 Tareas

Documentar arquitectura (diagramas y dependencias).

Añadir explicación de cada módulo en README.md.

Crear conclusiones técnicas:

Qué patrones se han encontrado.

Cómo responde la espera ante el clima.

Qué mejoras futuras se pueden implementar.

🕓 Duración total estimada

25–30 días de trabajo intensivo (unas 5-6 semanas razonables).
Ideal dedicación: 4-5 horas diarias.

💡 Resultado final

Parklytics se convierte en una plataforma de análisis y simulación predictiva basada en datos reales y públicos.
Permite:

Entender cómo afectan el clima y el calendario a la espera.

Predecir picos de afluencia y optimizar recursos.

Simular escenarios de operación y clima.
Y todo ello con un enfoque transparente, reproducible y explicable.