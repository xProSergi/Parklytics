# 📊 KPIs – Indicadores Clave del Proyecto ParkLytics

Los KPIs (Key Performance Indicators) definen cómo se medirá el éxito técnico y analítico del proyecto ParkLytics.

---

## 🎯 1. KPIs de datos
| Indicador | Descripción | Meta |
|------------|-------------|------|
| **Cobertura temporal** | % de días con datos válidos en el dataset | ≥ 95 % |
| **Calidad de datos** | Porcentaje de registros sin valores nulos | ≥ 98 % |
| **Integración de fuentes** | Fuentes correctamente integradas (Queue-Times + Clima) | 100 % |
| **Latencia de ingesta** | Tiempo desde obtención → almacenamiento | < 1 min (modo batch local) |

---

## ⚙️ 2. KPIs técnicos
| Indicador | Descripción | Meta |
|------------|-------------|------|
| **Tiempo medio de procesamiento Spark** | Procesamiento completo de un día de datos | < 10 s |
| **Uso de memoria estable** | Evitar saturación en sesiones Spark | Sin errores |
| **Pipeline reproducible** | Capacidad de ejecutar el flujo completo sin intervención manual | Sí (1 comando/notebook) |

---

## 🔮 3. KPIs de analítica y predicción
| Indicador | Descripción | Meta |
|------------|-------------|------|
| **Precisión del modelo (R² o RMSE)** | Grado de ajuste de la predicción de afluencia | R² ≥ 0.80 |
| **Explicabilidad (SHAP importance)** | Top 3 variables más influyentes identificadas | Sí |
| **Predicción climática correlativa** | Correlación clima ↔ afluencia | r ≥ 0.6 |

---

## 💰 4. KPIs de optimización económica
| Indicador | Descripción | Meta |
|------------|-------------|------|
| **Coste estimado por visitante** | Medida del coste medio operativo / afluencia | Disminución del 10 % |
| **Eficiencia de personal (simulación)** | Ajuste de personal según predicción | +15 % eficiencia |
| **ROI simulado (beneficio / coste)** | Impacto económico de aplicar las recomendaciones | ROI positivo |

---

## 🧩 5. KPIs de entregables
| Indicador | Descripción | Meta |
|------------|-------------|------|
| **Documentación completa** | Vision + KPIs + README | 100 % |
| **Repositorio Git limpio** | Estructura clara, commits descriptivos | Sí |
| **Notebook principal ejecutable** | Desde inicio hasta visualización final | Sin errores |
| **Informe técnico final** | Resumen completo del flujo y resultados | Entregado |

---

## ✅ Resultado esperado
Cumplir al menos el **80 % de los KPIs definidos** será considerado un éxito del proyecto, con especial foco en:
- Predicción fiable de afluencia.  
- Explicabilidad del modelo.  
- Valor económico potencial para la gestión real del parque.

