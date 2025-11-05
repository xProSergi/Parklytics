import pandas as pd
import holidays

# 1. Cargar dataset existente
df = pd.read_csv("data/clean/queue_times_weather.csv")

# 2. Convertir la columna 'date' a tipo datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# 3. Crear calendario de festivos de España (Madrid)
festivos_madrid = holidays.Spain(subdiv='MD')  # 'MD' = Madrid

# 4. Crear columna booleana 'festivo'
df['festivo'] = df['fecha'].dt.date.isin(festivos_madrid)


# 6. Guardar dataset enriquecido
df.to_csv("data/tiempos.csv", index=False)

print("✅ Dataset enriquecido con festivos guardado como 'queue_times_weather_festivos.csv'")
print(df[['fecha', 'festivo']].head(10))
