import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
import schedule

# === CONFIGURACIÓN ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUEUE_TIMES_URL = "https://queue-times.com/parks/298/queue_times.json"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "queue_times")
LOG_FILE = os.path.join(BASE_DIR, "data", "logs", "ingestion_log.txt")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_message(message: str):
    """Guarda un mensaje de log con timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"[{timestamp}] {message}\n")
    print(message)


def download_queue_times():
    """Descarga y guarda los datos de Queue Times con timestamp"""
    try:
        response = requests.get(QUEUE_TIMES_URL)
        response.raise_for_status()
        data = response.json()

        # Crear DataFrame
        rides_list = []
        for land in data.get("lands", []):
            for ride in land.get("rides", []):
                rides_list.append({
                    "zona": land["name"],
                    "atraccion": ride["name"],
                    "tiempo_espera": ride["wait_time"],
                    "abierta": ride["is_open"],
                    "ultima_actualizacion": ride["last_updated"]
                })

        df = pd.DataFrame(rides_list)

        # Procesamiento
        df["ultima_actualizacion"] = pd.to_datetime(df["ultima_actualizacion"])
        now = datetime.now()
        df["fecha"] = now.date()
        df["hora"] = now.time().replace(microsecond=0)
        df["dia_semana"] = now.strftime("%A")  # lunes, martes, etc.

        # Nombre del archivo: queue_times_YYYY-MM-DD_HH-MM.csv
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
        filename = f"queue_times_{timestamp_str}.csv"
        output_path = os.path.join(RAW_DATA_DIR, filename)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        log_message(f"✅ Ingesta completada: {len(df)} registros → {output_path}")

    except Exception as e:
        log_message(f"❌ ERROR durante la ingesta: {e}")


def run_scheduler(interval_minutes=15):
    """Ejecuta la descarga automática cada X minutos"""
    log_message(f"🚀 Iniciando ingesta automática cada {interval_minutes} minutos...")
    schedule.every(interval_minutes).minutes.do(download_queue_times)

    # Primera ejecución inmediata
    download_queue_times()

    # Bucle infinito (Ctrl + C para detener)
    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    run_scheduler(interval_minutes=15)
