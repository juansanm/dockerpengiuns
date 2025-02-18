from fastapi import FastAPI, Form
import joblib
import os
import logging

app = FastAPI()

MODEL_DIR = "/models"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "predictions.log")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(message)s")

def cargar_modelo_mas_reciente():
    """Carga el modelo más reciente de la carpeta /models"""
    modelos = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") and "scaler" not in f]
    escaladores = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl") and "scaler" in f]

    if not modelos or not escaladores:
        return None, None

    modelos.sort(key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)), reverse=True)
    escaladores.sort(key=lambda x: os.path.getctime(os.path.join(MODEL_DIR, x)), reverse=True)

    modelo_path = os.path.join(MODEL_DIR, modelos[0])
    scaler_path = os.path.join(MODEL_DIR, escaladores[0])

    modelo = joblib.load(modelo_path)
    scaler = joblib.load(scaler_path)

    return modelo, scaler

@app.post("/predict/")
async def predict_cluster(
    culmen_length: float = Form(...),
    culmen_depth: float = Form(...),
    flipper_length: float = Form(...),
    body_mass: float = Form(...),
    delta_15: float = Form(...),
    delta_13: float = Form(...)
):
    try:
        model, scaler = cargar_modelo_mas_reciente()
        if model is None or scaler is None:
            return {"error": "No hay modelos disponibles en /models"}

        X_new = [[culmen_length, culmen_depth, flipper_length, body_mass, delta_15, delta_13]]
        X_scaled = scaler.transform(X_new)
        cluster = model.predict(X_scaled)[0]

        log_message = f"Predicción: Cluster {cluster} | Datos: {X_new}"
        logging.info(log_message)

        return {"cluster": int(cluster)}

    except Exception as e:
        return {"error": str(e)}
