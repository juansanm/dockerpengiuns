from fastapi import FastAPI
import joblib
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from modelo import kmeansmodel
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
model, scaler, df_clusters = kmeansmodel("penguins_lter.xlsx")
joblib.dump(model, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
@app.get("/plot")
def get_plot():
    return FileResponse("static/clusters_plot.png", media_type="image/png")
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("static/cluster.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: No se encontró el archivo cluster.html</h1>", status_code=404)
@app.get("/predict/")
def predict_cluster(culmen_length: float, culmen_depth: float, flipper_length: float, 
                     body_mass: float, delta_15: float, delta_13: float):
    try:
        model = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        X_new = [[culmen_length, culmen_depth, flipper_length, body_mass, delta_15, delta_13]]
        X_scaled = scaler.transform(X_new)
        cluster = model.predict(X_scaled)[0]
        
        return {"cluster": int(cluster)}
    
    except Exception as e:
        return {"error": str(e)}
