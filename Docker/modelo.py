import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prepro import limpieza
def kmeansmodel(filepath: str):
    df = limpieza(filepath)
    features = [
        "Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)", 
        "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"
    ]
    df = df.dropna(subset=features)
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    optimal_k = 3
    model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(X_scaled) 
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Culmen Length (mm)"], df["Flipper Length (mm)"], c=df["Cluster"], cmap="viridis")
    plt.xlabel("Culmen Length (mm)")
    plt.ylabel("Flipper Length (mm)")
    plt.title("Clusters de Culmen Length vs Flipper Length")
    plt.colorbar(label="Cluster")
    plt.savefig("static/clusters_plot.png")  

    return model, scaler, df
