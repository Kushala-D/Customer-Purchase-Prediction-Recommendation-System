
import pandas as pd
from sklearn.cluster import KMeans

def run_recommendation():
    data = pd.DataFrame({
        "user_id": range(100),
        "spend": range(100),
        "visits": range(100)
    })

    kmeans = KMeans(n_clusters=3)
    data["cluster"] = kmeans.fit_predict(data[["spend","visits"]])

    print(data.head())

if __name__ == "__main__":
    run_recommendation()
