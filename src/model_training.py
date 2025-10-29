import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.data_preprocessing import preprocess_data

def train_models(df):
    df = df.dropna(subset=['Close'])
    X = df.drop(columns=['Close'])
    y = df['Close']

    X_processed, preprocessor = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }
        joblib.dump(model, f"./models/{name}_model.pkl")

    joblib.dump(preprocessor, "./models/preprocessor.pkl")
    return results
if __name__ == "__main__":
    data = pd.read_csv("./data/IBM_Stock_1980_2025.csv")
    training_results = train_models(data)
    for model_name, metrics in training_results.items():
        print(f"{model_name} - MSE: {metrics['MSE']}, R2: {metrics['R2']}")
        
        