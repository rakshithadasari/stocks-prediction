import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

@dataclass
class Report:
    mae: float
    rmse: float
    mape: float
    n_train: int
    n_test: int
    feature_importances: dict

def train_eval_time_split(df, feature_cols, target_col):
    X = df[feature_cols].values
    y = df[target_col].values
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    report = Report(mae, rmse, mape, len(y_train), len(y_test),
                    dict(zip(feature_cols, model.feature_importances_)))
    return model, report, (X_train, X_test, y_train, y_test)
