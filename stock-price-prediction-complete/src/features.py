import pandas as pd

def add_features(df: pd.DataFrame):
    df_feat = df.copy()
    df_feat['Return'] = df_feat['Close'].pct_change()
    df_feat['MA5'] = df_feat['Close'].rolling(5).mean()
    df_feat['MA10'] = df_feat['Close'].rolling(10).mean()
    df_feat['STD5'] = df_feat['Close'].rolling(5).std()
    df_feat['VolumeChange'] = df_feat['Volume'].pct_change()
    df_feat = df_feat.dropna()
    feature_cols = ['Return','MA5','MA10','STD5','VolumeChange']
    target_col = 'Close'
    return df_feat, feature_cols, target_col

def recursive_predict_next_n(df, model, feature_cols, n=5):
    preds = []
    df_copy = df.copy()
    for i in range(n):
        df_feat, feature_cols, target_col = add_features(df_copy)
        X = df_feat[feature_cols].iloc[[-1]]
        y_pred = model.predict(X)[0]
        next_date = df_copy.index[-1] + pd.tseries.offsets.BDay()
        new_row = df_copy.iloc[[-1]].copy()
        new_row.index = [next_date]
        new_row['Close'] = y_pred
        df_copy = pd.concat([df_copy, new_row])
        preds.append((next_date, y_pred))
    return pd.DataFrame(preds, columns=['Date','PredictedClose']).set_index('Date')
