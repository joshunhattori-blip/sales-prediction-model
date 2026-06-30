import os
import pandas as pd
import toml

def load_config():
    """設定ファイル(toml)を読み込む"""
    with open("config/config.toml", "r", encoding="utf-8") as f:
        return toml.load(f)

def run_preprocess():
    print("=== 前処理フェーズ開始 ===")
    config = load_config()
    
    raw_dir = config["data"]["raw_dir"]
    cache_dir = config["data"]["cache_dir"]
    processed_dir = config["data"]["processed_dir"]
    use_cache = config["preprocess"]["use_cache"]
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, "merged_data_cache.pkl")

    if use_cache and os.path.exists(cache_path):
        print("キャッシュを発見しました。結合処理をスキップします...")
        df_merged = pd.read_pickle(cache_path)
    else:
        print("キャッシュがないため、生データから結合処理を行います...")
        sales_path = os.path.join(raw_dir, "sales.csv")
        weather_path = os.path.join(raw_dir, "weather_informations.csv")
        
        df_sales = pd.read_csv(sales_path)
        df_weather = pd.read_csv(weather_path)

        df_sales['date'] = pd.to_datetime(df_sales['date'])
        df_weather['date'] = pd.to_datetime(df_weather['date'])

        df_merged = pd.merge(df_sales, df_weather, on=['area', 'date'], how='left')
        
        df_merged.to_pickle(cache_path)
        print(f"結合済みのデータをキャッシュとして保存しました: {cache_path}")

    # ==========================================
    # ② 特徴量エンジニアリング
    # ==========================================
    print("特徴量を作成しています...")
    
    df_merged['Month'] = df_merged['date'].dt.month
    df_merged['day_of_week'] = df_merged['date'].dt.dayofweek 

    df_merged = df_merged.sort_values(['area', 'item_index', 'date'])
    grouped = df_merged.groupby(['area', 'item_index'])
    
    #  1日前の売上と、7日前の売上
    df_merged['Lag1_Sales'] = grouped['recorded_sales'].shift(1)
    df_merged['Lag7_Sales'] = grouped['recorded_sales'].shift(7)
    
    #  過去7日間（昨日時点）の移動平均
    df_merged['Rolling7_Mean'] = grouped['recorded_sales'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    
    #  1週間前（Lag7）の時点から見た、さらに過去7日間の平均
    df_merged['Lag7_Rolling7_Mean'] = grouped['recorded_sales'].transform(lambda x: x.shift(7).rolling(window=7, min_periods=1).mean())
    
    # 気温の差分特徴量
    df_merged['Lag1_Temp'] = grouped['temperature'].shift(1)
    df_merged['TempChange'] = df_merged['temperature'] - df_merged['Lag1_Temp']
    
    # ==========================================
    # ③ 不要カラムの削除と欠損値処理
    # ==========================================
    drop_cols = ['true_sales', 'Lag1_Temp']
    df_final = df_merged.drop(columns=[col for col in drop_cols if col in df_merged.columns])
    
    df_final = df_final.dropna().reset_index(drop=True)

    out_path = os.path.join(processed_dir, "final_ml_data.csv")
    df_final.to_csv(out_path, index=False)
    print(f"学習用データの前処理が完了しました: {out_path}\n")

if __name__ == "__main__":
    run_preprocess()