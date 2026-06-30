import os
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from sklearn.metrics import mean_squared_error

def run_train():
    print("=== 学習フェーズ開始 ===")
    
    processed_dir = "data/processed"
    data_path = os.path.join(processed_dir, "final_ml_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"エラー: {data_path} が見つかりません。先に前処理を実行してください。")

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    df['area'] = df['area'].astype('category')
    df['weather_condition'] = df['weather_condition'].astype('category')

    # 変更：特徴量リストの更新
    features = [
        'area', 'item_index', 'Month', 'day_of_week', 
        'weather_condition',
        'temperature', 'max_temperature', 'min_temperature', 'amount_of_precipitation',
        'Lag1_Sales', 'Lag7_Sales', 
        'Rolling7_Mean', 
        'Lag7_Rolling7_Mean', 
        'TempChange'
    ]
    target = 'recorded_sales'

    latest_date = df['date'].max()
    split_date = latest_date - pd.DateOffset(years=1)
    train_start_date = split_date - pd.DateOffset(years=9)

    train_df = df[(df['date'] >= train_start_date) & (df['date'] < split_date)]
    test_df = df[df['date'] >= split_date]

    print(f"学習データ期間: {train_df['date'].min().date()} 〜 {train_df['date'].max().date()} ({len(train_df)}件)")
    print(f"テストデータ期間: {test_df['date'].min().date()} 〜 {test_df['date'].max().date()} ({len(test_df)}件)")

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    evals_result = {}
    params = {
        'objective': 'poisson',       # 需要予測に適したポアソン回帰に変更
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'verbosity': -1,
        'feature_fraction': 0.7       # 一部の特徴量への過度な依存を防ぐ
    }

    model = lgb.train(
        params, train_data, valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'], num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.record_evaluation(evals_result)]
    )

    y_pred = np.maximum(0, np.round(model.predict(X_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"学習完了！ テストデータのRMSE: {rmse:.2f}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = os.path.join("outputs", run_id)
    os.makedirs(run_out_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    lgb.plot_metric(evals_result, metric='rmse', figsize=(8, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(run_out_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    lgb.plot_importance(model, importance_type='gain', figsize=(8, 5), title='Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(os.path.join(run_out_dir, "feature_importance.png"))
    plt.close()

    sample_area = test_df['area'].iloc[0]
    sample_item = test_df['item_index'].iloc[0]
    sample_df = test_df[(test_df['area'] == sample_area) & (test_df['item_index'] == sample_item)].copy()
    
    if not sample_df.empty:
        sample_df['pred'] = model.predict(sample_df[features])
        
        weekly_df = sample_df.set_index('date')[[target, 'pred']].resample('W-MON').sum().reset_index()
        
        plt.figure(figsize=(10, 5))
        plt.plot(weekly_df['date'], weekly_df[target], label='Actual (Weekly)', marker='o')
        plt.plot(weekly_df['date'], weekly_df['pred'], label='Predicted (Weekly)', marker='x', linestyle='--')
        plt.title(f'Weekly Sales Prediction vs Actual ({sample_area}, Item {sample_item})')
        plt.xlabel('Date (Week Start)')
        plt.ylabel('Weekly Sales')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_out_dir, "predict_vs_actual_weekly.png"))
        plt.close()

    results_meta = {"rmse": rmse, "best_iteration": model.best_iteration, "features": features}
    with open(os.path.join(run_out_dir, "metrics.json"), "w") as f:
        json.dump(results_meta, f, indent=4)
        
    print(f"実行結果を {run_out_dir}/ にまとめて保存しました\n")

if __name__ == "__main__":
    run_train()