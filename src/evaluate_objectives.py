import os
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_poisson_deviance

def run_objective_experiment():
    print("=== 損失関数（Objective）の総当たり実験を開始します ===")
    
    data_path = "data/processed/final_ml_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"エラー: {data_path} が見つかりません。")

    # 1. データの作成と実行用フォルダの準備
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = os.path.join("outputs", f"experiment_{run_id}")
    os.makedirs(run_out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['area'] = df['area'].astype('category')
    df['weather_condition'] = df['weather_condition'].astype('category')

    features = [
        'area', 'item_index', 'Month', 'day_of_week', 'weather_condition',
        'temperature', 'max_temperature', 'min_temperature', 'amount_of_precipitation',
        'Lag1_Sales', 'Lag7_Sales', 'Rolling7_Mean', 'Lag7_Rolling7_Mean', 'TempChange'
    ]
    target = 'recorded_sales'

    # データの分割（直近1年をテスト、過去9年を学習）
    latest_date = df['date'].max()
    split_date = latest_date - pd.DateOffset(years=1)
    train_start_date = split_date - pd.DateOffset(years=9)

    train_df = df[(df['date'] >= train_start_date) & (df['date'] < split_date)]
    test_df = df[df['date'] >= split_date]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 実験エントリー：7つの損失関数
    objectives = [
        'regression', 'regression_l1', 'huber', 'fair', 'poisson', 'tweedie', 'mape'
    ]
    
    metrics_summary = []
    
    # 代表としてグラフを描画するサンプル（最初のエリアと商品）
    sample_area = test_df['area'].iloc[0]
    sample_item = test_df['item_index'].iloc[0]
    sample_df = test_df[(test_df['area'] == sample_area) & (test_df['item_index'] == sample_item)].copy()

    # 週次集計用の実測値（後でグラフのベースにする）
    weekly_actual = sample_df.set_index('date')[[target]].resample('W-MON').sum().reset_index()

    # 各Objectiveごとの予測結果をプロットするための準備
    plt.figure(figsize=(14, 7))
    plt.plot(weekly_actual['date'], weekly_actual[target], label='Actual (Weekly)', marker='o', color='black', linewidth=2, zorder=3)

    # 2. 総当たり学習・評価ループ
    for obj in objectives:
        print(f"学習中: Objective = '{obj}' ...")
        
        params = {
            'objective': obj,
            'metric': 'rmse', # 早回し・ストップ判定用
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'verbosity': -1,
            'feature_fraction': 0.7
        }

        model = lgb.train(
            params, train_data, valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'], num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # 予測と丸め処理
        y_pred_raw = model.predict(X_test)
        y_pred = np.maximum(0, np.round(y_pred_raw))
        
        # 評価指標の算出
        rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_score = mean_absolute_error(y_test, y_pred)
        mape_score = mean_absolute_percentage_error(y_test, y_pred)
        
        y_pred_p = np.maximum(1e-5, y_pred)
        y_test_p = np.maximum(1e-5, y_test)
        poisson_score = mean_poisson_deviance(y_test_p, y_pred_p)

        metrics_summary.append({
            'Objective': obj, 'RMSE': rmse_score, 'MAE': mae_score, 'MAPE': mape_score, 'Poisson Dev': poisson_score
        })

        # 個別の予測比較データの作成（週次集計）
        if not sample_df.empty:
            obj_sample = sample_df.copy()
            obj_sample['pred'] = model.predict(obj_sample[features])
            weekly_obj = obj_sample.set_index('date')[['pred']].resample('W-MON').sum().reset_index()
            
            # ① 各損失関数単体の予測比較グラフを保存
            plt.figure(figsize=(10, 5))
            plt.plot(weekly_actual['date'], weekly_actual[target], label='Actual', marker='o', color='black')
            plt.plot(weekly_obj['date'], weekly_obj['pred'], label=f'Predicted ({obj})', marker='x', linestyle='--')
            plt.title(f'Weekly Sales Prediction [{obj}] vs Actual ({sample_area}, Item {sample_item})')
            plt.xlabel('Date')
            plt.ylabel('Weekly Sales')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_out_dir, f"predict_vs_actual_{obj}.png"))
            plt.close()

    # 3. 評価指標の総当たり比較グラフ（棒グラフ）の生成・保存
    res_df = pd.DataFrame(metrics_summary)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Objective Performance Comparison (Lower is Better)', fontsize=16, fontweight='bold')
    
    metric_cols = ['RMSE', 'MAE', 'MAPE', 'Poisson Dev']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, col in enumerate(metric_cols):
        ax = axes[idx//2, idx%2]
        bars = ax.bar(res_df['Objective'], res_df[col], color=colors[idx], edgecolor='black', alpha=0.8)
        ax.set_title(f'Evaluation Metric: {col}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(res_df['Objective'], rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 棒のトップに数値をテキスト表示
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3pt vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(run_out_dir, "metrics_comparison_matrix.png"))
    plt.close()

    # メタデータ(JSON)も同封
    with open(os.path.join(run_out_dir, "results_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n" + "="*70)
    print(f"実験が正常に完了しました！成果物はすべて下記フォルダに格納されました。")
    print(f"出力先: {run_out_dir}/")
    print(f"   - metrics_comparison_matrix.png (全指標の総当たり棒グラフ)")
    print(f"   - predict_vs_actual_[損失関数名].png (各モデルの週次予測推移グラフ × 7枚)")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_objective_experiment()