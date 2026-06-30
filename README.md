## 実行方法 (How to Run)

本プロジェクトは `uv` を使用して環境・パッケージ管理を行っています。

```bash
# 1. パイプラインの一括実行（前処理 → 学習 → 評価結果の出力）
uv run main.py

# 2. 損失関数（Objective）の総当たり評価実験
uv run src/evaluate_objectives.py