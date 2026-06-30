from src.preprocess import run_preprocess
from src.train import run_train

def main():
    print("需要予測パイプラインを開始します...\n" + "="*40)
    
    # 1. キャッシュの確認と前処理
    run_preprocess()
    
    # 2. LightGBMによる学習と評価
    run_train()
    
    print("="*40 + "\n 全てのパイプラインが正常に完了しました！")

if __name__ == "__main__":
    main()