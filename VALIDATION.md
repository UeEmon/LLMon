# 検証記録

作成日: 2026-10-15（日本時間）

## PASS

- 独立した質問を使用したtrain 24 / valid 4 / test 4の読込
- 学習データと評価データで質問が完全一致した場合の拒否
- 回答とEOSにだけラベルを付け、質問部分を-100でマスク
- 系列長超過の拒否（回答を黙って切り捨てない）
- Pythonソースの構文確認
- 全Shellスクリプトのbash構文確認
- Compose YAMLとCPU設定JSONの読み取り、ARM64・ローカル接続設定の確認

## 未実施

- Docker Desktop / Composeの実起動
- ARM64 Docker imageのビルドと依存関係の解決
- Hugging Faceからのモデル取得
- 実トークナイザーでのサンプル系列長確認
- PyTorch CPU学習の完走と推論
- M4 Metal / MLX学習・評価・推論
- 学習時間、メモリ消費、応答品質の測定

作成環境にDocker、PyTorch、MLX、Apple Siliconがないため、実学習の成功は保証できません。
依存関係を導入した後、まず付属サンプルで20ステップの学習を行ってください。
