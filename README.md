# LLMon — Apple Silicon M4・24GB ローカルLLM学習環境

DockerでCPU LoRA学習を実施する環境と、macOS上でM4 GPUを利用するMLX QLoRA学習環境です。
このパッケージは環境構築用ファイルです。利用者のMacへのインストール・モデルの取得・実機学習は未実施です。

## 専門文書を参照するRAGを追加しました

PDF / DOCX / TXT / MDを登録し、文書名・ページ/段落位置と原文を確認しながら質問できます。
確認・修正した回答はLoRA用の学習候補として保存できます。文書登録だけでは追加学習は行いません。

```bash
bash scripts/Start-RAG.sh
```

画面: http://localhost:18083 （Mac自身からのみ接続）。詳しい取込・質問・追加学習手順は
[docs/RAG.md](docs/RAG.md)を参照してください。初回は埋め込みモデルと回答モデルの取得が必要です。
スキャンPDFのOCRは未実装です。文書・索引・実務学習データは`data/private/`配下で管理し、公開GitHubへ登録しないでください。

## 構成と初期設定

| 項目 | Docker CPU | Mac GPU（任意・実用向け） |
|---|---|---|
| 実行環境 | Linux ARM64 / Docker Desktop | macOSネイティブ / Python仮想環境 |
| 学習エンジン | PyTorch / Transformers / PEFT | MLX-LM |
| 初期モデル | Qwen3 0.6B | Qwen3 4B・4bit |
| 学習方式 | LoRA、q_proj/v_proj、rank 8 | QLoRA、最後の4層 |
| 初期バッチ | 1、勾配蓄積4 | 1 |
| 最大系列長 | 256 tokens（超過はエラー） | 512 tokens |
| 初期学習回数 | 20 optimizer steps | 20 iterations |
| 操作 | JupyterLab、ターミナル | Macのターミナル |
| 保存場所 | outputs/cpu | outputs/mlx |

通常のLinuxコンテナからApple GPU/Metalへ直接アクセスする構成は利用できません。
Docker Model RunnerのMac GPU推論対応は、コンテナ内のPyTorch学習へのGPU提供とは異なります。
そのため「学習処理もすべてDocker内」を満たす構成はCPU学習になります。
MLXはMac側で実行します。データと成果物は同じプロジェクトフォルダを利用しますが、
PyTorch用とMLX用のアダプターは別形式のため相互にそのまま利用できません。

## 1. MacとDocker Desktopの準備

1. Apple Silicon版Docker Desktopをインストールし、起動します。
2. Docker Desktop → Settings → ResourcesでCPUを6、メモリを12GB程度に設定します。
   これはMac全体24GBの一部です。macOSと他アプリのための余裕を残します。
   Resource Saver等により項目が異なる場合は現在の画面に従ってください。
3. 空きディスクは20〜30GB程度を確保してください。複数モデルを保存するならさらに必要です。
4. リポジトリを取得し、ターミナルでLLMonフォルダへ移動します。

```bash
git clone https://github.com/UeEmon/LLMon.git
cd LLMon
bash scripts/Check-Mac.sh
bash scripts/Start-Docker.sh
```

ブラウザで http://localhost:18082 を開きます。
Jupyterトークンは起動ログの`?token=...`に表示されます。ログのポート8888は
コンテナ側の番号です。Macからは18082を使い、トークン文字列を入力してください。

```bash
docker compose logs lab
```

JupyterLabでは学習データ・設定の編集、ログ確認、Terminalからの実行ができます。
画面上の学習専用ボタンは実装していません。
接続先はMac自身に限定しています。初回のビルド・モデル取得にはインターネットが必要です。

## 2. Docker内でCPU学習

Macのターミナルで実行します。

```bash
bash scripts/Train-CPU.sh
```

初回はQwen3 0.6Bを取得します。学習する重みはLoRA部分のみですが、ベースモデルの計算は行います。
FP32のベース重みだけでも約2.4GBで、実行時は活性値などのメモリが追加されます。
CPU処理はGPUより遅いため、初期設定は動作確認用の20ステップです。
JupyterLabを同時に起動していても学習コンテナは独立して実行されます。
同時に複数の学習を実行しないでください。

ログとモデルは、毎回別の`outputs/cpu/日時-PID`へ保存されます。

- `adapter/`: LoRAアダプターとトークナイザー（ベースモデル本体は含みません）
- `run.json`: 設定、ベースモデルの取得revision、データのSHA-256、ライブラリ版
- `test_metrics.json`: 学習後の未使用testデータに対する損失
- `checkpoint-*`: 学習途中のチェックポイント
- 同名の`.log`: 実行ログ

JupyterLabのTerminalから直接実行する場合:

```bash
python /opt/app/data_utils.py data
python /opt/app/train_cpu.py --output outputs/cpu/manual-001
```

`config/cpu.json`で学習回数、系列長、モデルなどを変更できます。
このコードはQwen3の会話テンプレートとq_proj/v_projを前提としています。
別系列のモデルを指定する場合はテンプレート・EOS・対象モジュールの調整が必要です。
0.6Bから1.7Bへの拡大は動作確認後に試してください。4B以上のDocker FP32学習は
24GB Macでの初期運用には推奨しません。

### 学習前後の応答比較

`RUN`は実際に作成されたディレクトリへ置き換えてください。

```bash
RUN=outputs/cpu/20261015-060000-12345
docker compose run --rm -T --no-deps lab python /opt/app/infer_cpu.py --run "$RUN" --base-only --prompt '作業状況を短く問い合わせてください。'
docker compose run --rm -T --no-deps lab python /opt/app/infer_cpu.py --run "$RUN" --prompt '作業状況を短く問い合わせてください。'
```

## 3. M4 GPUで学習する場合（MacネイティブMLX）

MacのARM64版Python 3.11を事前にインストールしてください。
Homebrewをすでに利用している場合は`brew install python@3.11`で準備できます。
次の操作はDockerコンテナ内ではなく、Macのターミナルで行います。

GPU学習時はDocker環境を停止し、ほかの大きなアプリを閉じてメモリを確保します。

```bash
docker compose down
bash scripts/Setup-MLX.sh
bash scripts/Train-MLX.sh
```

`python3.11`が見つからない場合は、そのPythonのパスを指定します。

```bash
PYTHON_BIN=/opt/homebrew/bin/python3.11 bash scripts/Setup-MLX.sh
```

MLX-LMは0.26.4を指定し、実際に解決された依存関係は`config/mlx-installed.txt`へ保存します。
初期モデルは`mlx-community/Qwen3-4B-4bit`です。4bitモデルに対するLoRA学習はQLoRAとなります。
4B・系列長512・バッチ1・4層学習は24GB向けの控えめな開始設定ですが、実測値ではありません。
メモリ不足時は系列長を256、層数を2へ減らすか、モデルを小さくしてください。

```bash
bash scripts/Train-MLX.sh --max-seq-length 256 --num-layers 2
```

学習回数を増やす場合:

```bash
bash scripts/Train-MLX.sh --iters 200
```

応答比較:

```bash
RUN=outputs/mlx/20261015-060000-12345
.venv-mlx/bin/python app/infer_mlx.py --run "$RUN" --base-only --prompt '作業状況を短く問い合わせてください。'
.venv-mlx/bin/python app/infer_mlx.py --run "$RUN" --prompt '作業状況を短く問い合わせてください。'
```

モデル・学習データを変更した場合は、モデル名、依存関係、学習設定、データ分割を記録してください。
学習成果物や実務データはGitHubへコミットせず、ローカルで管理してください。

## 4. 学習データ

`data/train.jsonl`、`valid.jsonl`、`test.jsonl`を使用します。
UTF-8で、1行が1件のJSONです。systemは任意、user→assistantの1往復を受け付けます。

```json
{"messages":[{"role":"system","content":"日本語で簡潔に回答してください。"},{"role":"user","content":"試験開始の条件は。"},{"role":"assistant","content":"確認メールを受信した後、試験を開始してください。"}]}
```

付属データは一般的な業務連絡の架空サンプルで、train 24件、valid 4件、test 4件です。
これは動作確認用であり、品質向上を保証する学習量ではありません。
業務向けには良質な質問・回答を収集し、同じ質問やその言い換えが評価データへ漏れないよう分割します。
バリデーターは完全一致した質問の重複を検出しますが、言い換えの判定は行いません。
本文の内容、文体、正確性は人が確認してください。

CPU側は質問部分を学習対象から除外し、回答とEOSに対する損失を計算します。
MLX側も`--mask-prompt`を指定します。
検証損失・test損失だけでなく、未学習の実務質問で応答の品質と既存能力の低下を確認します。
資料に基づく正確な回答が目的なら、追加学習とあわせてRAGを検討します。

## 5. 停止・再起動・データ保持

学習の停止は実行中ターミナルのCtrl+Cです。
コンテナ全体の停止:

```bash
docker compose down
```

再起動:

```bash
docker compose up -d
```

学習データと成果物はこのフォルダ内に残ります。Dockerのモデルキャッシュは
`hf-cache`ボリュームへ保存します。`docker compose down -v`はキャッシュも削除します。
MLXモデルキャッシュはMacユーザーの標準Hugging Faceキャッシュに保存されます。
モデル取得後の完全オフライン運用ではキャッシュを確保し、CPU側に
`HF_HUB_OFFLINE=1`と`TRANSFORMERS_OFFLINE=1`を設定してください。
MLX側は取得済みモデルのローカルパスを`MLX_MODEL`に指定すると管理しやすくなります。

## 6. 検証状況

作成環境ではDocker、Apple Silicon、Metal、PyTorch/MLXが利用できません。
したがって、ARM64イメージのビルド、依存関係の導入、実モデル取得、CPU/GPU学習の完走は未検証です。
データの整合性、学習時の回答マスクと系列長超過の検出、評価データへの完全一致漏洩の検出、
Python/Shellの構文、Compose YAMLの静的な読み取りを検証しています。
実機の初回起動・20ステップ学習が次の確認になります。

## 参考資料

- Docker GPU passthrough: https://docs.docker.com/desktop/features/gpu/
- MLX-LM学習: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- Qwen3 0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- Qwen3 4B MLX 4bit: https://huggingface.co/mlx-community/Qwen3-4B-4bit

コードと各モデルの利用条件は別です。利用するモデルのライセンスを確認してください。
