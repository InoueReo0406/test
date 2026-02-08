# SAM3 CPU Segmentation + IoU (COCO 2017 val)

CPU だけで SAM3 によるセグメンテーションを行い、COCO 2017 val の 1 枚画像で IoU を計算する最小チュートリアルです。

## 前提
- Python 3.12
- インターネット接続（COCO アノテーションと SAM3 モデルを取得）
- Hugging Face で `facebook/sam3` の利用許可が必要です: https://huggingface.co/facebook/sam3

## セットアップ（venv）
```bash
cd /Users/inouereo/git_research/turorial_sam3
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# HF 認証（どちらか1つ）
hf auth login
# もしくは
huggingface-cli login
```

## 実行（デフォルト: IoU > 0 まで自動リトライ）
```bash
python run_sam3_iou.py --text-prompt person --image-size 224
```

### 重要なオプション
- `--text-prompt` は COCO のカテゴリ名と一致させてください（例: `person`, `car`, `dog`）。
- デフォルトは `--min-iou 1e-9` なので、実質 `IoU > 0` になるまで継続します。
- `--max-attempts 0` がデフォルトで、無制限リトライです（画像をシャッフルし続ける）。
- 必要なら `--max-attempts N` で上限を設定できます。
- `--require-pred` を付けると、予測マスクが空の場合に再試行します。

## 出力
`outputs/` に以下を保存します（毎回上書き）。
- `image.jpg` 入力画像
- `gt_mask.png` GT マスク
- `pred_mask_best.png` 予測マスク（IoU最大）
- `overlay.png` GT=赤 / 予測=緑 の重ね合わせ

## メモ
- 初回は COCO アノテーション（`annotations_trainval2017.zip`）をダウンロードします（数百MB）。
- CPU 実行のため推論は時間がかかります。必要なら `--image-size 560` など小さめを推奨します。
