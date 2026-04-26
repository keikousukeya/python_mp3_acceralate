# python_mp3_acceralate

Whisper で日本語音声を文字起こしし、特定キーワードの検出タイミングに応じて MP3 の再生速度を動的に変更する Python スクリプトです。

## できること

- MP3 を文字起こし（Whisper）
- 単語タイムスタンプに基づくキーワード検出
- キーワードごとの速度変化量を反映して区間ごとに速度変更
- 速度編集後の MP3 を出力
- 参考として単純な 1.5 倍速 WAV も出力

## 動作環境

- Windows（PowerShell 想定）
- Python 3.10 以上推奨
- ffmpeg / ffprobe
- GPU があれば CUDA 実行可能（未指定時は自動判定）

## セットアップ

### 1) 仮想環境を有効化

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
```

### 2) 依存パッケージをインストール

最小構成（推奨）:

```powershell
pip install -r requirements_min.txt
```

補足:

- `requirements.txt` は環境全体の固定版を含むため、通常は `requirements_min.txt` の利用を推奨します。

### 3) ffmpeg / ffprobe を準備

以下のいずれかで参照できるようにしてください。

- PATH 上にある
- 実行ファイルと同じフォルダ
- `bin` サブフォルダ
- 実行時に `--ffmpeg-dir` で明示

## 使い方

### 基本実行

```powershell
python .\mp3_test.py --input .\sample02.mp3
```

### 主な引数

```text
--input          入力MP3ファイル（既定: sample02.mp3）
--output-dir     出力ディレクトリ（既定: 入力ファイルと同じ場所）
--model          Whisperモデル名（tiny/base/small/medium/large）
--language       文字起こし言語コード（既定: ja）
--device         推論デバイス（auto/cpu/cuda）
--ffmpeg-dir     ffmpeg.exe と ffprobe.exe があるフォルダ
```

例:

```powershell
python .\mp3_test.py --input .\sample02.mp3 --output-dir .\out --model medium --language ja --device auto
```

## 速度制御ルール（既定）

`mp3_test.py` 内の既定ルールは以下です。

- 光: +0.15
- 風: +0.15
- 夢: +0.15
- 雲: -0.15

速度は検出のたびに更新され、次区間から反映されます。

## 出力ファイル

入力が `sample02.mp3` の場合、既定では同じフォルダに以下が生成されます。

- `sample02.fast.wav`（1.5 倍速の参考出力）
- `sample02.keyword_dynamic_speed.mp3`（キーワード連動の速度編集結果）

## 実行時メモ

- 初回実行時は Whisper モデルのダウンロードが発生し、時間がかかる場合があります。
- `--device auto` は CUDA 利用可なら GPU、不可なら CPU を選びます。
- `--device cuda` 指定時に GPU が使えないとエラーになります。

## exe 配布（PyInstaller）

ビルド手順の詳細は `BUILD_INSTRUCTIONS.md` を参照してください。

概要:

```powershell
pip install pyinstaller
pyinstaller build_config.spec
```

出力先:

- `dist/mp3_test/mp3_test.exe`

配布時は `ffmpeg.exe` と `ffprobe.exe` の同梱が必要です。

## トラブルシュート

- `ffmpeg/ffprobe が見つかりません`
  - PATH を確認
  - `--ffmpeg-dir` を指定
  - exe 配布時は exe と同階層または `bin` に配置

- 処理が遅い
  - `--model tiny` など軽量モデルを試す
  - `--device auto` で GPU 使用を確認

- キーワードが反映されない
  - 文字起こし結果に対象語が含まれているか確認
  - `mp3_test.py` のキーワード定義を調整

## ライセンス

必要に応じて追記してください。
