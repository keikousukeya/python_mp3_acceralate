from pathlib import Path
import shutil
import winsound
import torch
import whisper
import librosa
import soundfile as sf
import unicodedata

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Whisper device: {device}")
if device == "cuda":
	print(f"GPU: {torch.cuda.get_device_name(0)}")

# モデルを読み込み（smallやmediumでも可、largeは高精度だが重い）
model = whisper.load_model("medium", device=device)


from pydub import AudioSegment


def resolve_tool_path(tool_name: str) -> str | None:
	path = shutil.which(tool_name)
	if path:
		return path

	local_app_data = Path.home() / "AppData" / "Local"
	candidates = sorted(
		local_app_data.glob(
			"Microsoft/WinGet/Packages/Gyan.FFmpeg_*/ffmpeg-*/bin/*.exe"
		),
		reverse=True,
	)
	for candidate in candidates:
		if candidate.stem.lower() == tool_name.lower():
			return str(candidate)

	return None


ffmpeg_path = resolve_tool_path("ffmpeg")
ffprobe_path = resolve_tool_path("ffprobe")

if not ffmpeg_path or not ffprobe_path:
	raise RuntimeError(
		"ffmpeg/ffprobe が見つかりません。インストール後にターミナルを再起動してください。"
	)

AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

mp3_path = Path(__file__).resolve().parent / "sample02.mp3"

if not mp3_path.exists():
	raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

audio_data = AudioSegment.from_mp3(mp3_path)
result = model.transcribe(str(mp3_path), language="ja", fp16=(device == "cuda"))

# 結果のテキストを表示
print(result["text"])

# pydubの一時ファイル再生で権限エラーが出る環境向けに、作業フォルダへ明示的にWAVを書き出して再生する
wav_path = mp3_path.with_suffix(".wav")
try:
	audio_data.export(wav_path, format="wav")
	# winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
finally:
	if wav_path.exists():
		wav_path.unlink()

# librosaで読み込んで速度を1.5倍にしてみる
y, sr = librosa.load(mp3_path)
y_fast = librosa.effects.time_stretch(y, rate=1.5)
sf.write(mp3_path.with_suffix(".fast.wav"), y_fast, sr)

def normalize_ja(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def find_keyword_times_sec(result: dict, keywords: list[str]) -> list[float]:
    keys = [normalize_ja(k) for k in keywords]
    times: list[float] = []

    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            token = normalize_ja(w.get("word", ""))
            if any(k in token for k in keys):
                t = float(w.get("end", seg.get("end", 0.0)))
                if t > 0:
                    times.append(t)

    times.sort()

    # 近すぎる境界をまとめる
    deduped: list[float] = []
    min_gap_sec = 0.2
    for t in times:
        if not deduped or (t - deduped[-1]) >= min_gap_sec:
            deduped.append(t)

    return deduped


# ここは1回だけ実行（先頭で transcribe している箇所は削除してOK）
result = model.transcribe(
    str(mp3_path),
    language="ja",
    fp16=(device == "cuda"),
    word_timestamps=True,
)

trigger_keywords = ["風", "雲", "夢", "光"]
trigger_times = find_keyword_times_sec(result, trigger_keywords)

if not trigger_times:
    print("指定単語が見つからなかったため、加速編集はスキップ")
else:
    audio = AudioSegment.from_mp3(mp3_path)

    # 区間境界: 0, キーワード時刻..., 音声末尾
    points_ms = [0] + [int(t * 1000) for t in trigger_times if 0 < int(t * 1000) < len(audio)] + [len(audio)]
    points_ms = sorted(set(points_ms))

    # 出現ごとに倍率を上げる設定
    base_speed = 1.0    # キーワード前
    speed_step = 0.15    # 1回出るごとに +0.15
    max_speed = 3.0     # pydub.speedup の実用上限

    edited = AudioSegment.silent(duration=0)

    for i in range(len(points_ms) - 1):
        start_ms = points_ms[i]
        end_ms = points_ms[i + 1]
        segment = audio[start_ms:end_ms]

        # i=0 はキーワード前、i=1 は1回目後、i=2 は2回目後...
        rate = min(base_speed + i * speed_step, max_speed)

        if abs(rate - 1.0) < 1e-9:
            segment_out = segment
        else:
            segment_out = segment.speedup(
                playback_speed=rate,
                chunk_size=120,
                crossfade=20,
            )

        edited += segment_out
        print(f"区間{i+1}: {start_ms/1000:.2f}s - {end_ms/1000:.2f}s, speed={rate:.2f}")

    out_path = mp3_path.with_name(mp3_path.stem + ".keyword_multi_accel.mp3")
    edited.export(out_path, format="mp3")
    print(f"編集済みファイル: {out_path}")
    print(f"検出時刻(秒): {[round(t, 2) for t in trigger_times]}")

    