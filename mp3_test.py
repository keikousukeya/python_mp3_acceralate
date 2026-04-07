from pathlib import Path
import shutil
import winsound
import torch
import whisper
import librosa
import soundfile as sf

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

mp3_path = Path(__file__).resolve().parent / "sample.mp3"

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