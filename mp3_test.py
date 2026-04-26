from __future__ import annotations

import argparse
import shutil
import sys
import unicodedata
from pathlib import Path

import librosa
import soundfile as sf
import torch
import whisper
from pydub import AudioSegment


def get_runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller one-file実行時は展開先(_MEIPASS)を優先
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


def resolve_tool_path(tool_name: str, search_dirs: list[Path] | None = None) -> str | None:
    if search_dirs:
        for base in search_dirs:
            candidate = base / f"{tool_name}.exe"
            if candidate.exists():
                return str(candidate)

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


def normalize_ja(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def find_speed_events(
    result: dict,
    keyword_speed_delta: dict[str, float],
) -> list[tuple[float, float, str]]:
    normalized_rules = {
        normalize_ja(keyword): delta for keyword, delta in keyword_speed_delta.items()
    }
    events: list[tuple[float, float, str]] = []

    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            token = normalize_ja(w.get("word", ""))
            if not token:
                continue

            for keyword, delta in normalized_rules.items():
                if keyword in token:
                    t = float(w.get("end", seg.get("end", 0.0)))
                    if t > 0:
                        events.append((t, delta, keyword))

    events.sort(key=lambda x: x[0])
    return events


def change_playback_speed(segment: AudioSegment, speed: float) -> AudioSegment:
    if speed <= 0:
        raise ValueError("speed must be > 0")

    if abs(speed - 1.0) < 1e-9:
        return segment

    changed = segment._spawn(
        segment.raw_data,
        overrides={"frame_rate": int(segment.frame_rate * speed)},
    )
    return changed.set_frame_rate(segment.frame_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisperで文字起こしし、キーワード検出でMP3速度を動的変更する"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "sample02.mp3",
        help="入力MP3ファイル",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="出力ディレクトリ（省略時は入力ファイルと同じ場所）",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Whisperモデル名（例: tiny, base, small, medium, large）",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="文字起こし言語コード（例: ja, en）",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="推論デバイス",
    )
    parser.add_argument(
        "--ffmpeg-dir",
        type=Path,
        default=None,
        help="ffmpeg.exe と ffprobe.exe があるフォルダ",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mp3_path = args.input.resolve()
    if not mp3_path.exists():
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    output_dir = args.output_dir.resolve() if args.output_dir else mp3_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_base = get_runtime_base_dir()
    app_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    search_dirs: list[Path] = []
    if args.ffmpeg_dir:
        search_dirs.append(args.ffmpeg_dir.resolve())
    search_dirs.extend(
        [
            runtime_base,
            runtime_base / "bin",
            app_dir,
            app_dir / "bin",
            Path.cwd(),
            Path.cwd() / "bin",
        ]
    )

    # 順序を維持したまま重複を除去
    dedup_search_dirs = list(dict.fromkeys(search_dirs))

    ffmpeg_path = resolve_tool_path("ffmpeg", dedup_search_dirs)
    ffprobe_path = resolve_tool_path("ffprobe", dedup_search_dirs)

    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError(
            "ffmpeg/ffprobe が見つかりません。インストール後にターミナルを再起動してください。"
        )

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda が指定されましたが、CUDAが利用できません。")

    print(f"Whisper device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = whisper.load_model(args.model, device=device)

    result = model.transcribe(
        str(mp3_path),
        language=args.language,
        fp16=(device == "cuda"),
        word_timestamps=True,
    )

    print(result["text"])

    audio_data = AudioSegment.from_mp3(mp3_path)

    fast_wav_path = output_dir / f"{mp3_path.stem}.fast.wav"
    y, sr = librosa.load(mp3_path)
    y_fast = librosa.effects.time_stretch(y, rate=1.5)
    sf.write(fast_wav_path, y_fast, sr)
    print(f"fast wav: {fast_wav_path}")

    keyword_speed_delta = {
        "光": +0.15,
        "風": +0.15,
        "夢": +0.15,
        "雲": -0.15,
    }
    speed_events = find_speed_events(result, keyword_speed_delta)

    if not speed_events:
        print("指定単語（光/風/夢/雲）が見つからなかったため、速度編集はスキップ")
        return

    base_speed = 1.0
    max_speed = 3.0
    min_speed = 0.5

    event_deltas_ms: dict[int, float] = {}
    event_labels_ms: dict[int, list[str]] = {}
    for t_sec, delta, keyword in speed_events:
        t_ms = int(t_sec * 1000)
        if 0 < t_ms < len(audio_data):
            event_deltas_ms[t_ms] = event_deltas_ms.get(t_ms, 0.0) + delta
            event_labels_ms.setdefault(t_ms, []).append(keyword)

    event_points_ms = sorted(event_deltas_ms.keys())
    points_ms = [0] + event_points_ms + [len(audio_data)]

    edited = AudioSegment.silent(duration=0)
    current_speed = base_speed

    for i in range(len(points_ms) - 1):
        start_ms = points_ms[i]
        end_ms = points_ms[i + 1]
        segment = audio_data[start_ms:end_ms]

        segment_out = change_playback_speed(segment, current_speed)
        edited += segment_out
        print(
            f"区間{i + 1}: {start_ms / 1000:.2f}s - {end_ms / 1000:.2f}s, "
            f"speed={current_speed:.2f}"
        )

        if end_ms in event_deltas_ms:
            delta = event_deltas_ms[end_ms]
            labels = ",".join(event_labels_ms.get(end_ms, []))
            current_speed = max(min(current_speed + delta, max_speed), min_speed)
            print(
                f"  -> キーワード[{labels}]検出: delta={delta:+.2f}, "
                f"次区間speed={current_speed:.2f}"
            )

    out_path = output_dir / f"{mp3_path.stem}.keyword_dynamic_speed.mp3"
    edited.export(out_path, format="mp3")
    print(f"動的速度編集ファイル: {out_path}")
    print(
        "検出イベント(秒, 単語, 変化量): "
        f"{[(round(t, 2), w, d) for t, d, w in speed_events]}"
    )


if __name__ == "__main__":
    main()
