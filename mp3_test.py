from __future__ import annotations

import argparse
import queue
import shutil
import sys
import threading
import unicodedata
from pathlib import Path
from typing import Any, Callable

import librosa
import soundfile as sf
import torch
import whisper
from pydub import AudioSegment

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None


def get_runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
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


DEFAULT_KEYWORD_SPEED_DELTAS: dict[str, float] = {
    "光": +0.15,
    "風": +0.15,
    "夢": +0.15,
    "雲": -0.15,
}


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
        "--gui",
        action="store_true",
        help="GUIを起動する",
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


def resolve_ffmpeg_paths(ffmpeg_dir: Path | None = None) -> tuple[str, str]:
    runtime_base = get_runtime_base_dir()
    app_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    search_dirs: list[Path] = []
    if ffmpeg_dir:
        search_dirs.append(ffmpeg_dir.resolve())
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

    dedup_search_dirs = list(dict.fromkeys(search_dirs))

    ffmpeg_path = resolve_tool_path("ffmpeg", dedup_search_dirs)
    ffprobe_path = resolve_tool_path("ffprobe", dedup_search_dirs)

    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError(
            "ffmpeg/ffprobe が見つかりません。インストール後にターミナルを再起動してください。"
        )

    return ffmpeg_path, ffprobe_path


def process_audio(
    mp3_path: Path,
    output_dir: Path | None = None,
    model_name: str = "medium",
    language: str = "ja",
    device_name: str = "auto",
    ffmpeg_dir: Path | None = None,
    keyword_speed_delta: dict[str, float] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Path | None]:
    def emit(message: str) -> None:
        if log:
            log(message)
        else:
            print(message)

    mp3_path = mp3_path.resolve()
    if not mp3_path.exists():
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    resolved_output_dir = output_dir.resolve() if output_dir else mp3_path.parent
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path, ffprobe_path = resolve_ffmpeg_paths(ffmpeg_dir)
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda が指定されましたが、CUDAが利用できません。")

    emit(f"Whisper device: {device}")
    if device == "cuda":
        emit(f"GPU: {torch.cuda.get_device_name(0)}")

    model = whisper.load_model(model_name, device=device)

    result = model.transcribe(
        str(mp3_path),
        language=language,
        fp16=(device == "cuda"),
        word_timestamps=True,
    )

    emit(result["text"])

    audio_data = AudioSegment.from_mp3(mp3_path)

    fast_wav_path = resolved_output_dir / f"{mp3_path.stem}.fast.wav"
    y, sr = librosa.load(mp3_path)
    y_fast = librosa.effects.time_stretch(y, rate=1.5)
    sf.write(fast_wav_path, y_fast, sr)
    emit(f"fast wav: {fast_wav_path}")

    keyword_speed_delta = keyword_speed_delta or DEFAULT_KEYWORD_SPEED_DELTAS
    if not keyword_speed_delta:
        raise ValueError("少なくとも1つのキーワードを選択してください。")

    speed_events = find_speed_events(result, keyword_speed_delta)

    if not speed_events:
        emit("指定キーワードが見つからなかったため、速度編集はスキップ")
        return {"fast_wav": fast_wav_path, "edited_mp3": None}

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
        emit(
            f"区間{i + 1}: {start_ms / 1000:.2f}s - {end_ms / 1000:.2f}s, speed={current_speed:.2f}"
        )

        if end_ms in event_deltas_ms:
            delta = event_deltas_ms[end_ms]
            labels = ",".join(event_labels_ms.get(end_ms, []))
            current_speed = max(min(current_speed + delta, max_speed), min_speed)
            emit(
                f"  -> キーワード[{labels}]検出: delta={delta:+.2f}, 次区間speed={current_speed:.2f}"
            )

    out_path = resolved_output_dir / f"{mp3_path.stem}.keyword_dynamic_speed.mp3"
    edited.export(out_path, format="mp3")
    emit(f"動的速度編集ファイル: {out_path}")
    emit(
        "検出イベント(秒, 単語, 変化量): "
        f"{[(round(t, 2), w, d) for t, d, w in speed_events]}"
    )

    return {"fast_wav": fast_wav_path, "edited_mp3": out_path}


def launch_gui() -> None:
    if tk is None or filedialog is None or messagebox is None or ttk is None:
        raise RuntimeError("Tkinter が利用できません。GUI を起動できない環境です。")

    root = tk.Tk()
    root.title("MP3 Keyword Speed")
    root.geometry("780x580")
    root.minsize(740, 540)

    input_path_var = tk.StringVar(value=str(Path(__file__).resolve().parent / "sample02.mp3"))
    output_dir_var = tk.StringVar(value="")
    model_var = tk.StringVar(value="medium")
    language_var = tk.StringVar(value="ja")
    device_var = tk.StringVar(value="auto")
    ffmpeg_dir_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="待機中")
    progress_var = tk.DoubleVar(value=0.0)
    keyword_rows: list[dict[str, Any]] = []

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    header = ttk.Frame(root, padding=16)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)

    ttk.Label(header, text="MP3 Keyword Speed", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
    ttk.Label(header, text="キーワード検出でMP3の速度を変えるデスクトップアプリ", foreground="#555").grid(row=1, column=0, sticky="w", pady=(4, 0))

    body = ttk.Frame(root, padding=(16, 0, 16, 16))
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(1, weight=1)
    body.rowconfigure(8, weight=1)

    def add_entry_row(row: int, label: str, var: Any) -> Any:
        ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", pady=6)
        entry = ttk.Entry(body, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        return entry

    add_entry_row(0, "入力MP3", input_path_var)
    add_entry_row(1, "出力先", output_dir_var)
    add_entry_row(2, "モデル", model_var)
    add_entry_row(3, "言語", language_var)

    ttk.Label(body, text="デバイス").grid(row=4, column=0, sticky="w", pady=6)
    device_combo = ttk.Combobox(body, textvariable=device_var, values=["auto", "cpu", "cuda"], state="readonly")
    device_combo.grid(row=4, column=1, sticky="ew", pady=6)

    keyword_frame = ttk.LabelFrame(body, text="キーワード編集", padding=10)
    keyword_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 8))
    keyword_frame.columnconfigure(0, weight=1)

    def refresh_keyword_rows() -> None:
        for index, row in enumerate(keyword_rows):
            row["frame"].grid(row=index, column=0, sticky="ew", pady=2)

    def remove_keyword_row(row_data: dict[str, Any]) -> None:
        if row_data not in keyword_rows:
            return
        row_data["frame"].destroy()
        keyword_rows.remove(row_data)
        refresh_keyword_rows()

    def add_keyword_row(keyword: str = "", delta: float = 0.15, enabled: bool = True) -> None:
        row_frame = ttk.Frame(keyword_frame)
        row_frame.columnconfigure(1, weight=1)
        row_frame.columnconfigure(3, weight=0)

        enabled_var = tk.BooleanVar(value=enabled)
        keyword_var = tk.StringVar(value=keyword)
        delta_var = tk.StringVar(value=f"{delta:+.2f}")

        row_data: dict[str, Any] = {
            "frame": row_frame,
            "enabled_var": enabled_var,
            "keyword_var": keyword_var,
            "delta_var": delta_var,
        }
        keyword_rows.append(row_data)

        ttk.Checkbutton(row_frame, variable=enabled_var).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(row_frame, textvariable=keyword_var, width=16).grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(row_frame, text="変化量").grid(row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Entry(row_frame, textvariable=delta_var, width=8).grid(row=0, column=3, sticky="w", padx=(0, 8))
        ttk.Button(row_frame, text="削除", command=lambda: remove_keyword_row(row_data)).grid(row=0, column=4, sticky="e")

        refresh_keyword_rows()

    for keyword, default_delta in DEFAULT_KEYWORD_SPEED_DELTAS.items():
        add_keyword_row(keyword=keyword, delta=default_delta, enabled=True)

    keyword_actions = ttk.Frame(keyword_frame)
    keyword_actions.grid(row=len(keyword_rows), column=0, sticky="w", pady=(8, 0))

    def add_empty_keyword_row() -> None:
        add_keyword_row(keyword="", delta=0.15, enabled=True)
        keyword_actions.grid(row=len(keyword_rows), column=0, sticky="w", pady=(8, 0))

    ttk.Button(keyword_actions, text="行を追加", command=add_empty_keyword_row).pack(side="left", padx=(0, 6))
    ttk.Button(keyword_actions, text="すべて有効", command=lambda: [row["enabled_var"].set(True) for row in keyword_rows]).pack(side="left", padx=(0, 6))
    ttk.Button(keyword_actions, text="すべて無効", command=lambda: [row["enabled_var"].set(False) for row in keyword_rows]).pack(side="left")

    add_entry_row(6, "ffmpeg フォルダ", ffmpeg_dir_var)

    progress = ttk.Progressbar(body, variable=progress_var, maximum=100)
    progress.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 8))

    log_box = tk.Text(body, height=16, wrap="word")
    log_box.grid(row=8, column=0, columnspan=2, sticky="nsew")
    log_box.configure(state="disabled")

    scrollbar = ttk.Scrollbar(body, orient="vertical", command=log_box.yview)
    scrollbar.grid(row=8, column=2, sticky="ns")
    log_box.configure(yscrollcommand=scrollbar.set)

    status = ttk.Label(root, textvariable=status_var, anchor="w", padding=(16, 0, 16, 12))
    status.grid(row=2, column=0, sticky="ew")

    log_queue: queue.Queue[str] = queue.Queue()
    worker_state = {"running": False}

    def write_log(message: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", message + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def drain_logs() -> None:
        try:
            while True:
                write_log(log_queue.get_nowait())
        except queue.Empty:
            pass

        if worker_state["running"]:
            root.after(100, drain_logs)

    def enqueue_log(message: str) -> None:
        log_queue.put(message)

    def browse_input() -> None:
        path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")])
        if path:
            input_path_var.set(path)

    def browse_output() -> None:
        path = filedialog.askdirectory()
        if path:
            output_dir_var.set(path)

    def browse_ffmpeg() -> None:
        path = filedialog.askdirectory()
        if path:
            ffmpeg_dir_var.set(path)

    button_bar = ttk.Frame(body)
    button_bar.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 10))

    ttk.Button(button_bar, text="入力を選ぶ", command=browse_input).pack(side="left", padx=(0, 6))
    ttk.Button(button_bar, text="出力先を選ぶ", command=browse_output).pack(side="left", padx=(0, 6))
    ttk.Button(button_bar, text="ffmpeg を選ぶ", command=browse_ffmpeg).pack(side="left", padx=(0, 6))

    run_button = ttk.Button(button_bar, text="処理開始")
    run_button.pack(side="right")

    def run_task() -> None:
        try:
            status_var.set("処理中...")
            progress_var.set(10)
            enqueue_log("処理を開始します")

            input_path = Path(input_path_var.get().strip())
            output_dir = Path(output_dir_var.get().strip()) if output_dir_var.get().strip() else None
            ffmpeg_dir = Path(ffmpeg_dir_var.get().strip()) if ffmpeg_dir_var.get().strip() else None

            selected_keywords: dict[str, float] = {}
            for row in keyword_rows:
                if not row["enabled_var"].get():
                    continue

                keyword = row["keyword_var"].get().strip()
                if not keyword:
                    continue

                try:
                    delta = float(row["delta_var"].get().strip())
                except ValueError as exc:
                    raise ValueError(f"キーワード '{keyword}' の変化量が数値ではありません") from exc

                selected_keywords[keyword] = delta

            if not selected_keywords:
                raise ValueError("少なくとも1つのキーワードを有効にして入力してください。")

            enqueue_log("選択キーワード: " + ", ".join(f"{k}({v:+.2f})" for k, v in selected_keywords.items()))

            result = process_audio(
                mp3_path=input_path,
                output_dir=output_dir,
                model_name=model_var.get().strip() or "medium",
                language=language_var.get().strip() or "ja",
                device_name=device_var.get().strip() or "auto",
                ffmpeg_dir=ffmpeg_dir,
                keyword_speed_delta=selected_keywords,
                log=enqueue_log,
            )

            progress_var.set(100)
            status_var.set("完了")
            enqueue_log(f"完了: {result}")
            root.after(0, lambda: messagebox.showinfo("完了", "処理が完了しました。"))
        except Exception as exc:
            status_var.set("エラー")
            enqueue_log(f"エラー: {exc}")
            root.after(0, lambda: messagebox.showerror("エラー", str(exc)))
        finally:
            worker_state["running"] = False
            root.after(0, lambda: run_button.configure(state="normal"))
            root.after(0, drain_logs)

    def start_task() -> None:
        if worker_state["running"]:
            return
        worker_state["running"] = True
        run_button.configure(state="disabled")
        threading.Thread(target=run_task, daemon=True).start()
        root.after(100, drain_logs)

    run_button.configure(command=start_task)
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.gui or len(sys.argv) == 1:
        launch_gui()
        return

    process_audio(
        mp3_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model,
        language=args.language,
        device_name=args.device,
        ffmpeg_dir=args.ffmpeg_dir,
    )


if __name__ == "__main__":
    main()
