import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import fire
import numpy as np


def _fix_sherpa_dylib():
    '''sherpa-onnx macOS 轮子缺 libonnxruntime.dylib，运行时自动从 onnxruntime 包补上'''
    import glob
    import importlib.util
    import shutil as _shutil

    try:
        spec = importlib.util.find_spec("sherpa_onnx")
        if spec is None or not spec.submodule_search_locations:
            return
        lib_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
        target = lib_dir / "libonnxruntime.dylib"
        if target.exists():
            return
        ort_spec = importlib.util.find_spec("onnxruntime")
        if ort_spec is None or not ort_spec.submodule_search_locations:
            return
        ort_dir = Path(list(ort_spec.submodule_search_locations)[0])
        candidates = sorted(glob.glob(str(ort_dir / "capi" / "libonnxruntime*.dylib")))
        if candidates:
            _shutil.copy(candidates[-1], target)
    except Exception:
        pass


_fix_sherpa_dylib()

SAMPLE_RATE = 16000

# VAD 参数：按静音切段，12s 只是连续说话不断气时的强制上限
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_DURATION = 0.5
VAD_MIN_SPEECH_DURATION = 0.25
VAD_MAX_SPEECH_DURATION = 12

# 模型：SenseVoice int8（~250MB），中日韩粤英
MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{MODEL_DIR_NAME}.tar.bz2"
)
VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad.onnx"
)

# 遇到这些标点就断句，标点本身丢弃（字幕不需要标点）
SPLIT_PUNCTS = set("，。？！、；：,…—·「」『』（）()[]{}<>《》“”\"\"''・､。.:;,?!")

# 无标点超长句的兜底：单行超 18 字或超 4 秒就硬拆
MAX_CHARS_PER_LINE = 18
MAX_SECONDS_PER_LINE = 4.0


def main():
    fire.Fire(convert_audio_to_subtitle)


def convert_audio_to_subtitle(audio_file):
    """
    将音频文件转换为字幕文件

    使用示例：
        audiosub audio.mp3

    将在当前目录生成 audio.srt 字幕文件
    """
    # 检查输入文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误：输入文件不存在 - {audio_file}")
        return

    if shutil.which("ffmpeg") is None:
        print("错误：需要先安装 ffmpeg（brew install ffmpeg）")
        return

    # 获取输入文件的基本名称（不含扩展名）
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    # 在当前工作目录生成输出文件
    output_file = f"{base_name}.srt"

    print(f"正在处理音频文件: {audio_file}")
    print(f"输出字幕文件: {output_file}")

    try:
        audio_to_subtitle(audio_file, output_file)
        print(f"✓ 字幕文件已生成: {output_file}")
    except Exception as e:
        print(f"错误：处理失败 - {str(e)}")
        raise


def audio_to_subtitle(audio_file_path, output_file_path):
    '''将音频文件转换为字幕文件'''
    import sherpa_onnx

    model_path, tokens_path, vad_path = ensure_models()

    print("正在解码音频...")
    samples = decode_audio(audio_file_path)
    duration = len(samples) / SAMPLE_RATE
    print(f"音频时长: {duration:.1f}s")

    print("正在创建识别器...")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        tokens=tokens_path,
        num_threads=4,
        use_itn=True,
        debug=False,
    )

    print("正在 VAD 切段...")
    segments = vad_cut(samples, vad_path)
    print(f"切出 {len(segments)} 个语音段")

    # 逐段识别 → 按标点拆句
    subtitles = []
    for i, (seg_start, seg_samples) in enumerate(segments):
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, seg_samples)
        recognizer.decode_stream(stream)

        result = stream.result
        text = (result.text or "").strip()
        if not text or text in (".", "The."):
            continue
        tokens = [str(t) for t in result.tokens]
        timestamps = [float(t) for t in result.timestamps]

        sentences = split_to_sentences(tokens, timestamps, seg_start)
        subtitles.extend(sentences)
        print(f"\r正在识别: {i + 1}/{len(segments)}", end="", flush=True)
    if segments:
        print()

    subtitles.sort(key=lambda s: s["start"])

    # 存放到文件中
    with open(output_file_path, "w", encoding="utf-8") as f:
        for idx, sub in enumerate(subtitles):
            f.write(
                f"{idx + 1}\n"
                f"{format_time(sub['start'])} --> {format_time(sub['end'])}\n"
                f"{sub['text']}\n\n"
            )
    print(f"共生成 {len(subtitles)} 条字幕")


def ensure_models():
    '''首次运行自动下载模型到 ~/.cache/audiosub/，返回 (model, tokens, vad) 路径'''
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "audiosub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_dir = cache_dir / MODEL_DIR_NAME
    model_path = model_dir / "model.int8.onnx"
    tokens_path = model_dir / "tokens.txt"
    vad_path = cache_dir / "silero_vad.onnx"

    if not (model_path.is_file() and tokens_path.is_file()):
        print(f"首次运行，正在下载 SenseVoice 模型（约 250MB）到 {cache_dir}...")
        archive = cache_dir / f"{MODEL_DIR_NAME}.tar.bz2"
        download(MODEL_URL, archive)
        print("正在解压...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(cache_dir)
        archive.unlink()

    if not vad_path.is_file():
        print("正在下载 VAD 模型...")
        download(VAD_URL, vad_path)

    return str(model_path), str(tokens_path), str(vad_path)


def download(url, dest):
    def _progress(blocks, block_size, total):
        done = blocks * block_size
        pct = min(100.0, done * 100 / total) if total > 0 else 0
        print(f"\r  下载中: {pct:.0f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def decode_audio(audio_file_path):
    '''用 ffmpeg 把任意音频解码成 16k 单声道 float32'''
    cmd = [
        "ffmpeg", "-v", "error", "-i", audio_file_path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败: {proc.stderr.decode()[:500]}")
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if len(samples) == 0:
        raise RuntimeError("音频解码结果为空")
    return samples


def vad_cut(samples, vad_model_path):
    '''VAD 按静音切段，返回 [(段起始秒, 段采样), ...]'''
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = vad_model_path
    config.silero_vad.threshold = VAD_THRESHOLD
    config.silero_vad.min_silence_duration = VAD_MIN_SILENCE_DURATION
    config.silero_vad.min_speech_duration = VAD_MIN_SPEECH_DURATION
    config.silero_vad.max_speech_duration = VAD_MAX_SPEECH_DURATION
    config.sample_rate = SAMPLE_RATE

    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
    window_size = config.silero_vad.window_size

    offset = 0
    n = len(samples)
    while offset + window_size <= n:
        vad.accept_waveform(samples[offset:offset + window_size])
        offset += window_size
    if offset < n:
        # 尾部不足一窗的补零喂入，避免丢音频
        tail = np.zeros(window_size, dtype=np.float32)
        tail[:n - offset] = samples[offset:]
        vad.accept_waveform(tail)
    vad.flush()

    segments = []
    while not vad.empty():
        front = vad.front
        start = front.start / SAMPLE_RATE
        seg_samples = np.array(front.samples, dtype=np.float32)
        vad.pop()
        if len(seg_samples) > 0:
            segments.append((start, seg_samples))
    return segments


def split_to_sentences(tokens, timestamps, seg_start):
    '''
    把一段识别结果按标点拆成短句，标点丢弃。
    用逐字时间戳定位每句起止；无标点超长句按 18 字 / 4s 硬拆兜底。
    '''
    # tokens 与 timestamps 对齐成 (字, 相对起, 相对止)
    chars = align_tokens(tokens, timestamps)

    # 按标点拆成子句
    clauses = []
    current = []
    for ch, start, end in chars:
        if not ch:
            continue
        if ch in SPLIT_PUNCTS:
            if current:
                clauses.append(current)
                current = []
            continue
        if len(ch) > 1 and ch[-1] in SPLIT_PUNCTS:
            ch = ch[:-1].strip()
            if ch:
                current.append((ch, start, end))
            if current:
                clauses.append(current)
                current = []
            continue
        current.append((ch, start, end))
    if current:
        clauses.append(current)

    # 子句转字幕，超长兜底硬拆
    subtitles = []
    for clause in clauses:
        for chunk in hard_split(clause):
            text = "".join(c[0] for c in chunk).strip()
            if not text:
                continue
            subtitles.append({
                "start": seg_start + chunk[0][1],
                "end": seg_start + chunk[-1][2],
                "text": text,
            })
    return subtitles


def align_tokens(tokens, timestamps):
    '''对齐 token 与时间戳，返回 [(字, 相对起, 相对止)]，过滤特殊标记'''
    pairs = []
    for i, tok in enumerate(tokens):
        if tok.startswith("<") and tok.endswith(">"):
            continue  # <|zh|> 这类标记跳过
        surface = tok.replace("▁", " ")
        if surface and not surface.strip():
            surface = " "
        end = timestamps[i] if i < len(timestamps) else None
        pairs.append((surface, end))

    if not pairs:
        return []

    # timestamps[i] 是第 i 个 token 在段内的结束时间
    chars = []
    prev_end = 0.0
    for i, (surface, end) in enumerate(pairs):
        if end is None:
            end = prev_end + 0.1  # 缺时间戳时估一个
        end = max(end, prev_end)
        chars.append((surface.strip() if surface.strip() else surface, prev_end, end))
        prev_end = end
    return [(c, s, e) for c, s, e in chars if c]


def hard_split(clause):
    '''单句超 18 字或超 4s 则按字数/时长贪心硬拆'''
    text_len = sum(len(c[0]) for c in clause)
    dur = clause[-1][2] - clause[0][1] if clause else 0
    if text_len <= MAX_CHARS_PER_LINE and dur <= MAX_SECONDS_PER_LINE:
        return [clause]

    chunks, current, cur_len, cur_start = [], [], 0, None
    for item in clause:
        ch, start, end = item
        if current and (cur_len + len(ch) > MAX_CHARS_PER_LINE
                        or end - cur_start > MAX_SECONDS_PER_LINE):
            chunks.append(current)
            current, cur_len, cur_start = [], 0, None
        if cur_start is None:
            cur_start = start
        current.append(item)
        cur_len += len(ch)
    if current:
        chunks.append(current)
    return chunks


def format_time(time):
    '''将float格式的时间转换为字幕要求的格式 hh:mm:ss,ms'''
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
