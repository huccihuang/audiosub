# audiosub

一个基于 SenseVoice 的命令行工具，用于将音频文件转换为字幕文件。

## 特点

- 全平台：macOS / Linux / Windows，CPU 即可跑（无需 Apple Silicon）
- 速度极快：SenseVoice int8 + silero-vad，M4 上 5 分钟音频约 5 秒
- 中文准确度高：中文识别明显优于 Whisper turbo，自带标点+ITN
- 标点即断句：按标点自动拆成短字幕，标点丢弃，无标点超 18 字/4 秒硬拆兜底
- 一键生成：简单命令即可生成 .srt 字幕文件

## 系统要求
- Python 3.12+
- ffmpeg（`brew install ffmpeg`）
- uv

## 使用方法

通过 uv 的 `uvx` 命令安装并使用。

```bash
uvx audiosub <filename>
# 将在当前目录生成 audio.srt 字幕文件
```

## 注意事项

- 支持的音频格式：MP3、WAV、M4A 等（经 ffmpeg 解码）
- 首次运行会自动下载 SenseVoice int8 + VAD 模型（约 250MB）到 `~/.cache/audiosub/`，请确保网络连接正常
- SenseVoice-Small 对中英混排的技术词汇（如 SQLite、Postgres 等）识别较弱；若主要处理此类内容且使用 Apple Silicon，可回退 0.3.x（mlx-whisper 版）


## 依赖

- Python 3.12+
- sherpa-onnx
- onnxruntime
- numpy

## 许可证

MIT