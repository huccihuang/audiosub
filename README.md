# audiosub

一个基于 Qwen3-ASR 的命令行工具，用于将音频文件转换为字幕文件。

## 特点

- Apple Silicon 专属优化：基于 MLX 框架，充分利用 Metal GPU 加速
- 准确度高：Qwen3-ASR-1.7B，中文及中英混排（技术词汇）识别优于 Whisper
- 智能断句：按标点拆分短字幕，标点丢弃；超长句在词间停顿处断开，兜底 24 字硬拆
- 词级时间戳：搭配 Qwen3-ForcedAligner，字幕与语音精准对齐
- 一键生成：简单命令即可生成 .srt 字幕文件

## 系统要求
- macOS（仅限 Apple Silicon 芯片，MLX 框架依赖 Metal GPU）
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

- 不支持 Intel 芯片的 Mac（MLX 框架仅支持 Apple Silicon）
- 支持的音频格式：MP3、WAV、M4A 等（经 ffmpeg 解码）
- 首次运行会自动下载 Qwen3-ASR-1.7B + ForcedAligner 模型（约 5GB）到 `~/.cache/huggingface/`，请确保网络连接正常
- 若模型下载中断报错（CAS Client Error），重试即可断点续传

## 依赖

- Python 3.12+
- mlx-qwen3-asr

## 许可证

MIT
