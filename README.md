# audiosub

一个基于 mlx-whisper 的命令行工具，用于将音频文件转换为字幕文件。

## 特点

- **资源占用小**：基于 mlx-whisper，专为 Apple Silicon 优化
- **转换速度快**：利用 Apple Metal 加速，处理速度快
- **准确度高**：使用 OpenAI Whisper 模型，支持多种语言
- **使用简单**：命令行操作，一键生成字幕文件

## Prerequisites

- Python 3.12+
- macOS
- uv

## 使用方法

通过 uv 的 `uvx` 命令安装并使用。

```bash
uvx audiosub <filename>
# 将在当前目录生成 audio.srt 字幕文件
```

## 注意事项

- **仅支持 macOS**：由于使用了 mlx 库，本工具仅能在 macOS 系统上运行
- **Apple Silicon 优化**：在 Apple Silicon (M1/M2/M3) 芯片上性能最佳
- **支持的音频格式**：支持常见的音频格式，如 MP3、WAV、M4A 等
- **首次运行**：首次使用时会自动下载 Whisper 模型，请确保网络连接正常

## 依赖

- Python 3.12+
- mlx-whisper
- click

## 许可证

MIT