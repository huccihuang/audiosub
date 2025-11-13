# audiosub

一个基于 mlx-whisper 的命令行工具，用于将音频文件转换为字幕文件。

## 特点

- Apple Silicon 专属优化：仅支持 M1/M2/M3 等 Apple 芯片
- 速度极快：基于 Metal 加速的 mlx 框架
- 占用极低：高效利用 GPU 资源
- 准确度高：采用 OpenAI Whisper 模型，支持多语言识别
- 一键生成：简单命令即可生成 .srt 字幕文件

## 系统要求
- macOS（仅限 Apple Silicon 芯片）
- Python 3.12+
- uv￼

## 使用方法

通过 uv 的 `uvx` 命令安装并使用。

```bash
uvx audiosub <filename>
# 将在当前目录生成 audio.srt 字幕文件
```

## 注意事项

- 不支持 Intel 芯片的 Mac（mlx 框架仅支持 Apple Silicon）
- 支持的音频格式：MP3、WAV、M4A 等
- 首次运行会自动下载 Whisper 模型，请确保网络连接正常


## 依赖

- Python 3.12+
- mlx-whisper
- click

## 许可证

MIT