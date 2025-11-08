import os
import click
import mlx_whisper


def format_time(time):
    '''将float格式的时间转换为字幕要求的格式 hh:mm:ss,ms'''
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def audio_to_subtitle(audio_file_path, output_file_path):
    '''将音频文件转换为字幕文件'''

    # 获取词级时间戳
    result = mlx_whisper.transcribe(
        audio_file_path,
        word_timestamps=True
    )

    # 存放到文件中
    with open(output_file_path, "w") as f:
        for idx, segment in enumerate(result["segments"]):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            f.write(
                f"{idx + 1}\n"
                f"{start_time} --> {end_time}\n"
                f"{segment['text']}\n\n"
            )


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
def main(audio_file):
    """将音频文件转换为字幕文件

    使用示例：
        mksub audio.mp3

    将在当前目录生成 audio.srt 字幕文件
    """
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        click.echo(f"错误：文件 '{audio_file}' 不存在", err=True)
        return

    # 获取输入文件的基本名称（不含扩展名）
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    # 在当前工作目录生成输出文件
    output_file = f"{base_name}.srt"

    click.echo(f"正在处理音频文件: {audio_file}")
    click.echo(f"输出字幕文件: {output_file}")

    try:
        audio_to_subtitle(audio_file, output_file)
        click.echo(f"✓ 字幕文件已生成: {output_file}")
    except Exception as e:
        click.echo(f"错误：处理失败 - {str(e)}", err=True)
        raise


if __name__ == "__main__":
    main()
