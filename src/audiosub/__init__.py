import os
import click
import mlx_whisper


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
def main(audio_file):
    """将音频文件转换为字幕文件

    使用示例：
        mksub audio.mp3

    将在当前目录生成 audio.srt 字幕文件
    """
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


def audio_to_subtitle(audio_file_path, output_file_path):
    '''将音频文件转换为字幕文件'''

    # 获取词级时间戳
    result = mlx_whisper.transcribe(
        audio_file_path,
        path_or_hf_repo = 'mlx-community/whisper-turbo',
        language = 'zh',
        word_timestamps = True,
    )


    # 处理每个分段
    segments = []
    for segment in result["segments"]:
        segments += process_segment(segment)

    # 存放到文件中
    with open(output_file_path, "w") as f:
        for idx, segment in enumerate(segments):
            f.write(
                f"{idx + 1}\n"
                f"{segment['start']} --> {segment['end']}\n"
                f"{segment['text']}\n\n"
            )


def process_segment(segment):
    '''
    处理每个分段，包括
    1. 去掉行末的标点符号
    2. 行中有标点符号的句子拆分成两段或者更多
    3. 处理时间

    返回处理后的分段列表，每个分段的key包含start, end, text
    '''
    processed_segments = []
    punctuations = (',', '。', '？', '！', '?', '!')

    # 去掉行末的标点符号
    if segment['text'].endswith(punctuations):
        segment['text'] = segment['text'][:-1]

    # 行中有标点符号的句子拆分
    if ',' in segment['text']: # 如果句子中间有标点符号
        # 从words里面找带有标点符号的词，并从这个词开始，将前后拆成两段
        new_segment = {'text': ''}
        for word in segment['words']:
            if 'start' not in new_segment: # 如果是第一个词，记录开始时间
                new_segment['start'] = word['start']
            if word['word'].endswith(punctuations): 
                # 如果是最后一个词
                new_segment['end'] = word['end'] # 记录结束时间
                new_segment['text'] += word['word'][:-1] # 去掉标点符号
                processed_segments.append(new_segment) # 添加到分段列表
                new_segment = {'text': ''} # 重置新分段
            else:     
                new_segment['text'] += word['word'] # 将词添加进去
    else:
        processed_segments.append(
            {
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
            }
        )
    
    # 处理时间
    for seg in processed_segments:
        seg['start'] = format_time(seg['start'])
        seg['end'] = format_time(seg['end'])
        seg['text'] = seg['text'].strip()
    
    return processed_segments


def format_time(time):
    '''将float格式的时间转换为字幕要求的格式 hh:mm:ss,ms'''
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


if __name__ == "__main__":
    main('./iflow_cli_hevc.mp4')
