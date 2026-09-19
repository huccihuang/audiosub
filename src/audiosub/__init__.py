import os
import fire

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# 字幕拆行规则：优先按标点断句；超长句在词间停顿处断开，
# 找不到停顿时放宽到 SOFT_MAX_CHARS，再不行硬切
MAX_CHARS_PER_LINE = 18
MAX_SECONDS_PER_LINE = 4.0
SOFT_MAX_CHARS = 24
MIN_GAP_SECONDS = 0.1

PUNCTUATIONS = set("，。！？；：、…—·,.;:!?\"'()（）")


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
    """将音频文件转换为字幕文件"""
    from mlx_qwen3_asr import transcribe

    def show_progress(event):
        if event.get("event") == "chunk_completed":
            print(
                f"\r正在识别: {event['chunk_index']}/{event['total_chunks']}",
                end="",
                flush=True,
            )
        elif event.get("event") == "completed":
            print()

    result = transcribe(
        audio_file_path,
        model=MODEL_ID,
        return_timestamps=True,
        on_progress=show_progress,
    )

    if not result.segments:
        raise RuntimeError("识别结果为空或未获取到词级时间戳")

    entries = build_subtitle_entries(result.segments, result.text)

    with open(output_file_path, "w") as f:
        for idx, entry in enumerate(entries):
            f.write(
                f"{idx + 1}\n"
                f"{entry['start']} --> {entry['end']}\n"
                f"{entry['text']}\n\n"
            )


def build_subtitle_entries(words, text):
    """
    用词级时间戳 + 全文标点构建字幕条目：
    1. 把 text 中的标点映射回 word 流作为断句点（标点是 ASR 加的，
       词级时间戳流里没有标点 token）
    2. 按标点切成句子
    3. 超长句在停顿处拆分
    """
    breaks_after = map_punctuation_breaks(words, text)

    sentences, cur = [], []
    for wi, w in enumerate(words):
        cur.append(w)
        if wi in breaks_after:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    entries = []
    for sent in sentences:
        for group in split_long_sentence(sent):
            seg_text = "".join(w["text"] for w in group)
            if seg_text.strip():
                entries.append(
                    {
                        "start": format_time(group[0]["start"]),
                        "end": format_time(group[-1]["end"]),
                        "text": seg_text,
                    }
                )
    return entries


def map_punctuation_breaks(words, text):
    """
    把 text 中紧跟在某个词后面的标点，映射为该词之后的断句点。
    返回需要断句的词下标集合。
    """
    breaks_after = set()
    pos = 0
    for wi, w in enumerate(words):
        idx = text.find(w["text"], pos)
        if idx < 0:
            continue  # 对齐失败则跳过该词
        pos = idx + len(w["text"])
        if pos < len(text) and text[pos] in PUNCTUATIONS:
            breaks_after.add(wi)
    return breaks_after


def split_long_sentence(sentence):
    """
    拆分超长句：超过 MAX_CHARS_PER_LINE 字或 MAX_SECONDS_PER_LINE 秒时，
    在目标位置附近找最大的词间间隔处断开；间隔太小且剩余不长则不拆
    （整句放出，由 SOFT_MAX_CHARS 兜底防止无限长）。
    """
    text_len = sum(len(w["text"]) for w in sentence)
    duration = sentence[-1]["end"] - sentence[0]["start"]
    if text_len <= MAX_CHARS_PER_LINE and duration <= MAX_SECONDS_PER_LINE:
        return [sentence]

    groups, start_i, acc = [], 0, 0
    for i in range(len(sentence)):
        acc += len(sentence[i]["text"])
        if acc > MAX_CHARS_PER_LINE or (
            sentence[i]["end"] - sentence[start_i]["start"]
        ) > MAX_SECONDS_PER_LINE:
            # 在目标位置附近 ±8 词内找最大词间间隔
            lo = max(start_i + 3, i - 8)
            hi = min(i + 1, len(sentence))
            best_j, best_gap = i, -1.0
            for j in range(lo, hi):
                gap = sentence[j]["start"] - sentence[j - 1]["end"]
                if gap > best_gap:
                    best_gap, best_j = gap, j
            # 间隔太小且剩余不长 → 不拆，整句放出
            rest = sum(len(x["text"]) for x in sentence[best_j:])
            if best_gap < MIN_GAP_SECONDS and rest + acc <= SOFT_MAX_CHARS:
                continue
            groups.append(sentence[start_i:best_j])
            start_i = best_j
            acc = sum(len(x["text"]) for x in sentence[start_i : i + 1])
    groups.append(sentence[start_i:])

    # 拆完仍超 SOFT_MAX_CHARS 的组递归再拆
    out = []
    for g in groups:
        if sum(len(w["text"]) for w in g) > SOFT_MAX_CHARS and len(g) > 3:
            out.extend(split_long_sentence(g))
        else:
            out.append(g)
    return out


def format_time(time):
    """将float格式的时间转换为字幕要求的格式 hh:mm:ss,ms"""
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    milliseconds = int((time - int(time)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
