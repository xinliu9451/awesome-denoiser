import numpy as np
import librosa

def detect_silence(y, sr, frame_length=2048, hop_length=512, threshold=-40, min_silence_duration=0.5):
    # 计算RMS能量
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # 将RMS转换为dB
    db_rms = librosa.amplitude_to_db(rms, ref=np.max)

    # 找到高于阈值的帧
    loud_frames = np.where(db_rms > threshold)[0]

    # 如果没有足够大的声音，返回空列表
    if len(loud_frames) == 0:
        return []

    # 计算帧之间的间隔
    gaps = np.diff(loud_frames)

    # 找到大于最小静音持续时间的间隔
    silence_gaps = np.where(gaps > int(min_silence_duration * sr / hop_length))[0]

    # 构建保留片段的列表
    segments = []
    start = loud_frames[0]
    for gap in silence_gaps:
        end = loud_frames[gap]
        segments.append((start * hop_length, end * hop_length))
        start = loud_frames[gap + 1]

    # 处理最后一段音频
    last_loud_frame = loud_frames[-1]
    if last_loud_frame * hop_length < len(y) - hop_length:  # 确保不会超出音频长度
        segments.append((start * hop_length, (last_loud_frame + 1) * hop_length))
    else:
        segments.append((start * hop_length, len(y)))

    return segments

def remove_silence_from_both(y1, y2, sr, frame_length=2048, hop_length=512, threshold=-40, min_silence_duration=0.5):
    # 在第一个音频上检测非静音片段
    segments = detect_silence(y2, sr, frame_length, hop_length, threshold, min_silence_duration)

    # 如果没有检测到任何非静音片段，返回空数组
    if len(segments) == 0:
        return np.array([]), np.array([])

    # 从两个音频中提取非静音片段
    y1_out = np.concatenate([y1[start:end] for start, end in segments])
    y2_out = np.concatenate([y2[start:end] for start, end in segments])

    return y1_out