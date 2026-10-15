#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップ3: 曲のキー（調）を解析して「ドレミ」で分かる形に変換する

  音声 → クロマ特徴量 (12音それぞれの強さ) → 24種類のキーの型と照合 → 一番近いキー

表示例:  キー：ミ♭・メジャー（E♭ Major）／変ホ長調

使い方:
    python key_analyzer.py 曲.mp3
    python key_analyzer.py cache\\song_xxxx\\accompaniment.wav

librosa があれば librosa.feature.chroma_cqt を使い、無ければ numpy/scipy の簡易版で計算します。
"""

import argparse
import json
import os
import sys

import numpy as np

# Krumhansl-Kessler のキープロファイル (人がそのキーで各音をどれだけ「収まりが良い」と感じるか)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]   # ドレミファソラシ
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]   # 自然的短音階 (ラシドレミファソ の形)

# キーごとの表記 (楽譜で一般的な綴り。長調は♭系、短調は慣例に合わせる)
EN_MAJOR = ["C", "D♭", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]
EN_MINOR = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "G♯", "A", "B♭", "B"]

SOLFEGE = {"C": "ド", "D": "レ", "E": "ミ", "F": "ファ", "G": "ソ", "A": "ラ", "B": "シ"}
JP_LETTER = {"C": "ハ", "D": "ニ", "E": "ホ", "F": "ヘ", "G": "ト", "A": "イ", "B": "ロ"}

# 画面の音名 (pitch_core と同じく # 表記) と、キーに合わせた ♭ 表記
SHARP_NAMES = ["ド", "ド#", "レ", "レ#", "ミ", "ファ", "ファ#", "ソ", "ソ#", "ラ", "ラ#", "シ"]
FLAT_NAMES = ["ド", "レ♭", "レ", "ミ♭", "ミ", "ファ", "ソ♭", "ソ", "ラ♭", "ラ", "シ♭", "シ"]


def _to_doremi(en):
    """'E♭' → 'ミ♭'、'F♯' → 'ファ♯'"""
    return SOLFEGE[en[0]] + en[1:]


def _to_japanese(en):
    """'E♭' → '変ホ'、'F♯' → '嬰ヘ'"""
    base = JP_LETTER[en[0]]
    if en.endswith("♭"):
        return "変" + base
    if en.endswith("♯"):
        return "嬰" + base
    return base


def describe_key(tonic_pc, mode):
    """主音 (0=ド〜11=シ) とモード ('major'/'minor') → 表示用の情報一式。"""
    if mode == "major":
        en = EN_MAJOR[tonic_pc]
        steps, en_mode, jp_mode, solf_mode = MAJOR_STEPS, "Major", "長調", "メジャー"
        rel_pc, rel_mode = (tonic_pc + 9) % 12, "minor"
    else:
        en = EN_MINOR[tonic_pc]
        steps, en_mode, jp_mode, solf_mode = MINOR_STEPS, "Minor", "短調", "マイナー"
        rel_pc, rel_mode = (tonic_pc + 3) % 12, "major"

    scale_pcs = [(tonic_pc + s) % 12 for s in steps]
    use_flats = "♭" in en or (mode == "major" and tonic_pc == 5) or (mode == "minor" and tonic_pc in (2, 7, 0, 5))
    names = FLAT_NAMES if use_flats else SHARP_NAMES
    english = f"{en} {en_mode}"
    doremi = f"{_to_doremi(en)}・{solf_mode}"
    japanese = f"{_to_japanese(en)}{jp_mode}"
    rel_en = (EN_MINOR if rel_mode == "minor" else EN_MAJOR)[rel_pc]
    return {
        "tonic_pc": tonic_pc,
        "mode": mode,
        "english": english,                 # E♭ Major
        "doremi": doremi,                   # ミ♭・メジャー
        "japanese": japanese,               # 変ホ長調
        "display": f"キー：{doremi}（{english}）／{japanese}",
        "short": f"{doremi}（{english}）",
        "scale_pcs": scale_pcs,             # 画面で音階の線を強調するのに使う
        "scale_doremi": [names[p] for p in scale_pcs],
        "relative": f"{_to_japanese(rel_en)}{'短調' if rel_mode == 'minor' else '長調'}",  # 平行調
        "accidentals": _accidentals(tonic_pc, mode, use_flats),
    }


def _accidentals(tonic_pc, mode, use_flats):
    """調号 (♯/♭ の数)。キーの綴り (♯系/♭系) に合わせて数える。"""
    major_pc = tonic_pc if mode == "major" else (tonic_pc + 3) % 12
    if use_flats:
        fifths = {0: 0, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5, 6: -6, 11: -7}.get(major_pc)
    else:
        fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7}.get(major_pc)
    if not fifths:
        return "調号なし"
    return f"♯×{fifths}" if fifths > 0 else f"♭×{-fifths}"


# ---------------------------------------------------------------------------
# クロマ特徴量
# ---------------------------------------------------------------------------
def load_mono(path, sr=22050, max_sec=None):
    try:
        import librosa

        y, sr = librosa.load(path, sr=sr, mono=True, duration=max_sec)
        return y, sr
    except ImportError:
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        from math import gcd

        file_sr, y = wavfile.read(path)
        y = y.astype(np.float32) / (32768.0 if y.dtype == np.int16 else 1.0)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if file_sr != sr:
            g = gcd(file_sr, sr)
            y = resample_poly(y, sr // g, file_sr // g)
        if max_sec:
            y = y[: int(max_sec * sr)]
        return y.astype(np.float32), sr


def chroma_numpy(y, sr, n_fft=8192, hop=2048, fmin=65.0, fmax=2100.0):
    """STFT からのクロマ (librosa が無い時用)。"""
    from scipy.signal import stft

    from scipy.ndimage import median_filter

    _, _, Z = stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    mag = np.abs(Z)
    # ドラムなどの雑音 (なだらかな成分) に対し、周囲より 12dB 以上突き出た「音程の山」だけ残す
    ratio = mag / (median_filter(mag, size=(31, 1), mode="nearest") + 1e-9)
    mag = np.where(ratio > 4.0, np.log(ratio), 0.0)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    sel = (freqs >= fmin) & (freqs <= fmax)
    notes = np.round(12 * np.log2(freqs[sel] / 440.0)).astype(int) + 69
    m = mag[sel]
    chroma = np.zeros((12, mag.shape[1]))
    # 半音ごとに最大値を取る (高い音ほどビン数が多くなる偏りを防ぐ) → オクターブを畳む
    for note in np.unique(notes):
        chroma[note % 12] += m[notes == note].max(axis=0)
    return chroma


def compute_chroma(y, sr):
    try:
        import librosa

        y_h = librosa.effects.harmonic(y, margin=4)  # 打楽器の影響を減らす
        return librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=2048, bins_per_octave=36), "librosa"
    except ImportError:
        return chroma_numpy(y, sr), "numpy"


def key_from_chroma(chroma):
    """クロマ (12 x フレーム) → 24 キーとの相関を高い順に。"""
    c = np.asarray(chroma, dtype=np.float64)
    # 各フレームを正規化してから平均 (大きい音の場面だけに偏らないように)
    c = np.nan_to_num(c)
    frame_max = c.max(axis=0, keepdims=True)
    # 無音に近いフレームは除外 (librosa のクロマは各フレームが最大1に揃っているので、>= で比較する)
    active = (frame_max[0] >= np.percentile(frame_max, 20)) & (frame_max[0] > 1e-6)
    if not active.any():
        active = frame_max[0] > 0
    if not active.any():
        raise ValueError("音声が無音のため、キーを判定できません")
    c = c[:, active] / frame_max[:, active]
    profile = c.mean(axis=1)

    scores = []
    for pc in range(12):
        for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(profile, np.roll(prof, pc))[0, 1]
            scores.append((float(r), pc, mode))
    scores.sort(reverse=True)
    return scores, profile


def analyze_key(path, cache_json=None, max_sec=None):
    """音声ファイルのキーを判定。cache_json があれば結果を保存/再利用。"""
    if cache_json and os.path.exists(cache_json):
        with open(cache_json, encoding="utf-8") as f:
            return json.load(f)

    y, sr = load_mono(path, max_sec=max_sec)
    chroma, engine = compute_chroma(y, sr)
    scores, profile = key_from_chroma(chroma)
    best_r, pc, mode = scores[0]
    result = describe_key(pc, mode)
    result["confidence"] = round(best_r, 3)
    result["margin"] = round(best_r - scores[1][0], 3)   # 2位との差 (小さいほど判定が微妙)
    result["candidates"] = [
        {"short": describe_key(p, m)["short"], "japanese": describe_key(p, m)["japanese"], "score": round(r, 3)}
        for r, p, m in scores[:3]
    ]
    result["chroma_profile"] = [round(float(v), 4) for v in profile]
    result["engine"] = engine

    if cache_json:
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    return result


def print_result(res):
    print("\n" + "=" * 46)
    print(f"  {res['display']}")
    print("=" * 46)
    print(f"  音階　: {' '.join(res['scale_doremi'])}")
    print(f"  調号　: {res['accidentals']}　　平行調: {res['relative']}")
    print(f"  確からしさ: {res['confidence']:.2f}（2位との差 {res['margin']:.2f}）")
    print("  候補　: " + " ／ ".join(f"{c['short']} {c['score']:.2f}" for c in res["candidates"]))
    if res["margin"] < 0.03:
        print("  ※ 候補どうしの差が小さいので、上の候補も参考にしてください")


def main():
    ap = argparse.ArgumentParser(description="曲のキー（調）を判定して、ドレミで表示します")
    ap.add_argument("audio", help="MP3 / WAV ファイル")
    ap.add_argument("--max-sec", type=float, default=None, help="先頭から何秒だけ解析するか")
    args = ap.parse_args()
    if not os.path.isfile(args.audio):
        sys.exit(f"ファイルが見つかりません: {args.audio}")
    print(f"解析中: {os.path.basename(args.audio)}")
    print_result(analyze_key(args.audio, max_sec=args.max_sec))


if __name__ == "__main__":
    main()
