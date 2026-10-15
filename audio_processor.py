#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップ2: AIによるボーカル分離と「お手本」音程データの自動抽出

  MP3 ──(demucs: AI音源分離)──▶ vocals.wav / accompaniment.wav
  vocals.wav ──(YIN ピッチ解析)──▶ pitch.json (お手本データ)

結果は cache/<曲名>_<ハッシュ>/ に保存され、2回目以降は解析を省略します。

使い方:
    python audio_processor.py 曲.mp3              # 分離 + 解析
    python audio_processor.py 曲.mp3 --force      # キャッシュを無視してやり直し
    python audio_processor.py --vocals vocals.wav # 分離済みボーカルの解析だけ

必要ライブラリ:
    pip install numpy scipy librosa soundfile demucs torchcrepe
    (demucs が無い場合は簡易分離、torchcrepe が無い場合は pYIN/YIN に自動で切り替えます)
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import medfilt, resample_poly

from pitch_core import YinPitchDetector, hz_to_midi, midi_to_doremi, midi_to_english

CACHE_VERSION = 3   # 解析方法を変えたら上げる → 古いお手本データは自動で作り直し
ANALYSIS_SR = 32000      # ピッチ解析用 (10ms = 320 サンプルちょうど。端数があると時間が少しずつずれる)
HOP_SEC = 0.01           # お手本データの時間分解能 (10ms)
FRAME_LEN = 2048         # 解析フレーム長 (32000Hzで64ms → 最低音 約60Hz まで)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 入出力
# ---------------------------------------------------------------------------
def file_hash(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:10]


def cache_dir_for(mp3_path, cache_root="cache"):
    stem = os.path.splitext(os.path.basename(mp3_path))[0]
    safe = "".join(c if c.isalnum() or c in "-_ ()" else "_" for c in stem).strip() or "song"
    return os.path.join(cache_root, f"{safe[:40]}_{file_hash(mp3_path)}")


def load_audio(path, sr, mono=False):
    """MP3/WAV などを読み込む → (channels, samples) の float32。"""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=mono)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    return y.astype(np.float32)


def save_wav(path, audio, sr):
    """(channels, samples) float → 16bit WAV (pygame で再生しやすい形式)。"""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.T
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:  # クリップ防止
        audio = audio / peak
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))


def read_wav_mono(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


# ---------------------------------------------------------------------------
# AI 音源分離
# ---------------------------------------------------------------------------
DEFAULT_SEP_MODEL = "htdemucs_ft"   # ボーカル向けに調整された高精度モデル (標準の約4倍の時間)


USE_GPU = True   # 既定でグラボ (NVIDIA / CUDA) を使う。False か --cpu で CPU に固定


def ai_device():
    """AI の計算に使う装置 → ("cuda", "GeForce RTX ...") / ("cpu", 理由)。"""
    try:
        import torch
    except ImportError:
        return "cpu", "PyTorch なし"
    if not USE_GPU:
        return "cpu", "CPU を使う設定"
    try:
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
        if torch.version.cuda is None:
            return "cpu", "CPU 専用版の PyTorch (setup.bat でグラボ対応版に入れ替えられます)"
        return "cpu", "グラボが見つかりません"
    except Exception as e:
        return "cpu", f"グラボを使えません ({e})"


def _device_label(dev, info):
    return f"グラボ ({info})" if dev == "cuda" else f"CPU ({info})"


def _is_gpu_error(e):
    msg = str(e).lower()
    return any(k in msg for k in ("cuda", "out of memory", "cudnn", "cublas", "device-side"))


def separate_demucs(mp3_path, model_name=DEFAULT_SEP_MODEL):
    """demucs でボーカルと伴奏に分離 → (vocals, accompaniment, sr, 使ったモデル名)。"""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    device, info = ai_device()
    log(f"  AIモデル {model_name} を準備中 (初回はモデルを自動ダウンロードします) / 使用デバイス: {_device_label(device, info)}")
    try:
        model = get_model(model_name)
    except Exception as e:
        if model_name == "htdemucs":
            raise
        log(f"  [注意] {model_name} を読み込めませんでした ({e})。標準モデル htdemucs で続けます")
        model_name = "htdemucs"
        model = get_model(model_name)
    model.eval()

    sr = model.samplerate
    audio = load_audio(mp3_path, sr, mono=False)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio = audio[: model.audio_channels]

    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    if device == "cuda":
        log(f"  分離中… (曲の長さ {audio.shape[1] / sr:.0f} 秒。グラボなので1〜2分ほどです。初回だけです)")
    else:
        slow = "10〜20分" if model_name.endswith("_ft") else "数分"
        log(f"  分離中… (曲の長さ {audio.shape[1] / sr:.0f} 秒。CPUだと{slow}かかります。初回だけです)")
    t_start = time.time()
    with torch.no_grad():
        try:
            sources = apply_model(model, ((wav - mean) / std)[None], device=device, split=True, overlap=0.25, progress=True)[0]
        except RuntimeError as e:
            if device != "cuda" or not _is_gpu_error(e):
                raise
            log(f"  [注意] グラボでの分離に失敗しました ({str(e)[:80]})。CPU でやり直します")
            torch.cuda.empty_cache()
            device = "cpu"
            sources = apply_model(model.cpu(), ((wav - mean) / std)[None], device="cpu", split=True, overlap=0.25, progress=True)[0]
    log(f"  分離完了 ({time.time() - t_start:.0f}秒 / {('グラボ' if device == 'cuda' else 'CPU')})")
    vi = model.sources.index("vocals")          # モデルを片付ける前に、ボーカルが何番目かを取っておく
    sources = (sources * std + mean).cpu().numpy()
    if device == "cuda":
        del model                                # グラボのメモリを空けて、次の CREPE で使えるように
        torch.cuda.empty_cache()

    vocals = sources[vi]
    accompaniment = sources.sum(axis=0) - vocals
    return vocals, accompaniment, sr, model_name


def separate_simple(mp3_path):
    """demucs が無いときの簡易分離 (librosa の REPET-SIM 法)。精度は低めです。"""
    import librosa

    sr = 44100
    audio = load_audio(mp3_path, sr, mono=False)
    log("  [注意] demucs が見つからないため、簡易分離を使います (pip install demucs を推奨)")
    vocals, accomp = [], []
    for ch in audio:
        S, phase = librosa.magphase(librosa.stft(ch))
        S_filter = librosa.decompose.nn_filter(
            S, aggregate=np.median, metric="cosine", width=int(librosa.time_to_frames(2, sr=sr))
        )
        S_filter = np.minimum(S, S_filter)
        mask_v = librosa.util.softmask(S - S_filter, 10 * S_filter, power=2)
        mask_a = librosa.util.softmask(S_filter, 2 * (S - S_filter), power=2)
        vocals.append(librosa.istft(mask_v * S * phase, length=len(ch)))
        accomp.append(librosa.istft(mask_a * S * phase, length=len(ch)))
    return np.array(vocals), np.array(accomp), sr


def separate(mp3_path, out_dir, method="auto", sep_model=DEFAULT_SEP_MODEL):
    vocals_path = os.path.join(out_dir, "vocals.wav")
    accomp_path = os.path.join(out_dir, "accompaniment.wav")
    if method in ("auto", "demucs"):
        try:
            vocals, accomp, sr, used = separate_demucs(mp3_path, sep_model)
        except ImportError:
            if method == "demucs":
                raise
            vocals, accomp, sr = separate_simple(mp3_path)
            used = "simple"
    else:
        vocals, accomp, sr = separate_simple(mp3_path)
        used = "simple"
    save_wav(vocals_path, vocals, sr)
    save_wav(accomp_path, accomp, sr)
    return vocals_path, accomp_path, used


# ---------------------------------------------------------------------------
# ボーカルの音程解析
# ---------------------------------------------------------------------------
def _resample(y, sr, target):
    if sr == target:
        return y, sr
    from math import gcd

    g = gcd(sr, target)
    return resample_poly(y, target // g, sr // g).astype(np.float32), target


def _frame_db(y, hop, n_frames):
    y_pad = np.concatenate([np.zeros(FRAME_LEN // 2, np.float32), y, np.zeros(FRAME_LEN, np.float32)])
    rms = np.array([np.sqrt(np.mean(y_pad[i * hop : i * hop + FRAME_LEN] ** 2) + 1e-12) for i in range(n_frames)])
    return 20 * np.log10(rms), y_pad


def pitch_yin(y, sr, hop_sec=HOP_SEC, progress=True):
    """改良版 YIN。戻り値: (MIDI 配列, 信頼度配列, 音量dB配列)。"""
    det = YinPitchDetector(sr, fmin=60.0, fmax=1100.0, threshold=0.15, fallback=0.40)
    hop = int(round(hop_sec * sr))
    assert abs(hop - hop_sec * sr) < 1e-6, "hop が整数でないと時間がずれていきます"
    n_frames = int(np.ceil(len(y) / hop))
    db, y_pad = _frame_db(y, hop, n_frames)
    gate = max(-55.0, float(np.percentile(db, 95)) - 40.0)
    midi = np.full(n_frames, np.nan)
    conf = np.zeros(n_frames)
    t_start = time.time()
    for i in range(n_frames):
        if db[i] < gate:
            continue
        f, c = det(y_pad[i * hop : i * hop + FRAME_LEN])
        if f > 0:
            midi[i], conf[i] = hz_to_midi(f), c
        if progress and i % 2000 == 0:
            sys.stdout.write(f"\r  音程解析中 (YIN)… {i / n_frames * 100:5.1f}%")
            sys.stdout.flush()
    if progress:
        sys.stdout.write(f"\r  音程解析中 (YIN)… 100.0% ({time.time() - t_start:.1f}秒)\n")
    return midi, conf, db, gate


def pitch_pyin(y, sr, n_frames, hop_sec=HOP_SEC):
    """librosa の pYIN (確率的YIN)。歌声の有声/無声の判定が得意。戻り値: (MIDI, 有声確率)。"""
    import librosa

    log("  pYIN で解析中… (5分の曲で1〜3分ほどかかります)")
    t_start = time.time()
    hop = int(round(hop_sec * sr))
    f0, _, vprob = librosa.pyin(
        y, fmin=60.0, fmax=1100.0, sr=sr, frame_length=FRAME_LEN, hop_length=hop, fill_na=None, center=True
    )
    midi = np.full(n_frames, np.nan)
    prob = np.zeros(n_frames)
    k = min(n_frames, len(f0))
    with np.errstate(divide="ignore", invalid="ignore"):
        midi[:k] = 69 + 12 * np.log2(np.asarray(f0[:k], dtype=np.float64) / 440.0)
    prob[:k] = np.nan_to_num(vprob[:k])
    log(f"  pYIN 完了 ({time.time() - t_start:.0f}秒)")
    return midi, prob


def best_available_engine():
    """このPCで使える一番精度の高い解析方法。"""
    for name, mod in (("crepe", "torchcrepe"), ("pyin", "librosa")):
        try:
            __import__(mod)
            return name
        except Exception:
            continue
    return "yin"


def pitch_crepe(y, sr, n_frames, hop_sec=HOP_SEC, model="full", chunk_sec=20.0):
    """torchcrepe (歌声・楽器を学習した AI) で音程を推定。戻り値: (MIDI, 確からしさ periodicity)。

    長い曲は 20 秒ごとに区切り、前後 0.32 秒の重なりを持たせて処理 (進み具合を表示するため)。
    """
    import torch
    import torchcrepe

    device, info = ai_device()
    y16, sr16 = _resample(y, sr, 16000)
    hop = int(round(hop_sec * sr16))            # 160 サンプル = 10ms
    ctx = 32 * hop                              # 重なり 0.32 秒
    chunk = int(chunk_sec * sr16) // hop * hop
    midi = np.full(n_frames, np.nan)
    per = np.zeros(n_frames)
    t_start = time.time()
    log(f"  AI (CREPE {model}) で音程を解析中… / 使用デバイス: {_device_label(device, info)}")
    for a in range(0, len(y16), chunk):
        b = min(len(y16), a + chunk)
        lo, hi = max(0, a - ctx), min(len(y16), b + ctx)
        audio = torch.from_numpy(np.ascontiguousarray(y16[lo:hi], dtype=np.float32))[None]
        try:
            p, c = torchcrepe.predict(
                audio, sr16, hop, 60.0, 1100.0, model, decoder=torchcrepe.decode.viterbi,
                return_periodicity=True, batch_size=1024, device=device, pad=True,
            )
        except RuntimeError as e:
            if device != "cuda" or not _is_gpu_error(e):
                raise
            log(f"  [注意] グラボでの音程解析に失敗しました ({str(e)[:80]})。CPU で続けます")
            device = "cpu"
            p, c = torchcrepe.predict(
                audio, sr16, hop, 60.0, 1100.0, model, decoder=torchcrepe.decode.viterbi,
                return_periodicity=True, batch_size=1024, device=device, pad=True,
            )
        c = torchcrepe.filter.median(c, 3)
        p, c = p[0].cpu().numpy(), c[0].cpu().numpy()
        # p[k] は時刻 (lo + k*hop) の値。担当範囲 [a, b) だけ取り出す
        k0 = (a - lo) // hop
        f0, f1 = a // hop, min(n_frames, (b + hop - 1) // hop)
        cnt = max(0, min(f1 - f0, len(p) - k0))
        with np.errstate(divide="ignore", invalid="ignore"):
            midi[f0 : f0 + cnt] = 69 + 12 * np.log2(p[k0 : k0 + cnt] / 440.0)
        per[f0 : f0 + cnt] = c[k0 : k0 + cnt]
        sys.stdout.write(f"\r  CREPE 解析中… {b / len(y16) * 100:5.1f}%")
        sys.stdout.flush()
        if a == 0 or (a // chunk) % 3 == 0:
            log(f"  CREPE 解析中… {b / len(y16) * 100:4.0f}% ({time.time() - t_start:.0f}秒経過)")
    sys.stdout.write("\n")
    log(f"  CREPE 完了 ({time.time() - t_start:.0f}秒)")
    return midi, per


def extract_pitch_track(y, sr, hop_sec=HOP_SEC, progress=True, engine="auto", crepe_model="full",
                        return_engine=False):
    """モノラル音声 → 10ms ごとの MIDI 値 (無声区間は nan) の配列。

    engine: "auto" (使える中で最高精度: crepe → pyin → yin) / "crepe" / "pyin" / "yin"
    """
    if engine == "auto":
        engine = best_available_engine()
    y, sr = _resample(y, sr, ANALYSIS_SR)
    midi_y, conf_y, db, gate = pitch_yin(y, sr, hop_sec, progress)
    n = len(midi_y)
    loud = db >= gate

    # 残響 (リバーブ) の尾は「歌っていない」扱い: 直前 0.4 秒のピークから 14dB 以上下がった部分
    from scipy.ndimage import maximum_filter1d

    w = int(0.4 / hop_sec)
    recent_peak = maximum_filter1d(db, size=w, origin=(w - 1) // 2)  # 過去側の最大値
    loud &= db >= recent_peak - 14.0

    # 1) YIN の結果: はっきりした声 (信頼度 0.6 以上)
    midi = np.where(loud & (conf_y >= 0.60), midi_y, np.nan)

    used = "yin"
    # 2a) CREPE (AI): 確からしさの高いフレームは CREPE を優先し、YIN の取りこぼしを埋める
    if engine == "crepe":
        try:
            midi_c, per = pitch_crepe(y, sr, n, hop_sec, crepe_model)
            use_c = loud & (per >= 0.30) & ~np.isnan(midi_c)
            agree = np.isnan(midi) | (np.abs(midi - midi_c) < 1.0)
            midi = np.where(use_c & agree, midi_c, midi)
            disagree = use_c & ~agree
            midi = np.where(disagree & (per >= 0.55), midi_c, midi)              # CREPE に自信があれば CREPE
            midi = np.where(disagree & (per < 0.55) & (conf_y < 0.80), np.nan, midi)
            # CREPE が「声なし」とはっきり判断した所は、YIN の自信が高い場合だけ残す
            midi = np.where(~use_c & (per < 0.15) & (conf_y < 0.85), np.nan, midi)
            used = "crepe"
        except ImportError:
            log("  (torchcrepe が無いため pYIN/YIN で解析します。pip install torchcrepe で高精度になります)")
            engine = "pyin"

    # 2b) pYIN が使えれば、有声確率の高いフレームは pYIN を優先し、YIN の取りこぼしを埋める
    if engine == "pyin":
        try:
            midi_p, prob = pitch_pyin(y, sr, n, hop_sec)
            use_p = loud & (prob >= 0.15) & ~np.isnan(midi_p)
            # YIN と pYIN がほぼ同じ (または YIN が無い) ところは pYIN を採用
            agree = np.isnan(midi) | (np.abs(midi - midi_p) < 1.0)
            midi = np.where(use_p & agree, midi_p, midi)
            # 食い違うときは、確率が高ければ pYIN、低ければ YIN (信頼度 0.75 以上) を残す
            disagree = use_p & ~agree
            midi = np.where(disagree & (prob >= 0.5), midi_p, midi)
            midi = np.where(disagree & (prob < 0.5) & (conf_y < 0.75), np.nan, midi)
            used = "pyin"
        except ImportError:
            log("  (librosa が無いため YIN のみで解析します)")

    midi = limit_to_vocal_range(midi, midi_y, conf_y)
    midi = clean_pitch_track(midi)
    return (midi, used) if return_engine else midi


def limit_to_vocal_range(midi, midi_y, conf_y):
    """歌声の音域から大きく外れた検出 (伴奏のベースの残りなど) を取り除く。"""
    sure = midi_y[(conf_y >= 0.85) & ~np.isnan(midi_y)]
    if sure.size < 50:
        return midi
    lo, hi = np.percentile(sure, 3) - 4, np.percentile(sure, 97) + 4
    out = midi.copy()
    out[(out < lo) | (out > hi)] = np.nan
    return out


def _runs(mask):
    """True が続く区間 [(開始, 終了)] を返す。"""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def clean_pitch_track(midi, min_run=4, med=5, max_gap=12):
    """揺れ・誤検出・オクターブ飛びを直し、子音などの短い途切れを埋める。"""
    midi = midi.copy()
    n = len(midi)

    # (a) オクターブ飛びの修正: 周囲 0.3 秒の中央値から 9半音以上ずれた点は 12 の倍数だけ戻す
    voiced_idx = np.where(~np.isnan(midi))[0]
    if voiced_idx.size:
        for _ in range(2):
            vals = midi[voiced_idx]
            ref = np.array([np.median(vals[max(0, k - 15) : k + 16]) for k in range(len(vals))])
            off = vals - ref
            fix = np.abs(off) > 9
            vals[fix] -= 12 * np.round(off[fix] / 12)
            midi[voiced_idx] = vals

    # (b) 一瞬だけ (40ms 未満) の検出は誤検出として削除
    for a, b in _runs(~np.isnan(midi)):
        if b - a < min_run:
            midi[a:b] = np.nan

    # (c) 子音・息つぎ程度の短い途切れ (120ms 以内) で前後の高さが近ければ、なめらかにつなぐ
    runs = _runs(~np.isnan(midi))
    for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
        gap = a2 - b1
        if 0 < gap <= max_gap:
            left = np.median(midi[max(a1, b1 - 3) : b1])
            right = np.median(midi[a2 : min(b2, a2 + 3)])
            if abs(left - right) <= 0.7:          # 同じ音 → なめらかにつなぐ
                midi[b1:a2] = np.linspace(left, right, gap + 2)[1:-1]
            elif abs(left - right) <= 7:          # 違う音 → 隙間の真ん中で切り替える
                h = gap // 2
                midi[b1 : b1 + h] = left
                midi[b1 + h : a2] = right

    # (d) 有声区間ごとのメディアン平滑化
    for a, b in _runs(~np.isnan(midi)):
        k = min(med, (b - a) if (b - a) % 2 else (b - a - 1))
        if k > 1:
            midi[a:b] = medfilt(midi[a:b], kernel_size=k)
    return midi


def segment_notes(midi, hop_sec=HOP_SEC, min_note=0.07, merge_gap=0.12, split_thr=0.7, hold=5):
    """MIDI 値の列 → お手本バー (開始・終了・音程) のリスト。

    - 今のバーの音から split_thr 半音以上離れた状態が hold フレーム (50ms) 続いたら次の音
    - 短すぎるかけらは捨てずに隣のバーへ吸収 (空白を作らない)
    """
    raw = []
    for a, b in _runs(~np.isnan(midi)):
        start = a
        cur = midi[a : a + 3]
        cur_note = float(np.median(cur))
        i = a
        while i < b:
            # 離れた状態が hold フレーム続くか
            if abs(midi[i] - cur_note) > split_thr:
                j = i
                while j < b and abs(midi[j] - cur_note) > split_thr and j - i < hold:
                    j += 1
                if j - i >= hold or (j == b and j - i >= 2):
                    raw.append([start, i, float(np.median(midi[start:i])) if i > start else cur_note])
                    start = i
                    cur_note = float(np.median(midi[i : min(b, i + hold)]))
                    i = j
                    continue
            # 今のバーの音程を、これまでの値の中央値で更新 (ビブラートに強い)
            if i - start >= 3:
                cur_note = float(np.median(midi[max(start, i - 25) : i + 1]))
            i += 1
        raw.append([start, b, float(np.median(midi[start:b]))])

    # 短いかけらを隣に吸収
    min_f = int(round(min_note / hop_sec))
    changed = True
    while changed and len(raw) > 1:
        changed = False
        for k, (s, e, m) in enumerate(raw):
            if e - s >= min_f:
                continue
            prev = raw[k - 1] if k > 0 and raw[k - 1][1] == s else None
            nxt = raw[k + 1] if k + 1 < len(raw) and raw[k + 1][0] == e else None
            if prev is None and nxt is None:
                if e - s < 3:          # 孤立した極短の音だけ削除
                    raw.pop(k)
                    changed = True
                    break
                continue
            target = prev if (nxt is None or (prev is not None and abs(prev[2] - m) <= abs(nxt[2] - m))) else nxt
            if target is prev:
                prev[1] = e
            else:
                nxt[0] = s
            raw.pop(k)
            changed = True
            break

    # 同じ音でごく短い隙間しかないバーは1本に
    notes = []
    for s, e, m in raw:
        nt = {"start": s * hop_sec, "end": e * hop_sec, "midi_f": m, "midi": int(round(m))}
        if notes and notes[-1]["midi"] == nt["midi"] and nt["start"] - notes[-1]["end"] <= merge_gap:
            notes[-1]["end"] = nt["end"]
        else:
            notes.append(nt)

    for nt in notes:
        nt["start"] = round(nt["start"], 3)
        nt["end"] = round(nt["end"], 3)
        nt["midi_f"] = round(nt["midi_f"], 2)
        nt["doremi"] = midi_to_doremi(nt["midi"])
        nt["name"] = midi_to_english(nt["midi"])
    return notes


def analyze_vocals(vocals_path, json_path=None, source=None, engine="auto", crepe_model="full"):
    y, sr = read_wav_mono(vocals_path)
    log(f"  ボーカル音声: {len(y) / sr:.1f} 秒 / {sr} Hz")
    midi, used = extract_pitch_track(y, sr, engine=engine, crepe_model=crepe_model, return_engine=True)
    notes = segment_notes(midi)
    voiced = midi[~np.isnan(midi)]

    data = {
        "version": CACHE_VERSION,
        "source": source or os.path.basename(vocals_path),
        "duration": round(len(y) / sr, 3),
        "engine": used,
        "hop_sec": HOP_SEC,
        # 10ms ごとの細かい音程 (なめらかな線の描画用)。null = 声なし
        "pitch": [None if np.isnan(m) else round(float(m), 2) for m in midi],
        # お手本バー
        "notes": notes,
        "range": {
            "min_midi": int(np.floor(np.percentile(voiced, 1))) if voiced.size else 60,
            "max_midi": int(np.ceil(np.percentile(voiced, 99))) if voiced.size else 72,
        },
    }
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    return data


# ---------------------------------------------------------------------------
# まとめ: MP3 → キャッシュ
# ---------------------------------------------------------------------------
def process(mp3_path, cache_root="cache", force=False, method="auto", sep_model=DEFAULT_SEP_MODEL,
            engine="auto", crepe_model="full"):
    """MP3 を処理し、ファイルパスとお手本データを返す。

    戻り値: {"dir", "vocals", "accompaniment", "pitch_json", "data"}
    """
    if not os.path.isfile(mp3_path):
        raise FileNotFoundError(mp3_path)
    out_dir = cache_dir_for(mp3_path, cache_root)
    os.makedirs(out_dir, exist_ok=True)
    vocals_path = os.path.join(out_dir, "vocals.wav")
    accomp_path = os.path.join(out_dir, "accompaniment.wav")
    json_path = os.path.join(out_dir, "pitch.json")
    meta_path = os.path.join(out_dir, "meta.json")

    log(f"曲: {os.path.basename(mp3_path)}")
    log(f"保存先: {out_dir}")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    prev_model = meta.get("model", "htdemucs" if meta.get("separator") == "demucs" else meta.get("separator"))
    want_model = sep_model if method != "simple" else "simple"
    have_wavs = os.path.exists(vocals_path) and os.path.exists(accomp_path)
    if have_wavs and not force and prev_model != want_model and prev_model in ("htdemucs", "simple") and want_model != "simple":
        log(f"[1/2] 前回より高精度なモデル ({want_model}) で分離し直します (前回: {prev_model})")
        have_wavs = False
    if force or not have_wavs:
        log(f"[1/2] AI でボーカルと伴奏を分離しています (モデル: {want_model})")
        _, _, used = separate(mp3_path, out_dir, method, sep_model)
        meta = {"separator": "simple" if used == "simple" else "demucs", "model": used, "source": os.path.abspath(mp3_path)}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        force = True  # 分離し直したら解析もやり直す
    else:
        log(f"[1/2] 分離済みの音源を再利用します (キャッシュ / モデル: {prev_model})")

    data = None
    if not force and os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != CACHE_VERSION:
            data = None
        elif engine == "auto" and data.get("engine") != best_available_engine():
            log(f"  より高精度な解析方法 ({best_available_engine()}) が使えるので、お手本を作り直します")
            data = None
        else:
            log("[2/2] 解析済みのお手本データを再利用します (キャッシュ)")
    if data is None:
        log("[2/2] ボーカルの音程を解析しています")
        data = analyze_vocals(vocals_path, json_path, source=os.path.basename(mp3_path), engine=engine,
                              crepe_model=crepe_model)

    return {"dir": out_dir, "vocals": vocals_path, "accompaniment": accomp_path, "pitch_json": json_path, "data": data}


def summarize(data):
    notes = data["notes"]
    r = data["range"]
    log(f"\n完了: お手本バー {len(notes)} 本 / 曲の長さ {data['duration']:.1f} 秒")
    log(f"  音域: {midi_to_doremi(r['min_midi'])} 〜 {midi_to_doremi(r['max_midi'])}"
        f" ({midi_to_english(r['min_midi'])} 〜 {midi_to_english(r['max_midi'])})")
    if notes:
        log("  最初のフレーズ:")
        for nt in notes[:8]:
            log(f"    {nt['start']:7.2f}〜{nt['end']:7.2f} 秒  {nt['doremi']:<5} ({nt['name']})")


def main():
    ap = argparse.ArgumentParser(description="MP3 からボーカルを分離し、お手本の音程データを作成します")
    ap.add_argument("mp3", nargs="?", help="解析する MP3 ファイル")
    ap.add_argument("--force", action="store_true", help="キャッシュを無視して作り直す")
    ap.add_argument("--cache", default="cache", help="保存先フォルダ (既定: cache)")
    ap.add_argument("--method", choices=["auto", "demucs", "simple"], default="auto", help="分離方法")
    ap.add_argument("--vocals", help="分離済みのボーカル WAV だけを解析する")
    ap.add_argument("--sep-model", default=DEFAULT_SEP_MODEL, help="分離モデル: htdemucs_ft (高精度/既定) / htdemucs (速い)")
    ap.add_argument("--engine", choices=["auto", "crepe", "pyin", "yin"], default="auto", help="音程の解析方法")
    ap.add_argument("--crepe-model", choices=["full", "tiny"], default="full", help="CREPE の大きさ (tiny は速いが精度少し下)")
    ap.add_argument("--cpu", action="store_true", help="グラボを使わず CPU で解析する")
    args = ap.parse_args()
    global USE_GPU
    USE_GPU = not args.cpu
    dev, info = ai_device()
    log(f"AI の計算: {_device_label(dev, info)}")

    if args.vocals:
        out = os.path.splitext(args.vocals)[0] + "_pitch.json"
        data = analyze_vocals(args.vocals, out, engine=args.engine, crepe_model=args.crepe_model)
        summarize(data)
        log(f"  保存: {out}")
    elif args.mp3:
        res = process(args.mp3, args.cache, args.force, args.method, args.sep_model, args.engine, args.crepe_model)
        summarize(res["data"])
        log(f"\n  ボーカル: {res['vocals']}\n  伴奏    : {res['accompaniment']}\n  お手本  : {res['pitch_json']}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
