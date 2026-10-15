# -*- coding: utf-8 -*-
"""
再生速度の変更 (音程はそのまま) と キーの変更 (速さはそのまま)

  位相ボコーダ (位相ロック付き) で長さだけを変え、リサンプルで高さを変えます。
    速度 s・移調 n 半音 のとき:
      1. 長さを p / s 倍にする (p = 2^(n/12))   … 音程はそのまま
      2. 1/p 倍に縮める (リサンプル)             … 高さが p 倍、長さは 1/s 倍
  numpy と scipy だけで動きます。
"""

import math
from fractions import Fraction

import numpy as np

N_FFT = 2048
HOP_S = 512          # 書き出し側の間隔 (N_FFT / 4)


def _owners(mag):
    """各周波数を、一番近い「山 (ピーク)」に割り当てる。mag: (フレーム, 周波数)"""
    f, b = mag.shape
    peak = np.zeros((f, b), bool)
    m = mag
    peak[:, 2:-2] = ((m[:, 2:-2] > m[:, 1:-3]) & (m[:, 2:-2] >= m[:, 3:-1]) &
                     (m[:, 2:-2] > m[:, :-4]) & (m[:, 2:-2] >= m[:, 4:]))
    idx = np.arange(b)[None, :].repeat(f, 0)
    lo = np.where(peak, idx, -1)
    lo = np.maximum.accumulate(lo, axis=1)                         # 左側で一番近い山
    hi = np.where(peak, idx, b * 4)
    hi = np.minimum.accumulate(hi[:, ::-1], axis=1)[:, ::-1]       # 右側で一番近い山
    use_hi = (lo < 0) | ((hi < b) & (hi - idx < idx - lo))
    own = np.where(use_hi, hi, lo)
    none = (lo < 0) & (hi >= b)                                    # 山が1つも無いフレーム
    own = np.where(none, idx, own)
    return own


def _stretch_channel(x, rate, progress=None, p0=0.0, p1=1.0):
    """1チャンネルの長さを rate 倍にする (音程はそのまま)。x: float32"""
    n = len(x)
    if n < N_FFT * 2 or abs(rate - 1.0) < 1e-6:
        return x.copy()
    win = np.hanning(N_FFT).astype(np.float32)
    ha = HOP_S / rate                                  # 読み取り側の間隔 (小数)
    pad = np.concatenate([np.zeros(N_FFT, np.float32), x, np.zeros(N_FFT, np.float32)])
    n_frames = int((len(pad) - N_FFT) / ha) + 1
    out_len = (n_frames - 1) * HOP_S + N_FFT
    out = np.zeros(out_len, np.float32)
    w2 = win * win
    bins = N_FFT // 2 + 1
    omega = 2 * np.pi * np.arange(bins) / N_FFT       # 1サンプルあたりの位相の進み
    all_pos = np.minimum(np.round(np.arange(n_frames) * ha).astype(np.int64), len(pad) - N_FFT)
    syn = None
    prev_ph = None
    prev_pos = None
    chunk = 512
    rows = np.arange(chunk)
    for c0 in range(0, n_frames, chunk):
        c1 = min(n_frames, c0 + chunk)
        m = c1 - c0
        pos = all_pos[c0:c1]
        frames = pad[pos[:, None] + np.arange(N_FFT)[None, :]] * win
        spec = np.fft.rfft(frames, axis=1)
        mag = np.abs(spec).astype(np.float32)
        ph = np.angle(spec).astype(np.float32)
        # 本当の周波数 (前のフレームとの位相の差から) をまとめて計算
        pph = np.vstack([prev_ph[None, :] if prev_ph is not None else ph[:1], ph[:-1]])
        ppos = np.concatenate([[prev_pos if prev_pos is not None else pos[0] - 1], pos[:-1]])
        step = np.maximum(1, pos - ppos)[:, None].astype(np.float64)
        dphi = ph - pph - omega[None, :] * step
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        adv = (omega[None, :] + dphi / step) * HOP_S
        own = _owners(mag)
        r = rows[:m, None]
        rel = ph - ph[r, own]                          # 山との位相の差 (位相ロック)
        syn_all = np.empty((m, bins), np.float32)
        adv = adv.astype(np.float32)
        for j in range(m):
            if syn is None:
                syn = ph[j].copy()
            else:
                a = syn + adv[j]
                syn = a[own[j]] + rel[j]
            syn_all[j] = syn
        y = np.fft.irfft(mag * np.exp(1j * syn_all), N_FFT, axis=1).astype(np.float32) * win
        # 重ね合わせ (4分の1ずつずらして足す) をまとめて行う
        q = N_FFT // HOP_S
        yb = y.reshape(m, q, HOP_S)
        ob = out[: (len(out) // HOP_S) * HOP_S].reshape(-1, HOP_S)
        for k in range(q):
            ob[c0 + k:c0 + k + m] += yb[:, k, :]
        prev_ph, prev_pos = ph[-1], pos[-1]
        if progress:
            progress(p0 + (p1 - p0) * c1 / n_frames)
    # 窓の重なりの分を割る (間隔が一定なので周期的な値)
    wsum = np.zeros(out_len, np.float32)
    base = np.zeros(N_FFT + HOP_S * 4, np.float32)
    for k in range(4 + N_FFT // HOP_S):
        base[k * HOP_S:k * HOP_S + N_FFT] += w2 if k * HOP_S + N_FFT <= len(base) else w2[:len(base) - k * HOP_S]
    norm = float(np.mean(base[N_FFT:N_FFT + HOP_S]))
    out /= max(norm, 1e-3)
    start = int(round(N_FFT * rate))                   # 頭の余白の分
    want = int(round(n * rate))
    out = out[start:start + want]
    if len(out) < want:
        out = np.concatenate([out, np.zeros(want - len(out), np.float32)])
    return out


def _ratio(factor):
    """リサンプルの比 (分母が小さいほど速い。高さの誤差は 2セント 以下)。"""
    return Fraction(factor).limit_denominator(160)


def _resample(x, fr):
    """長さを fr 倍にする (高さは 1/fr 倍)。"""
    if fr == 1:
        return x
    from scipy.signal import resample_poly

    return resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)


def process(audio, sr, speed=1.0, semis=0, progress=None, p0=0.0, p1=1.0):
    """audio: (サンプル, ch) int16 → 速度 speed 倍・semis 半音上げた int16。"""
    if abs(speed - 1.0) < 1e-6 and semis == 0:
        return audio
    p = 2.0 ** (semis / 12.0)
    fr = _ratio(1.0 / p) if semis else Fraction(1)
    rate = (1.0 / speed) / float(fr)          # 最後の長さがちょうど 1/speed 倍になるように
    x = audio.astype(np.float32) / 32768.0
    if x.ndim == 1:
        x = x[:, None]
    nc = x.shape[1]
    done = [0.0] * nc

    def one(c):
        def prog(f):
            done[c] = f
            if progress:
                progress(p0 + (p1 - p0) * sum(done) / nc)
        y = _stretch_channel(np.ascontiguousarray(x[:, c]), rate, prog, 0.0, 1.0)
        return _resample(y, fr)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=nc) as ex:      # 左右のチャンネルを同時に処理
        chans = list(ex.map(one, range(nc)))
    n = min(len(y) for y in chans)
    want = int(round(len(x) / speed))
    y = np.stack([c[:n] for c in chans], axis=1)
    if len(y) < want:
        y = np.concatenate([y, np.zeros((want - len(y), nc), np.float32)])
    y = y[:want]
    return np.ascontiguousarray((np.clip(y, -1, 1) * 32767).astype(np.int16))


def stretch_mono(x, speed):
    """録音 (float32 1次元) を速度 speed で再生したときの長さにする (音程はそのまま)。"""
    if abs(speed - 1.0) < 1e-6:
        return x
    return _stretch_channel(x.astype(np.float32), 1.0 / speed)
