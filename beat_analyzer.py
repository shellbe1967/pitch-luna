#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拍と小節の解析 (楽譜のタテ線を「小節ごと」に引くため)

  伴奏の音 → 音の立ち上がり (アタック) の強さ → テンポ (BPM) → 拍の位置 → 小節の頭 (1拍目)

  1. 音の立ち上がり : 周波数ごとの音量が急に大きくなった所 (ドラム・ベース・コードの入り)
  2. テンポ        : 立ち上がりが何秒おきに繰り返しているか (自己相関)
  3. 拍            : テンポどおりに並び、立ち上がりの強い所を通る列 (動的計画法 / Ellis 2007)
  4. 小節の頭      : 低い音 (キック・ベース) が強く、コードが変わる拍 → 1拍目
                     3拍子か4拍子かも、どちらの方がはっきり区切れるかで判定

使い方:
    python beat_analyzer.py cache\\曲_xxxx\\accompaniment.wav

numpy と scipy だけで動きます (librosa は不要)。
"""

import argparse
import json
import os
import sys

import numpy as np

VERSION = 1
SR = 11025          # 解析用のサンプリング周波数 (拍の解析には十分)
N_FFT = 1024
HOP = 128           # 約 11.6ms ごと
N_BANDS = 24


def _load(path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd

    sr, y = wavfile.read(path)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    else:
        y = y.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def _features(y):
    """フレームごとの (帯域の対数エネルギー, クロマ, 音量)。"""
    pad = N_FFT // 2
    y = np.concatenate([np.zeros(pad, np.float32), y, np.zeros(pad, np.float32)])
    n = max(1, 1 + (len(y) - N_FFT) // HOP)
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    edges = np.geomspace(30, 5000, N_BANDS + 1)
    band_of = np.digitize(freqs, edges) - 1                  # -1 / N_BANDS は使わない
    B = np.zeros((len(freqs), N_BANDS), np.float32)
    for i, b in enumerate(band_of):
        if 0 <= b < N_BANDS:
            B[i, b] = 1.0
    B /= np.maximum(B.sum(axis=0, keepdims=True), 1)
    C = np.zeros((len(freqs), 12), np.float32)               # クロマ (65Hz〜2kHz)
    ok = (freqs >= 65) & (freqs <= 2000)
    pcs = (np.round(12 * np.log2(np.maximum(freqs, 1) / 440.0)) + 9) % 12
    for i in np.where(ok)[0]:
        C[i, int(pcs[i])] = 1.0
    bands = np.zeros((n, N_BANDS), np.float32)
    chroma = np.zeros((n, 12), np.float32)
    rms = np.zeros(n, np.float32)
    step = 4096
    for s in range(0, n, step):
        e = min(n, s + step)
        idx = np.arange(s, e)[:, None] * HOP + np.arange(N_FFT)[None, :]
        fr = y[idx] * win
        mag = np.abs(np.fft.rfft(fr, axis=1)).astype(np.float32)
        p = mag ** 2
        bands[s:e] = np.log1p(1000.0 * (p @ B))
        chroma[s:e] = p @ C
        rms[s:e] = np.sqrt(np.mean(fr ** 2, axis=1))
    return bands, chroma, rms


def _normalize(x, win):
    """ゆっくりした変化を引いて、標準偏差で割る。"""
    k = np.ones(win) / win
    x = x - np.convolve(x, k, mode="same")
    x = np.maximum(x, 0)
    sd = x.std()
    return x / sd if sd > 0 else x


def _onset(bands):
    d = np.diff(bands, axis=0, prepend=bands[:1])
    d = np.maximum(d, 0)
    fps = SR / HOP
    env = _normalize(d.sum(axis=1), int(fps * 0.4))
    low = _normalize(d[:, :6].sum(axis=1), int(fps * 0.4))   # 30〜110Hz 付近 (キック・ベース)
    return env, low


def estimate_tempo(env, lo_bpm=55, hi_bpm=210, center=120):
    """自己相関でテンポを推定 → (BPM, 1拍のフレーム数)。"""
    fps = SR / HOP
    x = env - env.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-9)
    lags = np.arange(n)
    bpm = np.where(lags > 0, 60 * fps / np.maximum(lags, 1), 0)
    w = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1) / center) / 0.9) ** 2)
    sel = (bpm >= lo_bpm) & (bpm <= hi_bpm)
    score = np.zeros(n)
    # 倍の周期 (2拍) も同じように強ければ、その拍らしい
    for i in np.where(sel)[0]:
        s = ac[i]
        if 2 * i < n:
            s += 0.3 * ac[2 * i]
        score[i] = s * w[i]
    best = int(np.argmax(score))
    if score[best] <= 0:
        return None, None                      # 繰り返しが見つからない (リズムの無い音)
    # 放物線で小数点以下まで
    if 1 <= best < n - 1:
        a, b, c = score[best - 1], score[best], score[best + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        period = best + float(np.clip(off, -0.5, 0.5))
    else:
        period = float(best)
    return 60 * fps / period, period


def track_beats(env, period, tightness=100.0):
    """テンポどおりに並び、立ち上がりの強い所を通る拍の列 (フレーム番号)。"""
    n = len(env)
    p = period
    # 拍の周期の幅でぼかした立ち上がり
    t = np.arange(-int(p), int(p) + 1)
    g = np.exp(-0.5 * (t * 32.0 / p) ** 2)
    local = np.convolve(env, g, mode="same")
    lo_off, hi_off = int(round(p / 2)), int(round(2 * p))
    offs = np.arange(lo_off, hi_off + 1)
    penalty = -tightness * np.log(offs / p) ** 2
    cum = local.copy()
    back = -np.ones(n, dtype=int)
    for i in range(lo_off, n):
        prev = i - offs
        valid = prev >= 0
        if not valid.any():
            continue
        cand = np.where(valid, cum[np.maximum(prev, 0)] + penalty, -np.inf)
        j = int(np.argmax(cand))
        if cand[j] > 0:
            cum[i] = local[i] + cand[j]
            back[i] = prev[j]
    # 最後の拍: 終わりの 2拍分の中で一番よい所
    tail = cum[max(0, n - hi_off):]
    i = max(0, n - hi_off) + int(np.argmax(tail))
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return np.array(beats[::-1], dtype=int)


def _beat_sync(feat, beats):
    out = []
    for a, b in zip(beats[:-1], beats[1:]):
        out.append(feat[a:max(a + 1, b)].mean(axis=0))
    out.append(feat[beats[-1]:beats[-1] + 8].mean(axis=0) if len(feat) > beats[-1] else feat[-1])
    return np.array(out)


def find_downbeats(beats, low, chroma):
    """1小節の拍の数 (3 / 4) と、何番目の拍が1拍目か。→ (bpb, phase, はっきり度)"""
    if len(beats) < 8:
        return 4, 0, 0.0
    lo = np.array([low[max(0, b - 2):b + 4].max() for b in beats])
    ch = _beat_sync(chroma, beats)
    ch = ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    change = np.zeros(len(beats))
    change[1:] = 1 - (ch[1:] * ch[:-1]).sum(axis=1)          # 前の拍からコードがどれだけ変わったか
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    sal = z(lo) + 1.3 * z(change)
    res = {}
    for bpb in (3, 4):
        scores = []
        for k in range(bpb):
            on = sal[k::bpb]
            off = np.delete(sal, np.arange(k, len(sal), bpb))
            scores.append(on.mean() - off.mean())
        k = int(np.argmax(scores))
        res[bpb] = (scores[k], k)
    if res[3][0] > 1.25 * res[4][0] and res[3][0] > 0.4:
        bpb = 3
    else:
        bpb = 4
    return bpb, res[bpb][1], float(res[bpb][0])


def _period(times):
    """拍の間隔 (秒)。テンポが一定とみなして直線を当てはめる (フレームの丸めの影響を減らす)。"""
    times = np.asarray(times, dtype=float)
    ibi = np.diff(times)
    med = float(np.median(ibi))
    if len(times) < 8:
        return med
    # テンポが途中で変わる曲もあるので、極端な間隔を除いて平均
    good = ibi[np.abs(ibi - med) < 0.15 * med]
    return float(good.mean()) if len(good) else med


def analyze(path):
    y = _load(path)
    bands, chroma, rms = _features(y)
    env, low = _onset(bands)
    bpm, period = estimate_tempo(env)
    if bpm is None or len(env) < 4 * period:
        return {"version": VERSION, "tempo": 0.0, "beats": [], "bpb": 4, "phase": 0, "clarity": 0.0}
    frames = track_beats(env, period)
    # 音が鳴っていない所 (曲の前後の無音) の拍は使わない
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    alive = db > db.max() - 40
    frames = np.array([f for f in frames if alive[max(0, f - 20):f + 20].any()], dtype=int)
    if len(frames) < 4:
        return {"version": VERSION, "tempo": round(float(bpm), 2), "beats": [], "bpb": 4, "phase": 0, "clarity": 0.0}
    bpb, phase, clarity = find_downbeats(frames, low, chroma)
    times = frames * HOP / SR
    tempo = 60.0 / _period(times)
    return {"version": VERSION, "tempo": round(tempo, 2), "beats": [round(float(t), 4) for t in times],
            "bpb": int(bpb), "phase": int(phase), "clarity": round(clarity, 3)}


def analyze_beats(path, cache_json=None):
    """拍と小節を解析 (cache_json があれば結果を保存/再利用)。"""
    if cache_json and os.path.exists(cache_json):
        try:
            with open(cache_json, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("version") == VERSION:
                return d
        except Exception:
            pass
    d = analyze(path)
    if cache_json:
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump(d, f)
    return d


# ---------------------------------------------------------------------------
# 画面で使う「小節の線」(解析結果 + 自分で直した分)
# ---------------------------------------------------------------------------
class BeatGrid:
    """解析した拍の位置と、画面で直した内容 (拍子・1拍目・ずらし・手動テンポ) をまとめて扱う。"""

    USER_KEYS = ("bpb", "phase", "shift", "manual_bpm", "anchor", "anchor_bar")

    def __init__(self, info, duration, user_path=None):
        self.info = info
        self.duration = max(float(duration or 0), 1.0)
        self.user_path = user_path
        self.base = np.array(info.get("beats") or [], dtype=float)
        self.bpb = int(info.get("bpb", 4))
        self.phase = int(info.get("phase", 0))
        self.shift = 0.0
        self.manual_bpm = None            # 数字を入れると、そのテンポの等間隔の線になる
        self.anchor = 0.0                 # 手動テンポの基準の小節の頭 (秒)
        self.anchor_bar = 1
        self._load_user()
        self._build()

    # ----- 保存 -----
    def _load_user(self):
        if self.user_path and os.path.exists(self.user_path):
            try:
                with open(self.user_path, encoding="utf-8") as f:
                    u = json.load(f)
                for k in self.USER_KEYS:
                    if k in u:
                        setattr(self, k, u[k])
            except Exception:
                pass

    def save(self):
        if not self.user_path:
            return
        try:
            with open(self.user_path, "w", encoding="utf-8") as f:
                json.dump({k: getattr(self, k) for k in self.USER_KEYS}, f)
        except Exception:
            pass

    def reset(self):
        self.bpb = int(self.info.get("bpb", 4))
        self.phase = int(self.info.get("phase", 0))
        self.shift, self.manual_bpm, self.anchor, self.anchor_bar = 0.0, None, 0.0, 1
        self._build()

    # ----- 拍の列を作る -----
    @property
    def ok(self):
        return len(self.base) >= 4 or self.manual_bpm is not None

    @property
    def analyzed_bpm(self):
        return float(self.info.get("tempo") or 120.0)

    @property
    def bpm(self):
        if self.manual_bpm:
            return float(self.manual_bpm)
        if len(self.base) >= 2:
            return 60.0 / _period(self.base)
        return self.analyzed_bpm

    def _build(self):
        """画面に使う拍の時刻 self.t (曲の頭〜終わりまで延長) と、小節番号の基準を作る。"""
        self.bpb = int(min(12, max(2, self.bpb)))
        if self.manual_bpm:
            p = 60.0 / float(self.manual_bpm)
            a = self.anchor
            k0 = -int(np.ceil(a / p)) - 1
            k1 = int(np.ceil((self.duration - a) / p)) + 2
            ks = np.arange(k0, k1)
            self.t = a + ks * p + self.shift
            self.idx0 = -k0                           # self.t[idx0] = 基準の小節の頭
            self.first_bar_index = self.idx0 - (self.anchor_bar - 1) * self.bpb
            return
        b = self.base
        if len(b) < 2:
            self.t = np.array([])
            self.first_bar_index = 0
            return
        p = _period(b)
        n_pre = int(np.floor(b[0] / p))
        n_post = int(np.ceil(max(0.0, self.duration - b[-1]) / p))
        pre = b[0] - p * np.arange(n_pre, 0, -1)
        post = b[-1] + p * np.arange(1, n_post + 1)
        self.t = np.concatenate([pre, b, post]) + self.shift
        self.n_pre = n_pre
        # 1小節目 = 曲が始まってから最初の1拍目
        i = n_pre + (self.phase % self.bpb)
        self.first_bar_index = i

    def is_bar(self, i):
        return (i - self.first_bar_index) % self.bpb == 0

    def bar_number(self, i):
        """拍 i が属する小節の番号 (1から)。曲が始まる前は 0 以下。"""
        return (i - self.first_bar_index) // self.bpb + 1

    def lines(self, t0, t1):
        """画面に入る拍: [(時刻, 小節の頭か, 小節番号, 何拍目 (1〜))]"""
        if len(self.t) == 0:
            return []
        a = max(0, int(np.searchsorted(self.t, t0)) - 1)
        b = min(len(self.t), int(np.searchsorted(self.t, t1)) + 1)
        out = []
        for i in range(a, b):
            out.append((float(self.t[i]), self.is_bar(i), self.bar_number(i),
                        (i - self.first_bar_index) % self.bpb + 1))
        return out

    def nearest(self, t):
        if len(self.t) == 0:
            return None
        i = int(np.argmin(np.abs(self.t - t)))
        return i

    def bar_at(self, t):
        """時刻 t の小節番号と拍。"""
        if len(self.t) == 0:
            return None, None
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        i = max(0, i)
        return self.bar_number(i), (i - self.first_bar_index) % self.bpb + 1

    # ----- 画面からの調整 -----
    def set_downbeat_at(self, t):
        """時刻 t に一番近い拍を、小節の頭 (1拍目) にする。"""
        i = self.nearest(t)
        if i is None:
            return None
        if self.manual_bpm:
            # 手動テンポ: 基準をその拍にする (小節番号はそのまま近いものを使う)
            bar = self.bar_number(i) if self.is_bar(i) else self.bar_number(i) + 1
            self.anchor = float(self.t[i]) - self.shift
            self.anchor_bar = bar
        else:
            self.phase = (i - self.n_pre) % self.bpb
        self._build()
        return float(self.t[i])

    def move_downbeat(self, beats):
        """小節の頭を拍の単位でずらす (+1 = 1拍あと)。"""
        if self.manual_bpm:
            self.anchor += beats * 60.0 / float(self.manual_bpm)
        else:
            self.phase = (self.phase + beats) % self.bpb
        self._build()

    def set_bpb(self, bpb, keep_t=None):
        """拍子を変える。keep_t の近くの小節の頭はそのまま残す。"""
        keep = None
        if keep_t is not None and len(self.t):
            i = self.nearest(keep_t)
            # keep_t より前で一番近い小節の頭
            while i > 0 and not self.is_bar(i):
                i -= 1
            keep = i
        self.bpb = int(bpb)
        if keep is not None:
            if self.manual_bpm:
                self.anchor = float(self.t[keep]) - self.shift
                self.anchor_bar = max(1, self.bar_number(keep))
            else:
                self.phase = (keep - self.n_pre) % self.bpb
        self._build()

    def set_shift(self, s):
        self.shift = float(round(s, 4))
        self._build()

    def set_bpm(self, bpm, keep_t=None):
        """テンポを手動で指定 (keep_t の近くの小節の頭を基準に伸び縮み)。None で解析結果に戻す。"""
        if bpm is None:
            self.manual_bpm = None
            self._build()
            return
        bpm = float(min(300.0, max(30.0, round(bpm, 2))))
        if not self.manual_bpm:
            # 解析の線から手動へ: 基準を keep_t の手前の小節の頭にする
            ref = keep_t if keep_t is not None else 0.0
            i = self.nearest(ref) if len(self.t) else None
            if i is not None:
                while i > 0 and not self.is_bar(i):
                    i -= 1
                self.anchor = float(self.t[i]) - self.shift
                self.anchor_bar = self.bar_number(i)
            else:
                self.anchor, self.anchor_bar = 0.0, 1
        self.manual_bpm = bpm
        self._build()


def main():
    ap = argparse.ArgumentParser(description="伴奏の拍と小節を解析します")
    ap.add_argument("audio", help="WAV ファイル (分離した accompaniment.wav など)")
    args = ap.parse_args()
    if not os.path.isfile(args.audio):
        sys.exit(f"ファイルが見つかりません: {args.audio}")
    d = analyze(args.audio)
    print(f"テンポ: ♩ = {d['tempo']:.1f}  拍子: {d['bpb']}/4  拍の数: {len(d['beats'])}  はっきり度: {d['clarity']:.2f}")
    if d["beats"]:
        b = d["beats"]
        first = b[d["phase"]]
        print(f"最初の小節の頭: {first:.2f}秒")


if __name__ == "__main__":
    main()
