#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステップ1: マイク入力 → リアルタイムピッチ検出 → 「ドレミ」表示テスト

使い方:
    python test_mic_pitch.py                 # 既定のマイクで開始 (Ctrl+C で終了)
    python test_mic_pitch.py --list          # 入力デバイス一覧
    python test_mic_pitch.py --device 2      # デバイス番号を指定
    python test_mic_pitch.py --threshold -45 # ミュートする音量の閾値 (dBFS) を変更
    python test_mic_pitch.py --selftest      # マイク不要。合成音で検出精度を確認

必要ライブラリ: numpy, sounddevice  (aubio は任意。入っていれば --engine aubio で使用可)
"""

import argparse
import math
import queue
import sys
import time
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# 音名マッピング
# ---------------------------------------------------------------------------
JP_NOTE_NAMES = ["ド", "ド#", "レ", "レ#", "ミ", "ファ", "ファ#", "ソ", "ソ#", "ラ", "ラ#", "シ"]
EN_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
A4_FREQ = 440.0
A4_MIDI = 69


def hz_to_midi(freq_hz: float) -> float:
    """周波数(Hz) → MIDIノート番号(小数)。A4=440Hz=69。"""
    if freq_hz is None or freq_hz <= 0:
        return float("nan")
    return A4_MIDI + 12.0 * math.log2(freq_hz / A4_FREQ)


def midi_to_hz(midi: float) -> float:
    return A4_FREQ * 2.0 ** ((midi - A4_MIDI) / 12.0)


def midi_to_doremi(midi_note: int) -> str:
    """MIDIノート番号(整数) → 「ド4」「ファ#3」形式。MIDI 60 = ド4 (中央のド)。"""
    name = JP_NOTE_NAMES[midi_note % 12]
    octave = midi_note // 12 - 1
    return f"{name}{octave}"


def midi_to_english(midi_note: int) -> str:
    return f"{EN_NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


def hz_to_doremi(freq_hz: float):
    """周波数(Hz) → (ドレミ表記, 英語表記, 最寄りMIDI番号, セントずれ)。"""
    midi_f = hz_to_midi(freq_hz)
    if math.isnan(midi_f):
        return None
    nearest = int(round(midi_f))
    cents = (midi_f - nearest) * 100.0
    return midi_to_doremi(nearest), midi_to_english(nearest), nearest, cents


# ---------------------------------------------------------------------------
# ピッチ検出 (YIN アルゴリズム / numpy のみで動作)
# ---------------------------------------------------------------------------
class YinPitchDetector:
    """de Cheveigné & Kawahara (2002) の YIN をFFTで高速化した実装。"""

    def __init__(self, samplerate=44100, fmin=65.0, fmax=1100.0, threshold=0.15):
        self.sr = samplerate
        self.tau_min = max(2, int(samplerate / fmax))
        self.tau_max = int(samplerate / fmin)
        self.threshold = threshold

    def __call__(self, frame: np.ndarray):
        """1フレームの周波数(Hz)と信頼度(0〜1)を返す。検出なしは (0.0, 0.0)。"""
        x = frame.astype(np.float64)
        x = x - x.mean()
        n = len(x)
        w = n - self.tau_max  # 積分窓の長さ
        if w <= self.tau_max // 2:
            raise ValueError("フレームが短すぎます (fmin を上げるか、フレームを長くしてください)")

        # 差分関数 d(tau) = Σ (x[j] - x[j+tau])^2  をFFTで計算
        size = 1 << (2 * n - 1).bit_length()
        fx = np.fft.rfft(x, size)
        fw = np.fft.rfft(x[:w][::-1], size)
        corr = np.fft.irfft(fx * fw, size)[w - 1 : w - 1 + self.tau_max + 1]
        sq = np.cumsum(np.concatenate(([0.0], x * x)))
        energy0 = sq[w]
        energy_tau = sq[w : w + self.tau_max + 1] - sq[: self.tau_max + 1]
        d = energy0 + energy_tau - 2.0 * corr

        # 累積平均正規化差分関数 (CMNDF)
        cmndf = np.ones_like(d)
        cums = np.cumsum(d[1:])
        taus = np.arange(1, len(d))
        cmndf[1:] = d[1:] * taus / np.where(cums == 0, 1e-12, cums)

        # 閾値を下回る最初の谷を探す
        tau = -1
        search = cmndf[self.tau_min : self.tau_max]
        below = np.where(search < self.threshold)[0]
        if below.size:
            t = below[0] + self.tau_min
            while t + 1 < self.tau_max and cmndf[t + 1] < cmndf[t]:
                t += 1
            tau = t
        else:
            return 0.0, 0.0

        # 放物線補間でサブサンプル精度に
        if 1 <= tau < len(cmndf) - 1:
            a, b, c = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
            denom = a - 2 * b + c
            shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        else:
            shift = 0.0
        better_tau = tau + shift
        confidence = float(max(0.0, 1.0 - cmndf[tau]))
        return float(self.sr / better_tau), confidence


class AubioPitchDetector:
    """aubio が使える環境向け (任意)。"""

    def __init__(self, samplerate=44100, buf_size=2048, hop_size=1024):
        import aubio  # noqa: 遅延import

        self.p = aubio.pitch("yinfft", buf_size, hop_size, samplerate)
        self.p.set_unit("Hz")
        self.p.set_silence(-90)
        self.hop = hop_size

    def __call__(self, frame):
        f = self.p(frame[-self.hop :].astype(np.float32))[0]
        return float(f), float(self.p.get_confidence())


def rms_dbfs(frame: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 1e-10 else -120.0


# ---------------------------------------------------------------------------
# 表示
# ---------------------------------------------------------------------------
def tuning_meter(cents: float, width: int = 21) -> str:
    """-50〜+50セントを「----|----」型のメーターで表示。"""
    pos = int(round((cents + 50) / 100 * (width - 1)))
    pos = min(max(pos, 0), width - 1)
    chars = ["-"] * width
    chars[width // 2] = "|"
    chars[pos] = "●"
    return "".join(chars)


def format_line(freq, conf, db, threshold_db):
    level = int(min(max((db + 70) / 70, 0), 1) * 10)
    vol = "▮" * level + "▯" * (10 - level)
    if db < threshold_db:
        return f"音量 {vol} {db:6.1f}dB | (ミュート: 小さい音は無視中)"
    if freq <= 0:
        return f"音量 {vol} {db:6.1f}dB | (音程を検出できません)"
    name, en, _, cents = hz_to_doremi(freq)
    return (
        f"音量 {vol} {db:6.1f}dB | {freq:7.1f} Hz  →  {name:<5} ({en:<3}) "
        f"{cents:+5.0f}¢ [{tuning_meter(cents)}] 信頼度{conf:.2f}"
    )


# ---------------------------------------------------------------------------
# マイク入力ループ
# ---------------------------------------------------------------------------
def run_mic(args):
    try:
        import sounddevice as sd
    except OSError as e:
        sys.exit(
            f"sounddevice を読み込めません: {e}\n"
            "  Windows/macOS: pip install --upgrade sounddevice で解決することが多いです\n"
            "  Linux: sudo apt install libportaudio2"
        )

    if args.list:
        print(sd.query_devices())
        return

    sr = args.samplerate
    frame_len = args.frame
    hop = args.hop
    if args.engine == "aubio":
        detector = AubioPitchDetector(sr, frame_len, hop)
    else:
        detector = YinPitchDetector(sr, fmin=args.fmin, fmax=args.fmax)

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, t, status):
        if status:
            print(f"\n[警告] {status}", file=sys.stderr)
        q.put(indata[:, 0].copy())

    ring = np.zeros(frame_len, dtype=np.float32)
    history = deque(maxlen=args.smooth)  # メディアンで揺れを抑える

    dev_name = sd.query_devices(args.device, "input")["name"]
    print(f"入力デバイス: {dev_name}  /  {sr}Hz  /  閾値 {args.threshold} dBFS")
    print("声を出してみてください。Ctrl+C で終了します。\n")

    with sd.InputStream(
        device=args.device, channels=1, samplerate=sr, blocksize=hop, dtype="float32", callback=callback
    ):
        try:
            while True:
                block = q.get()
                ring = np.concatenate((ring[len(block) :], block))
                db = rms_dbfs(block)
                if db < args.threshold:
                    history.clear()
                    freq, conf = 0.0, 0.0
                else:
                    freq, conf = detector(ring)
                    if freq > 0 and conf >= args.min_conf:
                        history.append(freq)
                    freq = float(np.median(history)) if history else 0.0
                sys.stdout.write("\r\033[K" + format_line(freq, conf, db, args.threshold))
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n終了しました。")


# ---------------------------------------------------------------------------
# セルフテスト (マイク不要)
# ---------------------------------------------------------------------------
def run_selftest(args):
    sr = args.samplerate
    det = YinPitchDetector(sr, fmin=args.fmin, fmax=args.fmax)
    rng = np.random.default_rng(0)
    t = np.arange(args.frame) / sr

    print("== 1. 音名変換 ==")
    for hz, expect in [(261.63, "ド4"), (440.0, "ラ4"), (185.0, "ファ#3"), (130.81, "ド3"), (523.25, "ド5")]:
        got = hz_to_doremi(hz)[0]
        print(f"  {hz:7.2f} Hz → {got:<5} {'OK' if got == expect else 'NG (期待 ' + expect + ')'}")
        assert got == expect

    print("\n== 2. 声に近い合成音 (倍音 + ビブラート + ノイズ) の検出 ==")
    worst = 0.0
    for midi in range(43, 82, 3):  # ソ2 〜 ラ5 程度 (男声〜女声の範囲)
        f0 = midi_to_hz(midi)
        vib = 1 + 0.006 * np.sin(2 * np.pi * 5.5 * t)
        phase = 2 * np.pi * np.cumsum(f0 * vib) / sr
        sig = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 7))
        sig = 0.3 * sig / np.max(np.abs(sig)) + 0.01 * rng.standard_normal(len(t))
        hz, conf = det(sig)
        err = abs(1200 * math.log2(hz / f0)) if hz > 0 else 9999
        worst = max(worst, err)
        name = hz_to_doremi(hz)[0] if hz > 0 else "-"
        ok = name == midi_to_doremi(midi)
        print(f"  正解 {midi_to_doremi(midi):<5} {f0:7.1f}Hz → 検出 {hz:7.1f}Hz {name:<5} 誤差{err:5.1f}¢ {'OK' if ok else 'NG'}")
        assert ok

    print(f"  最大誤差: {worst:.1f} セント")

    print("\n== 3. ミュート処理 (小さいノイズは無視) ==")
    noise = 0.002 * rng.standard_normal(args.frame)
    db = rms_dbfs(noise)
    print("  " + format_line(0, 0, db, args.threshold))
    assert db < args.threshold

    print("\n== 4. 表示サンプル ==")
    for hz in (261.0, 268.0, 392.0, 600.0):
        print("  " + format_line(hz, 0.95, -20.0, args.threshold))

    t0 = time.perf_counter()
    for _ in range(200):
        det(sig)
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"\n== 5. 処理速度: 1フレーム {ms:.2f} ms (リアルタイム要件 {args.hop / sr * 1000:.1f} ms 以内) ==")
    print("\nセルフテスト: すべて合格")


def main():
    ap = argparse.ArgumentParser(description="マイク入力のピッチを「ドレミ」で表示するテスト")
    ap.add_argument("--list", action="store_true", help="入力デバイス一覧を表示")
    ap.add_argument("--device", type=int, default=None, help="入力デバイス番号")
    ap.add_argument("--samplerate", type=int, default=44100)
    ap.add_argument("--frame", type=int, default=2048, help="解析フレーム長 (サンプル)")
    ap.add_argument("--hop", type=int, default=1024, help="更新間隔 (サンプル)")
    ap.add_argument("--threshold", type=float, default=-40.0, help="この音量(dBFS)未満はミュート")
    ap.add_argument("--min-conf", type=float, default=0.8, help="この信頼度未満の検出は捨てる")
    ap.add_argument("--smooth", type=int, default=3, help="メディアン平滑化のフレーム数")
    ap.add_argument("--fmin", type=float, default=65.0)
    ap.add_argument("--fmax", type=float, default=1100.0)
    ap.add_argument("--engine", choices=["yin", "aubio"], default="yin")
    ap.add_argument("--selftest", action="store_true", help="マイクなしで動作確認")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
    else:
        run_mic(args)


if __name__ == "__main__":
    main()
