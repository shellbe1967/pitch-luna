# -*- coding: utf-8 -*-
"""
ピッチ検出と「ドレミ」変換の共通部品。
audio_processor.py / singing_trainer.py から利用します (numpy のみで動作)。
"""

import math

import numpy as np

JP_NOTE_NAMES = ["ド", "ド#", "レ", "レ#", "ミ", "ファ", "ファ#", "ソ", "ソ#", "ラ", "ラ#", "シ"]
EN_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
A4_FREQ = 440.0
A4_MIDI = 69


def hz_to_midi(freq_hz):
    """周波数(Hz) → MIDIノート番号(小数)。A4=440Hz=69。無効値は nan。"""
    if freq_hz is None or freq_hz <= 0:
        return float("nan")
    return A4_MIDI + 12.0 * math.log2(freq_hz / A4_FREQ)


def midi_to_hz(midi):
    return A4_FREQ * 2.0 ** ((midi - A4_MIDI) / 12.0)


def midi_to_doremi(midi_note, octave=True):
    """MIDI番号(整数) → 「ド4」「ファ#3」。MIDI 60 = ド4 (中央のド)。"""
    midi_note = int(midi_note)
    name = JP_NOTE_NAMES[midi_note % 12]
    return f"{name}{midi_note // 12 - 1}" if octave else name


def midi_to_english(midi_note, octave=True):
    midi_note = int(midi_note)
    name = EN_NOTE_NAMES[midi_note % 12]
    return f"{name}{midi_note // 12 - 1}" if octave else name


def hz_to_doremi(freq_hz):
    """周波数 → (ドレミ表記, 英語表記, 最寄りMIDI番号, セントずれ)。無効なら None。"""
    midi_f = hz_to_midi(freq_hz)
    if math.isnan(midi_f):
        return None
    nearest = int(round(midi_f))
    return midi_to_doremi(nearest), midi_to_english(nearest), nearest, (midi_f - nearest) * 100.0


def rms_dbfs(frame):
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 1e-10 else -120.0


class YinPitchDetector:
    """YIN (de Cheveigné & Kawahara 2002) を FFT で高速化した実装。

    det(frame) → (周波数Hz, 信頼度0〜1)。検出なしは (0.0, 0.0)。
    """

    def __init__(self, samplerate=44100, fmin=65.0, fmax=1100.0, threshold=0.15, fallback=None):
        """fallback: 閾値を下回る谷が無いとき、最小値がこの値未満なら「自信低め」として採用 (None=採用しない)。"""
        self.sr = samplerate
        self.tau_min = max(2, int(samplerate / fmax))
        self.tau_max = int(samplerate / fmin)
        self.threshold = threshold
        self.fallback = fallback

    def min_frame_length(self):
        return 2 * self.tau_max + 2

    def __call__(self, frame):
        x = np.asarray(frame, dtype=np.float64)
        x = x - x.mean()
        n = len(x)
        w = n - self.tau_max
        if w <= self.tau_max // 2:
            raise ValueError("フレームが短すぎます")

        size = 1 << (2 * n - 1).bit_length()
        fx = np.fft.rfft(x, size)
        fw = np.fft.rfft(x[:w][::-1], size)
        corr = np.fft.irfft(fx * fw, size)[w - 1 : w - 1 + self.tau_max + 1]
        sq = np.cumsum(np.concatenate(([0.0], x * x)))
        d = sq[w] + (sq[w : w + self.tau_max + 1] - sq[: self.tau_max + 1]) - 2.0 * corr

        cmndf = np.ones_like(d)
        cums = np.cumsum(d[1:])
        cmndf[1:] = d[1:] * np.arange(1, len(d)) / np.where(cums == 0, 1e-12, cums)

        search = cmndf[self.tau_min : self.tau_max]
        below = np.where(search < self.threshold)[0]
        if below.size:
            tau = below[0] + self.tau_min
            while tau + 1 < self.tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
        elif self.fallback is not None and search.size and search.min() < self.fallback:
            tau = int(np.argmin(search)) + self.tau_min   # 息や残響で濁った声でも一番それらしい周期を採用
        else:
            return 0.0, 0.0

        a, b, c = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(self.sr / (tau + shift)), float(max(0.0, 1.0 - cmndf[tau]))
