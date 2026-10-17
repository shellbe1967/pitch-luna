#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitch Luna — If Vocal (歌の音程練習ソフト)

  MP3 を読み込む → AIでボーカル分離・お手本の音程とキーを自動解析
  → 伴奏 (+ボーカル) を流しながら、お手本バーと自分の声 (●) を「ドレミ」の高さで表示

起動:
    python singing_trainer.py
      → 曲の一覧が開きます。「フォルダを開く」で曲の入ったフォルダを選び、
        曲をダブルクリック (または Enter / 「▶ 解析して練習」) すると、その曲だけ解析して練習画面へ進みます。
        解析済みの曲はすぐ開きます。同じ名前の .mid があれば自動でお手本に使います。
    python singing_trainer.py song.mp3     (一覧でその曲を選んだ状態で起動。解析は開いたときに行います)

操作:
    Space   再生 / 一時停止          V      ボーカル ON / OFF
    ← / →   巻き戻し / スキップ       ↑ / ↓  スキップ間隔 ±1秒
    [ / ]   お手本を1オクターブ下げる / 上げる (声の高さに合わせる)
    - / +   音量                     P      お手本の細い音程線 表示切替
    R       採点リセット              Home   最初から
    F11     全画面 ⇔ ウィンドウ (ウィンドウは最大化・最小化・サイズ変更できます)
    O       自分の歌声をスピーカー / ヘッドホンから出す ON / OFF (ヘッドホン推奨)
    F1      操作方法を表示            Ctrl+O 曲を開く   Ctrl+M MIDI を開く

マウス操作:
    メニューバー (ファイル / 再生 / 音声 / お手本 / 表示 / ヘルプ) と下のボタンで、ほぼすべての操作ができます
    上の青い進行バー: クリック・ドラッグで移動     譜面: ドラッグで前後へ、クリックでその時刻へ
    ホイール: 1秒ずつ移動                          Ctrl + ホイール: 横の拡大 / 縮小
    , / .   お手本のタイミングを 0.05秒 早める / 遅らせる (MIDI 使用時。次回も保持)
    M       お手本の切替: MIDI ⇔ 歌声から解析 (MIDI 使用時)
    プログレスバーをクリック → その位置へ移動          Esc  終了

必要ファイル (同じフォルダ): pitch_core.py, audio_processor.py, key_analyzer.py, midi_loader.py, ui_menu.py
必要ライブラリ: pygame numpy scipy sounddevice librosa soundfile demucs torchcrepe
"""

import argparse
import bisect
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np

from pitch_core import YinPitchDetector, hz_to_midi, midi_to_doremi, midi_to_english, rms_dbfs
from ui_menu import SEPARATOR, Button, MenuBar, MenuItem, PopupMenu, draw_tooltip
import lyrics_view
from lyrics_view import LyricsMixin, lyric_lane_height

lyrics_view.bind(globals())              # 歌詞の表示から、この画面の変数 (幅・高さなど) を使えるように

APP_TITLE = "Pitch Luna — If Vocal"
W, H = 1280, 720          # 現在の描画サイズ (ウィンドウに合わせて変わる)
DEFAULT_SIZE = (1280, 720)
MIN_W, MIN_H = 980, 600   # これより小さいウィンドウでは、この大きさで描いて縮小表示する
MENU_H = 28               # メニューバー
HEADER_H = MENU_H + 112   # メニューバー + 曲名・キー・情報
PROGRESS_H = 10
FOOTER_H = 48             # 下のツールバー (クリックで操作)
LYRIC_H = 0               # 楽譜の下の「歌詞の段」の高さ (歌詞があって表示するとき)
LABEL_W = 92
PX_PER_SEC = 160          # 横方向: 1秒あたりのピクセル数
PLAYHEAD_RATIO = 0.30     # 現在位置の線を左から何割の位置に置くか
TRAIL_SEC = 6.0           # 自分の声の軌跡を何秒分残すか

NATURAL_PCS = (0, 2, 4, 5, 7, 9, 11)
from theme import BG_GLOW, ROW_TINT
from theme import (ACCENT, BAD, BAR, BAR_HIT, BAR_PAST, BG, BG_DEEP, DIM, FAINT, FREE, GLOW, GOLD, GOOD, HAIR,
                   HAIR_DIM, HAIR_STRONG, LABEL, NEAR, OUTLINE, PANEL, PANEL_HI, ROW_OTHER, ROW_SCALE, USER,
                   USER_HIT, WHITE)

LINE, LINE_DO = HAIR, HAIR_STRONG
SHARP_COLOR = FAINT


def note_color(midi):
    return BAR


def fmt_time(t):
    t = max(0, int(t))
    return f"{t // 60}:{t % 60:02d}"


# ---------------------------------------------------------------------------
# お手本データ (pitch.json) の扱い
# ---------------------------------------------------------------------------
class Guide:
    def __init__(self, data, notes=None, rng=None, source="歌声から解析"):
        self.base_notes = notes if notes is not None else data["notes"]
        self.source = source
        self.offset = 0.0
        self._apply_offset()
        self.pitch = data.get("pitch", [])
        self.hop = data.get("hop_sec", 0.01)
        self.duration = data.get("duration", 0)
        r = rng or data["range"]
        self.min_midi = r["min_midi"]
        self.max_midi = r["max_midi"]

    def _apply_offset(self):
        o = self.offset
        self.notes = [dict(n, start=n["start"] + o, end=n["end"] + o) for n in self.base_notes] if o else list(self.base_notes)
        self.starts = [n["start"] for n in self.notes]

    def shift(self, d):
        """お手本バーのタイミングを d 秒ずらす。"""
        self.offset = round(self.offset + d, 3)
        self._apply_offset()

    def index_at(self, t):
        """時刻 t に鳴っているバーの番号 (無ければ -1)。"""
        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0 and self.notes[i]["start"] <= t < self.notes[i]["end"]:
            return i
        return -1

    def note_at(self, t):
        """時刻 t に鳴っているお手本バー (無ければ None)。"""
        i = bisect.bisect_right(self.starts, t) - 1
        if i >= 0 and self.notes[i]["start"] <= t < self.notes[i]["end"]:
            return self.notes[i]
        return None

    def visible(self, t0, t1):
        """t0〜t1 に少しでもかかるバーの範囲 (インデックス)。"""
        lo = max(0, bisect.bisect_left(self.starts, t0) - 50)
        hi = bisect.bisect_right(self.starts, t1)
        for i in range(lo, hi):
            if self.notes[i]["end"] >= t0:
                yield self.notes[i]

    def visible_idx(self, t0, t1):
        """visible と同じ。ただし (番号, バー) を返す。"""
        lo = max(0, bisect.bisect_left(self.starts, t0) - 50)
        hi = bisect.bisect_right(self.starts, t1)
        for i in range(lo, hi):
            if self.notes[i]["end"] >= t0:
                yield i, self.notes[i]

    def pitch_points(self, t0, t1):
        a = max(0, int(t0 / self.hop))
        b = min(len(self.pitch), int(t1 / self.hop) + 1)
        for i in range(a, b):
            yield i * self.hop, self.pitch[i]


VIEW_DEFAULT = {
    "semis": 20,          # 画面に入る半音の数 (少ないほど 1段が高く = ヨコ線の間隔が広い)
    "pxsec": 160,         # 1秒の幅 (タテ方向の時間の間隔)
    "vgrid": 1.0,         # タテ線の間隔 (秒)
    "line": 1.0,          # 線の濃さ (1.0 = 標準)
    "sharp_lines": True,  # ♯の段の点線
    "follow": True,       # 歌う音に合わせて上下に自動で動かす (音階を中央に)
    "vmode": "bar",       # タテ線: "bar" = 小節と拍 (解析) / "sec" = 秒ごと
    "beat_lines": True,   # 小節の中の拍の点線
    "toolbar2": True,     # 下の2段目 (練習ツール: A-B ループ・速度・キー・録音・編集)
    "lyrics": True,       # 歌詞の段 (歌詞を入れた曲だけ)
    "ruby": True,         # 歌詞のふりがな
    "lyric_color": True,  # 歌った歌詞の色を、音程が合っていたかで変える
    "lyric_size": 20,     # 歌詞の文字の大きさ
}
VGRID_CHOICES = (0.5, 1.0, 2.0, 5.0)
BPB_CHOICES = (2, 3, 4, 6)
BARS_CHOICES = (1, 2, 4, 8, 16)   # 画面に表示する小節の数
PXSEC_MIN, PXSEC_MAX = 30, 800
SEMIS_MIN, SEMIS_MAX = 6, 60      # 画面に入る段 (半音) の数
OCT_CHOICES = (1, 1.5, 2, 3, 4)   # 画面に入れるオクターブの数
LINE_CHOICES = (0, 50, 100, 150, 200)
RULER_H = 22              # 楽譜の上の「小節の目盛り」の高さ


def view_range(guide, octave_shift):
    """縦軸に表示する MIDI の範囲 (最低14半音、上下に余白)。"""
    lo = guide.min_midi + 12 * octave_shift - 2
    hi = guide.max_midi + 12 * octave_shift + 2
    while hi - lo < 14:
        lo -= 1
        hi += 1
    return lo, hi


def judge(user_midi, target_midi):
    """自分の声とお手本の差 → (色, セント差)。"""
    if user_midi is None:
        return None, None
    if target_midi is None:
        return FREE, None
    cents = (user_midi - target_midi) * 100
    a = abs(cents)
    return (GOOD if a <= 50 else NEAR if a <= 100 else BAD), cents


# ---------------------------------------------------------------------------
# 伴奏 + ボーカルの同期再生 (pygame.mixer)
# ---------------------------------------------------------------------------
def read_wav_int16(path):
    from scipy.io import wavfile

    sr, d = wavfile.read(path)
    if d.dtype != np.int16:
        d = np.clip(d.astype(np.float32), -1, 1)
        d = (d * 32767).astype(np.int16)
    if d.ndim == 1:
        d = np.stack([d, d], axis=1)
    return sr, np.ascontiguousarray(d[:, :2])


class FreePlayback:
    """曲を選ぶ前の空の楽譜用。音は出さず、時計だけ進む (自分の声の軌跡を流して見られる)。"""

    def __init__(self):
        self.duration = 0.0
        self.volume = 0.8
        self.vocal_on = True
        self.playing = False
        self.offset = 0.0
        self.started_at = 0.0
        self.loaded = True
        self.speed, self.semis = 1.0, 0
        self.fx_busy, self.fx_progress, self.fx_target = False, 1.0, (1.0, 0)
        self.take = None
        self.vocal_gain = 1.0

    def set_vocal_gain(self, g, restart=True):
        self.vocal_gain = round(min(2.0, max(0.0, g)), 2)

    def position(self):
        return self.offset + (time.perf_counter() - self.started_at) * self.speed if self.playing else self.offset

    def request_fx(self, speed, semis):
        """音が無いので、すぐに切り替える。"""
        pos = self.position()
        self.speed, self.semis = speed, semis
        self.fx_target = (speed, semis)
        self.offset, self.started_at = pos, time.perf_counter()
        return True

    def poll_fx(self):
        return False

    def play_take(self, take_audio, t0):
        self.seek(t0)
        if not self.playing:
            self.toggle_play()
        self.take = take_audio

    def stop_take(self):
        self.take = None

    def toggle_play(self):
        if self.playing:
            self.offset = self.position()
            self.playing = False
            self.take = None
        else:
            self.started_at = time.perf_counter()
            self.playing = True

    def seek(self, t):
        self.offset = max(0.0, t)
        self.started_at = time.perf_counter()
        self.take = None
        return self.offset

    def toggle_vocal(self):
        self.vocal_on = not self.vocal_on

    def change_volume(self, d):
        self.volume = min(1.0, max(0.0, round(self.volume + d, 2)))

    def close(self):
        pass


EMPTY_GUIDE_DATA = {"notes": [], "pitch": [], "hop_sec": 0.01, "duration": 0.0,
                    "range": {"min_midi": 55, "max_midi": 74}}   # ソ3〜レ5 (男声・女声どちらも入る高さ)


SONG_OUT = {"name": None, "used": None, "error": None}   # 曲を鳴らすスピーカー (None = Windows の既定)


def _norm_dev(name):
    return "".join(str(name).lower().split())


def match_device(want, names):
    """want (PortAudio の名前) に一番近い名前を names から探す。見つからなければ None。"""
    if not want:
        return None
    w = _norm_dev(want)
    for n in names:
        if _norm_dev(n) == w:
            return n
    for n in names:                      # MME は名前が31文字で切れる / 方式で少し違うことがある
        nn = _norm_dev(n)
        if nn.startswith(w[:24]) or w.startswith(nn[:24]):
            return n
    return None


def sdl_output_names(pygame):
    """pygame (SDL) から見えるスピーカーの名前。音の仕組みが止まっているときは一時的に起こして調べる。"""
    try:
        from pygame._sdl2 import audio as sdl_audio
    except Exception:
        return []
    try:
        names = list(sdl_audio.get_audio_device_names(False))
        if names:
            return names
    except Exception:
        pass
    try:
        pygame.mixer.init()
        names = list(sdl_audio.get_audio_device_names(False))
        pygame.mixer.quit()
        return names
    except Exception:
        return []


def mixer_init(pygame, frequency):
    """曲用のミキサーを、選んだスピーカーで開く (開けなければ既定のスピーカー)。"""
    SONG_OUT["error"] = None
    SONG_OUT["used"] = None
    want = SONG_OUT["name"]
    if want:
        dev = match_device(want, sdl_output_names(pygame))
        if dev:
            try:
                pygame.mixer.init(frequency=frequency, size=-16, channels=2, buffer=512, devicename=dev)
                SONG_OUT["used"] = dev
                return True
            except Exception as e:
                SONG_OUT["error"] = str(e)
        else:
            SONG_OUT["error"] = "見つかりません"
    pygame.mixer.init(frequency=frequency, size=-16, channels=2, buffer=512)
    return SONG_OUT["error"] is None


class Playback:
    """2つのトラックを同じ瞬間から鳴らし、シーク時は両方を同じ位置から鳴らし直す。"""

    def __init__(self, pygame, accomp_path, vocals_path):
        self.pg = pygame
        sr_a, acc = read_wav_int16(accomp_path)
        sr_v, voc = read_wav_int16(vocals_path)
        if sr_a != sr_v:
            raise RuntimeError("伴奏とボーカルのサンプリング周波数が違います")
        n = max(len(acc), len(voc))
        self.acc = np.zeros((n, 2), np.int16)
        self.voc = np.zeros((n, 2), np.int16)
        self.acc[: len(acc)] = acc
        self.voc[: len(voc)] = voc
        self.acc0, self.voc0 = self.acc, self.voc     # 元の音 (速度・キーを変える前)
        self.sr = sr_a
        self.duration = n / self.sr
        # 再生速度 (音程はそのまま) と キー (半音)
        self.speed, self.semis = 1.0, 0
        self.fx_target = (1.0, 0)
        self.fx_busy = False
        self.fx_progress = 1.0
        self.fx_error = None
        self._fx_ready = None
        self._fx_cache = {(1.0, 0): (self.acc0, self.voc0)}
        self._fx_job = 0
        self.take = None            # 再生中のテイク (自分の歌声の録音)
        self.vocal_gain = 1.0       # ボーカル (お手本の歌声) の大きさ (0〜2.0。1.0 = 元のまま)
        self._voc_boost = 1.0       # 鳴らしている音に掛けてある倍率 (100% より大きいとき)

        self._open_mixer()

        self.volume = 0.8
        self.vocal_on = True
        self.playing = False
        self.offset = 0.0           # 再生開始 (またはシーク) 時点の曲の位置 [秒]
        self.started_at = 0.0       # そのときの perf_counter
        self.loaded = False         # 現在の位置で Sound が用意されているか
        self._sounds = None

    def _open_mixer(self):
        pygame = self.pg
        pygame.mixer.quit()
        mixer_init(pygame, self.sr)
        pygame.mixer.set_num_channels(5)
        pygame.mixer.set_reserved(3)
        self.ch_acc = pygame.mixer.Channel(0)
        self.ch_voc = pygame.mixer.Channel(1)
        self.ch_take = pygame.mixer.Channel(2)       # 録音したテイク

    def reopen(self):
        """スピーカーを変えたとき: 同じ位置・同じ状態で鳴らし直す。"""
        self.stop_take()
        pos, was = self.position(), self.playing
        try:
            self.ch_acc.stop()
            self.ch_voc.stop()
        except Exception:
            pass
        self.playing = False
        self._sounds = None
        self._open_mixer()
        self.offset = pos
        self.loaded = False
        if was:
            self._start_at(pos)

    # --- 速度・キー ---
    def request_fx(self, speed, semis):
        """速度・キーを変える。処理は裏で行い、できたら poll_fx() で切り替える。戻り値: すぐ切り替わったか。"""
        key = (round(float(speed), 2), int(semis))
        self.fx_target = key
        if key in self._fx_cache:
            self._apply_fx(key, *self._fx_cache[key])
            return True
        self._fx_job += 1
        job = self._fx_job
        self.fx_busy, self.fx_progress, self.fx_error = True, 0.0, None

        def work():
            try:
                import audio_fx

                prog = {"a": 0.0, "v": 0.0}

                def pa(f):
                    prog["a"] = f
                    self.fx_progress = (prog["a"] + prog["v"]) / 2

                def pv(f):
                    prog["v"] = f
                    self.fx_progress = (prog["a"] + prog["v"]) / 2

                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=2) as ex:     # 伴奏とボーカルを同時に
                    fa = ex.submit(audio_fx.process, self.acc0, self.sr, key[0], key[1], pa)
                    fv = ex.submit(audio_fx.process, self.voc0, self.sr, key[0], key[1], pv)
                    acc, voc = fa.result(), fv.result()
                if job == self._fx_job:
                    self._fx_ready = (key, acc, voc)
            except Exception as e:
                if job == self._fx_job:
                    self.fx_error = str(e)
                    self.fx_busy = False

        threading.Thread(target=work, daemon=True).start()
        return False

    def poll_fx(self):
        """裏の処理ができていたら切り替える (画面の更新ごとに呼ぶ)。戻り値: 切り替えたか。"""
        r = self._fx_ready
        if r is None:
            return False
        self._fx_ready = None
        key, acc, voc = r
        if len(self._fx_cache) >= 3:                  # 覚えておくのは最近の数個だけ (メモリ節約)
            for k in list(self._fx_cache):
                if k != (1.0, 0) and k != key:
                    del self._fx_cache[k]
                    break
        self._fx_cache[key] = (acc, voc)
        self._apply_fx(key, acc, voc)
        return True

    def _apply_fx(self, key, acc, voc):
        pos, was = self.position(), self.playing
        self.stop_take()
        self.ch_acc.stop()
        self.ch_voc.stop()
        self.acc, self.voc = acc, voc
        self.speed, self.semis = key
        self.fx_busy, self.fx_progress = False, 1.0
        self.offset, self.loaded = pos, False
        self.playing = False
        if was:
            self._start_at(pos)

    # --- テイク (録音) の再生 ---
    def play_take(self, take_audio, t0):
        """曲を t0 秒から鳴らし、同時に録音 (int16 ステレオ・ミキサーと同じ周波数) を鳴らす。"""
        self._start_at(max(0.0, t0))
        snd = self.pg.mixer.Sound(buffer=np.ascontiguousarray(take_audio))
        self._take_snd = snd
        self.ch_take.set_volume(1.0)
        self.ch_take.play(snd)
        self.take = take_audio

    def stop_take(self):
        if self.take is not None:
            try:
                self.ch_take.stop()
            except Exception:
                pass
        self.take = None

    # --- 時計 ---
    def position(self):
        if self.playing:
            p = self.offset + (time.perf_counter() - self.started_at) * self.speed
            if p >= self.duration:
                self.playing = False
                self.offset = self.duration
                return self.duration
            return p
        return self.offset

    # --- 操作 ---
    def _start_at(self, t):
        pg = self.pg
        self.ch_acc.stop()
        self.ch_voc.stop()
        i = min(len(self.acc) - 1, int(t / self.speed * self.sr))
        snd_a = pg.mixer.Sound(buffer=self.acc[i:])
        boost = max(1.0, self.vocal_gain)
        voc = self.voc[i:]
        if boost > 1.0:                               # 100% より大きくするときは音そのものを大きくする
            voc = np.clip(voc.astype(np.int32) * boost, -32768, 32767).astype(np.int16)
        self._voc_boost = boost
        snd_v = pg.mixer.Sound(buffer=voc)
        self._sounds = (snd_a, snd_v)   # 参照を保持 (消えると音が止まるため)
        self._apply_volume()
        # 2つを続けて開始 → 同じミキサーで同じ瞬間から鳴る
        self.ch_acc.play(snd_a)
        self.ch_voc.play(snd_v)
        self.offset = t
        self.started_at = time.perf_counter()
        self.playing = True
        self.loaded = True

    def _apply_volume(self):
        self.ch_acc.set_volume(self.volume)
        g = min(1.0, self.vocal_gain)
        self.ch_voc.set_volume(self.volume * g if self.vocal_on else 0.0)

    def set_vocal_gain(self, g, restart=True):
        """ボーカルの大きさ (0〜2.0)。100% を超える分は、鳴らし直して音を大きくする。"""
        self.vocal_gain = round(min(2.0, max(0.0, g)), 2)
        if max(1.0, self.vocal_gain) != self._voc_boost and self.playing and restart:
            self._start_at(self.position())
        elif max(1.0, self.vocal_gain) != self._voc_boost:
            self.loaded = False                       # 次に再生するときに作り直す
        self._apply_volume()

    def toggle_play(self):
        if self.playing:
            self.offset = self.position()
            self.pg.mixer.pause()
            self.playing = False
            self.stop_take()
        elif self.offset >= self.duration - 0.05:
            self._start_at(0.0)
        elif self.loaded:
            self.pg.mixer.unpause()
            self.started_at = time.perf_counter()
            self.playing = True
        else:
            self.pg.mixer.unpause()
            self._start_at(self.offset)

    def seek(self, t):
        """伴奏とボーカルを同時に t 秒へ。戻り値は実際の移動先。"""
        t = min(max(0.0, t), max(0.0, self.duration - 0.05))
        self.stop_take()
        if self.playing:
            self._start_at(t)
        else:
            self.ch_acc.stop()
            self.ch_voc.stop()
            self.offset = t
            self.loaded = False
        return t

    def toggle_vocal(self):
        self.vocal_on = not self.vocal_on
        self._apply_volume()

    def change_volume(self, d):
        self.volume = min(1.0, max(0.0, round(self.volume + d, 2)))
        self._apply_volume()

    def close(self):
        self.stop_take()
        self.ch_acc.stop()
        self.ch_voc.stop()


# ---------------------------------------------------------------------------
# マイク入力のリアルタイムピッチ検出
# ---------------------------------------------------------------------------
class MicTracker:
    """マイク入力: 声の高さを調べる + (ON のとき) 自分の声をスピーカー/ヘッドホンへそのまま出す。"""

    MONITOR_BLOCK = 128          # 出力の遅れを小さくするため、細かい単位 (約3ms) でやりとりする

    # Windows の音の出し入れの方式 (遅れが小さい順に試す)。名前は PortAudio の表記
    API_CHOICES = [
        ("auto", "自動（おすすめ）"),
        ("Windows WASAPI", "WASAPI（遅れが小さい）"),
        ("Windows WDM-KS", "WDM-KS（さらに小さいが機器によっては不安定）"),
        ("Windows DirectSound", "DirectSound"),
        ("MME", "MME（Windows の標準・遅れが大きい）"),
    ]
    AUTO_ORDER = ["Windows WASAPI", "Windows DirectSound", "MME"]
    WIRELESS_WORDS = ("bluetooth", "hands-free", "handsfree", "airpods", "buds", "wf-", "wh-", "ワイヤレス", "headset (")

    def __init__(self, clock, latency=0.10, threshold_db=-45.0, device=None, out_device=None,
                 monitor=False, monitor_gain=0.8, api="auto", monitor_auto=True, noise="mid", noise_params=None,
                 in_pref=None, out_pref=None):
        self.clock = clock               # () -> (曲の位置, 再生中か)
        self.latency = latency
        self.threshold_db = threshold_db
        self.device = device
        self.out_device = out_device
        self.api = api or "auto"
        self.sr, self.hop, self.frame = 44100, 1024, 2048
        self.det = YinPitchDetector(self.sr, fmin=65.0, fmax=1100.0)
        self.lock = threading.Lock()
        self.current = None              # 今の声の高さ (MIDI 小数) / None = 無音
        self.level_db = -120.0
        self.points = deque(maxlen=3000)  # (曲の時刻, MIDI)
        self.error = None
        self.monitor = bool(monitor)     # 自分の声を出力するか
        self.monitor_gain = min(self.GAIN_MAX, max(0.0, float(monitor_gain)))
        # 自動で聞きやすい大きさに: マイクの声の大きさを測って、小さければ持ち上げる (最大 +AGC_MAX_DB)
        self.monitor_auto = bool(monitor_auto)
        self._agc_level = -32.0          # 歌っているときの声の大きさの目安 (dBFS)
        self._agc_db = 0.0               # 今かけている持ち上げ (dB)
        # 雑音を抑える: 周りの雑音の大きさを測り、それより十分大きい音 (歌声) のときだけ通す
        self.noise = "mid"
        self.gate_margin, self.gate_cut, self.gate_hold_sec, self.hp_freq = self.NOISE_LEVELS["mid"]
        self.set_noise(noise if noise in self.NOISE_LEVELS else "mid")
        if noise == "custom" and noise_params:
            for k, v in noise_params.items():
                self.set_noise_param(k, v)
        self.in_db = -120.0              # 今のマイクの大きさ (低い雑音を切った後)
        self._calib = 0                  # 「今の雑音を測る」の残りサンプル数
        self.noise_floor_db = -70.0      # 周りの雑音の大きさ (自動で測る)
        self._nf_mins = deque(maxlen=8)  # 0.5秒ごとの一番小さい音 (4秒分)
        self._nf_cur = 0.0
        self._nf_n = 0
        self._nf_total = 0               # 測り始めてからのサンプル数
        self._open_n = 0                 # 「声」と判定され続けているサンプル数
        self._gate_g = 1.0               # 今の通し具合 (1 = そのまま)
        self._gate_hold = 0
        self.gate_open = False
        self._hp = None                  # 低い雑音 (エアコン・振動・ハム) を切るフィルター
        self._hp_sr = None
        self.monitor_error = None        # 出力できない理由 (None = 出力できる)
        self.monitor_latency_ms = None
        self.api_used = None             # 実際に使っている方式
        self.out_name = ""
        self.in_name = ""
        self.in_pref = in_pref           # 選んだマイクの名前 (None = Windows の既定)
        self.out_pref = out_pref         # 選んだスピーカー / ヘッドホンの名前 (None = Windows の既定)
        self.pref_missing = []           # 選んだのに見つからなかった機器
        self.wireless = False            # ワイヤレス (Bluetooth) の出力先らしい
        self._q = queue.Queue(maxsize=512)
        self._rec = None                 # 録音中: {"chunks", "pos0", "n"}
        self.speed = 1.0                 # 曲の再生速度 (表示位置の遅れ補正に使う)
        self._stream = None
        self._running = False
        self._gen = 0

    # ----- 方式・機器の選択 -----
    @staticmethod
    def available_apis():
        try:
            import sounddevice as sd

            return [h["name"] for h in sd.query_hostapis()]
        except Exception:
            return []

    def _candidates(self, sd):
        """[(方式の名前, 入力デバイス, 出力デバイス, extra_settings)] を試す順に。"""
        hostapis = list(sd.query_hostapis())
        names = [h["name"] for h in hostapis]
        cands = []
        if self.device is not None or self.out_device is not None:      # 番号を指定されたときはそのまま
            cands.append(("指定のデバイス", self.device, self.out_device, None))
            return cands
        order = self.AUTO_ORDER if self.api == "auto" else [self.api]
        for api in order:
            if api not in names:
                continue
            hi = names.index(api)
            h = hostapis[hi]
            i, o = h.get("default_input_device", -1), h.get("default_output_device", -1)
            i = self._find(sd, hi, self.in_pref, "max_input_channels", i)
            o = self._find(sd, hi, self.out_pref, "max_output_channels", o)
            if i is None or o is None or i < 0 or o < 0:
                continue
            extra = None
            if api == "Windows WASAPI" and hasattr(sd, "WasapiSettings"):
                try:
                    extra = (sd.WasapiSettings(exclusive=False, auto_convert=True),) * 2
                except TypeError:          # 古い sounddevice (auto_convert なし)
                    extra = (sd.WasapiSettings(exclusive=False),) * 2
            cands.append((api, i, o, extra))
        if self.api == "auto" or not cands:
            cands.append(("既定", None, None, None))                     # 最後の手段: Windows の既定
        return cands

    def _find(self, sd, hostapi, want, chan_key, default):
        """方式 hostapi の中から、名前が want に合う機器の番号 (見つからなければ default)。"""
        if not want:
            return default
        try:
            devs = [(k, d) for k, d in enumerate(sd.query_devices()) if d.get("hostapi") == hostapi and d.get(chan_key, 0) > 0]
        except Exception:
            return default
        m = match_device(want, [d["name"] for _, d in devs])
        if m is None:
            if want not in self.pref_missing:
                self.pref_missing.append(want)
            return default
        return next(k for k, d in devs if d["name"] == m)

    SKIP_WORDS = ("サウンド マッパー", "sound mapper", "プライマリ", "primary sound")

    def list_devices(self):
        """選べる (マイクの名前の一覧, スピーカーの名前の一覧)。今使っている方式の中から。"""
        try:
            import sounddevice as sd

            hostapis = list(sd.query_hostapis())
            names = [h["name"] for h in hostapis]
            api = self.api_used if self.api_used in names else next((a for a in self.AUTO_ORDER if a in names), None)
            hi = names.index(api) if api else None
            ins, outs = [], []
            for d in sd.query_devices():
                if hi is not None and d.get("hostapi") != hi:
                    continue
                n = d["name"]
                if any(w in n.lower() for w in self.SKIP_WORDS):
                    continue
                if d.get("max_input_channels", 0) > 0 and n not in ins:
                    ins.append(n)
                if d.get("max_output_channels", 0) > 0 and n not in outs:
                    outs.append(n)
            return ins, outs
        except Exception:
            return [], []

    def set_devices(self, in_pref=False, out_pref=False):
        """マイク / スピーカーを選び直して開き直す (False = そのまま、None = Windows の既定)。"""
        if in_pref is not False:
            self.in_pref = in_pref
        if out_pref is not False:
            self.out_pref = out_pref
        self.pref_missing = []
        was = self.monitor
        ok = self.restart()
        if self.monitor_error is None:
            self.set_monitor(was)
        return ok

    def start(self):
        try:
            import sounddevice as sd
        except Exception as e:  # マイクが無くても練習画面は使えるようにする
            self.error = self.monitor_error = str(e)
            return False
        self.monitor_error = None
        errors = []
        try:
            cands = self._candidates(sd)
        except Exception:
            cands = [("既定", self.device, self.out_device, None)]
        for api, din, dout, extra in cands:
            # その機器が普段使っている周波数で開く (WASAPI は違う周波数だと開けないことがある)
            try:
                sr = int(sd.query_devices(din, "input")["default_samplerate"]) if din is not None else 44100
            except Exception:
                sr = 44100
            for rate in dict.fromkeys([sr, 48000, 44100]):
                try:
                    kw = {"extra_settings": extra} if extra else {}
                    st_ = sd.Stream(device=(din, dout), channels=(1, 2), samplerate=rate,
                                    blocksize=self.MONITOR_BLOCK, dtype="float32", latency="low",
                                    callback=self._duplex_callback, **kw)
                    st_.start()
                except Exception as e:
                    errors.append(f"{api} {rate}Hz: {e}")
                    continue
                self._stream = st_
                self._set_rate(rate)
                self.api_used = api
                try:
                    self.out_name = sd.query_devices(dout if dout is not None else sd.default.device[1])["name"]
                except Exception:
                    self.out_name = ""
                self.wireless = any(w in self.out_name.lower() for w in self.WIRELESS_WORDS)
                try:
                    self.in_name = sd.query_devices(din if din is not None else sd.default.device[0])["name"]
                except Exception:
                    self.in_name = ""
                try:
                    lat_in, lat_out = st_.latency
                    self.monitor_latency_ms = int(round((lat_in + lat_out) * 1000 + self.MONITOR_BLOCK / rate * 1000))
                except Exception:
                    self.monitor_latency_ms = None
                self._start_worker()
                return True
        # 出力と組み合わせられない場合は、入力だけで動かす (声の高さの表示はできる)
        self.monitor_error = "自分の声を出力できません (" + (errors[-1] if errors else "原因不明") + ")"
        try:
            self._stream = sd.InputStream(device=self.device, channels=1, samplerate=44100,
                                          blocksize=self.MONITOR_BLOCK, dtype="float32", callback=self._callback)
            self._stream.start()
            self._set_rate(44100)
            self.api_used = None
        except Exception as e2:
            self.error = str(e2)
            return False
        self._start_worker()
        return True

    def _set_rate(self, rate):
        if rate != self.sr or self.det is None:
            self.sr = rate
            self.det = YinPitchDetector(rate, fmin=65.0, fmax=1100.0)

    def _start_worker(self):
        self._gen += 1
        self._running = True
        threading.Thread(target=self._worker, args=(self._gen,), daemon=True).start()

    def restart(self, api=None):
        """方式を変えて開き直す。戻り値: 開けたか。"""
        if api is not None:
            self.api = api
        self.stop()
        with self._q.mutex:
            self._q.queue.clear()
        ok = self.start()
        if self.monitor_error:
            self.monitor = False
        return ok

    def set_monitor(self, on):
        self.monitor = bool(on) and self.monitor_error is None
        return self.monitor

    GAIN_MAX = 3.0          # 自分の声の音量の上限 (300%)
    AGC_TARGET_DB = -14.0   # 自動のとき: 歌声をこの大きさ (dBFS) くらいにそろえる
    AGC_MAX_DB = 24.0       # 自動のとき: 持ち上げる上限 (約16倍)
    AGC_GATE_DB = -52.0     # これより小さい音 (息・部屋の雑音) は声として測らない

    def set_monitor_gain(self, g):
        self.monitor_gain = min(self.GAIN_MAX, max(0.0, float(g)))

    def set_monitor_auto(self, on):
        self.monitor_auto = bool(on)

    # 雑音カットの強さ:
    #   (声とみなす大きさ = 雑音 + 何dB, 雑音だけのときに下げる量 dB, 声が止まってから絞るまでの秒数, 低い音を切る Hz)
    NOISE_LEVELS = {"off": None,
                    "weak": (6.0, -15.0, 0.25, 70.0),
                    "mid": (10.0, -30.0, 0.18, 90.0),
                    "strong": (14.0, -45.0, 0.15, 120.0)}
    NOISE_NAMES = {"off": "OFF", "weak": "弱", "mid": "中", "strong": "強", "custom": "カスタム"}
    # 細かい調整の範囲: 名前 → (最小, 最大)
    NOISE_PARAMS = {"margin": (0.0, 30.0), "cut": (-60.0, 0.0), "hold": (0.05, 0.8), "hp": (0.0, 250.0)}

    def set_noise(self, level):
        """強さを選ぶ (OFF / 弱 / 中 / 強)。細かい値もその強さの値になる。"""
        cfg = self.NOISE_LEVELS.get(level, False)
        if cfg is False:
            return
        self.noise = level
        if cfg is not None:
            self.gate_margin, self.gate_cut, self.gate_hold_sec, self.hp_freq = cfg

    def set_noise_param(self, name, value):
        """細かい値を変える → 強さは「カスタム」になる。"""
        lo, hi = self.NOISE_PARAMS[name]
        v = float(min(hi, max(lo, value)))
        attr = {"margin": "gate_margin", "cut": "gate_cut", "hold": "gate_hold_sec", "hp": "hp_freq"}[name]
        setattr(self, attr, v)
        self.noise = "custom"

    def noise_params(self):
        return {"margin": self.gate_margin, "cut": self.gate_cut, "hold": self.gate_hold_sec, "hp": self.hp_freq}

    @property
    def noise_on(self):
        return self.noise != "off"

    def calibrate_noise(self, sec=2.0):
        """「今の雑音を測る」: 静かにしている間の大きさを、雑音としてすぐに測り直す。"""
        self._nf_mins.clear()
        self._nf_n = 0
        self._calib = int(sec * self.sr)

    def _track_noise(self, db, n):
        """周りの雑音の大きさを測る。
        直近4秒の「一番静かな瞬間」の大きさ = 雑音 (歌っていても息つぎの間に測れる)。"""
        if self._nf_n == 0 or db < self._nf_cur:
            self._nf_cur = db
        self._nf_n += n
        if self._nf_n >= self.sr * 0.5:
            self._nf_mins.append(self._nf_cur)
            self._nf_n = 0
        target = min(self._nf_mins) if self._nf_mins else self._nf_cur
        nf = self.noise_floor_db
        if self._calib > 0:                     # 測り直し中: 今の音そのものを雑音とみなす
            self._calib -= n
            nf += (db - nf) * 0.1
            self.noise_floor_db = max(-95.0, min(-20.0, nf))
            self._nf_total += n
            return
        if target < nf:
            k = 0.05
        else:
            # 上がるのはゆっくり。歌っている間 (長く伸ばす音) はさらにゆっくり。
            # ただし、使い始めの1秒と、20秒以上ずっと「声」と判定されている (= たぶん雑音が大きくなった) ときは普通に。
            warmup = self._nf_total < self.sr and target < -45.0     # 使い始め: 静かめの音なら雑音として素早く測る
            singing = self.gate_open and not warmup and self._open_n < self.sr * 20
            k = 0.0003 if singing else 0.01
        self._nf_total += n
        self._open_n = self._open_n + n if self.gate_open else 0
        nf += (target - nf) * k
        self.noise_floor_db = max(-95.0, min(-20.0, nf))

    def _highpass(self, x):
        """hp_freq (例 90Hz) より低い音 (エアコン・机の振動・電源のハム) を切る。"""
        if self.hp_freq < 20:
            return x
        try:
            key = (self.sr, round(self.hp_freq))
            if self._hp_sr != key:
                from scipy.signal import butter, sosfilt_zi

                self._hp = [butter(2, float(self.hp_freq), "highpass", fs=self.sr, output="sos"), None]
                self._hp[1] = sosfilt_zi(self._hp[0]) * 0.0
                self._hp_sr = key
            from scipy.signal import sosfilt

            y, self._hp[1] = sosfilt(self._hp[0], x, zi=self._hp[1])
            return y.astype(np.float32)
        except Exception:
            return x

    def _update_gate(self, db, n):
        """今の音が歌声か (雑音より十分大きいか) を決める。"""
        if not self.noise_on:
            self.gate_open = True
            return
        if db > self.noise_floor_db + self.gate_margin:
            self.gate_open = True
            self._gate_hold = max(1, int(self.gate_hold_sec * self.sr / max(1, n)))   # 少しの間は開けたまま (語尾が切れない)
        elif self._gate_hold > 0:
            self._gate_hold -= 1
        else:
            self.gate_open = False

    def _gate(self, x, db):
        """歌声のときだけ通す (雑音だけのときは小さくする)。区切りが目立たないよう、なめらかに変える。"""
        if not self.noise_on:
            return x
        cut = self.gate_cut
        target = 1.0 if self.gate_open else 10 ** (cut / 20.0)
        prev = self._gate_g
        k = 0.5 if target > prev else 0.06      # 開くのは速く、閉じるのはゆっくり
        new = prev + (target - prev) * k
        self._gate_g = new
        return x * np.linspace(prev, new, len(x), dtype=np.float32)

    @property
    def monitor_boost_db(self):
        """自動でかけている持ち上げ (dB)。自動 OFF なら 0。"""
        return self._agc_db if self.monitor_auto else 0.0

    def _measure(self, x):
        """低い雑音を切って、大きさ (dB) を測り、周りの雑音の大きさを更新する。"""
        if self.noise_on:
            x = self._highpass(x)
        db = 20.0 * math.log10(float(np.sqrt(np.mean(x * x))) + 1e-9)
        self.in_db = db
        self._update_gate(db, len(x))
        self._track_noise(db, len(x))
        return x, db

    def monitor_out(self, x, db=None):
        """マイクの音 (1次元) → 出力する音。
        低い雑音を切る → 雑音だけの所は小さく → 自動の持ち上げ → 音量 → 割れないように柔らかく制限。"""
        if db is None:
            x, db = self._measure(x)
        x = self._gate(x, db)
        if self.monitor_auto:
            voice = db > self.AGC_GATE_DB and (not self.noise_on or self.gate_open)
            if voice:                            # 声が出ている所だけで大きさを測る (上がる時は速く、下がる時はゆっくり)
                a = 0.03 if db > self._agc_level else 0.004
                self._agc_level += (db - self._agc_level) * a
            want = min(self.AGC_MAX_DB, max(0.0, self.AGC_TARGET_DB - self._agc_level))
            self._agc_db += (want - self._agc_db) * 0.02
            g = (10.0 ** (self._agc_db / 20.0)) * self.monitor_gain
        else:
            g = self.monitor_gain
        return np.tanh(x * g)                    # 大きすぎる所だけ丸める (音割れ防止)

    # ----- 録音 (テイク) -----
    REC_MAX_SEC = 600

    def start_rec(self):
        self._rec = {"chunks": [], "pos0": None, "n": 0}

    @property
    def recording(self):
        return self._rec is not None

    def rec_seconds(self):
        r = self._rec
        return r["n"] / self.sr if r else 0.0

    def stop_rec(self, min_sec=1.0):
        """録音を終える → {"audio": float32, "sr", "pos0"} (短すぎるときは None)。"""
        r, self._rec = self._rec, None
        if not r or r["pos0"] is None or r["n"] < self.sr * min_sec:
            return None
        return {"audio": np.concatenate(r["chunks"]).astype(np.float32), "sr": self.sr, "pos0": r["pos0"]}

    def _record(self, x):
        r = self._rec
        if r is None:
            return
        if r["pos0"] is None:
            try:
                r["pos0"] = self.clock()[0]      # 最初の音が届いたときの曲の位置
            except Exception:
                r["pos0"] = 0.0
        r["chunks"].append(x.copy())
        r["n"] += len(x)
        if r["n"] > self.sr * self.REC_MAX_SEC:
            self._rec = None

    def _duplex_callback(self, indata, outdata, frames, t, status):
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass
        self._record(indata[:, 0])
        x, db = self._measure(indata[:, 0])     # 出力しないときも雑音の大きさは測っておく (声の高さの判定に使う)
        if self.monitor:
            outdata[:, 0] = self.monitor_out(x, db)
            outdata[:, 1:] = outdata[:, :1]     # 左右両方から
        else:
            outdata.fill(0)

    def _callback(self, indata, frames, t, status):
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass
        self._record(indata[:, 0])
        self._measure(indata[:, 0])             # 雑音の大きさを測る (声の高さの判定に使う)

    def _worker(self, gen=0):
        ring = np.zeros(self.frame, np.float32)
        hist = deque(maxlen=3)
        pending = []
        while self._running and gen == self._gen:
            try:
                block = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            pending.append(block)
            if sum(len(b) for b in pending) < self.hop:   # 約23ms ぶん貯まったら調べる
                continue
            block = np.concatenate(pending)
            pending = []
            ring = np.concatenate((ring, block))[-self.frame:]
            db = rms_dbfs(block)
            self._check_howl(block, db)
            cur = None
            thr = self.threshold_db
            if self.noise_on:                   # 周りの雑音より十分大きいときだけ、声の高さを調べる
                thr = max(thr, self.noise_floor_db + self.gate_margin)
            if db >= thr:
                f, conf = self.det(ring)
                if f > 0 and conf >= 0.75:
                    hist.append(hz_to_midi(f))
                if hist:
                    cur = float(np.median(hist))
            else:
                hist.clear()
            pos, playing = self.clock()
            with self.lock:
                self.current = cur
                self.level_db = db
                if playing and cur is not None:
                    self.points.append((pos - self.latency * self.speed, cur))

    HOWL_DB = -20.0          # これより大きく
    HOWL_RATIO = 0.85        # 1つの高さの音だけが目立ち (ピーッ・キーン。歌声は倍音があるので目立ちにくい)
    HOWL_SEC = 0.5           # 同じ高さのままこれだけ続いたらハウリングとみなす

    def _check_howl(self, block, db):
        """自分の声を出しているときにハウリング (ピーッ・キーン) が起きたら、すぐに出力を止める。
        ハウリング = 大きくて、1つの高さだけの音が、まったく揺れずに続く (歌声は倍音や揺れがある)。"""
        if not self.monitor:
            self._howl_f = []
            return
        f = None
        if db > self.HOWL_DB and len(block) >= 256:
            n = 1 << int(math.log2(len(block)))
            spec = np.abs(np.fft.rfft(block[:n] * np.hanning(n))) ** 2
            total = float(spec.sum()) + 1e-12
            k = int(np.argmax(spec))
            peak = float(spec[max(0, k - 2):k + 3].sum())
            if peak / total > self.HOWL_RATIO and 1 <= k < len(spec) - 1:
                a, b, c = np.log(spec[k - 1:k + 2] + 1e-20)            # 山の頂上を細かく求める
                d = a - 2 * b + c
                off = 0.5 * (a - c) / d if d != 0 else 0.0
                f = (k + off) * self.sr / n
                if f < 150:
                    f = None
        fs = getattr(self, "_howl_f", [])
        if f is None:
            fs = []
        else:
            fs.append(f)
            need = int(self.HOWL_SEC * self.sr / max(1, len(block)))
            fs = fs[-need:]
            if len(fs) >= need:
                spread = (max(fs) - min(fs)) / (sum(fs) / len(fs))
                if spread < 0.001:                                  # 0.1% (2セント) も揺れない = 機械的な音
                    self.monitor = False
                    self.howl_stopped = True
                    fs = []
        self._howl_f = fs

    def snapshot(self):
        with self.lock:
            return self.current, self.level_db

    def trail(self, t0, t1):
        with self.lock:
            return [(t, m) for t, m in self.points if t0 <= t <= t1]

    def clear(self):
        with self.lock:
            self.points.clear()

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------
class Fonts:
    CANDIDATES = "meiryo,yugothic,yugothicui,msgothic,notosanscjkjp,notosanscjk,hiraginosans,hiraginokakugothicpro,takaogothic,ipagothic"

    def __init__(self, pygame, path=None):
        self.pg = pygame
        local_regular, local_bold = self._local_fonts()
        self.path = path or local_regular or pygame.font.match_font(self.CANDIDATES)
        self.bold_path = None if path else local_bold   # 太字専用ファイルがあれば使う (無ければ疑似太字)
        if self.path is None:
            print("[注意] 日本語フォントが見つかりません。setup_check.py を実行するか、--font でフォントを指定してください", file=sys.stderr)
        self.cache = {}

    @staticmethod
    def _local_fonts():
        """プロジェクトの fonts フォルダのフォント (通常, 太字) を探す。"""
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
        if not os.path.isdir(d):
            return None, None
        files = [f for f in sorted(os.listdir(d)) if f.lower().endswith((".ttf", ".otf", ".ttc"))]
        bold = next((f for f in files if "bold" in f.lower()), None)
        regular = next((f for f in files if "regular" in f.lower()), None) or next((f for f in files if f != bold), None)
        join = lambda f: os.path.join(d, f) if f else None
        return join(regular or bold), join(bold)

    def get(self, size, bold=False):
        key = (size, bold)
        if key not in self.cache:
            if bold and self.bold_path:
                f = self.pg.font.Font(self.bold_path, size)
            else:
                f = self.pg.font.Font(self.path, size)
                f.set_bold(bold)
            self.cache[key] = f
        return self.cache[key]

    def render(self, text, size, color, bold=False):
        return self.get(size, bold).render(text, True, color)


class Layout:
    """座標計算 (描画ライブラリに依存しない部分)。"""

    def __init__(self, lo, hi, width, height):
        self.lo, self.hi = lo, hi
        self.top = HEADER_H + PROGRESS_H
        self.bottom = height - FOOTER_H - LYRIC_H
        self.left = LABEL_W
        self.right = width
        self.row_h = (self.bottom - self.top) / (hi - lo + 1)
        self.playhead_x = self.left + (self.right - self.left) * PLAYHEAD_RATIO

    def y_of(self, midi):
        """MIDI (小数可) → 行の中心の y 座標。"""
        return self.bottom - (midi - self.lo + 0.5) * self.row_h

    def x_of(self, t, now):
        return self.playhead_x + (t - now) * PX_PER_SEC

    def time_window(self, now):
        t0 = now - (self.playhead_x - self.left) / PX_PER_SEC
        t1 = now + (self.right - self.playhead_x) / PX_PER_SEC
        return t0, t1


class _Key:
    def __init__(self, key):
        self.key = key
        self.mod = 0


class Trainer(LyricsMixin):
    def __init__(self, pg, screen, fonts, title, guide, key_info, playback, mic, skip_sec=5, alt_guide=None,
                 on_offset_change=None):
        self.pg, self.screen, self.fonts = pg, screen, fonts
        self.title, self.guide, self.pb, self.mic = title, guide, playback, mic
        self.key0 = key_info                    # 解析したキー (移調する前)
        self.transpose = 0                      # キー変更 (半音)。曲の音とお手本バーに反映済みの値
        self.key = key_info
        self.alt_guide = alt_guide              # MIDI 使用時の「歌声から解析」したお手本 (M キーで切替)
        self.on_offset_change = on_offset_change
        self.song_path = None                   # None = 曲を選ぶ前の空の楽譜
        self.song_dir = None                    # 解析結果のフォルダ (練習の設定・テイク・編集版の保存先)
        # A-B ループ
        self.loop_a = None
        self.loop_b = None
        self.loop_on = False
        self.loop_count = 0
        # 録音・テイク
        self.rec_armed = False                  # 録音モード (再生している間ずっと録音)
        self.takes = []                         # 新しい順ではなく録った順
        self._rec_active = False
        self._tick_prev = None                  # (曲の位置, 時刻) … 移動 (ジャンプ) を見つけるため
        self.reviewing = None                   # 再生中のテイク
        self._review_vocal = None
        self.take_offset = 0.0                  # テイク再生のずれ補正 (秒、+ で遅らせる)
        self.on_take_offset = None
        self._take_no = 0
        self._take_btn_pos = (W // 2, H - 90)
        self._free_take_snd = None
        # お手本バーの編集
        self.edit_mode = False
        self.sel = set()                        # 選んでいるバー (guide.notes の番号)
        self._undo, self._redo = [], []
        self.edit_dirty = False
        self.edit_name = None                   # 保存した編集版の名前
        self._note_rects = []                   # 画面のバーの位置 [(Rect, 番号)]
        self.on_monitor_change = None           # 自分の声の出力設定が変わったとき (設定の保存用)
        self.show_mixer = False                 # 音量の調整パネル
        self.hits = {}                          # (お手本, バーの開始時刻) -> [[合い始め, 合い終わり], ...]
        self.view = dict(VIEW_DEFAULT)          # 縦・横のグリッドの設定
        self.on_view_change = None              # 設定が変わったとき (保存用)
        self.on_device_change = None            # マイク / スピーカーを選び直したとき (保存用)
        self.on_vocal_gain = None               # ボーカルの音量を変えたとき (保存用)
        self.update_auto = lambda: True         # 起動時にアップデートを確認するか (Shell が差し替える)
        self._dlg_buttons = []
        self._notified = None
        self._vocal_dirty = False
        self.show_grid_panel = False
        self.show_noise_panel = False           # 雑音カットの調整パネル
        self.beats = None                       # 小節の線 (beat_analyzer.BeatGrid)。曲を開くと入る
        self.popup = None                       # 右クリックのメニュー
        self._last_click = (0.0, (-99, -99))
        self._view_dirty = False
        self.label_size = 14
        self._label_cache = {}
        self._vdot_key = None
        self._last_frame = time.perf_counter()
        lo0, hi0 = view_range(guide, 0)
        self.v_center = (lo0 + hi0) / 2         # 画面の上下の中心 (MIDI)
        self.hold = None                        # 押しっぱなしのボタン
        self._monitor_dirty = False             # 自分の声の音量が変わって、まだ保存していない
        self._sliders = []                      # [(つまみを動かす範囲 Rect, "monitor" / "song")]
        self.skip_sec = skip_sec
        self.octave_shift = 0
        self.show_fine = True
        self.toast = ("", 0.0)
        self.score_hit = 0
        self.score_total = 0
        self._set_key_display()
        # マウス操作の状態
        self.mouse = (-1, -1)
        self.drag = None            # {"kind": "lane"/"progress", ...}
        self.buttons = []
        self.show_help = False
        self.requests = []          # main に頼む操作: "open_song" / "open_midi" / "fullscreen" / "quit"
        self.is_fullscreen = lambda: False
        self._lyrics_init()
        self.menu = MenuBar(pg, fonts, self._build_menus(), height=MENU_H)
        self._rebuild_background()

    @property
    def gshift(self):
        """お手本バーを動かす半音の数 (オクターブ + キー変更)。"""
        return 12 * self.octave_shift + self.transpose

    def _set_key_display(self):
        """キー変更に合わせて、画面のキーの表示と音階の行を変える。"""
        k0 = self.key0
        if k0 and self.transpose and "tonic_pc" in k0:
            try:
                from key_analyzer import describe_key

                k = dict(k0)
                k.update(describe_key((k0["tonic_pc"] + self.transpose) % 12, k0["mode"]))
                self.key = k
            except Exception:
                self.key = k0
        else:
            self.key = k0
        self.scale_pcs = set(self.key.get("scale_pcs", [])) if self.key else set()
        self.tonic_pc = self.key.get("tonic_pc") if self.key else None

    def set_screen(self, screen):
        """ウィンドウの大きさが変わったとき: 描画先を差し替えて背景を作り直す。"""
        self.screen = screen
        self._rebuild_background()

    # ---------- 背景 (ドレミの行とラベル) ----------
    def _update_layout(self):
        global FOOTER_H, LYRIC_H
        FOOTER_H = 92 if self.view.get("toolbar2", True) else 48     # 2段目 (練習ツール) の分
        LYRIC_H = lyric_lane_height(self.view, getattr(self, "lyrics", None) is not None)
        semis = self.view["semis"]
        lo = self.v_center - semis / 2
        self.layout = Layout(lo, lo + semis, W, H)
        return self.layout

    def _rebuild_background(self):
        pg = self.pg
        L = self._update_layout()
        bg = pg.Surface((W, H))
        bg.fill(BG)
        # 下の方にうっすら紫のグラデーション
        g0 = int(H * 0.45)
        for y in range(g0, H, 2):
            t = ((y - g0) / max(1, H - g0)) ** 1.6
            c = tuple(int(BG[i] + (BG_GLOW[i] - BG[i]) * t) for i in range(3))
            pg.draw.rect(bg, c, (0, y, W, 2))
        pg.draw.line(bg, HAIR, (L.left, L.top), (L.left, L.bottom), 1)
        self.background = bg
        self._vdot_key = None

    def _lc(self, c):
        """線の色を「線の濃さ」に合わせる (0 = 見えない, 1 = 標準, 2 = 濃い)。"""
        k = self.view["line"]
        return tuple(max(0, min(255, int(BG[i] + (c[i] - BG[i]) * k))) for i in range(3))

    def _label(self, key, text, size, color, bold=False):
        k = (key, size)
        if k not in self._label_cache:
            if len(self._label_cache) > 400:
                self._label_cache.clear()
            self._label_cache[k] = self.fonts.render(text, size, color, bold=bold)
        return self._label_cache[k]

    def _draw_rows(self, scr, L):
        """音の段 (行の色・ヨコ線・左の音名)。上下に動くので毎回描く。"""
        pg = self.pg
        label_size = int(max(10, min(20, L.row_h * 0.78)))
        self.label_size = label_size
        scr.set_clip(pg.Rect(0, L.top, W, L.bottom - L.top))
        tint = None
        for m in range(math.ceil(L.lo - 1), math.floor(L.hi + 1) + 1):
            pc = m % 12
            yc = round(L.y_of(m))
            if yc < L.top - L.row_h or yc > L.bottom + L.row_h:
                continue
            if self.scale_pcs and pc in self.scale_pcs:   # キーの音階の行だけ、ほんの少し明るく
                if tint is None:
                    tint = pg.Surface((L.right - L.left, math.ceil(L.row_h)), pg.SRCALPHA)
                    tint.fill(ROW_TINT)
                scr.blit(tint, (L.left, round(L.y_of(m) - L.row_h / 2)))
            if pc in NATURAL_PCS:                         # ドレミのヨコ線 (ドだけ少し明るく)
                if self.view["line"] > 0:
                    pg.draw.line(scr, self._lc(HAIR_STRONG if pc == 0 else HAIR), (L.left, yc), (L.right, yc), 1)
            elif self.view["sharp_lines"] and self.view["line"] > 0:   # ♯ の段は点線
                col = self._lc(HAIR_DIM)
                for x in range(L.left, L.right, 6):
                    pg.draw.line(scr, col, (x, yc), (x + 1, yc), 1)
            # 左の音名: 「ド」(太字) + 「C4」(小さく細く)。段が細いときは ド・ミ・ソ (さらに細いと ド) だけ
            show = pc in NATURAL_PCS and (L.row_h >= 12 or pc in ((0, 4, 7) if L.row_h >= 7 else (0,)))
            if show:
                jp = self._label(("jp", m), midi_to_doremi(m, octave=False), label_size, WHITE if pc == 0 else LABEL, True)
                en = self._label(("en", m), midi_to_english(m), max(9, label_size - 7), FAINT)
                scr.blit(jp, (L.left - jp.get_width() - 12, yc - jp.get_height() // 2))
                scr.blit(en, (L.left - jp.get_width() - en.get_width() - 18, yc - en.get_height() // 2 + 1))
                if pc == self.tonic_pc:                   # キーの主音に印
                    pg.draw.circle(scr, GOLD, (7, yc), 3)
            elif L.row_h >= 13:
                sh = self._label(("sh", m), midi_to_doremi(m, octave=False), max(9, label_size - 6), FAINT)
                scr.blit(sh, (L.left - sh.get_width() - 12, yc - sh.get_height() // 2))
        scr.set_clip(None)

    def _follow(self, now):
        """これから歌う音が画面の上下の真ん中に来るように、なめらかに動かす。"""
        t = time.perf_counter()
        dt = min(0.25, max(0.0, t - self._last_frame))
        self._last_frame = t
        if not self.view["follow"]:
            return
        shift = self.gshift
        ms = [n["midi"] + shift for n in self.guide.visible(now - 1.0, now + 6.0) if n["end"] >= now - 1.0]
        if ms:
            target = (min(ms) + max(ms)) / 2
        else:
            cur, _ = self.mic.snapshot()
            if cur is None:
                return
            target = cur                                  # お手本が無いところでは自分の声に合わせる
        diff = target - self.v_center
        if abs(diff) > 0.05:
            self.v_center += diff * (1 - math.exp(-dt * 3.0))

    # ---------- グリッドの調整 ----------
    def _view_changed(self, toast=None):
        if toast:
            self.show_toast(toast, 2.0)
        if self.drag and self.drag.get("kind") in ("slider", "zoom", "vzoom"):
            self._view_dirty = True             # スライダーを動かしている間は、離したときに保存
        elif self.on_view_change:
            self.on_view_change()

    def toggle_grid_panel(self):
        self.show_grid_panel = not self.show_grid_panel
        if self.show_grid_panel:
            self._fit_panels("grid")

    def toggle_noise_panel(self):
        self.show_noise_panel = not self.show_noise_panel
        if self.show_noise_panel:
            self._fit_panels("noise")

    def set_semis(self, n, toast=True):
        self.view["semis"] = int(min(SEMIS_MAX, max(SEMIS_MIN, round(n))))
        self._update_layout()
        oc = self.view["semis"] / 12
        self._view_changed(f"縦：画面に約 {oc:.1f} オクターブ（{self.view['semis']}段・1段 {self.layout.row_h:.0f}px）"
                           if toast else None)

    def set_octaves_on_screen(self, n):
        """画面に n オクターブがちょうど入るように縦を拡大・縮小する。"""
        self.set_semis(12 * n)

    def vzoom(self, factor):
        """縦の拡大 (factor > 1) / 縮小。1回で最低1段は変える。"""
        cur = self.view["semis"]
        new = round(cur / factor)
        if new == cur:
            new = cur - (1 if factor > 1 else -1)
        self.set_semis(new)

    def set_pxsec(self, px, toast=True):
        global PX_PER_SEC
        PX_PER_SEC = int(min(PXSEC_MAX, max(PXSEC_MIN, round(px))))
        self.view["pxsec"] = PX_PER_SEC
        if toast and self.has_beats():
            toast = f"横：画面に約 {self.bars_on_screen():.1f} 小節（1秒 = {PX_PER_SEC}px）"
        elif toast:
            toast = f"横：1秒 = {PX_PER_SEC}px（画面に約 {(self.layout.right - self.layout.left) / PX_PER_SEC:.0f}秒分）"
        self._view_changed(toast or None)

    def bars_on_screen(self):
        L = self.layout
        bar_sec = self.beats.bpb * 60.0 / max(1.0, self.beats.bpm)
        return (L.right - L.left) / PX_PER_SEC / bar_sec

    def set_bars_on_screen(self, n):
        """画面に n 小節がちょうど入るように横を拡大・縮小する。"""
        if not self.has_beats():
            return
        L = self.layout
        bar_sec = self.beats.bpb * 60.0 / max(1.0, self.beats.bpm)
        self.set_pxsec((L.right - L.left) / (n * bar_sec))

    def set_vgrid(self, sec):
        self.view["vgrid"] = sec
        self.view["vmode"] = "sec"
        self._view_changed(f"タテ線：{sec:g}秒ごと")

    # ---------- 小節の線 (拍の解析) ----------
    def set_beats(self, grid):
        self.beats = grid

    def has_beats(self):
        return self.beats is not None and self.beats.ok

    def bar_mode(self):
        return self.view["vmode"] == "bar" and self.has_beats()

    def set_vmode(self, mode):
        if mode == "bar" and not self.has_beats():
            self.show_toast("この曲では拍を見つけられませんでした（秒ごとの線になります）", 2.5)
            return
        self.view["vmode"] = mode
        self._view_changed("タテ線：小節と拍" if mode == "bar" else f"タテ線：{self.view['vgrid']:g}秒ごと")

    def toggle_beat_lines(self):
        self.view["beat_lines"] = not self.view["beat_lines"]
        self._view_changed("拍の点線：" + ("表示" if self.view["beat_lines"] else "なし（小節線だけ）"))

    def _bar_desc(self):
        b = self.beats
        return f"{b.bpb}拍子　♩ = {b.bpm:.1f}" + ("（手動）" if b.manual_bpm else "")

    def _grid_changed(self, toast):
        """小節の線を直したとき: 小節の線を表示にして、保存する。"""
        if self.view["vmode"] != "bar":
            self.view["vmode"] = "bar"
            self._view_changed()
        if toast:
            self.show_toast(toast, 2.2)
        if self.drag and self.drag.get("kind") == "ruler":
            return                              # ドラッグ中は離したときに保存
        self.beats.save()

    def set_downbeat_here(self, t):
        if not self.has_beats():
            return
        tt = self.beats.set_downbeat_at(t)
        bar, _ = self.beats.bar_at(tt + 1e-3)
        self._grid_changed(f"{fmt_time(tt)} を小節の頭（{bar}小節目の1拍目）にしました")

    def move_downbeat(self, n):
        if self.has_beats():
            self.beats.move_downbeat(n)
            self._grid_changed(f"小節の頭を {abs(n)}拍 {'後ろ' if n > 0 else '前'}へずらしました")

    def set_bpb(self, n):
        if self.has_beats():
            self.beats.set_bpb(n, keep_t=self.pb.position())
            self._grid_changed(f"拍子：{n}拍子（1小節 = {n}拍）")

    def step_bpm(self, d):
        if self.has_beats():
            self.beats.set_bpm(self.beats.bpm + d, keep_t=self.pb.position())
            self._grid_changed(f"テンポ：♩ = {self.beats.bpm:.1f}（手動。今の小節を基準に伸び縮み）")

    def reset_bpm(self):
        if self.has_beats() and self.beats.manual_bpm:
            self.beats.set_bpm(None)
            self._grid_changed(f"テンポを解析結果に戻しました（♩ = {self.beats.bpm:.1f}）")

    def nudge_bars(self, sec):
        if self.has_beats():
            self.beats.set_shift(self.beats.shift + sec)
            self._grid_changed(f"小節線の位置：{self.beats.shift:+.2f}秒")

    def reset_bars(self):
        if self.beats is not None:
            self.beats.reset()
            self._grid_changed(f"小節線を解析結果に戻しました（{self._bar_desc()}）")

    def set_line(self, k, toast=True):
        self.view["line"] = round(min(2.0, max(0.0, k)), 2)
        self._vdot_key = None
        self._view_changed(f"線の濃さ：{int(round(self.view['line'] * 100))}%" if toast else None)

    def toggle_sharp_lines(self):
        self.view["sharp_lines"] = not self.view["sharp_lines"]
        self._view_changed("♯の段の点線：" + ("表示" if self.view["sharp_lines"] else "なし"))

    def toggle_follow(self):
        self.view["follow"] = not self.view["follow"]
        self._view_changed("音階を中央に（自動で上下）：" + ("ON" if self.view["follow"] else "OFF（ホイールで上下に動かせます）"))

    def scroll_pitch(self, semis):
        """手動で上下に動かす (自動の追従は OFF にする)。"""
        if self.view["follow"]:
            self.view["follow"] = False
            self._view_changed("音階の自動の上下：OFF（表示メニューで戻せます）")
        self.v_center = min(108, max(24, self.v_center + semis))

    def reset_view(self):
        self.view = dict(VIEW_DEFAULT)
        global PX_PER_SEC
        PX_PER_SEC = VIEW_DEFAULT["pxsec"]
        self._vdot_key = None
        self._update_layout()
        self._view_changed("グリッドを元に戻しました")

    def apply_view(self, v):
        """保存してある設定を反映 (起動時)。"""
        global PX_PER_SEC
        for k in VIEW_DEFAULT:
            if k in (v or {}):
                self.view[k] = type(VIEW_DEFAULT[k])(v[k])
        PX_PER_SEC = int(self.view["pxsec"])
        self._update_layout()

    # ---------- 練習の設定 (曲ごとに保存: 速度・キー・A-B) ----------
    def _practice_path(self):
        return os.path.join(self.song_dir, "practice.json") if self.song_dir else None

    def save_practice(self):
        p = self._practice_path()
        if not p:
            return
        sp, se = getattr(self.pb, "fx_target", (1.0, 0))
        self.save_practice_extra(speed=sp, semis=se, loop_a=self.loop_a, loop_b=self.loop_b, loop_on=self.loop_on)

    def load_practice(self):
        p = self._practice_path()
        if not p or not os.path.exists(p):
            return
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return
        self.loop_a, self.loop_b = d.get("loop_a"), d.get("loop_b")
        self.loop_on = bool(d.get("loop_on")) and self.loop_a is not None and self.loop_b is not None
        sp, se = float(d.get("speed", 1.0)), int(d.get("semis", 0))
        if (sp, se) != (1.0, 0):
            self._request_fx(sp, se, quiet=True)
        if d.get("last_edit"):
            self.load_edit(d["last_edit"], quiet=True)

    # ---------- 再生速度・キー ----------
    SPEEDS = (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)

    def _request_fx(self, speed, semis, quiet=False):
        speed = round(min(1.0, max(0.7, speed)), 2)
        semis = int(min(6, max(-6, semis)))
        if not hasattr(self.pb, "request_fx"):
            return
        if self.pb.request_fx(speed, semis):
            self._fx_applied(quiet)
        elif not quiet:
            self.show_toast(f"準備しています…（速度 {speed:.2f}倍・キー {semis:+d}）数秒お待ちください", 3.0)
        self.save_practice()

    def _fx_applied(self, quiet=False):
        self.transpose = self.pb.semis
        self._set_key_display()
        self.mic.clear()
        if not quiet:
            k = f"　キー：{self.key['short'] if self.key else ''}" if self.transpose else ""
            self.show_toast(f"速度 {self.pb.speed:.2f}倍・キー {self.pb.semis:+d}{k}" if (self.pb.speed, self.pb.semis) != (1.0, 0)
                            else "速度・キーを元に戻しました", 2.5)

    def set_speed(self, v):
        self._request_fx(v, self.pb.fx_target[1])

    def step_speed(self, d):
        self.set_speed(self.pb.fx_target[0] + d)

    def set_key_shift(self, n):
        self._request_fx(self.pb.fx_target[0], n)

    def step_key(self, d):
        self.set_key_shift(self.pb.fx_target[1] + d)

    # ---------- A-B ループ ----------
    def _snap(self, t):
        """小節線があれば一番近い拍にそろえる。"""
        if self.bar_mode():
            i = self.beats.nearest(t)
            if i is not None:
                return float(self.beats.t[i])
        return t

    def set_loop_a(self, t=None):
        t = self.pb.position() if t is None else t
        self.loop_a = max(0.0, t)
        if self.loop_b is not None and self.loop_b <= self.loop_a + 0.2:
            self.loop_b = None
            self.loop_on = False
        self.show_toast(f"A 点：{fmt_time(self.loop_a)}" + ("　（B キーで終わりを決めます）" if self.loop_b is None else ""), 2.0)
        self.save_practice()

    def set_loop_b(self, t=None):
        t = self.pb.position() if t is None else t
        if self.loop_a is None:
            self.loop_a = 0.0
        if t <= self.loop_a + 0.2:
            self.show_toast("B 点は A 点より後ろにしてください", 2.0)
            return
        self.loop_b = t
        self.loop_on = True
        self.loop_count = 0
        self.show_toast(f"A-B ループ ON：{fmt_time(self.loop_a)} 〜 {fmt_time(self.loop_b)}", 2.5)
        self.save_practice()

    def set_loop(self, a, b, toast=True):
        a, b = sorted((max(0.0, a), b))
        if b - a < 0.2:
            return
        self.loop_a, self.loop_b, self.loop_on, self.loop_count = a, b, True, 0
        if toast:
            self.show_toast(f"A-B ループ ON：{fmt_time(a)} 〜 {fmt_time(b)}", 2.5)
        self.save_practice()

    def toggle_loop(self):
        if self.loop_a is None or self.loop_b is None:
            self.show_toast("先に A 点・B 点を決めてください（A / B キー、または Shift+ドラッグで範囲を選ぶ）", 3.0)
            return
        self.loop_on = not self.loop_on
        self.loop_count = 0
        if self.loop_on and not (self.loop_a - 0.05 <= self.pb.position() < self.loop_b):
            self._seek_to(self.loop_a, toast=False)
        self.show_toast("A-B ループ：" + ("ON" if self.loop_on else "OFF"))
        self.save_practice()

    def clear_loop(self):
        self.loop_a = self.loop_b = None
        self.loop_on = False
        self.show_toast("A-B ループを解除しました")
        self.save_practice()

    def loop_bars_at(self, t, n=1):
        """t を含む小節から n 小節をループにする。"""
        if not self.has_beats():
            self.set_loop(max(0.0, t - 2), t + 2)
            return
        b = self.beats
        i = int(np.searchsorted(b.t, t, side="right")) - 1
        while i > 0 and not b.is_bar(i):
            i -= 1
        a = float(b.t[max(0, i)])
        j = i + b.bpb * n
        e = float(b.t[j]) if j < len(b.t) else a + n * b.bpb * 60.0 / b.bpm
        self.set_loop(a, e, toast=False)
        self.show_toast(f"{b.bar_number(max(0, i))}小節目から {n}小節 をループ（{fmt_time(a)}〜{fmt_time(e)}）", 2.5)

    def _loop_tick(self, now):
        if not (self.loop_on and self.pb.playing and self.loop_a is not None and self.loop_b is not None):
            return
        if now >= self.loop_b:
            self.pb.seek(self.loop_a)
            self.loop_count += 1

    # ---------- 録音・テイク ----------
    def toggle_rec(self):
        self.rec_armed = not self.rec_armed
        if self.rec_armed:
            if getattr(self.mic, "error", None):
                self.rec_armed = False
                self.show_toast("マイクが使えないので録音できません（デバイスを確認してください）", 3.0)
                return
            self.show_toast("● 録音モード ON：再生している間の歌声をテイクとして録音します（止めるたびに1テイク）", 3.5)
        else:
            self._finish_take()
            self.show_toast("録音モード OFF")

    def _finish_take(self):
        if not self._rec_active:
            return
        self._rec_active = False
        r = self.mic.stop_rec()
        if not r:
            return
        sp = self.pb.speed
        self._take_no += 1
        t0 = r["pos0"] - self.mic.latency * sp
        dur = len(r["audio"]) / r["sr"]
        take = {"no": self._take_no, "t0": t0, "t1": t0 + dur * sp, "audio": r["audio"], "sr": r["sr"],
                "speed": sp, "semis": self.transpose, "points": None, "score": None,
                "time": time.strftime("%H:%M")}
        self.takes.append(take)
        if len(self.takes) > 12:                    # 古いものから消す (メモリ節約)
            self.takes.pop(0)
        threading.Thread(target=self._analyze_take, args=(take, self.guide, self.gshift), daemon=True).start()
        self.show_toast(f"テイク {take['no']} を録音しました（{fmt_time(take['t0'])}〜{fmt_time(take['t1'])}）　Y キーで聴き直す", 3.0)

    def _analyze_take(self, take, guide, shift):
        """テイクの音程を調べて (オレンジのバーで表示するため)、一致率を出す。"""
        try:
            x, sr = take["audio"], take["sr"]
            det = YinPitchDetector(sr, fmin=65.0, fmax=1100.0)
            hop, frame = 1024, 2048
            pts, hit, tot = [], 0, 0
            for i in range(0, max(0, len(x) - frame), hop):
                blk = x[i:i + frame]
                if rms_dbfs(blk[-hop:]) < max(-50.0, getattr(self.mic, "noise_floor_db", -70) + 8):
                    continue
                f, conf = det(blk)
                if f <= 0 or conf < 0.75:
                    continue
                m = hz_to_midi(f)
                t = take["t0"] + (i + frame) / sr * take["speed"]
                pts.append((t, m))
                n = guide.note_at(t)
                if n is not None:
                    tot += 1
                    hit += abs(m - (n["midi"] + shift)) <= 0.5
            take["points"] = pts
            take["score"] = (hit / tot) if tot >= 10 else None
        except Exception:
            take["points"] = []

    def _take_tick(self, now):
        """毎フレーム: 録音の開始・区切り、テイク再生の終わりを見張る。"""
        wall = time.perf_counter()
        prev, self._tick_prev = self._tick_prev, (now, wall)
        jumped = False
        if prev is not None and self.pb.playing:
            expect = prev[0] + (wall - prev[1]) * self.pb.speed
            jumped = abs(now - expect) > 0.3
        if self.reviewing is not None:
            t = self.reviewing
            if self.pb.take is None or jumped or not self.pb.playing or now > t["t1"] + 0.4:
                self.stop_review(pause=(not jumped and self.pb.playing))
            return
        if not self.rec_armed:
            return
        if self._rec_active and (not self.pb.playing or jumped):
            self._finish_take()
        if self.pb.playing and not self._rec_active:
            self.mic.start_rec()
            self._rec_active = True

    def _take_audio(self, take):
        """テイクを、今の速度・ミキサーの形 (int16 ステレオ) にする。"""
        x = take["audio"]
        if abs(take["speed"] - self.pb.speed) > 1e-3:      # 録ったときと速度が違う → 長さを合わせる
            import audio_fx

            x = audio_fx.stretch_mono(x, self.pb.speed / take["speed"])
        init = None
        try:
            init = self.pg.mixer.get_init()
        except Exception:
            pass
        msr, ch = (init[0], init[2]) if init else (44100, 2)
        if msr != take["sr"]:
            from fractions import Fraction

            from scipy.signal import resample_poly

            fr = Fraction(msr, take["sr"]).limit_denominator(1000)
            x = resample_poly(x, fr.numerator, fr.denominator)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        g = min(20.0, 0.7 / peak) if peak > 1e-4 else 1.0   # 小さく録れていても聴きやすい大きさに
        a = (np.clip(x * g, -1, 1) * 32767).astype(np.int16)
        return np.ascontiguousarray(np.stack([a] * ch, axis=1) if ch > 1 else a)

    def play_take(self, take=None):
        take = take or (self.takes[-1] if self.takes else None)
        if take is None:
            self.show_toast("まだテイクがありません（T キーで録音モードにして、再生しながら歌います）", 3.0)
            return
        if self._rec_active:
            self._finish_take()
        self.stop_review(pause=False)
        audio = self._take_audio(take)
        start = take["t0"] + self.take_offset
        if start < 0:                                   # 曲の頭より前の分は切る
            cut = int(-start / self.pb.speed * (self.pg.mixer.get_init() or (44100,))[0])
            audio = audio[cut:]
            start = 0.0
        self._review_vocal = self.pb.vocal_on
        if self.pb.vocal_on:                            # お手本の歌声は消して、伴奏 + 自分の声で聴く
            self.pb.toggle_vocal()
        if hasattr(self.pb, "ch_take"):
            self.pb.play_take(audio, start)
        else:                                           # 曲を選ぶ前: テイクだけ鳴らす
            self.pb.play_take(audio, start)
            try:
                self._free_take_snd = self.pg.mixer.Sound(buffer=audio)
                self._free_take_snd.play()
            except Exception:
                pass
        self.reviewing = take
        self._tick_prev = None
        sc = f"　一致率 {take['score'] * 100:.0f}%" if take.get("score") is not None else ""
        self.show_toast(f"▶ テイク {take['no']} を再生（伴奏 + 自分の声）{sc}　Space で止める", 3.0)

    def stop_review(self, pause=False):
        if self.reviewing is None:
            return
        self.reviewing = None
        self.pb.stop_take()
        if self._free_take_snd is not None:
            try:
                self._free_take_snd.stop()
            except Exception:
                pass
            self._free_take_snd = None
        if self._review_vocal and not self.pb.vocal_on:
            self.pb.toggle_vocal()
        self._review_vocal = None
        if pause and self.pb.playing:
            self.pb.toggle_play()

    def toggle_take_play(self):
        if self.reviewing is not None:
            self.stop_review(pause=True)
        else:
            self.play_take()

    def nudge_take_offset(self, d):
        self.take_offset = round(self.take_offset + d, 3)
        self.show_toast(f"テイクのずれ補正：{self.take_offset:+.2f}秒（+ = 自分の声を遅らせる）", 2.0)
        if self.on_take_offset:
            self.on_take_offset(self.take_offset)

    def save_take(self, take=None):
        take = take or (self.takes[-1] if self.takes else None)
        if take is None:
            self.show_toast("保存するテイクがありません", 2.0)
            return None
        folder = os.path.join(self.song_dir or os.path.dirname(os.path.abspath(__file__)), "takes")
        try:
            os.makedirs(folder, exist_ok=True)
            from scipy.io import wavfile

            name = f"テイク{take['no']}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            path = os.path.join(folder, name)
            x = take["audio"]
            peak = float(np.max(np.abs(x))) if len(x) else 0.0
            g = min(20.0, 0.9 / peak) if peak > 1e-4 else 1.0
            wavfile.write(path, take["sr"], (np.clip(x * g, -1, 1) * 32767).astype(np.int16))
            self.show_toast(f"テイク {take['no']} を保存しました：{path}", 4.0)
            return path
        except Exception as e:
            self.show_toast(f"保存できませんでした：{e}", 3.0)
            return None

    def clear_takes(self):
        self.stop_review(pause=True)
        self.takes = []
        self.show_toast("テイクをすべて消しました")

    def _take_items(self):
        items = []
        for tk in reversed(self.takes[-10:]):
            sc = f"　一致率 {tk['score'] * 100:.0f}%" if tk.get("score") is not None else ""
            sp = f"　({tk['speed']:.2f}倍)" if abs(tk["speed"] - 1.0) > 1e-3 else ""
            items.append(MenuItem(f"▶ テイク {tk['no']}　{fmt_time(tk['t0'])}〜{fmt_time(tk['t1'])}{sc}{sp}　{tk['time']}",
                                  (lambda tk=tk: self.play_take(tk)), checked=(lambda tk=tk: self.reviewing is tk)))
        if not items:
            items.append(MenuItem("（まだテイクがありません。T で録音モード → 再生して歌う）", lambda: None,
                                  enabled=lambda: False))
        items += [SEPARATOR,
                  MenuItem("■ テイクの再生を止める", lambda: self.stop_review(pause=True),
                           enabled=lambda: self.reviewing is not None),
                  MenuItem("最後のテイクを WAV で保存", lambda: self.save_take(), enabled=lambda: bool(self.takes)),
                  MenuItem(lambda: f"ずれ補正：自分の声を早める（今 {self.take_offset:+.2f}秒）",
                           lambda: self.nudge_take_offset(-0.02)),
                  MenuItem("ずれ補正：自分の声を遅らせる", lambda: self.nudge_take_offset(0.02)),
                  MenuItem("テイクをすべて消す", self.clear_takes, enabled=lambda: bool(self.takes))]
        return items

    def open_take_menu(self, pos):
        self.popup = PopupMenu(self.pg, self.fonts, self._take_items(), pos, (0, MENU_H, W, H - MENU_H),
                               font_size=14, title="テイク（録音した歌声）")

    # ---------- お手本バーの編集 ----------
    def toggle_edit(self):
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            if self.pb.playing:
                self.pb.toggle_play()
            self.show_toast("編集モード：バーをクリックで選択 → Delete で削除 / ↑↓ で半音 / Shift+↑↓ でオクターブ / "
                            "上下にドラッグで音程 / Ctrl+Z 元に戻す / Ctrl+S 保存", 6.0)
        else:
            self.sel.clear()
            msg = "編集モードを終わりました"
            if self.edit_dirty:
                msg += "（保存していない変更があります：お手本 → 編集版を保存）"
            self.show_toast(msg, 3.5)

    def _snapshot(self):
        return [dict(n) for n in self.guide.base_notes]

    def _push_undo(self):
        self._undo.append((id(self.guide), self._snapshot(), self.guide.offset))
        if len(self._undo) > 60:
            self._undo.pop(0)
        self._redo.clear()
        self.edit_dirty = True

    def _restore(self, state):
        gid, notes, off = state
        if gid == "lyrics":                     # 歌詞の変更
            self._restore_lyrics(notes)
            return True
        if gid != id(self.guide):
            return False
        self.guide.base_notes = notes
        self.guide.offset = off
        self.guide._apply_offset()
        self.sel = {i for i in self.sel if i < len(self.guide.notes)}
        return True

    def undo(self):
        if not self._undo:
            self.show_toast("元に戻す操作はありません")
            return
        st = self._undo.pop()
        lyr = st[0] == "lyrics"
        self._redo.append(self._lyric_state() if lyr else (id(self.guide), self._snapshot(), self.guide.offset))
        if self._restore(st):
            if not lyr:
                self.edit_dirty = True
            self.show_toast("元に戻しました (Ctrl+Z)" + ("　…歌詞" if lyr else ""))

    def redo(self):
        if not self._redo:
            return
        st = self._redo.pop()
        lyr = st[0] == "lyrics"
        self._undo.append(self._lyric_state() if lyr else (id(self.guide), self._snapshot(), self.guide.offset))
        if self._restore(st):
            if not lyr:
                self.edit_dirty = True
            self.show_toast("やり直しました (Ctrl+Y)" + ("　…歌詞" if lyr else ""))

    def _base_index(self, i):
        """表示用の番号 → base_notes の番号 (どちらも同じ並び)。"""
        return i

    def delete_selected(self):
        if not self.sel:
            self.show_toast("削除するバーを先にクリックで選んでください")
            return
        self._push_undo()
        n = len(self.sel)
        keep = [nt for i, nt in enumerate(self.guide.base_notes) if i not in self.sel]
        self.guide.base_notes = keep
        self.guide._apply_offset()
        self.sel.clear()
        self.hits = {}
        self.show_toast(f"バーを {n}本 削除しました（Ctrl+Z で元に戻せます）", 2.5)

    def move_selected(self, d, push=True):
        if not self.sel:
            self.show_toast("音程を直すバーを先にクリックで選んでください")
            return
        if push:
            self._push_undo()
        for i in self.sel:
            nt = dict(self.guide.base_notes[i])
            nt["midi"] = int(min(108, max(24, nt["midi"] + d)))
            self.guide.base_notes[i] = nt
        self.guide._apply_offset()
        names = {midi_to_doremi(self.guide.notes[i]["midi"] + self.gshift) for i in self.sel}
        what = "1オクターブ" if abs(d) == 12 else f"{abs(d)}半音"
        self.show_toast(f"{len(self.sel)}本 を {what}{'上' if d > 0 else '下'}げました → " + "・".join(sorted(names))[:40], 2.0)

    def select_all_visible(self):
        L = self.layout
        t0, t1 = L.time_window(self.pb.position())
        self.sel = {i for i, n in enumerate(self.guide.notes) if n["end"] >= t0 and n["start"] <= t1}

    def _note_at_pos(self, pos):
        for r, i in reversed(self._note_rects):
            if r.inflate(4, 6).collidepoint(pos):
                return i
        return None

    def _edits_dir(self):
        return os.path.join(self.song_dir, "edits") if self.song_dir else None

    def list_edits(self):
        d = self._edits_dir()
        if not d or not os.path.isdir(d):
            return []
        out = []
        for f in os.listdir(d):
            if f.endswith(".json"):
                p = os.path.join(d, f)
                out.append((os.path.getmtime(p), f[:-5], p))
        return [(n, p) for _, n, p in sorted(out, reverse=True)]

    def save_edit(self, name=None, ask=False):
        """編集したお手本を保存。ask=True なら名前を聞く (別名で保存)。"""
        d = self._edits_dir()
        if not d:
            self.show_toast("曲を開いているときだけ保存できます", 2.5)
            return None
        if ask or not (name or self.edit_name):
            default = self.edit_name or time.strftime("編集版 %m月%d日 %H時%M分")
            name = ask_text("お手本の編集版を保存", "名前をつけて保存します（同じ名前があれば上書き）：", default)
            if not name:
                return None
        name = name or self.edit_name
        safe = "".join(c if c not in '\\/:*?"<>|' else "_" for c in name).strip() or "編集版"
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, safe + ".json")
        notes = [dict(n) for n in self.guide.notes]          # タイミングのずらしも入れた位置で保存
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": safe, "source": self.guide.source, "notes": notes,
                       "saved": time.strftime("%Y-%m-%d %H:%M")}, f, ensure_ascii=False)
        self.edit_name = safe
        self.edit_dirty = False
        if not self.guide.source.startswith("編集版"):
            self.guide.source = f"編集版（{safe}）"
        self.save_practice_extra(last_edit=safe)
        self.show_toast(f"お手本の編集版「{safe}」を保存しました", 3.0)
        return path

    def load_edit(self, name, path=None, quiet=False):
        path = path or os.path.join(self._edits_dir() or "", name + ".json")
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            if not quiet:
                self.show_toast(f"読み込めませんでした：{e}", 3.0)
            return False
        notes = d.get("notes", [])
        g0 = self.guide
        ms = [n["midi"] for n in notes] or [g0.min_midi, g0.max_midi]
        data = {"notes": notes, "pitch": g0.pitch, "hop_sec": g0.hop, "duration": g0.duration,
                "range": {"min_midi": min(ms), "max_midi": max(ms)}}
        g = Guide(data, notes=notes, source=f"編集版（{d.get('name', name)}）")
        if not g0.source.startswith("編集版"):
            self.alt_guide = g0                 # M キーで元のお手本と切り替えられるように
        self.guide = g
        self.edit_name = d.get("name", name)
        self.edit_dirty = False
        self.sel.clear()
        self._undo.clear()
        self._redo.clear()
        self.hits = {}
        self.save_practice_extra(last_edit=self.edit_name)
        if not quiet:
            self.show_toast(f"お手本：編集版「{self.edit_name}」", 2.5)
        return True

    def save_practice_extra(self, **kw):
        p = self._practice_path()
        if not p:
            return
        try:
            d = {}
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
            d.update(kw)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
        except Exception:
            pass

    def _guide_items(self):
        """「お手本」メニューの、編集版の一覧部分。"""
        items = [MenuItem("お手本を編集する（編集モード）", self.toggle_edit, "E", checked=lambda: self.edit_mode),
                 MenuItem("編集版を保存（上書き）", lambda: self.save_edit(), "Ctrl+S",
                          enabled=lambda: self.song_dir is not None),
                 MenuItem("編集版を別名で保存…", lambda: self.save_edit(ask=True), "Ctrl+Shift+S",
                          enabled=lambda: self.song_dir is not None),
                 MenuItem("元に戻す", self.undo, "Ctrl+Z", enabled=lambda: bool(self._undo)),
                 MenuItem("やり直す", self.redo, "Ctrl+Y", enabled=lambda: bool(self._redo))]
        eds = self.list_edits()
        if eds:
            items.append(SEPARATOR)
            for n, p in eds[:8]:
                items.append(MenuItem(f"お手本：編集版「{n}」", (lambda n=n, p=p: self.load_edit(n, p)),
                                      checked=(lambda n=n: self.guide.source == f"編集版（{n}）")))
        return items

    # ---------- 操作 ----------
    def seek_by(self, d):
        before = self.pb.position()
        t = self.pb.seek(before + d)
        self.mic.clear()
        arrow = "◀◀ 巻き戻し" if d < 0 else "▶▶ スキップ"
        self.show_toast(f"{arrow} {abs(int(d))}秒  →  {fmt_time(t)}")

    def show_toast(self, text, sec=1.4):
        self.toast = (text, time.perf_counter() + sec)

    def handle_key(self, ev):
        pg, k = self.pg, ev.key
        mod = getattr(ev, "mod", 0) or 0
        ctrl = bool(mod & getattr(pg, "KMOD_CTRL", 0))
        shift = bool(mod & getattr(pg, "KMOD_SHIFT", 0))
        K = lambda name: getattr(pg, name, -999)
        alt = bool(mod & getattr(pg, "KMOD_ALT", 0))
        if self.lyric_handle_key(k, ctrl or shift or alt):     # 歌詞 (K・歌い出しの記録の Enter)
            return
        # --- 編集 (どのモードでも Ctrl+Z / Y / S は使える) ---
        if ctrl and k == K("K_z"):
            (self.redo if shift else self.undo)()
            return
        if ctrl and k == K("K_y"):
            self.redo()
            return
        if ctrl and k == K("K_s"):
            self.save_edit(ask=shift)
            return
        if ctrl and k == K("K_a") and self.edit_mode:
            self.select_all_visible()
            return
        if self.edit_mode:
            if k in (K("K_DELETE"), K("K_BACKSPACE")):
                self.delete_selected()
                return
            if k in (pg.K_UP, pg.K_DOWN):
                self.move_selected((12 if shift else 1) * (1 if k == pg.K_UP else -1))
                return
            if k in (K("K_PAGEUP"), K("K_PAGEDOWN")) and self.sel:
                self.move_selected(12 if k == K("K_PAGEUP") else -12)
                return
        # --- 速度・キー ---
        if shift and k in (pg.K_UP, pg.K_DOWN):
            self.step_key(1 if k == pg.K_UP else -1)
            return
        if shift and k in (pg.K_LEFT, pg.K_RIGHT):
            self.step_speed(0.05 if k == pg.K_RIGHT else -0.05)
            return
        # --- A-B ループ・録音・編集 ---
        if k == K("K_a"):
            self.set_loop_a()
            return
        if k == K("K_b"):
            self.set_loop_b()
            return
        if k == K("K_l"):
            self.toggle_loop()
            return
        if k == K("K_t"):
            self.toggle_rec()
            return
        if k == K("K_y"):
            self.toggle_take_play()
            return
        if k == K("K_e"):
            self.toggle_edit()
            return
        if k == pg.K_SPACE:
            self.pb.toggle_play()
        elif k == pg.K_LEFT:
            self.seek_by(-self.skip_sec)
        elif k == pg.K_RIGHT:
            self.seek_by(self.skip_sec)
        elif k == pg.K_UP:
            self.skip_sec = min(60, self.skip_sec + 1)
            self.show_toast(f"スキップ間隔：{self.skip_sec}秒")
        elif k == pg.K_DOWN:
            self.skip_sec = max(1, self.skip_sec - 1)
            self.show_toast(f"スキップ間隔：{self.skip_sec}秒")
        elif k == pg.K_v:
            self.pb.toggle_vocal()
            self.show_toast("ボーカル：ON（お手本の声あり）" if self.pb.vocal_on else "ボーカル：OFF（伴奏だけ）")
        elif k == pg.K_HOME:
            if self.loop_on and self.loop_a is not None:
                self.pb.seek(self.loop_a)
                self.show_toast("A 点に戻りました")
            else:
                self.pb.seek(0)
                self.show_toast("最初に戻りました")
            self.mic.clear()
        elif k in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
            self.octave_shift += -1 if k == pg.K_LEFTBRACKET else 1
            self.octave_shift = max(-2, min(2, self.octave_shift))
            self._rebuild_background()
            self.show_toast(f"お手本の高さ：{self.octave_shift:+d}オクターブ" if self.octave_shift else "お手本の高さ：原曲どおり")
        elif k in (pg.K_MINUS, pg.K_KP_MINUS):
            self.pb.change_volume(-0.1)
            self.show_toast(f"曲の音量：{int(round(self.pb.volume * 100))}%")
        elif k in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS, pg.K_SEMICOLON):
            self.pb.change_volume(+0.1)
            self.show_toast(f"曲の音量：{int(round(self.pb.volume * 100))}%")
        elif k == pg.K_p:
            self.show_fine = not self.show_fine
        elif k in (pg.K_COMMA, pg.K_PERIOD) and self.guide.source.startswith("MIDI"):
            self.guide.shift(-0.05 if k == pg.K_COMMA else 0.05)
            if self.on_offset_change:
                self.on_offset_change(self.guide.offset)
            self.show_toast(f"お手本のタイミング：{self.guide.offset:+.2f}秒 （, . で調整）")
        elif k == pg.K_m and self.alt_guide is not None:
            self.guide, self.alt_guide = self.alt_guide, self.guide
            self._rebuild_background()
            self.show_toast(f"お手本：{self.guide.source}")
        elif k == pg.K_r:
            self.score_hit = self.score_total = 0
            self.hits = {}
            self.show_toast("採点をリセットしました（光った部分も消えます）")
        elif k == pg.K_o:
            self.toggle_monitor()
        elif k in (pg.K_9, pg.K_0):
            self.step_monitor_gain(-5 if k == pg.K_9 else 5)
        elif k == getattr(pg, "K_n", -1):
            self.toggle_noise_panel()
        elif k == getattr(pg, "K_g", -1):
            self.toggle_grid_panel()
        elif k == getattr(pg, "K_c", -1):
            self.toggle_follow()
        elif k in (getattr(pg, "K_PAGEUP", -1), getattr(pg, "K_PAGEDOWN", -1)):
            self.scroll_pitch(2 if k == pg.K_PAGEUP else -2)

    # ---------- メニュー ----------
    def press(self, key):
        """キー操作と同じ処理を呼ぶ (メニュー・ボタンから使う)。"""
        self.handle_key(_Key(key))

    def set_skip(self, sec):
        self.skip_sec = sec
        self.show_toast(f"スキップ間隔：{sec}秒")

    def set_octave(self, n):
        self.octave_shift = n
        self._rebuild_background()
        self.show_toast(f"お手本の高さ：{n:+d}オクターブ" if n else "お手本の高さ：原曲どおり")

    def zoom(self, factor):
        self.set_pxsec(PX_PER_SEC * factor)

    def _bar_items(self, t=None, full=False):
        """小節の線の操作 (メニューバーの「小節」と、楽譜の右クリックで共通)。t = 右クリックした時刻。"""
        hb = self.has_beats
        here = (lambda: t) if t is not None else (lambda: self.pb.position())
        items = [
            MenuItem("ここを小節の頭（1拍目）にする" if t is not None else "今の位置を小節の頭（1拍目）にする",
                     lambda: self.set_downbeat_here(here()), "ダブルクリック" if t is not None else "", enabled=hb),
            MenuItem("小節の頭を 1拍 前へ", lambda: self.move_downbeat(-1), enabled=hb),
            MenuItem("小節の頭を 1拍 後ろへ", lambda: self.move_downbeat(1), enabled=hb),
            SEPARATOR,
        ]
        items += [MenuItem(f"{n}拍子", (lambda n=n: self.set_bpb(n)), enabled=hb,
                           checked=(lambda n=n: self.has_beats() and self.beats.bpb == n)) for n in BPB_CHOICES]
        items += [
            SEPARATOR,
            MenuItem(lambda: f"テンポを上げる（今 ♩ = {self.beats.bpm:.1f}）" if self.has_beats() else "テンポを上げる",
                     lambda: self.step_bpm(0.5), "Shift+ホイール↑" if t is not None else "", enabled=hb),
            MenuItem("テンポを下げる", lambda: self.step_bpm(-0.5), "Shift+ホイール↓" if t is not None else "", enabled=hb),
        ]
        if full:
            items += [
                MenuItem("テンポを解析結果に戻す", self.reset_bpm, enabled=lambda: hb() and bool(self.beats.manual_bpm)),
                SEPARATOR,
                MenuItem("小節線を少し後ろへ（0.01秒）", lambda: self.nudge_bars(0.01), enabled=hb),
                MenuItem("小節線を少し前へ（0.01秒）", lambda: self.nudge_bars(-0.01), enabled=hb),
            ]
        items += [
            MenuItem("小節線を解析結果に戻す", self.reset_bars, enabled=lambda: self.beats is not None),
            SEPARATOR,
            MenuItem("タテ線：小節と拍（解析）", lambda: self.set_vmode("bar"), enabled=hb, checked=self.bar_mode),
        ]
        if full:
            items += [MenuItem("拍の点線も引く", self.toggle_beat_lines, checked=lambda: self.view["beat_lines"])]
        return items

    def _grid_items(self, full=False):
        """表示のグリッド (秒の線・縦横の間隔など)。"""
        hb = self.has_beats
        zoom_items = [MenuItem(f"画面に {n}小節", (lambda n=n: self.set_bars_on_screen(n)), enabled=hb,
                               checked=(lambda n=n: self.has_beats() and abs(self.bars_on_screen() - n) < 0.05 * n))
                      for n in (BARS_CHOICES if full else (2, 4, 8))]
        items = zoom_items + [SEPARATOR] + [MenuItem(f"タテ線：{s_:g}秒ごと", (lambda s_=s_: self.set_vgrid(s_)),
                          checked=(lambda s_=s_: not self.bar_mode() and self.view["vgrid"] == s_))
                 for s_ in (VGRID_CHOICES if full else (1.0,))]
        if full:
            items += [
                SEPARATOR,
                MenuItem("縦：拡大（段の間隔を広げる）", lambda: self.vzoom(1.2), "Shift+ホイール↑"),
                MenuItem("縦：縮小（広い音域が見える）", lambda: self.vzoom(1 / 1.2), "Shift+ホイール↓"),
                MenuItem("横：拡大（ゆっくり流れる）", lambda: self.zoom(1.25), "Ctrl+ホイール↑"),
                MenuItem("横：縮小（先まで見える）", lambda: self.zoom(0.8), "Ctrl+ホイール↓"),
                MenuItem("♯の段の点線を表示", self.toggle_sharp_lines, checked=lambda: self.view["sharp_lines"]),
            ]
        items += [
            SEPARATOR,
            MenuItem("音階を中央に（自動で上下）", self.toggle_follow, "C", checked=lambda: self.view["follow"]),
            MenuItem("グリッドの調整パネル…", self.toggle_grid_panel, "G", checked=lambda: self.show_grid_panel),
        ]
        if full:
            items += [MenuItem("グリッドを元に戻す", self.reset_view)]
        return items

    # ---------- マイク / スピーカーの選択 ----------
    @staticmethod
    def _short(name, n=44):
        return name if len(name) <= n else name[: n - 1] + "…"

    def _device_items(self):
        """「デバイス」メニューの中身 (開くたびに機器の一覧を調べ直す)。"""
        mic = self.mic
        ins, outs = mic.list_devices() if hasattr(mic, "list_devices") else ([], [])
        cur_in = getattr(mic, "in_pref", None)
        cur_out = SONG_OUT["name"]
        in_used = getattr(mic, "in_name", "") or "―"
        out_used = SONG_OUT["used"] or "Windows の既定"
        items = [MenuItem(f"マイク（入力）　今：{self._short(in_used, 36)}", lambda: None, enabled=lambda: False),
                 MenuItem("　Windows の既定のマイク", lambda: self.choose_input(None), checked=lambda: cur_in is None)]
        items += [MenuItem("　" + self._short(n), (lambda n=n: self.choose_input(n)),
                           checked=(lambda n=n: cur_in is not None and match_device(cur_in, [n]) == n)) for n in ins]
        items += [SEPARATOR,
                  MenuItem(f"スピーカー / ヘッドホン（出力）　今：{self._short(out_used, 30)}", lambda: None, enabled=lambda: False),
                  MenuItem("　Windows の既定のスピーカー", lambda: self.choose_output(None), checked=lambda: cur_out is None)]
        items += [MenuItem("　" + self._short(n), (lambda n=n: self.choose_output(n)),
                           checked=(lambda n=n: cur_out is not None and match_device(cur_out, [n]) == n)) for n in outs]
        if not ins and not outs:
            items += [MenuItem("　（機器の一覧を読めませんでした）", lambda: None, enabled=lambda: False)]
        items += [SEPARATOR,
                  MenuItem("テスト音を鳴らす（選んだスピーカーから）", self.play_test_tone),
                  MenuItem("機器をつなぎ直したときは、もう一度このメニューを開くと一覧が新しくなります", lambda: None,
                           enabled=lambda: False)]
        return items

    def _device_changed(self, what, name):
        mic = self.mic
        missing = getattr(mic, "pref_missing", [])
        if name and name in missing:
            self.show_toast(f"{what}「{self._short(name, 30)}」を開けませんでした。Windows の既定を使います", 3.5)
        elif what == "マイク" and getattr(mic, "error", None) and not getattr(mic, "in_name", ""):
            self.show_toast("マイクを開けませんでした。別のマイクを選んでください", 3.5)
        else:
            used = getattr(mic, "in_name", "") if what == "マイク" else (SONG_OUT["used"] or "Windows の既定")
            extra = ""
            if what == "スピーカー" and getattr(mic, "monitor_error", None):
                extra = "（自分の声はこの組み合わせでは出せません）"
            self.show_toast(f"{what}：{self._short(used or '既定', 40)}{extra}", 3.0)
        if self.on_device_change:
            self.on_device_change()

    def choose_input(self, name):
        self.mic.set_devices(in_pref=name)
        self.mic.clear()
        self._device_changed("マイク", name)

    def choose_output(self, name):
        SONG_OUT["name"] = name
        self.mic.set_devices(out_pref=name)                 # 自分の声もこのスピーカーから
        if hasattr(self.pb, "reopen"):
            self.pb.reopen()                                 # 曲も同じ位置からこのスピーカーで
        else:
            try:
                self.pg.mixer.quit()
                mixer_init(self.pg, 44100)
            except Exception:
                pass
        if name and SONG_OUT["error"] and name not in self.mic.pref_missing:
            self.mic.pref_missing.append(name)
        self._device_changed("スピーカー", name)

    def play_test_tone(self):
        """選んだスピーカーから「ポーン」と短い音を鳴らす。"""
        pg = self.pg
        try:
            init = pg.mixer.get_init()
            if not init:
                mixer_init(pg, 44100)
                init = pg.mixer.get_init()
            sr, _, ch = init
            t = np.arange(int(sr * 0.5)) / sr
            tone = np.sin(2 * np.pi * 880 * t) * np.exp(-t * 6) * 0.35
            a = (tone * 32767).astype(np.int16)
            buf = np.ascontiguousarray(np.stack([a] * ch, axis=1) if ch > 1 else a)
            self._test_snd = pg.mixer.Sound(buffer=buf)
            self._test_snd.play()
            self.show_toast(f"テスト音：{self._short(SONG_OUT['used'] or 'Windows の既定のスピーカー', 40)}", 2.0)
        except Exception as e:
            self.show_toast(f"テスト音を鳴らせませんでした：{e}", 3.0)

    def _practice_items(self):
        """「練習」メニュー: A-B ループ・速度・キー・録音。"""
        sp, se = getattr(self.pb, "fx_target", (1.0, 0))
        has_ab = lambda: self.loop_a is not None and self.loop_b is not None
        items = [MenuItem("今の位置を A 点に（ループの始め）", lambda: self.set_loop_a(), "A"),
                 MenuItem("今の位置を B 点に（ループの終わり）", lambda: self.set_loop_b(), "B"),
                 MenuItem("A-B ループ再生", self.toggle_loop, "L", checked=lambda: self.loop_on, enabled=has_ab),
                 MenuItem("今の小節をループ", lambda: self.loop_bars_at(self.pb.position(), 1), enabled=self.has_beats),
                 MenuItem("ループを解除", self.clear_loop, enabled=lambda: self.loop_a is not None),
                 SEPARATOR]
        items += [MenuItem(f"再生速度 {v:.2f}倍" + ("（元の速さ）" if v == 1.0 else ""), (lambda v=v: self.set_speed(v)),
                           "Shift+← →" if v == 1.0 else "", checked=(lambda v=v: abs(sp - v) < 1e-6))
                  for v in (1.0, 0.9, 0.8, 0.7)]
        items += [MenuItem(lambda: f"キー：{self.pb.fx_target[1]:+d}　半音上げる", lambda: self.step_key(1), "Shift+↑",
                           enabled=lambda: self.pb.fx_target[1] < 6),
                  MenuItem("キー：半音下げる", lambda: self.step_key(-1), "Shift+↓", enabled=lambda: self.pb.fx_target[1] > -6),
                  MenuItem("速度・キーを元に戻す", lambda: self._request_fx(1.0, 0),
                           enabled=lambda: tuple(self.pb.fx_target) != (1.0, 0)),
                  SEPARATOR,
                  MenuItem("録音モード（再生中の歌声をテイクとして録音）", self.toggle_rec, "T", checked=lambda: self.rec_armed),
                  MenuItem("最後のテイクを再生 / 止める", self.toggle_take_play, "Y",
                           checked=lambda: self.reviewing is not None),
                  MenuItem("テイクの一覧…", lambda: self.open_take_menu((W // 2, MENU_H + 40))),
                  SEPARATOR,
                  MenuItem("下の2段目（練習ツール）を表示", self.toggle_toolbar2, checked=lambda: self.view.get("toolbar2", True))]
        return items

    def toggle_toolbar2(self):
        self.view["toolbar2"] = not self.view.get("toolbar2", True)
        self._update_layout()
        self._rebuild_background()
        self._view_changed()

    def open_popup(self, pos):
        """楽譜の右クリック: その場所でグリッドを操作するメニュー。"""
        L = self.layout
        t = self._time_at_x(pos[0]) if pos[0] >= L.left else None
        title = None
        if t is not None and self.has_beats():
            bar, beat = self.beats.bar_at(t)
            title = f"{fmt_time(t)}　{bar}小節目・{beat}拍目　（{self._bar_desc()}）" if bar and bar > 0 else fmt_time(t)
        elif t is not None:
            title = fmt_time(t)
        tt = t if t is not None else self.pb.position()
        loop = [MenuItem("ここを A 点に", lambda: self.set_loop_a(tt)),
                MenuItem("ここを B 点に", lambda: self.set_loop_b(tt)),
                MenuItem("この小節をループ" if self.has_beats() else "この前後 4秒 をループ", lambda: self.loop_bars_at(tt, 1)),
                MenuItem("この2小節をループ", lambda: self.loop_bars_at(tt, 2), enabled=self.has_beats),
                MenuItem("ループを解除", self.clear_loop, enabled=lambda: self.loop_a is not None)]
        edit = []
        if self.edit_mode:
            i = self._note_at_pos(pos)
            if i is not None and i not in self.sel:
                self.sel = {i}
            edit = [MenuItem(f"選んだバー ({len(self.sel)}本) を削除", self.delete_selected, "Delete", enabled=lambda: bool(self.sel)),
                    MenuItem("半音 上げる", lambda: self.move_selected(1), "↑", enabled=lambda: bool(self.sel)),
                    MenuItem("半音 下げる", lambda: self.move_selected(-1), "↓", enabled=lambda: bool(self.sel)),
                    MenuItem("1オクターブ 上げる", lambda: self.move_selected(12), "Shift+↑", enabled=lambda: bool(self.sel)),
                    MenuItem("1オクターブ 下げる", lambda: self.move_selected(-12), "Shift+↓", enabled=lambda: bool(self.sel)),
                    MenuItem("元に戻す", self.undo, "Ctrl+Z", enabled=lambda: bool(self._undo)),
                    MenuItem("編集版を保存", lambda: self.save_edit(), "Ctrl+S"),
                    SEPARATOR]
            items = edit + loop
        else:
            items = loop + [SEPARATOR] + self._bar_items(t) + [SEPARATOR] + self._grid_items()
        self.popup = PopupMenu(self.pg, self.fonts, items, pos, (0, MENU_H, W, H - MENU_H),
                               font_size=14, title=title)

    def _shift_held(self):
        try:
            return bool(self.pg.key.get_mods() & getattr(self.pg, "KMOD_SHIFT", 0))
        except Exception:
            return False

    def _time_at_x(self, x):
        L = self.layout
        now = self.pb.position()
        return max(0.0, now + (x - L.playhead_x) / PX_PER_SEC)

    def _labels_rect(self):
        L = self.layout
        return self.pg.Rect(0, L.top, L.left, L.bottom - L.top)

    def _rband_rect(self, d, pos):
        x0, y0 = d["x0"], d["y0"]
        return self.pg.Rect(min(x0, pos[0]), min(y0, pos[1]), abs(pos[0] - x0) + 1, abs(pos[1] - y0) + 1)

    def _loop_flag_at(self, pos):
        """A / B の旗 (楽譜の上端) の上か → "A" / "B" / None"""
        if self.loop_a is None and self.loop_b is None:
            return None
        L = self.layout
        if not (L.top - 2 <= pos[1] <= L.top + RULER_H + 6):
            return None
        now = self.pb.position()
        for name, t in (("B", self.loop_b), ("A", self.loop_a)):
            if t is not None and abs(pos[0] - L.x_of(t, now)) <= 9:
                return name
        return None

    def _ruler_rect(self):
        L = self.layout
        return self.pg.Rect(L.left, L.top, L.right - L.left, RULER_H)

    def _build_menus(self):
        pg, K = self.pg, self.press
        is_midi = lambda: self.guide.source.startswith("MIDI")
        menus = [
            ("ファイル", [
                MenuItem("曲の一覧から選ぶ…", lambda: self.requests.append("library"), "Ctrl+L"),
                MenuItem("フォルダを開く…", lambda: self.requests.append("open_folder"), "Ctrl+F"),
                MenuItem("曲ファイルを直接開く…", lambda: self.requests.append("open_song"), "Ctrl+O"),
                MenuItem("この曲に MIDI を合わせる…", lambda: self.requests.append("open_midi"), "Ctrl+M",
                         enabled=lambda: self.song_path is not None),
                SEPARATOR,
                MenuItem("終了", lambda: self.requests.append("quit")),
            ]),
            ("再生", [
                MenuItem(lambda: "一時停止" if self.pb.playing else "再生", lambda: K(pg.K_SPACE), "Space"),
                MenuItem("最初から", lambda: K(pg.K_HOME), "Home"),
                SEPARATOR,
                MenuItem(lambda: f"巻き戻し（{self.skip_sec}秒）", lambda: K(pg.K_LEFT), "←"),
                MenuItem(lambda: f"スキップ（{self.skip_sec}秒）", lambda: K(pg.K_RIGHT), "→"),
                SEPARATOR,
                MenuItem("スキップ間隔を 1秒 長く", lambda: K(pg.K_UP), "↑"),
                MenuItem("スキップ間隔を 1秒 短く", lambda: K(pg.K_DOWN), "↓"),
            ] + [MenuItem(f"スキップ間隔：{n}秒", (lambda n=n: self.set_skip(n)), checked=(lambda n=n: self.skip_sec == n))
                 for n in (1, 3, 5, 10, 15)]),
            ("音声", [
                MenuItem("ボーカル（お手本の声）を鳴らす", lambda: K(pg.K_v), "V", checked=lambda: self.pb.vocal_on),
                MenuItem("自分の歌声をスピーカー / ヘッドホンから出す", self.toggle_monitor, "O",
                         checked=lambda: self.mic.monitor, enabled=lambda: self.mic.monitor_error is None),
                SEPARATOR,
                MenuItem(lambda: f"音量の調整（スライダー）…　自分の声 {round(self.mic.monitor_gain * 100)}% / 曲 {round(self.pb.volume * 100)}%",
                         self.toggle_mixer, checked=lambda: self.show_mixer),
                SEPARATOR,
                MenuItem(lambda: "雑音カットの調整…　今：" + MicTracker.NOISE_NAMES.get(getattr(self.mic, "noise", "mid"), "中"),
                         self.toggle_noise_panel, "N", checked=lambda: self.show_noise_panel),
                MenuItem("自分の声を自動で聞きやすい大きさに（おすすめ）", self.toggle_monitor_auto,
                         checked=lambda: getattr(self.mic, "monitor_auto", False), enabled=lambda: self.mic.monitor_error is None),
            ] + [MenuItem(f"　自分の声 {v}%", (lambda v=v: self.set_monitor_gain(v / 100)),
                          checked=(lambda v=v: round(self.mic.monitor_gain * 100) == v),
                          enabled=lambda: self.mic.monitor_error is None) for v in (50, 100, 200, 300)] + [
                SEPARATOR,
                MenuItem(lambda: f"自分の声の遅れ：{self._latency_text()}", lambda: None, enabled=lambda: False),
            ] + [MenuItem(f"　方式：{label}", (lambda a=a: self.set_monitor_api(a)),
                          checked=(lambda a=a: getattr(self.mic, "api", "auto") == a),
                          enabled=(lambda a=a: a == "auto" or a in self._apis()))
                 for a, label in MicTracker.API_CHOICES] + [
                SEPARATOR,
                MenuItem("曲の音量を上げる (10%)", lambda: K(pg.K_EQUALS), "+"),
                MenuItem("曲の音量を下げる (10%)", lambda: K(pg.K_MINUS), "-"),
            ]),
            ("練習", self._practice_items),
            ("お手本", lambda: [
                MenuItem("1オクターブ上げる", lambda: K(pg.K_RIGHTBRACKET), "]"),
                MenuItem("1オクターブ下げる", lambda: K(pg.K_LEFTBRACKET), "["),
                MenuItem("原曲の高さに戻す", lambda: self.set_octave(0), checked=lambda: self.octave_shift == 0),
                SEPARATOR,
                MenuItem("お手本を MIDI にする", lambda: None if is_midi() else K(pg.K_m), "M", checked=is_midi,
                         enabled=lambda: self.alt_guide is not None or is_midi()),
                MenuItem("お手本を歌声から解析したものにする", lambda: K(pg.K_m) if is_midi() else None, "M",
                         checked=lambda: not is_midi(), enabled=lambda: self.alt_guide is not None or not is_midi()),
                MenuItem("タイミングを 0.05秒 早める", lambda: K(pg.K_COMMA), ",", enabled=is_midi),
                MenuItem("タイミングを 0.05秒 遅らせる", lambda: K(pg.K_PERIOD), ".", enabled=is_midi),
                SEPARATOR,
                MenuItem("採点をリセット", lambda: K(pg.K_r), "R"),
                SEPARATOR,
            ] + self._guide_items()),
            ("歌詞", self._lyrics_items),
            ("表示", [
                MenuItem("全画面表示", lambda: self.requests.append("fullscreen"), "F11", checked=lambda: self.is_fullscreen()),
                MenuItem("お手本の細い音程線", lambda: K(pg.K_p), "P", checked=lambda: self.show_fine),
                SEPARATOR,
                MenuItem("音階を中央に（自動で上下に合わせる）", self.toggle_follow, "C",
                         checked=lambda: self.view["follow"]),
            ]),
            ("小節", self._bar_items(None, full=True)),
            ("グリッド", self._grid_items(full=True)),
            ("デバイス", self._device_items),
            ("ヘルプ", [
                MenuItem("操作方法", self.toggle_help, "F1", checked=lambda: self.show_help),
                SEPARATOR,
                MenuItem(lambda: "アップデートを確認…" + (f"　★ 新しい版 {UPDATE.notify} があります" if UPDATE.notify else ""),
                         lambda: UPDATE.check(manual=True)),
                MenuItem("ファイル (ZIP) からアップデート…", lambda: self.requests.append("update_zip")),
                MenuItem(lambda: f"バージョン情報（{UPDATE.current_version()}）", UPDATE.open_version),
                MenuItem("起動時にアップデートを確認する", lambda: self.requests.append("toggle_update_auto"),
                         checked=lambda: self.update_auto()),
            ]),
        ]
        return menus

    # ---------- 自分の声の出力 ----------
    def toggle_monitor(self):
        self.set_monitor(not self.mic.monitor)

    def set_monitor(self, on):
        if on and self.mic.monitor_error:
            self.show_toast("自分の声を出力できません（マイクか出力先を確認してください）")
            return
        self.mic.set_monitor(on)
        if self.mic.monitor:
            if getattr(self.mic, "wireless", False):
                self.show_toast("自分の声：ON  ※ワイヤレス (Bluetooth) 機器は遅れが大きくなります。有線がおすすめ", 4.0)
            else:
                self.show_toast(f"自分の声：ON（{self._latency_text()}）  ※ハウリング防止にヘッドホン推奨", 3.0)
        else:
            self.show_toast("自分の声：OFF")
        if self.on_monitor_change:
            self.on_monitor_change()

    def _latency_text(self):
        m = self.mic
        used = getattr(m, "api_used", None)
        api = {"Windows WASAPI": "WASAPI", "Windows WDM-KS": "WDM-KS", "Windows DirectSound": "DirectSound"}.get(
            used, used or "-")
        ms = getattr(m, "monitor_latency_ms", None)
        return f"{api} / 遅れ 約{ms}ms" if ms else api

    def set_monitor_api(self, api):
        """自分の声の出し入れの方式を変える (遅れを減らす)。"""
        was_on = self.mic.monitor
        ok = self.mic.restart(api)
        if self.mic.monitor_error:
            self.show_toast("この方式では自分の声を出力できません。別の方式を選んでください", 3.5)
        else:
            self.mic.set_monitor(was_on)
            self.show_toast(f"自分の声の方式：{self._latency_text()}" +
                            ("  ※ワイヤレス機器は遅れが大きくなります" if self.mic.wireless else ""), 3.5)
        if self.on_monitor_change:
            self.on_monitor_change()

    def set_noise(self, level):
        self.mic.set_noise(level)
        self.show_toast("雑音カット：" + {"off": "OFF（マイクの音そのまま）", "weak": "弱", "mid": "中（おすすめ）",
                                       "strong": "強（うるさい場所向け）"}.get(level, level), 2.2)
        if self.on_monitor_change:
            self.on_monitor_change()

    def _noise_param_text(self, name):
        m = self.mic
        return {"margin": f"声とみなす大きさ：雑音より {m.gate_margin:.0f}dB 大きい音から",
                "cut": f"雑音だけのときに下げる量：{m.gate_cut:.0f}dB" + ("（下げない）" if m.gate_cut > -1 else ""),
                "hold": f"余韻：声が止まってから {m.gate_hold_sec:.2f}秒 は通す",
                "hp": "低い音を切る：" + (f"{m.hp_freq:.0f}Hz より下" if m.hp_freq >= 20 else "OFF")}[name]

    def calibrate_noise(self):
        self.mic.calibrate_noise(2.0)
        self.show_toast("今の雑音を測っています… 2秒ほど静かにしてください", 2.5)

    def cycle_noise(self):
        order = ["off", "weak", "mid", "strong"]
        cur = getattr(self.mic, "noise", "mid")
        self.set_noise(order[(order.index(cur) + 1) % 4] if cur in order else "mid")

    def toggle_monitor_auto(self):
        self.mic.set_monitor_auto(not getattr(self.mic, "monitor_auto", False))
        self.show_toast("自分の声を自動で聞きやすい大きさに：" + ("ON（小さい声を持ち上げます）" if self.mic.monitor_auto else "OFF（マイクの音そのまま）"), 2.5)
        if self.on_monitor_change:
            self.on_monitor_change()

    def set_monitor_gain(self, g):
        self.mic.set_monitor_gain(g)
        self.show_toast(f"自分の声の音量：{int(round(self.mic.monitor_gain * 100))}%")
        if self.on_monitor_change:
            self.on_monitor_change()

    def step_monitor_gain(self, pct, save=True):
        """自分の声の音量を pct % (例 +1, -5) だけ変える。save=False なら保存は後で (ボタンを離したとき)。"""
        g = round(self.mic.monitor_gain * 100 + pct) / 100
        if save:
            self.set_monitor_gain(g)
        else:
            self.mic.set_monitor_gain(g)
            self._monitor_dirty = True
            self.show_toast(f"自分の声の音量：{int(round(self.mic.monitor_gain * 100))}%")

    def step_vocal_gain(self, pct, save=True):
        """ボーカル (お手本の歌声) の大きさを pct % 変える。"""
        self.set_vocal_gain(round(getattr(self.pb, "vocal_gain", 1.0) * 100 + pct) / 100, save)

    def set_vocal_gain(self, g, save=True):
        self.pb.set_vocal_gain(g, restart=save)       # 押しっぱなしの間は鳴らし直さない (音が途切れないように)
        v = round(self.pb.vocal_gain * 100)
        self.show_toast(f"ボーカル（お手本の歌声）の音量：{v}%" + ("（今は OFF。V キーで ON）" if not self.pb.vocal_on else ""))
        if save and self.on_vocal_gain:
            self.on_vocal_gain(self.pb.vocal_gain)
        elif not save:
            self._vocal_dirty = True

    def step_song_volume(self, pct):
        self._set_volume(round(self.pb.volume * 100 + pct) / 100)

    def _repeat_hold(self):
        """押しっぱなしのボタンを繰り返し押す (毎フレーム呼ぶ)。"""
        h = self.hold
        if not h:
            return
        b = next((b for b in self.buttons if b.label == h["label"] and b.tooltip == h["tip"]), None)
        if b is None or not b.rect.collidepoint(self.mouse) or not b.enabled:
            return                              # ボタンの外に出ている間は止める
        now = time.perf_counter()
        while now >= h["next"]:
            b.action()
            h["n"] += 1
            h["next"] += 0.07 if h["n"] < 10 else 0.03     # 10回目からは速く

    def _apis(self):
        if not hasattr(self, "_api_cache"):
            self._api_cache = MicTracker.available_apis()
        return self._api_cache

    def toggle_mixer(self):
        self.show_mixer = not self.show_mixer
        if self.show_mixer:
            self._fit_panels("mixer")

    def _set_volume(self, v):
        v = min(1.0, max(0.0, v))
        self.pb.change_volume(v - self.pb.volume)
        self.show_toast(f"曲の音量：{int(round(self.pb.volume * 100))}%")

    def toggle_help(self):
        self.show_help = not self.show_help

    # ---------- マウス ----------
    def _progress_rect(self):
        return self.pg.Rect(0, HEADER_H - 4, W, PROGRESS_H + 10)

    def _lane_rect(self):
        L = self.layout
        return self.pg.Rect(L.left, L.top, L.right - L.left, L.bottom - L.top)

    def _seek_to(self, t, toast=True):
        t = self.pb.seek(t)
        self.mic.clear()
        if toast:
            self.show_toast(f"移動  →  {fmt_time(t)}")
        return t

    def handle_mouse_down(self, pos, button=1):
        self.mouse = pos
        if self.show_help:
            self.show_help = False
            return
        if UPDATE.show and UPDATE.state:        # アップデートの画面: その画面のボタンだけ
            for b in self._dlg_buttons:
                if b.rect.collidepoint(pos) and b.enabled:
                    b.action()
            return
        if self.popup is not None:              # 右クリックのメニューが開いている
            self.popup.on_mouse_down(pos)
            if self.popup.closed:
                self.popup = None
            return
        L = self.layout
        if (self.lyrics is not None and not self.menu.is_open and self._lyric_lane_rect().collidepoint(pos)
                and not any(r.collidepoint(pos) for r in self._panel_rects())
                and not any(b.rect.collidepoint(pos) for b in self.buttons)):
            now_c = time.perf_counter()
            lt, lp = self._last_click
            double = now_c - lt < 0.4 and abs(pos[0] - lp[0]) <= 6 and abs(pos[1] - lp[1]) <= 6
            self._last_click = (0.0, (-99, -99)) if double else (now_c, pos)
            if self.lyric_mouse_down(pos, button, double):    # 歌詞の段
                return
        in_score = L.top <= pos[1] <= L.bottom and pos[0] < L.right
        if button == 3:                         # 右クリック: 楽譜の上ならグリッドのメニュー
            if self.menu.is_open:
                self.menu.close()
            elif in_score and not any(r.collidepoint(pos) for r in self._panel_rects()):
                self.open_popup(pos)
            return
        if button != 1:
            return
        if self.menu.on_mouse_down(pos):
            return
        for track, kind in self._sliders:
            if track.inflate(16, 18).collidepoint(pos):
                self.drag = {"kind": "slider", "which": kind, "track": track}
                self._slide_to(pos[0])
                return
        for b in reversed(self.buttons):        # 後から描いたもの (パネル) が上
            if b.rect.collidepoint(pos):
                if b.enabled:
                    b.action()
                    if b.repeat:     # 押しっぱなし: 0.35秒後から繰り返す (だんだん速く)
                        self.hold = {"label": b.label, "tip": b.tooltip, "next": time.perf_counter() + 0.35, "n": 0}
                return
        if any(r.collidepoint(pos) for r in self._panel_rects()):
            return                              # パネルの中の空いた所: 何もしない
        now_c = time.perf_counter()
        lt, lp = self._last_click
        double = now_c - lt < 0.4 and abs(pos[0] - lp[0]) <= 6 and abs(pos[1] - lp[1]) <= 6
        self._last_click = (now_c, pos)
        flag = self._loop_flag_at(pos)
        if flag:                                # A / B の旗をつかんで動かす
            self.drag = {"kind": "loop" + flag, "moved": False}
            return
        if self.bar_mode() and self._ruler_rect().collidepoint(pos):
            if double:                          # 目盛りをダブルクリック: ここを小節の頭に
                self.set_downbeat_here(self._time_at_x(pos[0]))
                self._last_click = (0.0, (-99, -99))
                return
            if self._shift_held():              # Shift+ドラッグ = 小節線を移動
                self.drag = {"kind": "ruler", "x0": pos[0], "shift0": self.beats.shift, "moved": False}
            else:                               # ドラッグ = 横の拡大・縮小 (右へ = 拡大)
                self.drag = {"kind": "zoom", "x0": pos[0], "px0": PX_PER_SEC, "moved": False}
            return
        if in_score and pos[0] < L.left:
            if self._shift_held():              # Shift+ドラッグ = 音階を上下に動かす
                self.drag = {"kind": "pitch", "y0": pos[1], "c0": self.v_center, "moved": False}
            else:                               # ドラッグ = 縦の拡大・縮小 (上へ = 拡大)
                self.drag = {"kind": "vzoom", "y0": pos[1], "s0": self.view["semis"], "moved": False}
            return
        if self._progress_rect().collidepoint(pos) and self.pb.duration > 0:
            self.drag = {"kind": "progress"}
            self._seek_to(pos[0] / W * self.pb.duration, toast=False)
            return
        if self.edit_mode and self._lane_rect().collidepoint(pos):
            i = self._note_at_pos(pos)
            if i is not None:                   # バーを選ぶ (Shift で追加・外す) → 上下にドラッグで音程
                if self._shift_held():
                    self.sel ^= {i}
                elif i not in self.sel:
                    self.sel = {i}
                self.drag = {"kind": "nmove", "y0": pos[1], "moved": False, "d": 0,
                             "orig": {j: self.guide.base_notes[j]["midi"] for j in self.sel}}
            else:                               # 何もない所: 四角で囲んで選ぶ
                if not self._shift_held():
                    self.sel = set()
                self.drag = {"kind": "rband", "x0": pos[0], "y0": pos[1], "moved": False}
            return
        if self._shift_held() and self._lane_rect().collidepoint(pos):
            # Shift+ドラッグ: ループする範囲を選ぶ
            self.drag = {"kind": "range", "x0": pos[0], "t0": self._time_at_x(pos[0]), "moved": False}
            return
        if self._lane_rect().collidepoint(pos):
            # 譜面をつかんで左右に動かす (離したときにその位置へ移動)。動かさずに離したらその時刻へジャンプ
            self.drag = {"kind": "lane", "x0": pos[0], "t0": self.pb.position(), "moved": False,
                         "was_playing": self.pb.playing}
            if self.pb.playing:
                self.pb.toggle_play()

    def handle_mouse_up(self, pos, button=1):
        if getattr(self, "hold", None):
            self.hold = None
        if self._monitor_dirty and self.on_monitor_change:
            self._monitor_dirty = False
            self.on_monitor_change()            # ボタンを離したときに設定を保存
        if self._vocal_dirty:
            self._vocal_dirty = False
            self.pb.set_vocal_gain(self.pb.vocal_gain)           # 離したときに 100% を超える分を反映
            if self.on_vocal_gain:
                self.on_vocal_gain(self.pb.vocal_gain)
        d, self.drag = self.drag, None
        if self._view_dirty:
            self._view_dirty = False
            if self.on_view_change:
                self.on_view_change()
        if not d:
            return
        if self.lyric_mouse_up(d):
            return
        if d["kind"] in ("loopA", "loopB"):
            if d["moved"]:
                self.loop_on = self.loop_a is not None and self.loop_b is not None
                self.save_practice()
                self.show_toast(f"A-B：{fmt_time(self.loop_a or 0)} 〜 {fmt_time(self.loop_b or 0)}", 2.0)
            return
        if d["kind"] == "range":
            if d["moved"] and d.get("t1") is not None:
                a, b = sorted((d["t0"], d["t1"]))
                a, b = self._snap(a), self._snap(b)
                if b - a >= 0.3:
                    self.set_loop(a, b)
                    if not (a <= self.pb.position() < b):
                        self._seek_to(a, toast=False)
            return
        if d["kind"] == "nmove":
            if d["moved"]:
                self.edit_dirty = True
            return
        if d["kind"] == "rband":
            if d["moved"]:
                r = self._rband_rect(d, pos)
                hit = {i for rr, i in self._note_rects if rr.colliderect(r)} if hasattr(r, "colliderect") else \
                    {i for rr, i in self._note_rects if _rects_overlap(rr, r)}
                self.sel |= hit
                self.show_toast(f"{len(self.sel)}本 選んでいます（Delete で削除 / ↑↓ で音程）", 2.0)
            return
        if d["kind"] in ("zoom", "vzoom"):
            pass                                # 保存は上の _view_dirty で
        elif d["kind"] == "ruler":
            if d["moved"]:
                self.beats.save()
                self.show_toast(f"小節線を移動しました（{self.beats.shift:+.2f}秒）", 1.6)
        elif d["kind"] == "pitch":
            pass
        elif d["kind"] == "slider":
            if (d["which"] == "monitor" or d["which"].startswith("n_")) and self.on_monitor_change:
                self.on_monitor_change()        # 離したときに設定を保存
        elif d["kind"] == "progress":
            self.show_toast(f"移動  →  {fmt_time(self.pb.position())}")
        elif d["kind"] == "lane":
            L = self.layout
            if d["moved"]:
                t = d["t0"] - (pos[0] - d["x0"]) / PX_PER_SEC
            else:
                t = d["t0"] + (pos[0] - L.playhead_x) / PX_PER_SEC   # クリックした時刻へ
            self._seek_to(t)
            if d["was_playing"]:
                self.pb.toggle_play()

    def handle_mouse_move(self, pos):
        self.mouse = pos
        self.menu.on_mouse_move(pos)
        if self.popup is not None:
            self.popup.on_mouse_move(pos)
        if self.lyric_mouse_move(pos):          # 歌詞の文字をドラッグ中
            return
        d = self.drag
        if d and d["kind"] in ("loopA", "loopB"):
            d["moved"] = True
            t = self._time_at_x(pos[0])
            t = t if self._shift_held() else self._snap(t)
            if d["kind"] == "loopA":
                self.loop_a = min(t, (self.loop_b - 0.2) if self.loop_b is not None else t)
            else:
                self.loop_b = max(t, (self.loop_a + 0.2) if self.loop_a is not None else t)
            return
        if d and d["kind"] == "range":
            if abs(pos[0] - d["x0"]) > 4:
                d["moved"] = True
                d["t1"] = self._time_at_x(pos[0])
            return
        if d and d["kind"] == "nmove":
            dd = round((d["y0"] - pos[1]) / max(1.0, self.layout.row_h))
            if dd != d["d"]:
                if not d["moved"]:
                    self._push_undo()
                    d["moved"] = True
                for j, m0 in d["orig"].items():
                    if j < len(self.guide.base_notes):
                        nt = dict(self.guide.base_notes[j])
                        nt["midi"] = int(min(108, max(24, m0 + dd)))
                        self.guide.base_notes[j] = nt
                self.guide._apply_offset()
                d["d"] = dd
            return
        if d and d["kind"] == "rband":
            if abs(pos[0] - d["x0"]) + abs(pos[1] - d["y0"]) > 4:
                d["moved"] = True
            return
        if d and d["kind"] == "vzoom":
            if abs(pos[1] - d["y0"]) > 2:
                d["moved"] = True
            if d["moved"]:
                n = round(d["s0"] * 2 ** ((pos[1] - d["y0"]) / 160.0))
                if n != self.view["semis"]:
                    self.set_semis(n)
        elif d and d["kind"] == "zoom":
            if abs(pos[0] - d["x0"]) > 2:
                d["moved"] = True
            if d["moved"]:
                self.set_pxsec(d["px0"] * 2 ** ((pos[0] - d["x0"]) / 160.0))
        elif d and d["kind"] == "ruler":
            if abs(pos[0] - d["x0"]) > 2:
                d["moved"] = True
            if d["moved"]:
                self.beats.set_shift(d["shift0"] + (pos[0] - d["x0"]) / PX_PER_SEC)
                if self.view["vmode"] != "bar":
                    self.view["vmode"] = "bar"
        elif d and d["kind"] == "pitch":
            dy = pos[1] - d["y0"]
            if abs(dy) > 2 and not d["moved"]:
                d["moved"] = True
                if self.view["follow"]:
                    self.view["follow"] = False
                    self._view_changed("音階の自動の上下：OFF（C キーで戻せます）")
            if d["moved"]:
                self.v_center = min(108, max(24, d["c0"] + dy / max(1.0, self.layout.row_h)))
        elif d and d["kind"] == "slider":
            self._slide_to(pos[0])
        elif d and d["kind"] == "progress" and self.pb.duration > 0:
            self._seek_to(min(max(pos[0], 0), W) / W * self.pb.duration, toast=False)
        elif d and d["kind"] == "lane" and abs(pos[0] - d["x0"]) > 4:
            d["moved"] = True

    def handle_wheel(self, dy, ctrl=False, shift=False):
        """ホイール: 1秒ずつ移動 / Ctrl+ホイール: 横の幅 / Shift+ホイール: 縦の間隔 / 左の音名の上: 上下に移動。"""
        if self.menu.is_open or self.popup is not None:
            return
        for b in reversed(self.buttons):                  # 下の % 表示や ± ボタンの上でもホイールで 1% ずつ
            if b.rect.collidepoint(self.mouse) and b.wheel:
                b.wheel(dy)
                if "自分の声" in b.tooltip and self.on_monitor_change:
                    self.on_monitor_change()        # 保存
                return
        for track, kind in self._sliders:       # スライダーの上ではホイールで 1% ずつ
            if track.inflate(16, 18).collidepoint(self.mouse):
                self._slider_wheel(kind, dy)
                return
        if any(r.collidepoint(self.mouse) for r in self._panel_rects()):
            return
        L = self.layout
        up = 1 if dy > 0 else -1
        if self.bar_mode() and self._ruler_rect().collidepoint(self.mouse) and not ctrl:
            if shift:
                self.step_bpm(0.5 * up)             # 目盛りの上で Shift+ホイール: テンポ
            else:
                self.zoom(1.15 if dy > 0 else 1 / 1.15)  # 目盛りの上でホイール: 横の拡大・縮小
            return
        if self.mouse[0] < L.left and L.top <= self.mouse[1] <= L.bottom and not ctrl:
            if shift:
                self.scroll_pitch(up)                       # 左の音名の上で Shift+ホイール: 上下に動かす
            else:
                self.vzoom(1.12 if dy > 0 else 1 / 1.12)    # 左の音名の上でホイール: 縦の拡大・縮小
            return
        if ctrl:
            self.zoom(1.25 if dy > 0 else 0.8)
        elif shift:
            self.vzoom(1.12 if dy > 0 else 1 / 1.12)
        elif self._lane_rect().collidepoint(self.mouse) or self._progress_rect().collidepoint(self.mouse):
            self._seek_to(self.pb.position() - dy * 1.0)

    def cursor_kind(self):
        """マウスカーソルの形: "hand" / "drag" / "arrow"。"""
        pos = self.mouse
        if self.drag:
            return "hand" if self.drag["kind"] == "slider" else "zoom" if self.drag["kind"] == "zoom" else \
                "vdrag" if self.drag["kind"] in ("pitch", "vzoom", "nmove") else \
                "arrow" if self.drag["kind"] == "rband" else "drag"
        if self.popup is not None:
            return "hand" if self.popup.is_clickable(pos) else "arrow"
        if any(t.inflate(16, 18).collidepoint(pos) for t, _ in self._sliders):
            return "hand"
        if self.menu.is_clickable(pos) or any(b.rect.collidepoint(pos) and b.enabled for b in self.buttons):
            return "hand"
        if self._progress_rect().collidepoint(pos):
            return "hand"
        if any(r.collidepoint(pos) for r in self._panel_rects()):
            return "arrow"
        L = self.layout
        if not self.menu.is_open and L.top <= pos[1] <= L.bottom and 0 <= pos[0] < L.left:
            return "vdrag"                      # 左の音名: つかんで上下
        if self._loop_flag_at(pos) and not self.menu.is_open:
            return "zoom"
        if self.bar_mode() and self._ruler_rect().collidepoint(pos) and not self.menu.is_open:
            return "zoom"
        if self.edit_mode and self._lane_rect().collidepoint(pos) and not self.menu.is_open:
            return "hand" if self._note_at_pos(pos) is not None else "arrow"
        if self._lane_rect().collidepoint(pos) and not self.menu.is_open:
            return "drag"
        return "arrow"

    def handle_click(self, pos):
        """互換用: クリック1回 (押して離す)。"""
        self.handle_mouse_down(pos)
        self.handle_mouse_up(pos)

    def _review_points(self, t0, t1):
        pts = (self.reviewing or {}).get("points") or []
        i = bisect.bisect_left(pts, (t0, -1.0))
        out = []
        while i < len(pts) and pts[i][0] <= t1:
            out.append(pts[i])
            i += 1
        return out

    def _user_segments(self, t0, t1, pts=None):
        """自分の声の点の列 → [(開始, 終了, 音 (整数), お手本と合っているか)]"""
        segs = []
        gap = 0.09 if pts is None else 0.06
        for t, m in (self.mic.trail(t0, t1) if pts is None else pts):
            note = int(round(m))
            tg = self.target_midi(t)
            hit = tg is not None and abs(m - tg) <= 0.5
            if segs and segs[-1][2] == note and segs[-1][3] == hit and t - segs[-1][1] < gap:
                segs[-1][1] = t + 0.025
            else:
                segs.append([t, t + 0.025, note, hit])
        return segs

    def target_midi(self, t):
        n = self.guide.note_at(t)
        return None if n is None else n["midi"] + self.gshift

    # ---------- 描画 ----------
    def draw(self):
        self._repeat_hold()
        if getattr(self.mic, "howl_stopped", False):     # ハウリングで自分の声を止めたとき
            self.mic.howl_stopped = False
            self.show_toast("ハウリングしたので自分の声を OFF にしました。ヘッドホンを使うか、音量を下げてから ON にしてください", 5.0)
            if self.on_monitor_change:
                self.on_monitor_change()
        pg, scr, F = self.pg, self.screen, self.fonts
        if self.pb.poll_fx():                  # 速度・キーの準備ができた → 切り替え
            self._fx_applied()
        self.mic.speed = getattr(self.pb, "speed", 1.0)
        self._loop_tick(self.pb.position())
        now = self.pb.position()
        self._take_tick(now)
        d = self.drag
        if d and d["kind"] == "lane" and d["moved"]:
            now = max(0.0, d["t0"] - (self.mouse[0] - d["x0"]) / PX_PER_SEC)   # つかんで動かしている位置
        self._follow(now)
        L = self._update_layout()
        t0, t1 = L.time_window(now)
        scr.blit(self.background, (0, 0))
        lab_hot = (self._labels_rect().collidepoint(self.mouse) and not self.menu.is_open and self.popup is None) or \
            (self.drag and self.drag["kind"] in ("pitch", "vzoom"))
        if lab_hot:                             # 左の音名: つかめることが分かるように少し明るく
            pg.draw.rect(scr, PANEL, self._labels_rect())
        self._draw_rows(scr, L)
        lane = pg.Rect(L.left, L.top, L.right - L.left, L.bottom - L.top)
        scr.set_clip(lane)
        if self._vdot_key != (L.bottom - L.top, self.view["line"]):     # タテの点線の絵 (濃さ・高さが変わったら作り直す)
            vd = pg.Surface((1, max(1, L.bottom - L.top)), pg.SRCALPHA)
            for y in range(0, L.bottom - L.top, 6):
                pg.draw.line(vd, self._lc(HAIR_DIM) + (255,), (0, y), (0, y + 1), 1)
            self._vdot, self._vdot_key = vd, (L.bottom - L.top, self.view["line"])

        self._draw_loop_band(L, now)
        bar_lines = self.beats.lines(t0, t1) if self.bar_mode() else None
        if bar_lines is not None:
            # 小節線 (実線) と 拍の点線
            if self.view["line"] > 0:
                gap = PX_PER_SEC * 60.0 / max(1.0, self.beats.bpm)
                for tt, is_bar, bar, beat in bar_lines:
                    x = round(L.x_of(tt, now))
                    if is_bar:
                        pg.draw.line(scr, self._lc(HAIR_STRONG), (x, L.top), (x, L.bottom), 1)
                    elif self.view["beat_lines"] and gap >= 8:
                        scr.blit(self._vdot, (x, L.top))
        else:
            # 秒ごとのタテ線
            iv = self.view["vgrid"]
            if self.view["line"] > 0:
                for k in range(math.floor(t0 / iv), math.ceil(t1 / iv) + 1):
                    tt = k * iv
                    x = round(L.x_of(tt, now))
                    major = abs(tt / 5 - round(tt / 5)) < 1e-6          # 5秒ごとは実線 + 時刻
                    if major:
                        pg.draw.line(scr, self._lc(HAIR), (x, L.top), (x, L.bottom), 1)
                    else:
                        scr.blit(self._vdot, (x, L.top))
            for s5 in range(math.floor(t0 / 5) * 5, int(t1) + 5, 5):        # 時刻の数字は線の濃さに関係なく表示
                if s5 >= 0:
                    scr.blit(self._label(("time", s5), fmt_time(s5), 11, FAINT), (round(L.x_of(s5, now)) + 4, L.top + 3))

        # 今の声と、今歌うべき音
        cur, level = self.mic.snapshot()
        if self.reviewing is not None:          # テイクを聴いているとき: 録音した声の高さを使う
            near = self._review_points(now - 0.06, now + 0.01)
            cur = near[-1][1] if near else None
        target = self.target_midi(now)
        col_now, cents = judge(cur, target)
        matching = cur is not None and target is not None and abs(cents) <= 50
        cur_idx = self.guide.index_at(now)
        gkey = id(self.guide)
        if matching and self.pb.playing and cur_idx >= 0:      # 合っている部分を記録 (あとで光らせる)
            k = (gkey, self.guide.notes[cur_idx]["start"])
            segs = self.hits.setdefault(k, [])
            if segs and now - segs[-1][1] < 0.12:
                segs[-1][1] = now
            else:
                segs.append([now, now + 0.02])

        # お手本バー (水色の単色)
        shift = self.gshift
        bar_h = max(5, L.row_h * 0.56)
        small = int(max(9, min(13, bar_h * 0.9)))
        rad = min(4, int(bar_h / 2))
        self._note_rects = []
        for idx, n in self.guide.visible_idx(t0, t1):
            x0, x1 = L.x_of(n["start"], now), L.x_of(n["end"], now)
            m = n["midi"] + shift
            yc = L.y_of(m)
            rect = pg.Rect(round(x0), round(yc - bar_h / 2), max(3, round(x1 - x0)), round(bar_h))
            current = n["start"] <= now < n["end"]
            glowing = current and matching
            if glowing:                                    # 今合っている → まわりを光らせる
                for k_, alpha in ((12, 26), (7, 48), (3, 80)):
                    g = rect.inflate(k_ * 2, k_ * 2)
                    surf = pg.Surface((g.width, g.height), pg.SRCALPHA)
                    pg.draw.rect(surf, GLOW + (alpha,), (0, 0, g.width, g.height), border_radius=rad + k_)
                    scr.blit(surf, (g.x, g.y))
            col = BAR_PAST if n["end"] < now else BAR
            pg.draw.rect(scr, BAR_HIT if glowing else col, rect, border_radius=rad)
            # 合っていた部分は明るいまま残す
            for h0, h1 in self.hits.get((gkey, n["start"]), ()):
                hx0, hx1 = max(rect.x, round(L.x_of(h0, now))), min(rect.right, round(L.x_of(h1, now)))
                if hx1 > hx0:
                    pg.draw.rect(scr, BAR_HIT, (hx0, rect.y, hx1 - hx0, rect.height), border_radius=min(rad, 4))
            if current and not glowing:                    # 今のバー: 下に細い線 (フラットに目印だけ)
                pg.draw.line(scr, WHITE, (rect.x, rect.bottom + 2), (rect.right - 1, rect.bottom + 2), 2)
            if rect.width > 34 and bar_h >= 10:
                lab = F.render(midi_to_doremi(m, octave=False), small, (12, 30, 40), bold=True)
                scr.blit(lab, (rect.x + 5, rect.centery - lab.get_height() // 2))
            if self.edit_mode:
                self._note_rects.append((rect, idx))
                if idx in self.sel:                        # 選んでいるバー: 白い枠
                    pg.draw.rect(scr, WHITE, rect.inflate(6, 6), 2, border_radius=rad + 3)

        # お手本の細い音程線 (元の歌い方の揺れ)
        if self.show_fine:
            pts, seg = [], []
            for t, m in self.guide.pitch_points(t0, t1):
                if m is None:
                    if len(seg) > 1:
                        pts.append(seg)
                    seg = []
                else:
                    seg.append((L.x_of(t, now), L.y_of(m + shift)))
            if len(seg) > 1:
                pts.append(seg)
            for s_ in pts:
                pg.draw.lines(scr, (150, 176, 206), False, s_, 1)

        # 自分の声のバー (リアルタイム): 同じ高さが続く間を1本のバーにする
        ubar_h = max(4, bar_h * 0.5)
        rv = self.reviewing is not None
        segs = (self._user_segments(t0, t1, self._review_points(t0, t1)) if rv else
                self._user_segments(now - TRAIL_SEC, now + 0.05))
        for seg in segs:
            t_a, t_b, note, hit = seg
            xa, xb = L.x_of(t_a, now), (L.x_of(t_b, now) if rv else min(L.x_of(t_b, now), L.playhead_x))
            r = pg.Rect(round(xa), round(L.y_of(note) - ubar_h / 2), max(3, round(xb - xa)), round(ubar_h))
            pg.draw.rect(scr, USER_HIT if hit else USER, r, border_radius=int(ubar_h / 2))

        # 現在位置の線
        pg.draw.line(scr, (200, 212, 236), (round(L.playhead_x), L.top), (round(L.playhead_x), L.bottom), 1)
        if bar_lines is not None:
            self._draw_ruler(bar_lines, now)
        self._draw_loop_flags(L, now)
        self._draw_edit_overlay(L, now)

        # 今の声 ●
        col = col_now
        if cur is not None:
            y = L.y_of(cur)
            y_clamped = min(max(y, L.top + 12), L.bottom - 12)
            pg.draw.circle(scr, BG, (round(L.playhead_x), round(y_clamped)), 11)
            pg.draw.circle(scr, col, (round(L.playhead_x), round(y_clamped)), 8)
            if y != y_clamped:  # 画面の外: 矢印で方向を示す
                d = -1 if y < y_clamped else 1
                cx, cy = round(L.playhead_x), round(y_clamped + d * 20)
                pg.draw.polygon(scr, col, [(cx - 7, cy - d * 4), (cx + 7, cy - d * 4), (cx, cy + d * 6)])
            if self.pb.playing and target is not None and self.reviewing is None:
                self.score_total += 1
                self.score_hit += abs(cents) <= 50
        scr.set_clip(None)
        self._draw_lyrics(L, now, bar_lines)    # 楽譜の下の歌詞の段
        if cur is not None and L.lo <= round(cur) <= L.hi:     # 左の音名: 今歌っている音を強調
            m = round(cur)
            yc = round(L.y_of(m))
            hh = max(14, round(L.row_h * 0.9))
            pill = pg.Rect(4, yc - hh // 2, L.left - 8, hh)
            pg.draw.rect(scr, USER, pill, border_radius=hh // 2)
            t_ = F.render(midi_to_doremi(m, octave=False), self.label_size or 14, (34, 18, 6), bold=True)
            e_ = F.render(midi_to_english(m), max(9, (self.label_size or 14) - 7), (90, 48, 16))
            scr.blit(t_, (pill.right - t_.get_width() - 8, yc - t_.get_height() // 2))
            scr.blit(e_, (pill.right - t_.get_width() - e_.get_width() - 14, yc - e_.get_height() // 2 + 1))

        self._draw_header(now, cur, cents, target, level)
        self._draw_footer()
        if not self.guide.notes:
            self._draw_no_vocal_notice()
        self._sliders = []
        if self.show_mixer:
            self._draw_mixer()
        if self.show_noise_panel:
            self._draw_noise_panel()
        if self.show_grid_panel:
            self._draw_grid_panel()
        self._draw_toast()
        self.menu.draw(scr, self.mouse)
        if self.popup is not None:
            self.popup.draw(scr)
        hover = next((b for b in reversed(self.buttons) if b.rect.collidepoint(self.mouse)), None)
        quiet = self.menu.is_open or self.popup is not None
        if hover and hover.tooltip and not quiet and not self.drag:
            draw_tooltip(pg, scr, F, hover.tooltip, hover.rect.midtop)
        elif not quiet and not self.drag and self._labels_rect().collidepoint(self.mouse):
            draw_tooltip(pg, scr, F, "上下にドラッグ / ホイール：縦の拡大・縮小　Shift+ドラッグ：上下に移動　右クリック：メニュー",
                         (self.mouse[0] + 260, self.mouse[1] - 4))
        elif (bar_lines is not None and not quiet and not self.drag and self._ruler_rect().collidepoint(self.mouse)
              and not any(r.collidepoint(self.mouse) for r in self._panel_rects())):
            draw_tooltip(pg, scr, F, "左右にドラッグ / ホイール：拡大・縮小　ダブルクリック：ここを1拍目に　Shift+ドラッグ：小節線を移動　右クリック：メニュー",
                         (self.mouse[0], self._ruler_rect().bottom + 34))
        elif (not quiet and not self.drag and self._lyric_rects and self._lyric_lane_rect().collidepoint(self.mouse)
              and self.mouse[0] > self.layout.left):
            draw_tooltip(pg, scr, F, "文字をドラッグ：タイミング　Shift+ドラッグ：行の後ろも一緒に　ダブルクリック：ふりがな　右クリック：メニュー",
                         (self.mouse[0], self._lyric_lane_rect().y - 10))
        elif self._progress_rect().collidepoint(self.mouse) and not self.menu.is_open and self.pb.duration:
            draw_tooltip(pg, scr, F, fmt_time(self.mouse[0] / W * self.pb.duration), (self.mouse[0], HEADER_H - 2))
        if self.show_help:
            self._draw_help()
        if UPDATE.notify and self._notified != UPDATE.notify:   # 起動時の確認で新しい版が見つかった
            self._notified = UPDATE.notify
            self.show_toast(f"新しい版 {UPDATE.notify} があります　→　ヘルプ → アップデートを確認", 6.0)
        if UPDATE.show and UPDATE.state:
            self._draw_update_dialog()

    def _draw_loop_band(self, L, now):
        """A-B の範囲をうっすら色付け (ループ ON のときは濃く)。範囲を選んでいる途中も表示。"""
        pg, scr = self.pg, self.screen
        d = self.drag
        a, b = self.loop_a, self.loop_b
        live = d and d["kind"] == "range" and d.get("t1") is not None
        if live:
            a, b = sorted((self._snap(d["t0"]), self._snap(d["t1"])))
        if a is None:
            return
        xa = round(L.x_of(a, now))
        if b is not None:
            xb = round(L.x_of(b, now))
            x0, x1 = max(L.left, xa), min(L.right, xb)
            if x1 > x0:
                band = pg.Surface((x1 - x0, L.bottom - L.top), pg.SRCALPHA)
                band.fill(ACCENT + ((30 if (self.loop_on or live) else 12),))
                scr.blit(band, (x0, L.top))
            if L.left <= xb <= L.right:
                pg.draw.line(scr, ACCENT, (xb, L.top), (xb, L.bottom), 2)
        if L.left <= xa <= L.right:
            pg.draw.line(scr, ACCENT, (xa, L.top), (xa, L.bottom), 2)

    def _draw_loop_flags(self, L, now):
        """A / B の旗 (つかんで動かせる)。"""
        pg, scr, F = self.pg, self.screen, self.fonts
        for name, t in (("A", self.loop_a), ("B", self.loop_b)):
            if t is None:
                continue
            x = round(L.x_of(t, now))
            if not (L.left - 10 <= x <= L.right + 10):
                continue
            hot = self._loop_flag_at(self.mouse) == name or (self.drag and self.drag["kind"] == "loop" + name)
            r = pg.Rect(x - (22 if name == "B" else 0), L.top, 22, 18)
            pg.draw.rect(scr, WHITE if hot else ACCENT, r, border_radius=3)
            s = F.render(name, 13, BG_DEEP, bold=True)
            scr.blit(s, (r.centerx - s.get_width() // 2, r.centery - s.get_height() // 2))
        if self.loop_on and self.loop_count:
            s = F.render(f"ループ {self.loop_count + 1}回目", 12, ACCENT, bold=True)
            scr.blit(s, (L.right - s.get_width() - 10, L.top + RULER_H + 4))

    def _draw_edit_overlay(self, L, now):
        pg, scr, F = self.pg, self.screen, self.fonts
        d = self.drag
        if d and d["kind"] == "rband" and d["moved"]:
            r = self._rband_rect(d, self.mouse)
            sf = pg.Surface((max(1, r.width), max(1, r.height)), pg.SRCALPHA)
            sf.fill(WHITE + (24,))
            scr.blit(sf, r)
            pg.draw.rect(scr, WHITE, r, 1)
        if not self.edit_mode:
            return
        msg = (f"編集モード　選択 {len(self.sel)}本　｜　クリック / 囲んで選択・Delete 削除・↑↓ 半音・Shift+↑↓ オクターブ・"
               f"上下ドラッグ・Ctrl+Z 元に戻す・Ctrl+S 保存　｜　E で終了")
        s = F.render(msg, 12, BG_DEEP, bold=True)
        r = pg.Rect(L.left, L.bottom - 22, L.right - L.left, 22)
        pg.draw.rect(scr, NEAR, r)
        scr.blit(s, (r.x + 10, r.y + 4))
        if self.edit_dirty:
            u = F.render("● 未保存", 12, BAD, bold=True)
            scr.blit(u, (r.right - u.get_width() - 10, r.y + 4))

    def _draw_ruler(self, bar_lines, now):
        """楽譜の上の小節の目盛り (小節番号)。つかんで左右に動かすと小節線が動く。"""
        pg, scr, F, L = self.pg, self.screen, self.fonts, self.layout
        r = self._ruler_rect()
        hot = r.collidepoint(self.mouse) or (self.drag and self.drag["kind"] in ("ruler", "zoom"))
        band = pg.Surface(r.size, pg.SRCALPHA)
        band.fill(BG_DEEP + (235 if hot else 205,))
        scr.blit(band, r)
        cur_bar, _ = self.beats.bar_at(now)
        bar_px = PX_PER_SEC * 60.0 / max(1.0, self.beats.bpm) * self.beats.bpb
        every = 1 if bar_px >= 34 else 2 if bar_px >= 17 else 4
        bars = [(tt, bar) for tt, is_bar, bar, _ in bar_lines if is_bar]
        for i, (tt, bar) in enumerate(bars):              # 今の小節は少し明るく
            if bar == cur_bar:
                x0 = round(L.x_of(tt, now))
                x1 = round(L.x_of(bars[i + 1][0], now)) if i + 1 < len(bars) else x0 + round(bar_px)
                pg.draw.rect(scr, PANEL_HI, (x0, r.y, x1 - x0, r.height))
        for tt, is_bar, bar, beat in bar_lines:
            x = round(L.x_of(tt, now))
            if is_bar:
                pg.draw.line(scr, ACCENT if hot else DIM, (x, r.y + 3), (x, r.bottom - 1), 1)
                if bar >= 1 and (bar - 1) % every == 0:
                    col = WHITE if bar == cur_bar else (LABEL if hot else DIM)
                    scr.blit(self._label(("bar", bar, col), str(bar), 12, col, bar == cur_bar), (x + 4, r.y + 3))
            elif bar_px >= 40:
                pg.draw.line(scr, FAINT, (x, r.bottom - 6), (x, r.bottom - 1), 1)
        pg.draw.line(scr, HAIR_STRONG if hot else HAIR, (r.x, r.bottom), (r.right, r.bottom), 1)
        px = round(L.playhead_x)
        pg.draw.polygon(scr, WHITE, [(px - 5, r.y), (px + 5, r.y), (px, r.y + 6)])

    def _draw_header(self, now, cur, cents, target, level):
        pg, scr, F = self.pg, self.screen, self.fonts
        oy = MENU_H
        pg.draw.rect(scr, BG, (0, oy, W, HEADER_H - oy))
        # 1行目: 曲名 (小さく) と 再生状態
        scr.blit(F.render("♪  " + self.title[:56], 14, DIM), (18, oy + 8))
        playing = self.pb.playing
        st = F.render("● 再生中" if playing else "Ⅱ 一時停止中　Space で再生", 13, GOOD if playing else FAINT, bold=playing)
        scr.blit(st, (W - st.get_width() - 18, oy + 9))

        # 2行目: キー (大きく・太く) + 英語と日本語の名前 (細く)
        x = 18
        if self.key:
            big = F.render(self.key["doremi"], 28, WHITE, bold=True)
            scr.blit(big, (x, oy + 26))
            x += big.get_width() + 14
            sub = F.render(f"{self.key['english']}　{self.key['japanese']}", 15, DIM)
            scr.blit(sub, (x, oy + 30))
            sc = F.render("  ".join(self.key["scale_doremi"]), 12, FAINT)
            scr.blit(sc, (x, oy + 50))
        else:
            msg = "キーを解析できませんでした" if self.song_path else "キー ―　曲を選ぶと表示されます"
            scr.blit(F.render(msg, 22, DIM, bold=True), (x, oy + 30))

        # 3行目: 情報 (ラベルは細く薄く・値は太く)。箱は使わず、細い区切り線だけ
        items = [
            ("時間", f"{fmt_time(now)} / {fmt_time(self.pb.duration)}" if self.pb.duration else fmt_time(now), WHITE),
        ]
        if getattr(self.pb, "fx_busy", False):
            items.append(("準備中", f"{int(self.pb.fx_progress * 100)}%", NEAR))
        if self.rec_armed:
            items.append(("録音", f"● {fmt_time(self.mic.rec_seconds())}" if self._rec_active else "● 待機", BAD))
        if self.reviewing is not None:
            items.append(("再生中", f"テイク {self.reviewing['no']}", GOOD))
        if getattr(self.pb, "speed", 1.0) != 1.0 or self.transpose:
            items.append(("速度・キー", f"{self.pb.speed:.2f}倍 {self.transpose:+d}", ACCENT))
        if self.has_beats():
            bar, beat = self.beats.bar_at(now)
            items.append(("小節", f"{max(0, bar)}・{beat}拍" if bar and bar > 0 else "―", WHITE))
            items.append(("テンポ", f"♩{self.beats.bpm:.0f}  {self.beats.bpb}拍子", ACCENT if self.beats.manual_bpm else WHITE))
        items += [
            ("お手本", self.guide.source.replace("MIDI（", "MIDI ").rstrip("）"),
             ACCENT if self.guide.source.startswith("MIDI") else WHITE),
            ("スキップ", f"{self.skip_sec}秒", WHITE),
            ("ボーカル", ("ON" if self.pb.vocal_on else "OFF") +
             (f" {round(self.pb.vocal_gain * 100)}%" if getattr(self.pb, "vocal_gain", 1.0) != 1.0 else ""),
             GOOD if self.pb.vocal_on else BAD),
            ("自分の声", "ON" if self.mic.monitor else "OFF", FREE if self.mic.monitor else FAINT),
            ("曲の音量", f"{int(round(self.pb.volume * 100))}%", WHITE),
        ]
        if self.octave_shift:
            items.append(("お手本の高さ", f"{self.octave_shift:+d}oct", NEAR))
        if self.score_total > 30:
            items.append(("一致率", f"{self.score_hit / self.score_total * 100:.0f}%", GOOD))
        x, y = 18, oy + 80
        for i, (lab, val, col) in enumerate(items):
            ls = F.render(lab, 12, FAINT)
            vs = F.render(val, 15, col, bold=True)
            w = ls.get_width() + 8 + vs.get_width()
            if x + w > W - 300:                   # 右の「お手本 / あなた」と重ならないように
                break
            if i:
                pg.draw.line(scr, HAIR, (x - 12, y + 3), (x - 12, y + 18), 1)
            scr.blit(ls, (x, y + 4))
            scr.blit(vs, (x + ls.get_width() + 8, y))
            x += w + 24

        # 右側: 「お手本」と「あなた」の音名 (小さなラベル + 大きな値)
        if self.mic.error:
            you, ycol = "マイクなし", BAD
        elif cur is None:
            you, ycol = "―", FAINT
        else:
            you = midi_to_doremi(round(cur)) + (f" {cents:+.0f}¢" if cents is not None else "")
            ycol, _ = judge(cur, target)
        stats = [("お手本", midi_to_doremi(target) if target is not None else "―", WHITE if target is not None else FAINT),
                 ("あなた", you, ycol)]
        right = W - 18
        for lab, val, col in reversed(stats):
            vs = F.render(val, 24, col, bold=True)
            ls = F.render(lab, 12, FAINT)
            w = max(vs.get_width(), ls.get_width(), 92)
            scr.blit(ls, (right - w, oy + 32))
            scr.blit(vs, (right - w, oy + 48))
            right -= w + 26
        pg.draw.line(scr, HAIR, (0, HEADER_H - 1), (W, HEADER_H - 1), 1)

        # 進行バー (細い線。マウスを乗せると太く・つまみ)
        hot = self._progress_rect().collidepoint(self.mouse) or (self.drag and self.drag["kind"] == "progress")
        th = 5 if hot else 3
        ty = HEADER_H + (PROGRESS_H - th) // 2
        pg.draw.rect(scr, HAIR, (0, ty, W, th))
        if self.pb.duration:
            px = int(W * now / self.pb.duration)
            pg.draw.rect(scr, ACCENT, (0, ty, px, th))
            if self.loop_a is not None and self.loop_b is not None:     # A-B の範囲
                xa = int(W * self.loop_a / self.pb.duration)
                xb = int(W * self.loop_b / self.pb.duration)
                pg.draw.rect(scr, GOLD if self.loop_on else DIM, (xa, ty - 2, max(2, xb - xa), th + 4))
            if hot:
                pg.draw.circle(scr, WHITE, (px, HEADER_H + PROGRESS_H // 2), 6)

    def _draw_footer(self):
        """下のツールバー: クリックで操作できるボタン。"""
        pg, scr, F = self.pg, self.screen, self.fonts
        y = H - FOOTER_H
        pg.draw.rect(scr, BG_DEEP, (0, y, W, FOOTER_H))
        pg.draw.line(scr, HAIR, (0, y), (W, y), 1)
        K = self.press
        bh, gap = 36, 6
        ok_mon = self.mic.monitor_error is None
        specs = [
            ("|◀", lambda: K(pg.K_HOME), "最初から (Home)", False, True, None, 46),
            (f"◀◀ {self.skip_sec}秒", lambda: K(pg.K_LEFT), "巻き戻し (←)", False, True, None, 84),
            ("Ⅱ 一時停止" if self.pb.playing else "▶ 再生", lambda: K(pg.K_SPACE), "再生 / 一時停止 (Space)",
             self.pb.playing, True, GOOD if not self.pb.playing else None, 110),
            (f"{self.skip_sec}秒 ▶▶", lambda: K(pg.K_RIGHT), "スキップ (→)", False, True, None, 84),
            ("−", lambda: K(pg.K_DOWN), "スキップ間隔を短く (↓)", False, self.skip_sec > 1, None, 32),
            ("＋", lambda: K(pg.K_UP), "スキップ間隔を長く (↑)", False, self.skip_sec < 60, None, 32),
            None,
            ("ボーカル " + ("ON" if self.pb.vocal_on else "OFF"), lambda: K(pg.K_v), "お手本の声を鳴らす / 消す (V)",
             self.pb.vocal_on, True, GOOD if self.pb.vocal_on else BAD, 112),
            ("−", lambda: self.step_vocal_gain(-1, save=False),
             "ボーカル（お手本の歌声）の音量を 1% 下げる (押しっぱなしで連続)", False, self.pb.vocal_gain > 0,
             GOOD if self.pb.vocal_on else None, 32, "repeat"),
            (f"{round(self.pb.vocal_gain * 100)}%", None, "ボーカル（お手本の歌声）の音量 (ホイールで 1% ずつ)",
             False, True, GOOD if self.pb.vocal_on else None, 54, "static_vocal"),
            ("＋", lambda: self.step_vocal_gain(1, save=False),
             "ボーカル（お手本の歌声）の音量を 1% 上げる (押しっぱなしで連続・最大 200%)", False, self.pb.vocal_gain < 2.0,
             GOOD if self.pb.vocal_on else None, 32, "repeat"),
            None,
            ("自分の声 " + ("ON" if self.mic.monitor else "OFF"), self.toggle_monitor,
             "自分の歌声をスピーカー / ヘッドホンから出す (O)" if not self.mic.monitor_error else "自分の声を出力できません",
             self.mic.monitor, self.mic.monitor_error is None, FREE if self.mic.monitor else None, 112),
            ("音量 −", lambda: self.step_monitor_gain(-1, save=False),
             "自分の声の音量を 1% 下げる (押しっぱなしで連続 / 9 キーで 5%)",
             False, ok_mon and self.mic.monitor_gain > 0, FREE, 70, "repeat"),
            (f"{round(self.mic.monitor_gain * 100)}%", None, "自分の声の音量 (ホイールで 1% ずつ)",
             False, ok_mon, FREE if ok_mon else None, 58, "static_monitor"),
            ("音量 ＋", lambda: self.step_monitor_gain(1, save=False),
             "自分の声の音量を 1% 上げる (押しっぱなしで連続 / 0 キーで 5%)",
             False, ok_mon and self.mic.monitor_gain < MicTracker.GAIN_MAX, FREE, 70, "repeat"),
            None,
            ("曲 −", lambda: self.step_song_volume(-1), "曲の音量を 1% 下げる (押しっぱなしで連続 / - キーで 10%)",
             False, self.pb.volume > 0, None, 56, "repeat"),
            (f"{int(round(self.pb.volume * 100))}%", None, "曲の音量 (ホイールで 1% ずつ)", False, True, None, 54, "static_song"),
            ("曲 ＋", lambda: self.step_song_volume(1), "曲の音量を 1% 上げる (押しっぱなしで連続 / + キーで 10%)",
             False, self.pb.volume < 1, None, 56, "repeat"),
            None,
            ("オクターブ ↓", lambda: K(pg.K_LEFTBRACKET), "お手本を1オクターブ下げる ([)", False, self.octave_shift > -2, None, 104),
            ("オクターブ ↑", lambda: K(pg.K_RIGHTBRACKET), "お手本を1オクターブ上げる (])", False, self.octave_shift < 2, None, 104),
        ]
        if self.alt_guide is not None:
            specs += [None, ("お手本：MIDI" if self.guide.source.startswith("MIDI") else "お手本：歌声",
                             lambda: K(pg.K_m), "お手本を切り替える (M)", True, True, ACCENT, 120)]
        self.buttons = []
        wheel = {"static_monitor": lambda dy: self.step_monitor_gain(1 if dy > 0 else -1),
                 "static_song": lambda dy: self.step_song_volume(1 if dy > 0 else -1),
                 "static_speed": lambda dy: self.step_speed(0.05 if dy > 0 else -0.05),
                 "static_key": lambda dy: self.step_key(1 if dy > 0 else -1),
                 "static_vocal": lambda dy: self.step_vocal_gain(1 if dy > 0 else -1)}
        self._draw_button_row(specs, y, bh, gap, wheel)
        if self.view.get("toolbar2", True):
            pg.draw.line(scr, HAIR_DIM, (0, y + 44), (W, y + 44), 1)
            self._draw_button_row(self._practice_specs(), y + 44, bh, gap, wheel)

    def _practice_specs(self):
        """2段目: 練習ツール (A-B ループ・速度・キー・録音・編集)。"""
        pb = self.pb
        sp, se = getattr(pb, "fx_target", (1.0, 0))
        busy = getattr(pb, "fx_busy", False)
        has_ab = self.loop_a is not None and self.loop_b is not None
        rec_lab = ("● 録音中" if self._rec_active else "● 録音待ち") if self.rec_armed else "● 録音"
        return [
            ("A 点", lambda: self.set_loop_a(), "今の位置を A 点 (ループの始め) に (A キー)", self.loop_a is not None, True,
             ACCENT if self.loop_a is not None else None, 58),
            ("B 点", lambda: self.set_loop_b(), "今の位置を B 点 (ループの終わり) に (B キー)", self.loop_b is not None, True,
             ACCENT if self.loop_b is not None else None, 58),
            ("ループ " + ("ON" if self.loop_on else "OFF"), self.toggle_loop,
             "A-B ループ再生 ON / OFF (L キー)。楽譜を Shift+ドラッグでも範囲を選べます", self.loop_on, has_ab,
             ACCENT if self.loop_on else None, 100),
            ("解除", self.clear_loop, "A-B ループを解除", False, has_ab or self.loop_a is not None, None, 50),
            None,
            ("速度", None, "再生速度 (音程はそのまま)", False, True, None, 44, "static_speed"),
            ("−", lambda: self.step_speed(-0.05), "遅くする 0.05倍 (Shift+←)", False, sp > 0.7 + 1e-6, None, 32),
            (f"{sp:.2f}倍" + ("…" if busy and sp != pb.speed else ""), None, "再生速度 (ホイールでも変えられます)",
             False, True, ACCENT if sp < 1.0 else None, 70, "static_speed"),
            ("＋", lambda: self.step_speed(0.05), "速くする 0.05倍 (Shift+→)", False, sp < 1.0 - 1e-6, None, 32),
            None,
            ("キー", None, "キー (半音)。曲とお手本バーを一緒に上げ下げ", False, True, None, 44, "static_key"),
            ("♭", lambda: self.step_key(-1), "半音下げる (Shift+↓)", False, se > -6, None, 32),
            (f"{se:+d}" + ("…" if busy and se != pb.semis else ""), None, "キー (ホイールでも変えられます)", False, True,
             ACCENT if se else None, 54, "static_key"),
            ("♯", lambda: self.step_key(1), "半音上げる (Shift+↑)", False, se < 6, None, 32),
            None,
            (rec_lab, self.toggle_rec, "録音モード (T キー)。再生している間の歌声を録音し、止めるたびに1テイク",
             self.rec_armed, getattr(self.mic, "error", None) is None, BAD if self.rec_armed else None, 104),
            ("■ 停止" if self.reviewing is not None else "▶ テイク", self.toggle_take_play,
             "最後のテイクを、伴奏と頭出しを合わせて再生 (Y キー)", self.reviewing is not None, True,
             GOOD if self.reviewing is not None else None, 92),
            (f"一覧 ({len(self.takes)})", lambda: self.open_take_menu(self._take_btn_pos), "録ったテイクの一覧・保存・ずれ補正",
             False, True, None, 84),
            None,
            ("編集中" if self.edit_mode else "編集", self.toggle_edit, "お手本バーを直す (E キー)", self.edit_mode, True,
             NEAR if self.edit_mode else None, 90),
        ]

    def _draw_button_row(self, specs, y, bh, gap, wheel):
        pg, scr, F = self.pg, self.screen, self.fonts
        total = sum((s[6] if s else 14) for s in specs) + gap * len(specs)
        scale = min(1.0, (W - 20) / total)
        x = max(10, (W - total * scale) / 2)
        for sp in specs:
            if sp is None:
                x += 14 * scale
                continue
            label, action, tip, active, enabled, accent, bw = sp[:7]
            kind = sp[7] if len(sp) > 7 else ""
            r = pg.Rect(round(x), y + 6, round(bw * scale), bh)
            btn = Button(r, label, action or (lambda: None), tip, active, enabled, accent,
                         repeat=(kind == "repeat"), static=kind.startswith("static"), wheel=wheel.get(kind))
            btn.draw(pg, scr, F, r.collidepoint(self.mouse) and not self.menu.is_open, 15 if scale > 0.85 else 13)
            self.buttons.append(btn)
            if label.startswith("一覧 ("):
                self._take_btn_pos = (r.x, r.y - 4)
            x += (bw + gap) * scale

    def _draw_no_vocal_notice(self):
        """お手本バーが1本も無いときの案内 (曲を選ぶ前 / 歌声を見つけられなかった曲)。"""
        pg, scr, F, L = self.pg, self.screen, self.fonts, self.layout
        if self.song_path is None:
            self._draw_welcome()
            return
        lines = [
            ("この曲では、お手本にする歌声を見つけられませんでした", 20, NEAR, True),
            ("伴奏だけの曲 (インスト / カラオケ音源) か、歌声がとても小さい曲の可能性があります。", 15, WHITE, False),
            ("同じ曲のメロディの MIDI があれば、「ファイル → この曲に MIDI を合わせる」でお手本にできます。", 15, WHITE, False),
            ("お手本が無くても、伴奏を流しながら自分の声 (●) の高さを見て練習できます。", 15, DIM, False),
        ]
        surfs = [F.render(t, sz, c, bold=b) for t, sz, c, b in lines]
        bw = max(sf.get_width() for sf in surfs) + 48
        bh = sum(sf.get_height() + 8 for sf in surfs) + 32
        box = pg.Rect((L.left + L.right - bw) // 2, (L.top + L.bottom - bh) // 2, bw, bh)
        bg = pg.Surface(box.size, pg.SRCALPHA)
        bg.fill(PANEL + (235,))
        scr.blit(bg, box)
        pg.draw.rect(scr, OUTLINE, box, 1, border_radius=10)
        y = box.y + 18
        for sf in surfs:
            scr.blit(sf, (box.x + 24, y))
            y += sf.get_height() + 8

    # ---------- 音量の調整パネル ----------
    SLIDER_MAX = {"monitor": 300, "song": 100}
    # スライダーの範囲 (左端, 右端)。semis は「右へ動かすほど間隔が広い」ように逆向き
    SLIDER_RANGE = {"n_margin": (0, 30), "n_cut": (0, -60), "n_hold": (50, 800), "n_hp": (0, 250), "monitor": (0, 300), "song": (0, 100), "semis": (SEMIS_MAX, SEMIS_MIN), "pxsec": (PXSEC_MIN, PXSEC_MAX), "line": (0, 200)}

    def _mixer_rect(self):
        w, h = self.PANEL_SIZES["mixer"]
        return self.pg.Rect(self._panel_x("mixer"), HEADER_H + PROGRESS_H + 12, w, h)

    def _fit_panels(self, keep):
        """パネルが画面に入りきらないときは、keep 以外の古いものを閉じる。"""
        for name in ("grid", "noise", "mixer"):
            if name == keep:
                continue
            if self._panel_x("grid" if self.show_grid_panel else "noise" if self.show_noise_panel else "mixer") >= LABEL_W + 20:
                return
            setattr(self, {"grid": "show_grid_panel", "noise": "show_noise_panel", "mixer": "show_mixer"}[name], False)

    PANEL_SIZES = {"mixer": (420, 246), "noise": (420, 388), "grid": (420, 386)}

    def _panel_x(self, name):
        """開いているパネルを右から 音量 → 雑音カット → グリッド の順に並べたときの x。"""
        x = W - 16
        for n, shown in (("mixer", self.show_mixer), ("noise", self.show_noise_panel), ("grid", self.show_grid_panel)):
            if not shown and n != name:
                continue
            x -= self.PANEL_SIZES[n][0]
            if n == name:
                return x
            x -= 10
        return x

    def _grid_rect(self):
        w, h = self.PANEL_SIZES["grid"]
        return self.pg.Rect(self._panel_x("grid"), HEADER_H + PROGRESS_H + 12, w, h)

    def _noise_rect(self):
        w, h = self.PANEL_SIZES["noise"]
        return self.pg.Rect(self._panel_x("noise"), HEADER_H + PROGRESS_H + 12, w, h)

    def _panel_rects(self):
        rs = []
        if self.show_mixer:
            rs.append(self._mixer_rect())
        if self.show_noise_panel:
            rs.append(self._noise_rect())
        if self.show_grid_panel:
            rs.append(self._grid_rect())
        return rs

    def _slider_value(self, which):
        return {"monitor": lambda: round(self.mic.monitor_gain * 100), "song": lambda: round(self.pb.volume * 100),
                "semis": lambda: self.view["semis"], "pxsec": lambda: PX_PER_SEC,
                "line": lambda: round(self.view["line"] * 100),
                "n_margin": lambda: round(self.mic.gate_margin), "n_cut": lambda: round(self.mic.gate_cut),
                "n_hold": lambda: round(self.mic.gate_hold_sec * 1000), "n_hp": lambda: round(self.mic.hp_freq)}[which]()

    def _set_slider(self, which, v):
        if which.startswith("n_"):
            name = which[2:]
            self.mic.set_noise_param(name, v / 1000 if name == "hold" else v)
            self.show_toast(self._noise_param_text(name), 1.5)
        elif which == "monitor":
            self.mic.set_monitor_gain(v / 100)
        elif which == "song":
            self.pb.change_volume(v / 100 - self.pb.volume)
        elif which == "semis":
            if round(v) != self.view["semis"]:
                self.set_semis(v)
        elif which == "pxsec":
            if round(v) != PX_PER_SEC:
                self.set_pxsec(v)
        elif which == "line":
            if round(v) != round(self.view["line"] * 100):
                self.set_line(v / 100)

    def _slider_wheel(self, which, dy):
        up = 1 if dy > 0 else -1
        if which.startswith("n_"):
            step = {"n_margin": 1, "n_cut": -3, "n_hold": 50, "n_hp": 10}[which]
            self._set_slider(which, self._slider_value(which) + step * up)
            if self.on_monitor_change and not (self.drag and self.drag.get("kind") == "slider"):
                self.on_monitor_change()
        elif which == "monitor":
            self.step_monitor_gain(up)
        elif which == "song":
            self.step_song_volume(up)
        elif which == "semis":
            self.set_semis(self.view["semis"] - up)          # ホイール↑ = 間隔を広げる
        elif which == "pxsec":
            self.set_pxsec(PX_PER_SEC + 10 * up)
        elif which == "line":
            self.set_line(self.view["line"] + 0.05 * up)

    def _slide_to(self, x):
        d = self.drag
        track, which = d["track"], d["which"]
        frac = min(1.0, max(0.0, (x - track.x) / max(1, track.width)))
        a, b = self.SLIDER_RANGE[which]
        self._set_slider(which, round(a + (b - a) * frac))

    def _draw_mixer(self):
        pg, scr, F = self.pg, self.screen, self.fonts
        box = self._mixer_rect()
        bg = pg.Surface(box.size, pg.SRCALPHA)
        bg.fill(PANEL + (245,))
        scr.blit(bg, box)
        pg.draw.rect(scr, OUTLINE, box, 1, border_radius=10)
        scr.blit(F.render("音量の調整", 16, WHITE, bold=True), (box.x + 16, box.y + 12))
        lt = F.render("自分の声：" + self._latency_text(), 12, DIM)
        scr.blit(lt, (box.right - 50 - lt.get_width(), box.y + 16))
        close = Button(pg.Rect(box.right - 40, box.y + 8, 30, 28), "×", self.toggle_mixer, "閉じる")
        close.draw(pg, scr, F, close.rect.collidepoint(self.mouse), 16)
        self.buttons.append(close)

        rows = [
            ("monitor", "自分の声", round(self.mic.monitor_gain * 100), FREE, self.mic.monitor_error is None),
            ("song", "曲（伴奏・ボーカル）", round(self.pb.volume * 100), GOOD, True),
        ]
        y = box.y + 50
        for which, label, pct, col, enabled in rows:
            mx = self.SLIDER_MAX[which]
            scr.blit(F.render(label, 15, WHITE if enabled else DIM, bold=True), (box.x + 16, y))
            if which == "monitor":
                st = "ON" if self.mic.monitor else "OFF"
                tb = Button(pg.Rect(box.x + 104, y - 3, 58, 26), st, self.toggle_monitor,
                            "自分の声の出力 ON / OFF (O)", self.mic.monitor, enabled, FREE if self.mic.monitor else None)
                tb.draw(pg, scr, F, tb.rect.collidepoint(self.mouse), 13)
                self.buttons.append(tb)
                auto = getattr(self.mic, "monitor_auto", False)
                at = f"自動 +{self.mic.monitor_boost_db:.0f}dB" if auto else "自動 OFF"
                ab = Button(pg.Rect(box.x + 168, y - 3, 104, 26), at, self.toggle_monitor_auto,
                            "マイクの声が小さいとき、自動で聞きやすい大きさまで持ち上げる (おすすめ)", auto, enabled,
                            FREE if auto else None)
                ab.draw(pg, scr, F, ab.rect.collidepoint(self.mouse), 12)
                self.buttons.append(ab)
            v = F.render(f"{pct}%", 18, col if enabled else DIM, bold=True)
            scr.blit(v, (box.right - 16 - v.get_width(), y - 2))
            # − [スライダー] ＋
            yy = y + 34
            step = (lambda w: (lambda: self.step_monitor_gain(w))) if which == "monitor" else (lambda w: (lambda: self.step_song_volume(w)))
            minus = Button(pg.Rect(box.x + 14, yy - 14, 30, 28), "−", step(-1), "1% 下げる (ホイールでも)", False, enabled)
            plus = Button(pg.Rect(box.right - 44, yy - 14, 30, 28), "＋", step(1), "1% 上げる (ホイールでも)", False, enabled)
            for b in (minus, plus):
                b.draw(pg, scr, F, b.rect.collidepoint(self.mouse), 16)
                self.buttons.append(b)
            track = pg.Rect(box.x + 60, yy - 3, box.width - 120, 6)
            pg.draw.rect(scr, HAIR, track, border_radius=3)
            fx = track.x + round(track.width * min(pct, mx) / mx)
            if enabled:
                pg.draw.rect(scr, col, (track.x, track.y, fx - track.x, track.height), border_radius=3)
            if which == "monitor":       # 100% の位置に目印
                m100 = track.x + round(track.width * 100 / mx)
                pg.draw.line(scr, FAINT, (m100, track.y - 6), (m100, track.bottom + 5), 1)
            hot = enabled and (track.inflate(16, 18).collidepoint(self.mouse) or
                               (self.drag and self.drag.get("which") == which))
            pg.draw.circle(scr, WHITE if enabled else DIM, (fx, track.centery), 10 if hot else 8)
            if enabled:
                self._sliders.append((track, which))
            y += 68
        # 雑音を抑える
        mic = self.mic
        lv = getattr(mic, "noise", None)
        if lv is not None:
            scr.blit(F.render("雑音カット", 14, WHITE, bold=True), (box.x + 16, y + 5))
            x = box.x + 104
            for key in ("off", "weak", "mid", "strong"):
                on = lv == key
                b = Button(pg.Rect(x, y, 44, 28), MicTracker.NOISE_NAMES[key], (lambda k=key: self.set_noise(k)),
                           {"off": "雑音を抑えない (マイクの音そのまま)", "weak": "少しだけ抑える (小さな声も通す)",
                            "mid": "ふつう (おすすめ)", "strong": "強く抑える (うるさい場所向け。小さな声は切れることも)"}[key],
                           on, True, FREE if on else None)
                b.draw(pg, scr, F, b.rect.collidepoint(self.mouse), 13)
                self.buttons.append(b)
                x += 48
            if lv == "custom":
                scr.blit(F.render("カスタム", 11, FREE), (x - 2, y - 12))
            b = Button(pg.Rect(x + 8, y, box.right - 16 - (x + 8), 28), "細かく調整…", self.toggle_noise_panel,
                       "雑音カットの細かい調整とメーター (N)", self.show_noise_panel, True,
                       FREE if self.show_noise_panel else None)
            b.draw(pg, scr, F, b.rect.collidepoint(self.mouse), 13)
            self.buttons.append(b)
        scr.blit(F.render("つまみをドラッグ / − ＋ / ホイールで 1% ずつ　　9・0 キーで自分の声 ±5%", 12, DIM),
                 (box.x + 16, box.bottom - 22))

    # ---------- 雑音カットの調整パネル ----------
    def _draw_noise_panel(self):
        pg, scr, F = self.pg, self.screen, self.fonts
        box = self._noise_rect()
        P = self._panel_button
        mic = self.mic
        bg = pg.Surface(box.size, pg.SRCALPHA)
        bg.fill(PANEL + (245,))
        scr.blit(bg, box)
        pg.draw.rect(scr, OUTLINE, box, 1, border_radius=10)
        scr.blit(F.render("雑音カットの調整", 16, WHITE, bold=True), (box.x + 16, box.y + 12))
        scr.blit(F.render("NOISE", 11, FAINT), (box.x + 156, box.y + 17))
        P(pg.Rect(box.right - 40, box.y + 8, 30, 28), "×", self.toggle_noise_panel, "閉じる (N)", size=16)

        # 強さ
        y = box.y + 44
        scr.blit(F.render("強さ", 14, WHITE, bold=True), (box.x + 16, y + 5))
        x = box.x + 76
        tips = {"off": "雑音を抑えない (マイクの音そのまま)", "weak": "少しだけ抑える (小さな声も通す)",
                "mid": "ふつう (おすすめ)", "strong": "強く抑える (うるさい場所向け)"}
        for key in ("off", "weak", "mid", "strong"):
            on = mic.noise == key
            b = Button(pg.Rect(x, y, 50, 28), MicTracker.NOISE_NAMES[key], (lambda k=key: self.set_noise(k)), tips[key],
                       on, True, FREE if on else None)
            b.draw(pg, scr, F, b.rect.collidepoint(self.mouse), 13)
            self.buttons.append(b)
            x += 54
        if mic.noise == "custom":
            c = F.render("カスタム（下で調整中）", 12, FREE, bold=True)
            scr.blit(c, (x + 4, y + 7))
        y += 38

        # メーター: 今の大きさ / 雑音 / しきい値
        lo_db, hi_db = -80.0, 0.0
        meter = pg.Rect(box.x + 16, y + 18, box.width - 32, 12)
        fx = lambda d: meter.x + round(meter.width * (min(hi_db, max(lo_db, d)) - lo_db) / (hi_db - lo_db))
        pg.draw.rect(scr, BG_DEEP, meter, border_radius=4)
        cur = getattr(mic, "in_db", -120.0)
        op = mic.gate_open if mic.noise_on else True
        pg.draw.rect(scr, GOOD if op else FAINT, (meter.x, meter.y, fx(cur) - meter.x, meter.height), border_radius=4)
        if mic.noise_on:
            nf = mic.noise_floor_db
            thr = nf + mic.gate_margin
            pg.draw.line(scr, WHITE, (fx(nf), meter.y - 3), (fx(nf), meter.bottom + 2), 2)
            pg.draw.line(scr, NEAR, (fx(thr), meter.y - 5), (fx(thr), meter.bottom + 4), 2)
            st_ = F.render("● 声を通しています" if op else "● 雑音だけ（小さくしています）", 12, GOOD if op else DIM, bold=op)
            scr.blit(st_, (box.x + 16, y))
            info = F.render(f"今 {max(-99, cur):.0f}dB　雑音 {nf:.0f}dB（白）　声とみなす {thr:.0f}dB〜（黄）", 11, FAINT)
            scr.blit(info, (box.x + 16, meter.bottom + 4))
        else:
            scr.blit(F.render("雑音カット OFF（マイクの音そのまま）", 12, DIM), (box.x + 16, y))
            scr.blit(F.render(f"今 {max(-99, cur):.0f}dB", 11, FAINT), (box.x + 16, meter.bottom + 4))
        y += 54

        # 細かい調整
        en = mic.noise_on
        rows = [("n_margin", "声とみなす大きさ", f"雑音 +{mic.gate_margin:.0f}dB", "大きいほど雑音を通しにくい"),
                ("n_cut", "雑音を下げる量", f"{mic.gate_cut:.0f}dB", "右ほど静かに"),
                ("n_hold", "余韻（語尾）", f"{mic.gate_hold_sec:.2f}秒", "語尾が切れるときは長く"),
                ("n_hp", "低い音を切る", f"{mic.hp_freq:.0f}Hz" if mic.hp_freq >= 20 else "OFF", "ブーン・ゴーという音")]
        for which, label, val, hint in rows:
            if en:
                self._grid_slider(box, y, which, label, val, hint)
            else:
                scr.blit(F.render(label, 15, DIM, bold=True), (box.x + 16, y))
            y += 50
        # 今の雑音を測る
        P(pg.Rect(box.x + 16, y + 2, 230, 28), "今の雑音を測る（2秒 静かに）", self.calibrate_noise,
          "歌わずに静かにしている間の音を、雑音として測り直します", enabled=en)
        scr.blit(F.render("ふだんは自動で測っています", 11, FAINT), (box.x + 256, y + 9))

    # ---------- グリッドの調整パネル ----------
    def _grid_slider(self, box, y, which, label, value_text, hint):
        """ラベル + 値 / − [スライダー] ＋ の1行。"""
        pg, scr, F = self.pg, self.screen, self.fonts
        scr.blit(F.render(label, 15, WHITE, bold=True), (box.x + 16, y))
        h = F.render(hint, 11, FAINT)
        v = F.render(value_text, 15, ACCENT, bold=True)
        scr.blit(v, (box.right - 16 - v.get_width(), y))
        scr.blit(h, (box.right - 16 - v.get_width() - 10 - h.get_width(), y + 4))
        yy = y + 34
        minus = Button(pg.Rect(box.x + 14, yy - 14, 30, 28), "−", lambda: self._slider_wheel(which, -1),
                       f"{label}：少し小さく (押しっぱなし・ホイールでも)")
        plus = Button(pg.Rect(box.right - 44, yy - 14, 30, 28), "＋", lambda: self._slider_wheel(which, 1),
                      f"{label}：少し大きく (押しっぱなし・ホイールでも)")
        for b in (minus, plus):
            b.repeat = True
            b.draw(pg, scr, F, b.rect.collidepoint(self.mouse), 16)
            self.buttons.append(b)
        track = pg.Rect(box.x + 60, yy - 3, box.width - 120, 6)
        pg.draw.rect(scr, HAIR, track, border_radius=3)
        a, b_ = self.SLIDER_RANGE[which]
        frac = min(1.0, max(0.0, (self._slider_value(which) - a) / (b_ - a)))
        fx = track.x + round(track.width * frac)
        pg.draw.rect(scr, ACCENT, (track.x, track.y, fx - track.x, track.height), border_radius=3)
        hot = track.inflate(16, 18).collidepoint(self.mouse) or (self.drag and self.drag.get("which") == which)
        pg.draw.circle(scr, WHITE, (fx, track.centery), 10 if hot else 8)
        self._sliders.append((track, which))

    def _panel_button(self, rect, label, action, tip, on=False, enabled=True, size=13):
        b = Button(rect, label, action, tip, on, enabled, ACCENT if on else None)
        b.draw(self.pg, self.screen, self.fonts, b.rect.collidepoint(self.mouse), size)
        self.buttons.append(b)
        return b

    def _draw_grid_panel(self):
        pg, scr, F = self.pg, self.screen, self.fonts
        box = self._grid_rect()
        L = self.layout
        P = self._panel_button
        bg = pg.Surface(box.size, pg.SRCALPHA)
        bg.fill(PANEL + (245,))
        scr.blit(bg, box)
        pg.draw.rect(scr, OUTLINE, box, 1, border_radius=10)
        scr.blit(F.render("グリッドの調整", 16, WHITE, bold=True), (box.x + 16, box.y + 12))
        scr.blit(F.render("GRID", 11, FAINT), (box.x + 136, box.y + 17))
        P(pg.Rect(box.right - 40, box.y + 8, 30, 28), "×", self.toggle_grid_panel, "閉じる (G)", size=16)
        P(pg.Rect(box.right - 132, box.y + 8, 84, 28), "元に戻す", self.reset_view, "グリッドの表示を最初の設定に戻す")

        y = box.y + 40
        hb = self.has_beats()
        self._grid_slider(box, y, "semis", "縦：拡大・縮小", f"{L.row_h:.0f}px/段",
                          f"画面に約{self.view['semis'] / 12:.1f}オクターブ")
        y += 52
        # 画面に入れるオクターブの数
        scr.blit(F.render("オクターブ", 14, WHITE, bold=True), (box.x + 16, y + 5))
        x = box.x + 104
        for n in OCT_CHOICES:
            P(pg.Rect(x, y, 56, 28), f"{n:g}", (lambda n=n: self.set_octaves_on_screen(n)),
              f"画面に {n:g}オクターブ（{round(12 * n)}段）が入るように拡大・縮小", self.view["semis"] == round(12 * n))
            x += 60
        y += 34
        secs = (L.right - L.left) / PX_PER_SEC
        self._grid_slider(box, y, "pxsec", "横：拡大・縮小", f"{PX_PER_SEC}px/秒",
                          f"画面に約{self.bars_on_screen():.1f}小節" if hb else f"画面に約{secs:.0f}秒")
        y += 52
        # 画面に入れる小節の数 (ワンタッチで拡大・縮小)
        scr.blit(F.render("小節の数", 14, WHITE if hb else DIM, bold=True), (box.x + 16, y + 5))
        x = box.x + 104
        cur = self.bars_on_screen() if hb else 0
        for n in BARS_CHOICES:
            P(pg.Rect(x, y, 56, 28), f"{n}小節", (lambda n=n: self.set_bars_on_screen(n)),
              f"画面に {n}小節 が入るように拡大・縮小", hb and abs(cur - n) < 0.05 * n, hb)
            x += 60
        y += 34
        # 線の濃さ
        scr.blit(F.render("線の濃さ", 14, WHITE, bold=True), (box.x + 16, y + 5))
        x = box.x + 104
        for v in LINE_CHOICES:
            P(pg.Rect(x, y, 56, 28), f"{v}%", (lambda v=v: self.set_line(v / 100)),
              "線を消す" if v == 0 else f"線の濃さ {v}%", round(self.view["line"] * 100) == v)
            x += 60
        y += 34

        bar = self.bar_mode()
        # タテ線: 小節 / 秒
        scr.blit(F.render("タテ線", 14, WHITE, bold=True), (box.x + 16, y + 5))
        x = box.x + 84
        P(pg.Rect(x, y, 62, 28), "小節", lambda: self.set_vmode("bar"),
          "小節ごとに線を引く (拍の解析)" if hb else "この曲では拍を見つけられませんでした", bar, hb)
        x += 68
        for s_ in VGRID_CHOICES:
            P(pg.Rect(x, y, 56, 28), f"{s_:g}秒", (lambda s_=s_: self.set_vgrid(s_)), f"{s_:g}秒ごとに線を引く",
              not bar and self.view["vgrid"] == s_)
            x += 60
        y += 34
        # 拍子と 1拍目
        scr.blit(F.render("拍子", 14, WHITE if hb else DIM, bold=True), (box.x + 16, y + 5))
        x = box.x + 84
        for n in BPB_CHOICES:
            P(pg.Rect(x, y, 38, 28), str(n), (lambda n=n: self.set_bpb(n)), f"{n}拍子 (1小節 = {n}拍)",
              hb and self.beats.bpb == n, hb)
            x += 42
        scr.blit(F.render("1拍目", 13, DIM), (x + 6, y + 6))
        x += 52
        P(pg.Rect(x, y, 40, 28), "◀", lambda: self.move_downbeat(-1), "小節の頭を 1拍 前へ", enabled=hb)
        P(pg.Rect(x + 44, y, 40, 28), "▶", lambda: self.move_downbeat(1), "小節の頭を 1拍 後ろへ", enabled=hb)
        y += 34
        # テンポ
        scr.blit(F.render("テンポ", 14, WHITE if hb else DIM, bold=True), (box.x + 16, y + 5))
        x = box.x + 84
        b = P(pg.Rect(x, y, 32, 28), "−", lambda: self.step_bpm(-0.5), "テンポを 0.5 下げる (押しっぱなしで連続)", enabled=hb, size=16)
        b.repeat = True
        val = f"♩ {self.beats.bpm:.1f}" if hb else "―"
        v = F.render(val, 15, ACCENT if hb and self.beats.manual_bpm else WHITE, bold=True)
        scr.blit(v, (x + 36 + (78 - v.get_width()) // 2, y + 4))
        b = P(pg.Rect(x + 118, y, 32, 28), "＋", lambda: self.step_bpm(0.5), "テンポを 0.5 上げる (押しっぱなしで連続)", enabled=hb, size=16)
        b.repeat = True
        P(pg.Rect(x + 158, y, 154, 28), "小節線を解析に戻す", self.reset_bars, "拍子・1拍目・テンポ・位置を解析した結果に戻す",
          enabled=self.beats is not None)
        y += 34
        # ON / OFF
        x = box.x + 16
        for label, on, act, tip in (
                ("音階を中央に", self.view["follow"], self.toggle_follow, "歌う音に合わせて上下に自動で動かす (C)"),
                ("拍の点線", self.view["beat_lines"], self.toggle_beat_lines, "小節の中の拍にも点線を引く"),
                ("♯の段の点線", self.view["sharp_lines"], self.toggle_sharp_lines, "♯ (黒鍵) の段に点線を引く")):
            text = f"{label} {'ON' if on else 'OFF'}"
            w = F.render(text, 12, WHITE, bold=True).get_width() + 18
            P(pg.Rect(x, y, w, 28), text, act, tip, on, size=12)
            x += w + 6

    def _draw_welcome(self):
        """曲を選ぶ前の案内 + 「曲を選ぶ」ボタン。"""
        pg, scr, F, L = self.pg, self.screen, self.fonts, self.layout
        lines = [
            ("練習する曲を選びましょう", 24, WHITE, True),
            ("下のボタン、またはメニューの「ファイル → 曲の一覧から選ぶ」(Ctrl+L) で曲を選ぶと、", 15, WHITE, False),
            ("その曲を解析して、お手本のバーがここに流れます。", 15, WHITE, False),
            ("曲を選ばなくても、マイクで歌うと自分の声の高さがオレンジのバーで流れます (Space で開始)。", 14, DIM, False),
        ]
        surfs = [F.render(t, sz, c, bold=b) for t, sz, c, b in lines]
        bw = max(sf.get_width() for sf in surfs) + 56
        bh = sum(sf.get_height() + 8 for sf in surfs) + 96
        box = pg.Rect((L.left + L.right - bw) // 2, (L.top + L.bottom - bh) // 2, bw, bh)
        if self.show_mixer and not self.show_grid_panel:      # 音量パネルと重ならないよう下へずらす
            box.y = min(max(box.y, self._mixer_rect().bottom + 12), L.bottom - bh - 4)
        bg = pg.Surface(box.size, pg.SRCALPHA)
        bg.fill(PANEL + (240,))
        scr.blit(bg, box)
        pg.draw.rect(scr, OUTLINE, box, 1, border_radius=12)
        y = box.y + 20
        for sf in surfs:
            scr.blit(sf, (box.x + 28, y))
            y += sf.get_height() + 8
        R = self.requests.append
        specs = [("曲の一覧から選ぶ", lambda: R("library"), "フォルダの曲を一覧から選ぶ (Ctrl+L)", GOOD),
                 ("曲ファイルを直接開く…", lambda: R("open_song"), "曲のファイルを1つ選んで開く (Ctrl+O)", None),
                 ("フォルダを開く…", lambda: R("open_folder"), "曲が入っているフォルダを選ぶ (Ctrl+F)", None)]
        x = box.x + 28
        for label, act, tip, acc in specs:
            w = F.render(label, 16, WHITE, bold=True).get_width() + 34
            b = Button(pg.Rect(x, y + 14, w, 40), label, act, tip, acc is not None, True, acc)
            b.draw(pg, scr, F, b.rect.collidepoint(self.mouse) and not self.menu.is_open, 16)
            self.buttons.append(b)
            x += w + 10

    def _draw_update_dialog(self):
        """アップデートの画面 (確認中 / 最新 / 新しい版あり / ダウンロード中 / 準備完了 / エラー / バージョン情報)。"""
        pg, scr, F = self.pg, self.screen, self.fonts
        shade = pg.Surface((W, H), pg.SRCALPHA)
        shade.fill(BG_DEEP + (190,))
        scr.blit(shade, (0, 0))
        st_, info = UPDATE.state, UPDATE.info or {}
        cur = info.get("current") or UPDATE.current_version()
        lines, buttons = [], []
        title = "アップデート"
        if st_ == "checking":
            lines = [("最新版を確かめています…", WHITE)]
        elif st_ == "latest":
            lines = [(f"お使いの版 {cur} は最新版です。", GOOD)]
            buttons = [("閉じる", UPDATE.close, True)]
        elif st_ == "error":
            lines = [(UPDATE.message, NEAR)]
            buttons = [("もう一度確かめる", lambda: UPDATE.check(True), False), ("閉じる", UPDATE.close, True)]
        elif st_ == "available":
            title = f"新しい版 {info.get('latest')} があります"
            lines = [(f"今の版：{cur}　→　新しい版：{info.get('latest')}", WHITE), ("", WHITE), ("変更点", ACCENT)]
            lines += [("・" + n, DIM) for n in (info.get("notes") or ["（説明なし）"])[:8]]
            if info.get("needs_build"):                 # 中のライブラリが変わった版 (ビルド版のとき)
                lines += [("", WHITE), ("この版は、ソフトの中の部品も新しくなっています。", NEAR),
                          ("GitHub から新しい ZIP をダウンロードして、上書きしてください（設定や曲のデータはそのまま）。", NEAR)]

                def open_page(url=info.get("download_url")):
                    import webbrowser

                    webbrowser.open(url)
                    UPDATE.close()

                buttons = [("ダウンロードページを開く", open_page, True), ("あとで", UPDATE.close, False)]
            else:
                lines += [("", WHITE), ("設定・解析した曲・テイク・お手本の編集版はそのまま残ります。", FAINT)]
                buttons = [("今すぐ更新", UPDATE.download, True), ("あとで", UPDATE.close, False)]
        elif st_ == "downloading":
            lines = [(f"ダウンロードしています… {int(UPDATE.progress * 100)}%", WHITE)]
        elif st_ == "ready":
            title = "アップデートの準備ができました"
            lines = [(f"{cur}　→　{info.get('latest')}", WHITE), ("", WHITE),
                     ("「再起動して更新」を押すと、ソフトをいったん閉じてファイルを入れ替え、", DIM),
                     ("自動で起動し直します（入れ替え前のファイルは backup フォルダに残します）。", DIM)]
            buttons = [("再起動して更新", lambda: self.requests.append("apply_update"), True), ("あとで", UPDATE.close, False)]
        elif st_ == "version":
            title = "バージョン情報"
            lines = [(f"{APP_TITLE}　バージョン {cur}", WHITE)]
            try:
                import updater

                m = updater.local_manifest()
                if m.get("date"):
                    lines.append((f"更新日：{m['date']}", DIM))
                hist = [{"version": m.get("version"), "notes": m.get("notes", [])}] + m.get("history", [])
                for h in hist[:3]:
                    if h.get("notes"):
                        lines.append(("", WHITE))
                        lines.append((f"{h.get('version')} の変更点", ACCENT))
                        lines += [("・" + n, DIM) for n in h["notes"][:5]]
            except Exception:
                pass
            buttons = [("アップデートを確認", lambda: UPDATE.check(True), False), ("閉じる", UPDATE.close, True)]
        surfs = [(F.render(t, 15, c) if t else None) for t, c in lines]
        ts = F.render(title, 20, WHITE, bold=True)
        bw = max([ts.get_width()] + [sf.get_width() for sf in surfs if sf] + [360]) + 64
        bw = min(bw, W - 40)
        bh = 70 + sum((sf.get_height() + 6) if sf else 10 for sf in surfs) + (64 if buttons else 20)
        box = pg.Rect((W - bw) // 2, max(MENU_H + 10, (H - bh) // 2), bw, bh)
        pg.draw.rect(scr, PANEL, box, border_radius=12)
        pg.draw.rect(scr, OUTLINE, box, 1, border_radius=12)
        scr.blit(ts, (box.x + 32, box.y + 22))
        y = box.y + 62
        for sf in surfs:
            if sf:
                scr.blit(sf, (box.x + 32, y))
                y += sf.get_height() + 6
            else:
                y += 10
        if st_ == "downloading":
            bar = pg.Rect(box.x + 32, y + 4, box.width - 64, 8)
            pg.draw.rect(scr, HAIR, bar, border_radius=4)
            pg.draw.rect(scr, ACCENT, (bar.x, bar.y, int(bar.width * UPDATE.progress), bar.height), border_radius=4)
        self._dlg_buttons = []
        x = box.right - 32
        for label, act, main in buttons:
            w = F.render(label, 15, WHITE, bold=True).get_width() + 40
            x -= w
            b = Button(pg.Rect(x, box.bottom - 54, w, 38), label, act, "", main, True, GOOD if main else None)
            b.draw(pg, scr, F, b.rect.collidepoint(self.mouse), 15)
            self._dlg_buttons.append(b)
            x -= 10

    def _draw_help(self):
        pg, scr, F = self.pg, self.screen, self.fonts
        shade = pg.Surface((W, H), pg.SRCALPHA)
        shade.fill(BG_DEEP + (200,))
        scr.blit(shade, (0, 0))
        rows = [
            ("マウス", ""),
            ("メニューバー", "ほぼすべての操作ができます"),
            ("下のボタン", "再生・移動・ボーカル・音量・オクターブ"),
            ("上の青いバーをクリック / ドラッグ", "その位置へ移動"),
            ("譜面をつかんで左右にドラッグ", "前後へ移動 (離した位置から再開)"),
            ("譜面をクリック", "クリックした時刻へ移動"),
            ("ホイール", "1秒ずつ前後へ移動"),
            ("Ctrl + ホイール / Shift + ホイール", "横の幅 / 縦の間隔 (グリッド)"),
            ("左の音名を上下にドラッグ / ホイール", "縦の拡大・縮小 (Shift を押しながら：上下に移動)"),
            ("楽譜の上の目盛り (小節番号)", "左右にドラッグ / ホイール：拡大・縮小　ダブルクリック：1拍目に"),
            ("Shift + 目盛りをドラッグ", "小節線を移動"),
            ("楽譜を右クリック / Shift+ドラッグ", "メニュー / A-B ループの範囲を選ぶ"),
            ("歌詞の文字をドラッグ / 右クリック", "タイミングを直す (Shift：行の後ろも) / ふりがな・1つ前後のバーへ"),
            ("キーボード", ""),
            ("Space", "再生 / 一時停止"), ("← →", f"巻き戻し / スキップ（{self.skip_sec}秒）"),
            ("↑ ↓", "スキップ間隔"), ("V", "ボーカル ON / OFF"), ("[ ]", "お手本のオクターブ"),
            ("- +", "曲の音量"), ("O / 9 0", "自分の声 ON・OFF / 音量 ±5%"), (", .", "MIDI のタイミング"), ("M", "お手本の切替"),
            ("P / R / Home", "音程線 / 採点リセット / 最初から"), ("G / C", "グリッドの調整 / 音階を中央に"),
            ("A / B / L", "A 点 / B 点 / A-B ループ ON・OFF"), ("Shift+← → / Shift+↑ ↓", "再生速度 / キー (半音)"),
            ("T / Y", "録音モード / 最後のテイクを再生"), ("E", "お手本の編集 (Delete・↑↓・Ctrl+Z・Ctrl+S)"), ("F11 / F1 / Ctrl+L", "全画面 / この画面 / 曲の一覧"),
            ("K / Enter", "歌詞の表示 ON・OFF / 歌い出しの記録中：今の行の歌い出し"),
        ]
        rh = int(min(26, (H - MENU_H - 70) / len(rows)))
        bw, bh = min(W - 40, 800), 40 + len(rows) * rh
        box = pg.Rect((W - bw) // 2, max(MENU_H + 10, (H - bh) // 2), bw, bh)
        pg.draw.rect(scr, PANEL, box, border_radius=12)
        pg.draw.rect(scr, OUTLINE, box, 1, border_radius=12)
        y = box.y + 16
        fs = int(max(11, min(15, rh - 4)))
        for k, v in rows:
            if not v:
                scr.blit(F.render(k, fs + 1, ACCENT, bold=True), (box.x + 24, y))
            else:
                scr.blit(F.render(k, fs, WHITE, bold=True), (box.x + 40, y + 1))
                scr.blit(F.render(v, fs, DIM), (box.x + 340, y + 1))
            y += rh
        s = F.render("クリックするか F1 で閉じます", 13, DIM)
        scr.blit(s, (box.right - s.get_width() - 16, box.bottom - 24))

    def _draw_toast(self):
        text, until = self.toast
        if not text or time.perf_counter() > until:
            return
        pg, scr = self.pg, self.screen
        cx, right = W // 2, W - 16
        panels = self._panel_rects()
        if panels:                              # パネルが開いているときは、その左の空いた所へ
            right = min(r.x for r in panels) - 12
            cx = (self.layout.left + right) // 2
        ty = HEADER_H + PROGRESS_H + 46
        if panels and right - self.layout.left < 520:     # 左に場所が無い → 譜面の下の方に出す
            cx, right, ty = W // 2, W - 16, self.layout.bottom - 34
        size = 20
        s = self.fonts.render(text, size, WHITE, bold=True)
        while s.get_width() + 36 > right - self.layout.left and size > 14:   # 入りきらなければ文字を小さく
            size -= 2
            s = self.fonts.render(text, size, WHITE, bold=True)
        r = s.get_rect(center=(cx, ty)).inflate(36, 18)
        box = pg.Surface(r.size, pg.SRCALPHA)
        box.fill(PANEL + (240,))
        scr.blit(box, r)
        pg.draw.rect(scr, OUTLINE, r, 1, border_radius=8)
        scr.blit(s, s.get_rect(center=r.center))


# ---------------------------------------------------------------------------
# 起動・解析中画面
# ---------------------------------------------------------------------------
_window_state = {"fullscreen": False, "size": DEFAULT_SIZE}


def open_window(pg, size=DEFAULT_SIZE, fullscreen=False):
    """サイズ変更・最大化・最小化ができるウィンドウ (fullscreen=True で全画面)。"""
    if fullscreen:
        return pg.display.set_mode((0, 0), pg.FULLSCREEN)
    return pg.display.set_mode(size, pg.RESIZABLE)


def toggle_fullscreen(pg):
    st = _window_state
    if not st["fullscreen"]:
        st["size"] = pg.display.get_surface().get_size()
        st["fullscreen"] = True
    else:
        st["fullscreen"] = False
    return open_window(pg, st["size"], st["fullscreen"])


def canvas_for_window(pg, win, current=None):
    """ウィンドウの大きさに合わせた描画先を返す。小さすぎるときは最小サイズで描いて縮小する。"""
    global W, H
    ww, wh = win.get_size()
    W, H = max(ww, MIN_W), max(wh, MIN_H)
    if (W, H) == (ww, wh):
        return win
    if current is not None and current is not win and current.get_size() == (W, H):
        return current
    return pg.Surface((W, H))


# ゲームパッド / コントローラーの抜き差しの知らせ (このアプリでは使わない)。
# pygame 2.6 は、Bluetooth 機器などがつながったときに、これを受け取ると
# 「KeyError: 0 / SystemError: <built-in function get>」で止まることがあるため、受け取らない。
_DEVICE_EVENTS = ("JOYDEVICEADDED", "JOYDEVICEREMOVED", "CONTROLLERDEVICEADDED", "CONTROLLERDEVICEREMOVED",
                  "CONTROLLERDEVICEREMAPPED", "JOYAXISMOTION", "JOYBALLMOTION", "JOYHATMOTION", "JOYBUTTONDOWN",
                  "JOYBUTTONUP", "CONTROLLERAXISMOTION", "CONTROLLERBUTTONDOWN", "CONTROLLERBUTTONUP",
                  "CONTROLLERTOUCHPADDOWN", "CONTROLLERTOUCHPADMOTION", "CONTROLLERTOUCHPADUP",
                  "CONTROLLERSENSORUPDATE", "JOYBATTERYLEVEL")
_event_errors = {"n": 0}


def block_device_events(pg):
    types = [getattr(pg, n) for n in _DEVICE_EVENTS if isinstance(getattr(pg, n, None), int)]
    try:
        if types:
            pg.event.set_blocked(types)
    except Exception:
        pass
    for mod in ("joystick", "_sdl2.controller"):          # 使わないので止めておく
        try:
            m = pg
            for part in mod.split("."):
                m = getattr(m, part)
            m.quit()
        except Exception:
            pass


def safe_events(pg):
    """pg.event.get() が pygame の内部エラーで失敗しても、アプリを止めずに続ける。"""
    try:
        return pg.event.get()
    except (SystemError, KeyError) as e:
        _event_errors["n"] += 1
        if _event_errors["n"] <= 3:
            print(f"[注意] 入力イベントの読み込みで pygame の内部エラー（無視して続けます）: {type(e).__name__}: {e}",
                  flush=True)
        if _event_errors["n"] == 1:
            block_device_events(pg)
        return []


def _rects_overlap(a, b):
    return a.x < b.x + b.width and b.x < a.x + a.width and a.y < b.y + b.height and b.y < a.y + a.height


class UpdateCtl:
    """アップデートの確認・ダウンロードの状態 (裏で行い、画面はこれを見て表示する)。"""

    def __init__(self):
        self.state = None          # None / checking / latest / available / downloading / ready / error / version
        self.info = None           # updater.check() の結果
        self.progress = 0.0
        self.stage = None          # 入れ替えの準備ができたフォルダ
        self.message = ""
        self.notify = None         # 起動時の確認で見つかった新しい版 (お知らせ用)
        self.show = False          # 画面に出すか

    def _run(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def current_version(self):
        try:
            import updater

            return updater.local_version()
        except Exception:
            return "?"

    def check(self, manual=True):
        if self.state in ("checking", "downloading"):
            return
        if manual:
            self.state, self.show = "checking", True

        def work():
            try:
                import updater

                r = updater.check()
            except Exception as e:
                r = {"error": str(e), "available": False, "current": self.current_version(), "latest": None, "notes": []}
            self.info = r
            if not manual:                       # 起動時の確認: 新しい版があるときだけお知らせ
                if r.get("available"):
                    self.notify = r["latest"]
                return
            if r.get("error"):
                self.state, self.message = "error", r["error"]
            elif r.get("available"):
                self.state = "available"
            else:
                self.state = "latest"

        self._run(work)

    def download(self):
        if not self.info or not self.info.get("manifest"):
            return
        self.state, self.progress, self.show = "downloading", 0.0, True

        def work():
            try:
                import updater

                def prog(f, rel):
                    self.progress = f

                self.stage = updater.download(self.info["manifest"], prog)
                self.state = "ready"
            except Exception as e:
                self.state, self.message = "error", f"ダウンロードに失敗しました：{e}"

        self._run(work)

    def from_zip(self, path):
        self.state, self.show = "downloading", True

        def work():
            try:
                import updater

                self.stage, m = updater.stage_from_zip(path)
                self.info = {"current": updater.local_version(), "latest": m.get("version"), "notes": m.get("notes", []),
                             "manifest": m, "available": True}
                self.state = "ready"
            except Exception as e:
                self.state, self.message = "error", f"ZIP から準備できませんでした：{e}"

        self._run(work)

    def open_version(self):
        self.state, self.show = "version", True

    def close(self):
        if self.state not in ("checking", "downloading"):
            self.state = None
        self.show = False


UPDATE = UpdateCtl()


def ask_text(title, prompt, default=""):
    """名前などを入力してもらう小さな画面 (使えないときは default を返す)。"""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        v = simpledialog.askstring(title, prompt, initialvalue=default, parent=root)
        root.destroy()
        return v.strip() if v else None
    except Exception:
        return default


def choose_file(kind="song", initial_dir=None):
    """ファイル選択画面。kind: "song" (曲) / "midi"。"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if kind == "zip":
            path = filedialog.askopenfilename(title="アップデートの ZIP ファイルを選んでください", initialdir=initial_dir,
                                              filetypes=[("ZIP ファイル", "*.zip"), ("すべて", "*.*")])
        elif kind == "midi":
            path = filedialog.askopenfilename(title="お手本にする MIDI ファイルを選んでください", initialdir=initial_dir,
                                              filetypes=[("MIDI ファイル", "*.mid *.midi *.kar"), ("すべて", "*.*")])
        else:
            path = filedialog.askopenfilename(title="練習する曲を選んでください", initialdir=initial_dir,
                                              filetypes=[("音楽ファイル", "*.mp3 *.wav *.m4a *.flac"), ("すべて", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        return None


_CURSORS = {}


def set_cursor(pg, kind):
    """マウスカーソルの形を変える (pygame 2 の標準カーソル)。"""
    if _CURSORS.get("now") == kind:
        return
    names = {"hand": "SYSTEM_CURSOR_HAND", "drag": "SYSTEM_CURSOR_SIZEWE", "zoom": "SYSTEM_CURSOR_SIZEWE",
             "vdrag": "SYSTEM_CURSOR_SIZENS", "arrow": "SYSTEM_CURSOR_ARROW"}
    try:
        pg.mouse.set_cursor(getattr(pg, names[kind]))
        _CURSORS["now"] = kind
    except Exception:
        pass


def run_analysis(mp3, state, opts, midi_path=None):
    """別スレッドで ステップ2 (分離・お手本) と ステップ3 (キー) を実行。"""
    try:
        import audio_processor
        import key_analyzer
        from library import CACHE_ROOT

        def log(msg):
            print(msg, flush=True)
            state["log"].append(str(msg))

        audio_processor.log = log  # 進み具合を画面にも出す
        audio_processor.USE_GPU = opts.use_gpu
        res = audio_processor.process(mp3, cache_root=CACHE_ROOT, sep_model=opts.sep_model, engine=opts.engine,
                                      crepe_model=opts.crepe_model)
        midi_guide = None
        if midi_path:
            import midi_loader

            log(f"[MIDI] {os.path.basename(midi_path)} を読み込んでいます")
            try:
                midi_guide = midi_loader.build_guide(midi_path, res["data"], res["dir"], track=opts.midi_track,
                                                     offset=opts.midi_offset, log=log)
            except Exception as e:
                log(f"MIDI を使えませんでした: {e}（歌声から解析したお手本で続けます）")
        log("[3/3] 曲のキー（調）を解析しています")
        try:
            key = key_analyzer.analyze_key(mp3, cache_json=os.path.join(res["dir"], "key.json"))
            log(key["display"])
        except Exception as e:
            log(f"キーの解析に失敗しました: {e}")
            key = None
        beats = None
        try:
            import beat_analyzer

            log("[拍] 拍と小節を解析しています")
            beats = beat_analyzer.analyze_beats(res["accompaniment"], cache_json=os.path.join(res["dir"], "beats.json"))
            if beats.get("beats"):
                log(f"テンポ ♩ = {beats['tempo']:.0f}・{beats['bpb']}拍子")
            else:
                log("拍を見つけられませんでした（タテ線は秒ごとになります）")
        except Exception as e:
            log(f"拍の解析に失敗しました: {e}（タテ線は秒ごとになります）")
        state["result"] = (res, key, midi_guide, beats)
    except Exception as e:
        traceback.print_exc()
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        state["done"] = True


COLORS = {"BG": BG, "WHITE": WHITE, "DIM": DIM, "GOLD": GOLD, "GOOD": GOOD, "NEAR": NEAR}


class Shell:
    """アプリ全体: 曲の一覧 → (解析中) → 練習画面 を切り替える。"""

    def __init__(self, pg, args):
        import library

        self.pg, self.args = pg, args
        self.lib = library
        self.settings = library.load_settings()
        self.fonts = Fonts(pg, args.font)
        self.requests = []
        self.mode = "practice"         # "practice" (楽譜) / "library" (曲の一覧) / "loading" (解析中)
        self.trainer = None
        self.playback = None
        self.loading = None            # {"song", "midi", "state", "t0"}
        self.mouse = (-1, -1)
        self.running = True
        self.mic = MicTracker(self._clock, args.latency, args.threshold, args.device, out_device=getattr(args, "out_device", None),
                              monitor=False,       # 起動時はいつも OFF (ハウリング防止。必要なときに ON にする)
                              monitor_gain=self.settings.get("monitor_gain", 0.8),
                              api=self.settings.get("monitor_api", "auto"),
                              monitor_auto=self.settings.get("monitor_auto", True),
                              noise=self.settings.get("noise", "mid"),
                              noise_params=self.settings.get("noise_params"),
                              in_pref=self.settings.get("in_device_name"), out_pref=self.settings.get("out_device_name"))
        SONG_OUT["name"] = self.settings.get("out_device_name")
        if SONG_OUT["name"]:                    # 曲を鳴らすスピーカーも、選んであるものにする
            try:
                pg.mixer.quit()
                mixer_init(pg, 44100)
            except Exception:
                pass
        if not self.mic.start():
            print(f"[注意] マイクを開けませんでした: {self.mic.error}", file=sys.stderr)
        self.settings.setdefault("use_gpu", True)
        if args.cpu:
            self.settings["use_gpu"] = False
        args.use_gpu = self.settings["use_gpu"]
        self.libview = library.LibraryView(pg, self.fonts, self.settings, self.open_song, self._library_menu,
                                           requests=self.requests)
        self.libview.device_text = "解析：確認中…"
        threading.Thread(target=self._detect_device, daemon=True).start()
        if self.settings.get("update_auto", True) and not os.environ.get("PITCHLUNA_NO_UPDATE_CHECK") \
                and not getattr(args, "autoquit", 0):
            UPDATE.check(manual=False)           # 新しい版があるか、裏でそっと確かめる
        self._empty_score()

    def _empty_score(self):
        """起動直後の空の楽譜 (曲は後から選ぶ)。"""
        self.playback = FreePlayback()
        self.trainer = Trainer(self.pg, None, self.fonts, "曲が選ばれていません", Guide(EMPTY_GUIDE_DATA), None,
                               self.playback, self.mic, skip_sec=self.args.skip)
        self._wire(self.trainer)
        # 起動直後は再生しない (Space / 再生ボタンで開始)
        self.mode = "practice"

    def _wire(self, t):
        t.requests = self.requests
        t.is_fullscreen = lambda: _window_state["fullscreen"]
        t.on_monitor_change = self._save_monitor
        t.on_device_change = self._save_devices
        t.take_offset = float(self.settings.get("take_offset", 0.0))
        t.on_take_offset = self._save_take_offset
        t.on_vocal_gain = self._save_vocal_gain
        t.update_auto = lambda: self.settings.get("update_auto", True)
        if hasattr(t.pb, "set_vocal_gain"):
            t.pb.set_vocal_gain(float(self.settings.get("vocal_gain", 1.0)))
        t.apply_view(self.settings.get("view"))
        t.on_view_change = lambda: self._save_view(t)

    def _save_view(self, t):
        self.settings["view"] = dict(t.view)
        self.lib.save_settings(self.settings)

    def _save_vocal_gain(self, v):
        self.settings["vocal_gain"] = round(v, 2)
        self.lib.save_settings(self.settings)

    def _save_take_offset(self, v):
        self.settings["take_offset"] = v
        self.lib.save_settings(self.settings)

    def _save_devices(self):
        self.settings["in_device_name"] = getattr(self.mic, "in_pref", None)
        self.settings["out_device_name"] = SONG_OUT["name"]
        self._save_monitor()

    def _save_monitor(self):
        self.settings["monitor"] = self.mic.monitor
        self.settings["monitor_gain"] = round(self.mic.monitor_gain, 2)
        self.settings["monitor_api"] = self.mic.api
        self.settings["monitor_auto"] = bool(getattr(self.mic, "monitor_auto", True))
        self.settings["noise"] = getattr(self.mic, "noise", "mid")
        if hasattr(self.mic, "noise_params"):
            self.settings["noise_params"] = {k: round(v, 3) for k, v in self.mic.noise_params().items()}
        self.lib.save_settings(self.settings)

    def _detect_device(self):
        """グラボが使えるかを裏で調べて一覧画面に表示 (torch の読み込みに数秒かかるため)。"""
        try:
            import audio_processor

            audio_processor.USE_GPU = self.args.use_gpu
            dev, info = audio_processor.ai_device()
            self.gpu_ok = dev == "cuda"
            self.libview.device_text = f"解析：グラボ（{info}）" if dev == "cuda" else f"解析：CPU（{info}）"
        except Exception as e:
            self.libview.device_text = f"解析：CPU（{e}）"

    def toggle_gpu(self):
        self.args.use_gpu = self.settings["use_gpu"] = not self.settings.get("use_gpu", True)
        self.lib.save_settings(self.settings)
        self.libview.device_text = "解析：確認中…"
        threading.Thread(target=self._detect_device, daemon=True).start()

    # ----- 共通 -----
    def _clock(self):
        pb = self.playback
        return (pb.position(), pb.playing) if (pb and self.mode == "practice") else (0.0, False)

    def _library_menu(self, view):
        R = self.requests.append
        recent = [MenuItem((lambda f=f: "最近："+(f if len(f) < 60 else "…"+f[-58:])), (lambda f=f: view.set_folder(f)))
                  for f in self.settings.get("recent_folders", [])[:6]]
        return MenuBar(self.pg, self.fonts, [
            ("ファイル", [
                MenuItem("楽譜の画面に戻る", lambda: R("score"), "Esc"),
                SEPARATOR,
                MenuItem("フォルダを開く…", lambda: R("open_folder"), "Ctrl+F"),
                MenuItem("曲ファイルを直接開く…", lambda: R("open_song"), "Ctrl+O"),
            ] + ([SEPARATOR] + recent if recent else []) + [
                SEPARATOR,
                MenuItem("終了", lambda: R("quit")),
            ]),
            ("一覧", [
                MenuItem("選んだ曲で練習（解析して開く）", view.open_selected, "Enter"),
                SEPARATOR,
                MenuItem("サブフォルダも探す", view.toggle_recursive, checked=lambda: self.settings.get("recursive", True)),
                MenuItem("名前順", lambda: view.set_sort("name"), checked=lambda: view.sort_key == "name"),
                MenuItem("新しい順", lambda: view.set_sort("new"), checked=lambda: view.sort_key == "new"),
                MenuItem("解析済みを上に", lambda: view.set_sort("done"), checked=lambda: view.sort_key == "done"),
                SEPARATOR,
                MenuItem("一覧を更新", view.reload, "F5"),
            ]),
            ("表示", [
                MenuItem("全画面表示", lambda: R("fullscreen"), "F11", checked=lambda: _window_state["fullscreen"]),
            ]),
            ("設定", [
                MenuItem("解析にグラボ（GPU）を使う", self.toggle_gpu, checked=lambda: self.settings.get("use_gpu", True)),
            ]),
            ("ヘルプ", [
                MenuItem("使い方：フォルダを選び、曲をダブルクリック", lambda: None),
            ]),
        ], height=MENU_H)

    # ----- 曲を開く -----
    def open_song(self, path, midi=None):
        """一覧やメニューで選ばれた曲を解析して練習画面へ (解析は裏で行う)。"""
        if self.mode == "loading":
            return
        if not path or not os.path.isfile(path):
            self.libview.message = "ファイルが見つかりません"
            return
        if midi is None:
            from midi_loader import find_midi_for

            midi = find_midi_for(path)
        if self.playback and self.playback.playing:
            self.playback.toggle_play()            # 今の曲は止めておく (解析を中止したら戻れるように残す)
        state = {"log": [], "done": False, "result": None, "error": None}
        self.loading = {"song": path, "midi": midi, "state": state, "t0": time.time()}
        self.mode = "loading"
        threading.Thread(target=run_analysis, args=(path, state, self.args, midi), daemon=True).start()

    def _finish_loading(self):
        ld = self.loading
        res, key, midi_guide, beats = (tuple(ld["state"]["result"]) + (None,))[:4]
        pg = self.pg
        self._stop_practice()                      # 前の曲を閉じてから新しい曲を読み込む
        self.playback = Playback(pg, res["accompaniment"], res["vocals"])
        vocal_guide = Guide(res["data"], source="歌声から解析")
        save_offset = None
        if midi_guide:
            import midi_loader

            inf = midi_guide["info"]
            guide = Guide(res["data"], notes=midi_guide["notes"], rng=midi_guide["range"], source=f"MIDI（{inf['name'][:12]}）")
            alt = vocal_guide
            base_offset = inf["offset"]
            save_offset = lambda o: midi_loader.save_offset(inf["cache_path"], base_offset + o)
        else:
            guide, alt = vocal_guide, None
        title = os.path.splitext(os.path.basename(ld["song"]))[0]
        self.trainer = Trainer(pg, self.canvas, self.fonts, title, guide, key, self.playback, self.mic,
                               skip_sec=self.args.skip, alt_guide=alt, on_offset_change=save_offset)
        self._wire(self.trainer)
        if beats and beats.get("beats"):
            from beat_analyzer import BeatGrid

            self.trainer.set_beats(BeatGrid(beats, self.playback.duration,
                                            user_path=os.path.join(res["dir"], "grid_user.json")))
        self.trainer.song_path = ld["song"]
        self.trainer.song_dir = res["dir"]
        self.trainer.load_practice()            # この曲の速度・キー・A-B・編集版
        self.trainer.load_lyrics()              # この曲の歌詞
        self.trainer.show_toast("Space で再生スタート！")
        self.mic.clear()
        self.libview.checker.refresh(ld["song"])
        self.libview.select_path(ld["song"])
        self.loading = None
        self.mode = "practice"

    def _stop_practice(self):
        if self.trainer is not None:
            try:
                self.trainer.stop_review()
                self.trainer._finish_take()
            except Exception:
                pass
        try:
            self.mic.stop_rec()
        except Exception:
            pass
        if self.playback:
            self.playback.close()
        self.playback = None
        self.trainer = None

    def back_to_library(self):
        """曲の一覧を開く (今の曲は一時停止して残しておく。Esc で楽譜に戻れる)。"""
        if self.playback and self.playback.playing:
            self.playback.toggle_play()
        if self.trainer is not None:
            self.trainer.stop_review()
            self.trainer._finish_take()
        self.mode = "library"
        self.libview.reload()

    def back_to_score(self):
        self.loading = None
        if self.trainer is None:
            self._empty_score()
        self.mode = "practice"

    # ----- メニューからの依頼 -----
    def handle_requests(self):
        while self.requests:
            req = self.requests.pop(0)
            pb_playing = self.playback.playing if self.playback else False
            if req == "quit":
                self.running = False
            elif req == "fullscreen":
                toggle_fullscreen(self.pg)
            elif req == "library":
                self.back_to_library()
            elif req == "open_folder":
                if pb_playing:
                    self.playback.toggle_play()
                f = self.lib.choose_folder(self.libview.folder)
                if f:
                    self.libview.set_folder(f)
                    self.libview.menu = self._library_menu(self.libview)   # 最近のフォルダを更新
                    self.mode = "library"
            elif req == "open_song":
                if pb_playing:
                    self.playback.toggle_play()
                path = choose_file("song", self.libview.folder)
                if path:
                    self.open_song(path)
            elif req == "apply_update" and UPDATE.stage:
                try:
                    import updater

                    if self.trainer is not None:
                        self.trainer._finish_take()
                    updater.launch_apply(UPDATE.stage, restart=True)
                    self.running = False             # 終了 → 入れ替え → 自動で起動し直す
                except Exception as e:
                    UPDATE.state, UPDATE.message = "error", f"入れ替えを始められませんでした：{e}"
            elif req == "update_zip":
                path = choose_file("zip", os.path.join(os.path.expanduser("~"), "Downloads"))
                if path:
                    UPDATE.from_zip(path)
            elif req == "toggle_update_auto":
                self.settings["update_auto"] = not self.settings.get("update_auto", True)
                self.lib.save_settings(self.settings)
            elif req == "score":
                self.back_to_score()
            elif req == "open_midi" and self.trainer and self.trainer.song_path:
                if pb_playing:
                    self.playback.toggle_play()
                path = choose_file("midi", os.path.dirname(self.trainer.song_path))
                if path:
                    self.open_song(self.trainer.song_path, midi=path)   # 解析済みなので MIDI の照合だけ

    # ----- 描画 -----
    def draw_loading(self, scr):
        pg, F = self.pg, self.fonts
        ld = self.loading
        st = ld["state"]
        scr.fill(BG)
        pg.draw.rect(scr, BG_DEEP, (0, 0, W, MENU_H))
        pg.draw.line(scr, HAIR, (0, MENU_H - 1), (W, MENU_H - 1), 1)
        scr.blit(F.render("解析中は操作できません（Esc で中止して一覧に戻ります）", 14, DIM), (14, 5))
        title = F.render(APP_TITLE, 32, WHITE, bold=True)
        scr.blit(title, title.get_rect(center=(W // 2, MENU_H + 90)))
        scr.blit(F.render("♪ " + os.path.basename(ld["song"]), 20, WHITE), (120, MENU_H + 150))
        if ld["midi"]:
            scr.blit(F.render("MIDI：" + os.path.basename(ld["midi"]), 15, ACCENT), (140, MENU_H + 180))
        if st["error"]:
            scr.blit(F.render("エラーが発生しました（Esc で一覧に戻ります）", 22, BAD, bold=True), (120, MENU_H + 210))
            scr.blit(F.render(st["error"][:110], 15, WHITE), (120, MENU_H + 250))
        else:
            sp = "◐◓◑◒"[int((time.time() - ld["t0"]) * 4) % 4]
            msg = f"{sp}  解析中…  {int(time.time() - ld['t0'])}秒経過（初回は数分〜十数分。2回目からはすぐ始まります）"
            scr.blit(F.render(msg, 19, FREE, bold=True), (120, MENU_H + 210))
            for i, line in enumerate(st["log"][-10:]):
                scr.blit(F.render(line.split("\r")[-1][:100], 16, DIM), (140, MENU_H + 255 + i * 27))

    # ----- メインループ -----
    def run(self, autoquit=0, start_song=None):
        pg = self.pg
        clock = pg.time.Clock()
        pg.key.set_repeat(350, 120)
        pg.key.start_text_input()
        win = pg.display.get_surface()
        self.canvas = canvas_for_window(pg, win)
        quit_at = None
        if start_song:
            self.libview.set_folder(os.path.dirname(os.path.abspath(start_song)))
            self.libview.select_path(start_song)
            if autoquit > 0:
                self.open_song(start_song)       # 動作確認のときだけ自動で解析して開く

        def to_canvas(p):
            ww, wh = pg.display.get_surface().get_size()
            return (p[0] * W / max(1, ww), p[1] * H / max(1, wh))

        while self.running:
            for ev in safe_events(pg):
                self._event(ev, to_canvas)
            self.handle_requests()

            if self.mode == "loading" and self.loading["state"]["done"] and not self.loading["state"]["error"]:
                self._finish_loading()
                if autoquit > 0 and quit_at is None:
                    self.playback.toggle_play()
                    quit_at = time.time() + autoquit
                    print(f"[動作確認] 再生を開始しました。{autoquit:.0f}秒後に自動で終了します")
            if autoquit > 0 and self.mode == "loading" and self.loading["state"]["error"]:
                print(f"[動作確認] 解析エラー: {self.loading['state']['error']}")
                self.running = False

            win = pg.display.get_surface()
            if min(win.get_size()) <= 0 or not pg.display.get_active():
                clock.tick(10)          # 最小化中は描画を休む (音とマイクはそのまま動く)
                continue
            size_before = (W, H)
            new_canvas = canvas_for_window(pg, win, self.canvas)
            resized = new_canvas is not self.canvas or (W, H) != size_before
            self.canvas = new_canvas
            if self.mode == "practice":
                if resized or self.trainer.screen is not self.canvas or self.trainer.screen.get_size() != (W, H):
                    self.trainer.set_screen(self.canvas)
                self.trainer.draw()
                set_cursor(pg, self.trainer.cursor_kind())
            elif self.mode == "loading":
                self.draw_loading(self.canvas)
                set_cursor(pg, "arrow")
            else:
                self.libview.draw(self.canvas, W, H, COLORS)
                set_cursor(pg, self.libview.cursor_kind(W, H))
            if self.canvas is not win:
                win.blit(pg.transform.smoothscale(self.canvas, win.get_size()), (0, 0))
            pg.display.flip()
            clock.tick(60)
            if quit_at and time.time() >= quit_at:
                print(f"[動作確認] 画面表示・再生 OK (再生位置 {self.playback.position():.1f}秒 / マイク: "
                      f"{'OK' if not self.mic.error else 'なし'})")
                self.running = False

        self.mic.stop()
        self._stop_practice()

    def _event(self, ev, to_canvas):
        pg = self.pg
        if ev.type == pg.QUIT:
            self.running = False
            return
        mode = self.mode
        view = self.trainer if mode == "practice" else self.libview
        if ev.type == pg.KEYDOWN:
            ctrl = ev.mod & pg.KMOD_CTRL
            if ev.key == pg.K_F11:
                toggle_fullscreen(pg)
            elif mode == "loading":
                if ev.key == pg.K_ESCAPE:     # 解析は裏で続くが、結果は使わずに楽譜へ戻る
                    self.back_to_score()
            elif ev.key == pg.K_ESCAPE:
                if mode == "practice" and UPDATE.show and UPDATE.state:
                    UPDATE.close()
                elif view.menu.is_open:
                    view.menu.close()
                elif mode == "practice" and self.trainer.popup is not None:
                    self.trainer.popup = None
                elif mode == "practice" and self.trainer.lyric_tap is not None:
                    self.trainer.stop_lyric_tap()       # 歌い出しの記録を終わる
                elif mode == "practice" and self.trainer.edit_mode:
                    if self.trainer.sel:
                        self.trainer.sel.clear()        # まず選択を外す
                    else:
                        self.trainer.toggle_edit()       # もう一度で編集モードを終わる
                elif mode == "practice" and self.trainer.show_help:
                    self.trainer.show_help = False
                elif _window_state["fullscreen"]:
                    toggle_fullscreen(pg)
                elif mode == "practice":
                    pass                        # 楽譜の画面では Esc で終了しない (終了はウィンドウの × かメニュー)
                elif self.libview.filter_text:
                    self.libview.filter_text = ""
                    self.libview.apply_filter()
                else:
                    self.back_to_score()        # 曲の一覧の Esc は楽譜へ戻る
            elif ctrl and ev.key == pg.K_o:
                self.requests.append("open_song")
            elif ctrl and ev.key == pg.K_f:
                self.requests.append("open_folder")
            elif ctrl and ev.key == pg.K_l:
                self.requests.append("library" if mode == "practice" else "score")
            elif ctrl and ev.key == pg.K_m and mode == "practice":
                self.requests.append("open_midi")
            elif mode == "practice" and UPDATE.show and UPDATE.state:
                pass                                # アップデートの画面を出している間はキーを使わない
            elif mode == "practice":
                if ev.key == pg.K_F1:
                    self.trainer.toggle_help()
                else:
                    self.trainer.handle_key(ev)
            elif mode == "library":
                self.libview.handle_key(ev, W, H)
        elif ev.type == getattr(pg, "TEXTINPUT", -1) and mode == "library":
            if not (pg.key.get_mods() & pg.KMOD_CTRL):
                self.libview.handle_text(ev.text)
        elif mode == "loading":
            return
        elif ev.type == pg.MOUSEBUTTONDOWN and ev.button in (1, 3):
            if mode == "practice":
                self.trainer.handle_mouse_down(to_canvas(ev.pos), ev.button)
            else:
                self.libview.handle_mouse_down(to_canvas(ev.pos), W, H, ev.button)
        elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1 and mode == "practice":
            self.trainer.handle_mouse_up(to_canvas(ev.pos), ev.button)
        elif ev.type == pg.MOUSEMOTION:
            view.handle_mouse_move(to_canvas(ev.pos))
        elif ev.type == pg.MOUSEWHEEL:
            if mode == "practice":
                mods = pg.key.get_mods()
                self.trainer.handle_wheel(ev.y, ctrl=bool(mods & pg.KMOD_CTRL),
                                          shift=bool(mods & getattr(pg, "KMOD_SHIFT", 0)))
            else:
                self.libview.handle_wheel(ev.y, W, H)
        elif ev.type == getattr(pg, "WINDOWLEAVE", -1):
            view.handle_mouse_move((-1, -1))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("mp3", nargs="?", help="一覧で最初に選んでおく曲 (解析は曲を開いたときに行います)")
    ap.add_argument("--skip", type=int, default=5, help="スキップ間隔の初期値 (秒)")
    ap.add_argument("--device", type=int, default=None, help="マイクのデバイス番号 (test_mic_pitch.py --list で確認)")
    ap.add_argument("--out-device", type=int, default=None, help="自分の声を出すスピーカー/ヘッドホンのデバイス番号 (既定の出力)")
    ap.add_argument("--threshold", type=float, default=-45.0, help="この音量 (dBFS) 未満の音は無視")
    ap.add_argument("--latency", type=float, default=0.10, help="声の表示位置の遅れ補正 (秒)")
    ap.add_argument("--font", default=None, help="日本語フォントのファイルパス")
    ap.add_argument("--fullscreen", action="store_true", help="全画面で起動 (F11 で切替)")
    ap.add_argument("--autoquit", type=float, default=0, help="動作確認用: 指定の曲を解析・再生し、指定秒数後に終了")
    ap.add_argument("--midi-track", default=None, help="MIDI のパートを手動指定 (例 2:0。python midi_loader.py 曲.mid で一覧)")
    ap.add_argument("--midi-offset", type=float, default=None, help="MIDI のタイミングを手動指定 (秒)")
    ap.add_argument("--sep-model", default="htdemucs_ft", help="分離モデル: htdemucs_ft (高精度/既定) / htdemucs (速い)")
    ap.add_argument("--engine", choices=["auto", "crepe", "pyin", "yin"], default="auto", help="音程の解析方法 (auto=使える中で最高精度)")
    ap.add_argument("--crepe-model", choices=["full", "tiny"], default="full", help="CREPE の大きさ (tiny は速い)")
    ap.add_argument("--cpu", action="store_true", help="グラボを使わず CPU で解析する (既定はグラボ)")
    args = ap.parse_args()
    if args.mp3 and not os.path.isfile(args.mp3):
        sys.exit(f"ファイルが見つかりません: {args.mp3}")

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()  # 高解像度画面でぼやけないように
        except Exception:
            pass

    import pygame as pg

    pg.init()
    block_device_events(pg)
    _window_state["fullscreen"] = args.fullscreen
    open_window(pg, DEFAULT_SIZE, args.fullscreen)
    pg.display.set_caption(APP_TITLE)
    try:                                         # ウィンドウのアイコン (三日月と音符)
        pg.display.set_icon(pg.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")))
    except Exception:
        pass
    shell = Shell(pg, args)
    shell.run(autoquit=args.autoquit, start_song=args.mp3)
    pg.quit()


if __name__ == "__main__":
    main()
