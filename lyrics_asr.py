# -*- coding: utf-8 -*-
"""
AI (OpenAI Whisper) で歌声から歌詞を聞き取る (使う人がオンにしたときだけ)

  ・聞き取るのは、曲の解析で分けた歌声 (vocals.wav)。
  ・言葉ごとの「いつ歌っているか」も出す → 入力した歌詞のタイミング合わせの手がかりにする。
  ・AI のモデルは、初めて使うときにダウンロードする (ソフトのフォルダの models\\whisper\\)。
  ・聞き取った結果は曲のフォルダの asr.json に保存。パソコンの外には出さない。
"""

import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

import vendor_lib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "whisper")
REPO = "shellbe1967/pitch-luna"
_OPENAI = "https://openaipublic.azureedge.net/main/whisper/models"

# 使うモデル (高精度 / 軽い)。sha256 は OpenAI の公開しているファイルの値
MODELS = {
    "turbo": {"file": "large-v3-turbo.pt", "label": "高精度（約1.6GB）", "mb": 1550,
              "sha": "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a"},
    "small": {"file": "small.pt", "label": "軽い（約0.5GB・精度は低め）", "mb": 460,
              "sha": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"},
}
VENDOR_FILES = ["openai_whisper-20250625-py3-none-any.zip",
                "tiktoken-0.14.0-cp312-cp312-win_amd64.whl",
                "regex-2026.9.29-cp312-cp312-win_amd64.whl"]


class Cancelled(Exception):
    pass


def model_path(name):
    return os.path.join(MODEL_DIR, MODELS[name]["file"])


def model_ready(name):
    p = model_path(name)
    return os.path.exists(p) and os.path.exists(p + ".ok")


def _urls(name):
    m = MODELS[name]
    return [f"{_OPENAI}/{m['sha']}/{m['file']}",
            f"https://github.com/{REPO}/releases/download/whisper-models/{m['file']}"]


def download_model(name, progress=None, cancel=None):
    """モデルをダウンロードする。progress(割合 0〜1, 文字)。確認用の値 (sha256) で確かめる。"""
    m = MODELS[name]
    os.makedirs(MODEL_DIR, exist_ok=True)
    dst = model_path(name)
    if model_ready(name):
        return dst
    errors = []
    for url in _urls(name):
        part = dst + ".part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PitchLuna"})
            with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0) or m["mb"] * 1024 * 1024
                h = hashlib.sha256()
                done = 0
                t0 = time.time()
                while True:
                    if cancel and cancel():
                        raise Cancelled()
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress:
                        sp = done / max(0.5, time.time() - t0) / 1e6
                        progress(min(1.0, done / total), f"AI モデルをダウンロード中 {done / 1e6:.0f} / {total / 1e6:.0f} MB（{sp:.1f} MB/秒）")
            if h.hexdigest() != m["sha"]:
                raise IOError("ダウンロードしたファイルが壊れています（確認用の値が違います）")
            os.replace(part, dst)
            open(dst + ".ok", "w").close()
            return dst
        except Cancelled:
            _rm(part)
            raise
        except Exception as e:
            errors.append(f"{url.split('/')[2]}: {e}")
            _rm(part)
    raise IOError("AI モデルをダウンロードできませんでした（" + " / ".join(errors) + "）")


def _rm(p):
    try:
        os.remove(p)
    except Exception:
        pass


def load_whisper():
    """whisper の部品を使えるようにする (同梱の部品を展開)。"""
    return vendor_lib.use("whisper", VENDOR_FILES)


def _read_vocals_16k(path):
    """歌声の WAV → 16kHz・モノラル・float32 (Whisper が受け取る形)。"""
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd

    sr, y = wavfile.read(path)
    if y.dtype.kind == "i":
        y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        y = resample_poly(y, 16000 // g, int(sr) // g).astype(np.float32)
    return y


def _energy_db(y, t0, t1):
    import numpy as np

    a, b = max(0, int(t0 * 16000)), min(len(y), int(t1 * 16000))
    if b - a < 160:
        return -120.0
    seg = y[a:b]
    return float(20 * np.log10(np.sqrt(np.mean(seg * seg)) + 1e-9))


def transcribe(vocals_path, name="turbo", use_gpu=True, progress=None, cancel=None):
    """歌声を聞き取る → {"model", "segments": [{"start","end","text","words":[{"w","s","e","p"}]}]}"""
    whisper = load_whisper()
    import torch

    dev = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    if progress:
        progress(0.0, "AI モデルを読み込み中…" + ("（グラボ）" if dev == "cuda" else "（CPU：時間がかかります）"))
    path = model_path(name)
    model = whisper.load_model(path, device=dev)
    try:
        from whisper import _ALIGNMENT_HEADS

        heads = _ALIGNMENT_HEADS.get("large-v3-turbo" if name == "turbo" else name)
        if heads:
            model.set_alignment_heads(heads)          # 言葉ごとの時刻を正確にする設定
    except Exception:
        pass
    audio = _read_vocals_16k(vocals_path)
    dur = len(audio) / 16000.0

    # 進み具合を受け取る (whisper の中の tqdm を差し替える)
    import whisper.transcribe  # noqa: F401  (whisper.transcribe は関数と同じ名前なので、モジュールは sys.modules から取る)
    wt = sys.modules["whisper.transcribe"]

    class _Bar:
        def __init__(self, total=None, **k):
            self.total = total or 1
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def update(self, n=1):
            self.n += n
            if cancel and cancel():
                raise Cancelled()
            if progress:
                progress(min(1.0, self.n / self.total), f"AI が歌詞を聞き取り中… {min(100, int(self.n / self.total * 100))}%")

    old = wt.tqdm
    wt.tqdm = type("tq", (), {"tqdm": _Bar})
    try:
        res = model.transcribe(audio, language="ja", task="transcribe", word_timestamps=True, verbose=False,
                               fp16=(dev == "cuda"), condition_on_previous_text=False,
                               no_speech_threshold=0.6, compression_ratio_threshold=2.2,
                               temperature=(0.0, 0.2, 0.4))
    finally:
        wt.tqdm = old
        del model
        if dev == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    segs = []
    prev_text = None
    repeat = 0
    for s in res.get("segments", []):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        if s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0:
            continue                                  # 声の無い所の聞き違い
        repeat = repeat + 1 if text == prev_text else 0
        prev_text = text
        if repeat >= 2:                               # 同じ文を何度もくり返す聞き違い
            continue
        words = []
        for w in s.get("words") or []:
            ws, we = float(w["start"]), float(w["end"])
            if we <= ws or _energy_db(audio, ws, we) < -50:   # 歌声がほぼ無い所は使わない
                continue
            words.append({"w": w["word"].strip(), "s": round(ws, 3), "e": round(we, 3),
                          "p": round(float(w.get("probability", 0.0)), 3)})
        words = [w for w in words if w["w"]]
        if words:
            segs.append({"start": words[0]["s"], "end": words[-1]["e"], "text": text, "words": words})
    return {"model": name, "duration": round(dur, 2), "made": time.strftime("%Y-%m-%d %H:%M"), "segments": segs}


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
