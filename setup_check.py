#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitch Luna — If Vocal  セットアップ & 動作確認

  1. 必要なライブラリを確認し、足りないものを自動インストール
  2. 日本語フォントを確認 (無ければ Noto Sans JP を fonts フォルダへダウンロード)
  3. テスト用の曲 (WAV / MP3 / MIDI) を tests フォルダに作成
  4. マイク・スピーカー・画面の確認
  5. テストドライラン (分離・音程解析・キー判定・MIDI照合 → アプリを数秒起動して自動終了)

使い方:
    python setup_check.py              すべて実行
    python setup_check.py --quick      解析を速い設定で (htdemucs + CREPE tiny)
    python setup_check.py --no-install インストールせずに確認だけ
    python setup_check.py --no-launch  最後のアプリ起動をしない

結果は setup_log.txt にも保存されます。
"""

import argparse
import importlib
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
TEST_DIR = os.path.join(HERE, "tests")
LOG_PATH = os.path.join(HERE, "setup_log.txt")

# (import 名, pip で入れる候補 (順に試す), 必須か, 用途, import で確認するモジュール)
PACKAGES = [
    ("numpy", ["numpy"], True, "数値計算", "numpy"),
    ("scipy", ["scipy"], True, "信号処理", "scipy.signal"),
    ("sounddevice", ["sounddevice"], True, "マイク入力", "sounddevice"),
    ("pygame", ["pygame"], True, "画面表示・伴奏とボーカルの再生", "pygame"),
    ("soundfile", ["soundfile"], True, "音声ファイルの読み書き", "soundfile"),
    ("librosa", ["librosa"], True, "MP3 読み込み・キー解析・pYIN 音程解析", "librosa"),
    ("torch", ["torch"], True, "AI の土台 (demucs / CREPE)", "torch"),
    ("demucs", ["demucs", "https://github.com/adefossez/demucs/archive/refs/heads/main.zip"], True,
     "AI ボーカル分離 (htdemucs_ft)", "demucs.pretrained"),
    ("torchcrepe", ["torchcrepe"], False, "AI 音程検出 (CREPE) ※推奨", "torchcrepe"),
    ("lameenc", ["lameenc"], False, "テスト用 MP3 の作成", "lameenc"),
    ("PyInstaller", ["pyinstaller"], False, "将来の exe 化 (ビルド用。今回は使いません)", "PyInstaller"),
]

# 指定されたが入れないもの (理由つき)
NOT_NEEDED = {
    "aubio": "Python 3.12 以降向けの配布が無くインストールできません。このアプリでは自前の YIN と torchcrepe で置き換え済みです",
    "spleeter": "TensorFlow が必要で Python 3.10 までしか動きません。このアプリでは上位の demucs で置き換え済みです",
}

FONT_FILES = {
    "NotoSansJP-Regular.otf": "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/JP/NotoSansJP-Regular.otf",
    "NotoSansJP-Bold.otf": "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/JP/NotoSansJP-Bold.otf",
    "LICENSE-NotoSansJP.txt": "https://github.com/notofonts/noto-cjk/raw/main/Sans/LICENSE",
}
SYSTEM_FONTS = "meiryo,yugothic,yugothicui,msgothic,notosanscjkjp,notosanscjk,hiraginosans"

PROJECT_FILES = ["pitch_core.py", "audio_processor.py", "key_analyzer.py", "midi_loader.py", "singing_trainer.py",
                 "ui_menu.py", "library.py", "test_mic_pitch.py"]

RESULTS = []  # (項目, 状態 OK/WARN/NG, 詳細)


# ---------------------------------------------------------------------------
# 表示・ログ
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()


def head(title):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64, flush=True)


def result(item, status, detail=""):
    mark = {"OK": "[ OK ]", "WARN": "[注意]", "NG": "[失敗]", "SKIP": "[省略]"}[status]
    print(f"  {mark} {item}  {detail}", flush=True)
    RESULTS.append((item, status, detail))


def run(cmd, timeout=None, show=True):
    """コマンドを実行し、出力を流しながら (戻り値, 出力) を返す。"""
    if show:
        print("    $ " + " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    lines = []
    t0 = time.time()
    for line in p.stdout:
        lines.append(line)
        if show:
            # 進捗バー (\r) は最後の部分だけ表示
            print("      " + line.rstrip().split("\r")[-1][:150], flush=True)
        if timeout and time.time() - t0 > timeout:
            p.kill()
            lines.append("\n[時間切れ]\n")
            break
    p.wait()
    return p.returncode, "".join(lines)


def import_ok(module):
    """別プロセスで import できるか (インストール直後でも正しく判定するため)。"""
    code, out = run([sys.executable, "-c", f"import {module}"], show=False)
    return code == 0, out.strip().splitlines()[-1] if out.strip() else ""


def version_of(module):
    code, out = run([sys.executable, "-c", f"import {module} as m; print(getattr(m, '__version__', '?'))"], show=False)
    return out.strip().splitlines()[-1] if code == 0 and out.strip() else "?"


# ---------------------------------------------------------------------------
# 1. ライブラリ
# ---------------------------------------------------------------------------
def check_python():
    head("0. Python の確認")
    v = sys.version_info
    print(f"  Python {platform.python_version()} ({platform.architecture()[0]}) / {platform.system()} {platform.release()}")
    print(f"  場所: {sys.executable}")
    if v < (3, 9):
        result("Python のバージョン", "NG", "3.9 以上が必要です (3.11 または 3.12 を推奨)")
        return False
    if v >= (3, 14):
        result("Python のバージョン", "WARN", "3.14 以降は AI ライブラリ (torch) が未対応の場合があります。3.12 を推奨")
    else:
        result("Python のバージョン", "OK", platform.python_version())
    if platform.architecture()[0] != "64bit":
        result("64bit 版 Python", "NG", "torch (AI) は 64bit 版の Python が必要です")
    return True


# ---------------------------------------------------------------------------
# グラボ (NVIDIA / CUDA)
# ---------------------------------------------------------------------------
# PyTorch が配布しているグラボ対応版 (新しい順)。(名前, 必要な CUDA のバージョン)
CUDA_BUILDS = [("cu130", 13.0), ("cu128", 12.8), ("cu126", 12.6), ("cu124", 12.4)]
TORCH_INDEX = "https://download.pytorch.org/whl/{}"

GPU_CHECK_CODE = (
    "import torch,sys\n"
    "ok=torch.cuda.is_available()\n"
    "if ok:\n"
    "    x=torch.randn(256,256,device='cuda'); (x@x).sum().item()\n"
    "print('TORCH', torch.__version__, torch.version.cuda, ok, torch.cuda.get_device_name(0) if ok else '-')\n"
)


def nvidia_gpu():
    """NVIDIA のグラボとドライバーが対応する CUDA のバージョン → (名前, 12.8) / None。"""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=20).stdout
        name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=20).stdout.strip().splitlines()
    except Exception:
        return None
    import re

    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
    if not name or not m:
        return None
    return name[0].strip(), float(f"{m.group(1)}.{m.group(2)}")


def torch_status():
    """(入っているか, torch のバージョン, CUDA 版か, グラボで計算できたか, グラボ名)"""
    code, out = run([sys.executable, "-c", GPU_CHECK_CODE], show=False, timeout=300)
    line = next((l for l in out.splitlines() if l.startswith("TORCH ")), None)
    if not line:
        return False, None, False, False, None
    _, ver, cuda, ok, *name = line.split(" ")
    return True, ver, cuda != "None", ok == "True", " ".join(name)


def install_gpu_torch(gpu):
    """グラボ対応版の PyTorch (と torchaudio) に入れ替える。うまくいかなければ CPU 版に戻す。"""
    name, drv = gpu
    builds = [b for b in CUDA_BUILDS if b[1] <= drv + 1e-9]
    if not builds:
        result("グラボ対応版 PyTorch", "WARN",
               f"グラボのドライバーが古いため使えません (対応 CUDA {drv})。NVIDIA のドライバーを更新してから setup.bat をもう一度実行してください")
        return False
    print(f"\n  → グラボ ({name}) で解析できるように、グラボ対応版の PyTorch を入れます", flush=True)
    print("    ダウンロードが大きい (約2〜3GB) ため、10分以上かかることがあります", flush=True)
    for tag, _ in builds:
        print(f"\n    CUDA {tag[2:-1]}.{tag[-1]} 版を試します", flush=True)
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio"], timeout=600)
        code, out = run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "torch", "torchaudio",
                         "--index-url", TORCH_INDEX.format(tag)], timeout=3600)
        inst, ver, is_cuda, works, gname = torch_status()
        if works:
            result("グラボ対応版 PyTorch", "OK", f"torch {ver} をインストールしました → {gname} で解析します")
            return True
        print(f"    (この版ではグラボを使えませんでした)", flush=True)
    print("\n    グラボ対応版が使えなかったため、CPU 版に戻します", flush=True)
    run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "torch", "torchaudio"], timeout=3600)
    result("グラボ対応版 PyTorch", "WARN", "入れられませんでした。CPU で解析します (時間はかかりますが使えます)")
    return False


def check_gpu(install=True):
    """グラボが使えるか確認し、必要ならグラボ対応版に入れ替える。"""
    head("1-2. グラボ（GPU）の確認")
    gpu = nvidia_gpu()
    inst, ver, is_cuda, works, gname = torch_status()
    if not gpu:
        result("NVIDIA のグラボ", "SKIP", "見つかりません。CPU で解析します (NVIDIA 以外のグラボは非対応です)")
        return
    print(f"  グラボ: {gpu[0]} / ドライバーの対応 CUDA: {gpu[1]}")
    if works:
        result("グラボで解析", "OK", f"{gname} を使います (torch {ver})")
        return
    if not install:
        result("グラボで解析", "WARN", f"今の PyTorch は {'CPU 専用版' if not is_cuda else 'グラボを使えない状態'} です")
        return
    install_gpu_torch(gpu)


def check_packages(install=True):
    head("1. ライブラリの確認とインストール")
    for name, why in NOT_NEEDED.items():
        installed, _ = import_ok(name)
        note = "（入っていますが使いません）" if installed else ""
        result(f"{name}", "SKIP", f"インストールしません{note}: {why}")

    missing_required = []
    for name, specs, required, purpose, module in PACKAGES:
        ok, _ = import_ok(module)
        if ok:
            result(f"{name}", "OK", f"v{version_of(name if name != 'PyInstaller' else 'PyInstaller')}  … {purpose}")
            continue
        if not install:
            result(f"{name}", "NG" if required else "WARN", f"未インストール … {purpose}")
            if required:
                missing_required.append(name)
            continue
        print(f"\n  → {name} をインストールします ({purpose})", flush=True)
        if name == "torch":
            gpu = nvidia_gpu()
            if gpu and install_gpu_torch(gpu):
                continue
        done = False
        for spec in specs:
            code, out = run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", spec], timeout=1800)
            ok, err = import_ok(module)
            if ok:
                done = True
                break
            print(f"    (この方法では入りませんでした: {err[:120]}) ", flush=True)
        if done:
            result(f"{name}", "OK", f"インストールしました v{version_of(name)}  … {purpose}")
        else:
            result(f"{name}", "NG" if required else "WARN", f"インストールできませんでした … {purpose}")
            if required:
                missing_required.append(name)

    # torchcrepe などで torch が入れ替わっていないか、demucs がまだ動くか確認
    for module in ("torch", "demucs.pretrained", "demucs.apply"):
        ok, err = import_ok(module)
        if not ok:
            result(f"{module} の再確認", "NG", err[:150])
    return not missing_required


# ---------------------------------------------------------------------------
# 2. フォント
# ---------------------------------------------------------------------------
def check_font():
    head("2. 日本語フォントの確認")
    local = [f for f in os.listdir(FONT_DIR) if f.lower().endswith((".otf", ".ttf", ".ttc"))] if os.path.isdir(FONT_DIR) else []
    if local:
        result("日本語フォント", "OK", f"fonts フォルダ: {', '.join(local)}")
        return True
    sys_font = None
    try:
        import pygame

        pygame.font.init()
        sys_font = pygame.font.match_font(SYSTEM_FONTS)
    except Exception:
        pass
    if sys_font:
        result("日本語フォント", "OK", f"PC に入っているフォントを使います: {os.path.basename(sys_font)}（ダウンロード不要）")
        return True
    print("  PC に日本語フォントが見つからないため、Noto Sans JP をダウンロードします (約9MB)")
    os.makedirs(FONT_DIR, exist_ok=True)
    try:
        for fname, url in FONT_FILES.items():
            dest = os.path.join(FONT_DIR, fname)
            print(f"    {fname} … ", end="", flush=True)
            urllib.request.urlretrieve(url, dest)
            print(f"{os.path.getsize(dest) // 1024} KB", flush=True)
        result("日本語フォント", "OK", "Noto Sans JP (Regular / Bold) を fonts フォルダに配置しました")
        return True
    except Exception as e:
        result("日本語フォント", "NG", f"ダウンロード失敗: {e}")
        return False


# ---------------------------------------------------------------------------
# 3. テスト用の曲
# ---------------------------------------------------------------------------
TEST_BPM = 100
TEST_INTRO = 2.0  # 歌が始まるまで (MIDI はここを 0 として書くので、自動でずれを合わせられるか確認できる)
# (拍数, MIDI 番号) … ハ長調のメロディ (ドレミファミレド / ミファソラソファミ / ド ド ド ド / ドドレレミミファファミレド)
TEST_MELODY = [(1, 60), (1, 62), (1, 64), (1, 65), (1, 64), (1, 62), (2, 60),
               (1, 64), (1, 65), (1, 67), (1, 69), (1, 67), (1, 65), (2, 64),
               (2, 60), (2, 60), (2, 60), (2, 60),
               (0.5, 60), (0.5, 60), (0.5, 62), (0.5, 62), (0.5, 64), (0.5, 64), (0.5, 65), (0.5, 65),
               (1, 64), (1, 62), (2, 60)]
TEST_CHORDS = [(48, 52, 55), (53, 57, 60), (55, 59, 62), (48, 52, 55)]  # C F G C


def melody_events():
    beat = 60.0 / TEST_BPM
    t, out = 0.0, []
    for beats, m in TEST_MELODY:
        out.append((t, t + beats * beat * 0.92, m))
        t += beats * beat
    return out, t


def make_test_song(vocal_only=False):
    import numpy as np

    sr = 44100
    beat = 60.0 / TEST_BPM
    mel, mel_len = melody_events()
    total = TEST_INTRO + mel_len + 2.0
    n = int(total * sr)
    vocal = np.zeros(n)
    rng = np.random.default_rng(7)
    for s, e, m in mel:
        a, b = int((s + TEST_INTRO) * sr), int((e + TEST_INTRO) * sr)
        tt = np.arange(b - a) / sr
        f0 = 440 * 2 ** ((m - 69) / 12) * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * tt) * np.clip(tt / 0.3, 0, 1))
        ph = 2 * np.pi * np.cumsum(f0) / sr
        v = sum(a_ * np.sin((k + 1) * ph) for k, a_ in enumerate([1, .8, .9, .55, .35, .25, .15, .1]))
        env = np.minimum(1, tt / 0.03) * np.clip((tt[-1] - tt) / 0.05, 0, 1)
        vocal[a:b] += v * env * 0.10
    acc = np.zeros(n)
    t = np.arange(n) / sr
    bar = 4 * beat
    k = 0
    while (k * bar) < total - 0.5:
        a, b = int(k * bar * sr), min(n, int((k + 1) * bar * sr))
        ch = TEST_CHORDS[k % 4]
        for m in ch:
            f = 440 * 2 ** ((m - 69) / 12)
            acc[a:b] += 0.05 * np.sin(2 * np.pi * f * t[a:b]) * np.exp(-(t[a:b] - k * bar) * 0.6)
        fb = 440 * 2 ** ((ch[0] - 12 - 69) / 12)   # ベース
        acc[a:b] += 0.08 * np.sin(2 * np.pi * fb * t[a:b])
        for j in range(4):                         # ドラム (ノイズ)
            s0 = int((k * bar + j * beat) * sr)
            ln = min(int(0.08 * sr), n - s0)
            if ln > 0:
                acc[s0:s0 + ln] += rng.standard_normal(ln) * np.exp(-np.arange(ln) / sr * 40) * (0.12 if j % 2 == 0 else 0.06)
        k += 1
    if vocal_only:
        acc[:] = 0
    left = vocal + acc * 0.9
    right = vocal + acc * 1.1
    mix = np.stack([left, right], axis=1)
    mix = mix / np.max(np.abs(mix)) * 0.8
    return sr, (mix * 32767).astype(np.int16)


def write_wav(path, sr, data):
    from scipy.io import wavfile

    wavfile.write(path, sr, data)


def write_mp3(path, sr, data):
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(192)
    enc.set_in_sample_rate(sr)
    enc.set_channels(2)
    enc.set_quality(2)
    mp3 = enc.encode(data.tobytes()) + enc.flush()
    with open(path, "wb") as f:
        f.write(mp3)


def write_test_midi(path):
    """メロディ (Vocal) と和音 (Piano) の 2 パート。歌い出しを 0 秒として書く。"""
    div = 480
    beat_ticks = div

    def vl(x):
        b = [x & 0x7F]
        x >>= 7
        while x:
            b.insert(0, (x & 0x7F) | 0x80)
            x >>= 7
        return bytes(b)

    def track(events, name):
        events = sorted(events, key=lambda e: (e[0], e[1][0] & 0xF0 == 0x90))  # 同時刻は note off を先に
        body, last = b"\x00\xff\x03" + vl(len(name)) + name, 0
        for t, msg in events:
            body += vl(t - last) + msg
            last = t
        body += b"\x00\xff\x2f\x00"
        return b"MTrk" + struct.pack(">I", len(body)) + body

    tempo = b"\x00\xff\x51\x03" + int(60e6 / TEST_BPM).to_bytes(3, "big") + b"\x00\xff\x2f\x00"
    t0 = b"MTrk" + struct.pack(">I", len(tempo)) + tempo
    ev, t = [], 0
    for beats, m in TEST_MELODY:
        d = int(beats * beat_ticks)
        ev += [(t, bytes([0x90, m, 100])), (t + int(d * 0.92), bytes([0x80, m, 64]))]
        t += d
    ev_c = []
    for k in range(int(t / (4 * beat_ticks)) + 1):
        for m in TEST_CHORDS[k % 4]:
            ev_c += [(k * 4 * beat_ticks, bytes([0x91, m, 70])), ((k + 1) * 4 * beat_ticks - 10, bytes([0x81, m, 64]))]
    data = b"MThd" + struct.pack(">IHHH", 6, 1, 3, div) + t0 + track(ev, b"Vocal") + track(ev_c, b"Piano")
    with open(path, "wb") as f:
        f.write(data)


def make_test_files():
    head("3. テスト用の曲を作成 (tests フォルダ)")
    os.makedirs(TEST_DIR, exist_ok=True)
    try:
        sr, data = make_test_song()
    except Exception as e:
        result("テスト曲の作成", "NG", str(e))
        return None
    wav = os.path.join(TEST_DIR, "test_song.wav")
    write_wav(wav, sr, data)
    result("test_song.wav", "OK", f"{len(data) / sr:.0f}秒 (歌声の代わりの音 + 伴奏・ベース・ドラム)")
    target = wav
    try:
        mp3 = os.path.join(TEST_DIR, "test_song.mp3")
        write_mp3(mp3, sr, data)
        result("test_song.mp3", "OK", f"{os.path.getsize(mp3) // 1024} KB")
        target = mp3
    except Exception as e:
        result("test_song.mp3", "WARN", f"作成できませんでした ({e})。WAV で確認を続けます")
    try:
        mid = os.path.splitext(target)[0] + ".mid"
        write_test_midi(mid)
        result("test_song.mid", "OK", f"曲と同じ名前なので、アプリ起動時に自動で読み込まれます (歌い出しを {TEST_INTRO:.0f}秒ずらして作成)")
    except Exception as e:
        result("test_song.mid", "WARN", str(e))
    return target


# ---------------------------------------------------------------------------
# 4. 機器の確認
# ---------------------------------------------------------------------------
def check_devices():
    head("4. マイク・スピーカー・画面の確認")
    # マイク
    try:
        import numpy as np
        import sounddevice as sd

        dev = sd.query_devices(kind="input")
        print(f"  マイク: {dev['name']}")
        print("  2秒間マイクの音を録音します (何か声を出してみてください)…", flush=True)
        rec = sd.rec(int(2 * 44100), samplerate=44100, channels=1, dtype="float32")
        sd.wait()
        db = 20 * np.log10(np.sqrt(np.mean(rec ** 2)) + 1e-12)
        status = "OK" if db > -70 else "WARN"
        result("マイク入力", status, f"{dev['name']} / 音量 {db:.0f} dB" + ("" if status == "OK" else "（ほぼ無音です。マイクの接続・許可を確認してください）"))
    except Exception as e:
        result("マイク入力", "WARN", f"使用できません: {e}（練習画面は使えますが自分の声は表示されません）")

    # 画面・スピーカー
    try:
        import numpy as np
        import pygame

        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        scr = pygame.display.set_mode((560, 200))
        pygame.display.set_caption("セットアップ確認")
        sys.path.insert(0, HERE)
        from singing_trainer import Fonts, block_device_events, safe_events

        block_device_events(pygame)

        fonts = Fonts(pygame)
        t = np.arange(int(0.6 * 44100)) / 44100
        tone = (np.sin(2 * np.pi * 523.25 * t) * np.minimum(1, (0.6 - t) * 8) * 0.25 * 32767).astype(np.int16)
        snd = pygame.sndarray.make_sound(np.ascontiguousarray(np.stack([tone, tone], axis=1)))
        snd.play()
        t_end = time.time() + 2.0
        while time.time() < t_end:
            for _ in safe_events(pygame):
                pass
            scr.fill((14, 16, 22))
            scr.blit(fonts.render("キー：ド・メジャー（C Major）／ハ長調", 26, (255, 214, 102), bold=True), (20, 40))
            scr.blit(fonts.render("ド レ ミ ファ ソ ラ シ  ♯ ♭ ● ▶  (日本語の表示テスト)", 20, (236, 238, 244)), (20, 100))
            pygame.display.flip()
            time.sleep(0.03)
        pygame.quit()
        result("画面表示・日本語フォント", "OK", f"フォント: {os.path.basename(fonts.path or '標準(日本語なし)')}")
        result("スピーカー", "OK", "「ポーン」とドの音が鳴りました（聞こえなければ音量・出力先を確認）")
    except Exception as e:
        result("画面表示・スピーカー", "NG", str(e))


# ---------------------------------------------------------------------------
# 5. テストドライラン
# ---------------------------------------------------------------------------
def dry_run(song, quick=False, launch=True):
    head("5. テストドライラン")
    missing = [f for f in PROJECT_FILES if not os.path.exists(os.path.join(HERE, f))]
    if missing:
        result("プロジェクトのファイル", "NG", "見つかりません: " + ", ".join(missing))
        return
    result("プロジェクトのファイル", "OK", f"{len(PROJECT_FILES)} ファイルそろっています")

    code, out = run([sys.executable, "-c", "import pitch_core, audio_processor, key_analyzer, midi_loader, ui_menu, library, singing_trainer"], show=False)
    if code != 0:
        result("プログラムの読み込み", "NG", out.strip().splitlines()[-1] if out.strip() else "")
        return
    result("プログラムの読み込み", "OK")

    code, out = run([sys.executable, "test_mic_pitch.py", "--selftest"], show=False)
    result("音程検出のセルフテスト", "OK" if "すべて合格" in out else "NG", "test_mic_pitch.py --selftest")

    sep = ["--sep-model", "htdemucs", "--crepe-model", "tiny"] if quick else []
    print("\n  AI 分離と音程解析を実行します (初回は AI モデルのダウンロードがあり、数分かかります)…", flush=True)
    t0 = time.time()
    code, out = run([sys.executable, "audio_processor.py", song] + sep, timeout=3600)
    ok = code == 0 and "完了" in out
    engine = "crepe" if "CREPE 完了" in out else ("pyin" if "pYIN 完了" in out else "yin")
    sep_used = "htdemucs_ft" if "htdemucs_ft" in out and "標準モデル htdemucs で続け" not in out else "htdemucs"
    result("AI ボーカル分離 + お手本作成", "OK" if ok else "NG",
           f"{time.time() - t0:.0f}秒 / 分離 {sep_used} / 音程解析 {engine}" if ok else out.strip().splitlines()[-1][:150])
    if not ok:
        return
    bars = [l for l in out.splitlines() if "完了: お手本バー" in l]
    if bars:
        print("    " + bars[0].strip())
        if " 0 本" in bars[0]:
            print("    ※ テスト曲の「声」は機械で作った音なので、AI が歌声ではなく楽器と判断することがあります。")
            print("      アプリや AI の不具合ではありません (本物の歌の曲では、ちゃんとお手本が作られます)。")

    code, out = run([sys.executable, "key_analyzer.py", song], show=False)
    key_line = next((l.strip() for l in out.splitlines() if "キー：" in l), "")
    result("キーの判定", "OK" if "ド・メジャー" in key_line else "WARN",
           key_line + ("" if "ド・メジャー" in key_line else "（テスト曲はハ長調で作っています）"))

    mid = os.path.splitext(song)[0] + ".mid"
    if os.path.exists(mid):
        # MIDI の照合は「声だけ」のテスト音で確かめる (AI 分離の結果に左右されないように)
        sr_v, voc = make_test_song(vocal_only=True)
        voc_path = os.path.join(TEST_DIR, "test_vocal_only.wav")
        write_wav(voc_path, sr_v, voc)
        code, out = run([sys.executable, "-c",
                         "import audio_processor as ap, midi_loader as ml, sys;"
                         "ap.log=lambda m: None;"
                         f"d=ap.analyze_vocals({voc_path!r});"
                         f"g=ml.build_guide({mid!r}, d, None, log=lambda m: None);"
                         "i=g['info']; print('MIDI_RESULT', i['name'], round(i['offset'],2), i['stretch'], i['octave'], round(i['score'],2))"],
                        show=False)
        line = next((l for l in out.splitlines() if l.startswith("MIDI_RESULT")), "")
        if line:
            _, name, off, st, octv, sc = line.split()
            good = name == "Vocal" and abs(float(off) - TEST_INTRO) < 0.15
            result("MIDI の自動照合", "OK" if good else "WARN",
                   f"パート「{name}」/ ずれ {float(off):+.2f}秒 (正解 +{TEST_INTRO:.2f}) / 一致率 {float(sc):.0%}")
        else:
            result("MIDI の自動照合", "NG", out.strip().splitlines()[-1][:150] if out.strip() else "")

    if launch:
        print("\n  アプリを起動します。約8秒間、練習画面と再生を確認してから自動で閉じます…", flush=True)
        code, out = run([sys.executable, "singing_trainer.py", song, "--autoquit", "8"] + sep, timeout=600)
        ok = "[動作確認] 画面表示・再生 OK" in out
        line = next((l.strip() for l in out.splitlines() if "[動作確認] 画面表示" in l), out.strip().splitlines()[-1] if out.strip() else "")
        result("アプリの起動 (ドライラン)", "OK" if ok else "NG", line[:150])
    else:
        result("アプリの起動 (ドライラン)", "SKIP", "--no-launch 指定")


def summary(song):
    head("まとめ")
    ng = [r for r in RESULTS if r[1] == "NG"]
    warn = [r for r in RESULTS if r[1] == "WARN"]
    print(f"  OK {sum(r[1] == 'OK' for r in RESULTS)} 件 / 注意 {len(warn)} 件 / 失敗 {len(ng)} 件")
    for item, st, detail in ng + warn:
        print(f"   - [{'失敗' if st == 'NG' else '注意'}] {item}: {detail}")
    if not ng:
        print("\n  準備完了です。練習を始めるには:")
        print("      python singing_trainer.py   (曲の一覧が開きます。フォルダを選んで曲をダブルクリック)")
        if song:
            print(f"  テスト曲で試すには:\n      python singing_trainer.py {os.path.relpath(song, HERE)}")
    else:
        print("\n  失敗した項目があります。この画面 (または setup_log.txt) の内容を Claude に貼ってください。")
    print(f"\n  ログ: {LOG_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Pitch Luna — If Vocal セットアップ & 動作確認")
    ap.add_argument("--no-install", action="store_true", help="インストールせずに確認だけ")
    ap.add_argument("--quick", action="store_true", help="解析を速い設定で (htdemucs + CREPE tiny)")
    ap.add_argument("--no-launch", action="store_true", help="最後のアプリ起動をしない")
    ap.add_argument("--no-devices", action="store_true", help="マイク・画面の確認をしない")
    args = ap.parse_args()

    os.chdir(HERE)
    sys.stdout = Tee(LOG_PATH)
    print(f"Pitch Luna — If Vocal セットアップ  ({time.strftime('%Y-%m-%d %H:%M')})")
    print(f"プロジェクトフォルダ: {HERE}")
    if not check_python():
        summary(None)
        return
    pk = check_packages(install=not args.no_install)
    if pk:
        check_gpu(install=not args.no_install)
    if not pk:
        print("\n  必須ライブラリが足りないため、ここで止めます。")
        summary(None)
        return
    check_font()
    song = make_test_files()
    if not args.no_devices:
        check_devices()
    if song:
        dry_run(song, quick=args.quick, launch=not args.no_launch)
    summary(song)


if __name__ == "__main__":
    main()
