# -*- coding: utf-8 -*-
"""
Pitch Luna — If Vocal の起動用 (ビルド版の PitchLuna.exe の中身)

  ・ソフト本体 (singing_trainer.py などの .py) は exe の横に置いたファイルを読み込みます。
    → 「ヘルプ → アップデート」で .py を入れ替えれば、ビルドし直さなくても新しい版になります。
  ・exe の中には Python と必要なライブラリ (pygame・torch・demucs など) だけが入っています。

  PitchLuna.exe                 ソフトを起動
  PitchLuna.exe --updater ...   アップデートの入れ替え (ソフトから自動で呼ばれる)
  PitchLuna.exe --selftest      動作確認 (ライブラリの読み込みと、短い曲の解析)
"""

import multiprocessing
import os
import runpy
import sys
import traceback


def base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _console():
    """画面の無い exe でも、文字を出す黒い画面を開く (アップデートの進み具合を見せる)。"""
    if os.name == "nt" and getattr(sys, "frozen", False):
        try:
            import ctypes

            ctypes.windll.kernel32.AllocConsole()      # すでに黒い画面があるときは何もしない
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
            sys.stderr = sys.stdout
            sys.stdin = open("CONIN$", "r", encoding="utf-8")
        except Exception:
            pass


def _log_to_file(base):
    """画面の無い exe では print の行き先が無いので、ログファイルに書く (困ったときの手がかり)。"""
    if sys.stdout is None or sys.stderr is None:
        try:
            f = open(os.path.join(base, "pitchluna_log.txt"), "w", encoding="utf-8", buffering=1)
        except Exception:
            f = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f


def selftest(base):
    """ビルドの確認: すべてのライブラリが読み込めて、短い曲を最後まで解析できるか。"""
    import json
    import time
    import numpy as np

    out = {"ok": False, "steps": []}

    def step(name, fn):
        t0 = time.time()
        try:
            r = fn()
            out["steps"].append({"name": name, "ok": True, "sec": round(time.time() - t0, 1), "info": str(r)[:200]})
            print(f"OK  {name} ({time.time() - t0:.1f}s) {str(r)[:120]}", flush=True)
            return r
        except Exception as e:
            out["steps"].append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()[-1500:]})
            print(f"NG  {name}: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            raise

    try:
        import pitchluna_deps  # noqa: F401  (ビルドに入れたライブラリ)

        step("pygame", lambda: __import__("pygame").version.ver)
        step("sounddevice", lambda: __import__("sounddevice").__version__)
        step("torch", lambda: (__import__("torch").__version__, __import__("torch").version.cuda))
        step("torchaudio", lambda: __import__("torchaudio").__version__)
        step("librosa", lambda: __import__("librosa").__version__)
        step("torchcrepe", lambda: __import__("torchcrepe").__name__)
        step("demucs", lambda: __import__("demucs.pretrained", fromlist=["get_model"]).__name__)
        step("tkinter", lambda: __import__("tkinter").TkVersion)
        step("app modules", lambda: [__import__(m).__name__ for m in (
            "pitch_core", "theme", "ui_menu", "library", "audio_processor", "key_analyzer", "beat_analyzer",
            "midi_loader", "audio_fx", "updater", "singing_trainer")])

        tmp = os.path.join(base, "_selftest")
        os.makedirs(tmp, exist_ok=True)

        def make_song():
            from scipy.io import wavfile

            sr = 44100
            t = np.arange(int(sr * 12)) / sr
            mel = [60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 67, 72]
            f = np.array([440 * 2 ** ((mel[min(len(mel) - 1, int(x))] - 69) / 12) for x in t])
            voice = 0.25 * np.sin(2 * np.pi * np.cumsum(f) / sr) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
            beat = np.zeros_like(t)
            for k in range(24):
                i = int(k * 0.5 * sr)
                n = min(len(t) - i, 2000)
                beat[i:i + n] += 0.4 * np.exp(-np.arange(n) / 300) * np.sin(2 * np.pi * 60 * np.arange(n) / sr)
            chord = 0.08 * (np.sin(2 * np.pi * 130.8 * t) + np.sin(2 * np.pi * 164.8 * t) + np.sin(2 * np.pi * 196 * t))
            y = voice + beat + chord
            y = np.stack([y, y], 1)
            p = os.path.join(tmp, "selftest_song.wav")
            wavfile.write(p, sr, (y / np.abs(y).max() * 0.8 * 32767).astype(np.int16))
            return p

        song = step("テスト曲を作る", make_song)

        def analyze():
            import audio_processor

            audio_processor.USE_GPU = False
            res = audio_processor.process(song, cache_root=os.path.join(tmp, "cache"), sep_model="htdemucs",
                                          engine="crepe", crepe_model="tiny")
            return f"お手本バー {len(res['data']['notes'])} 本"

        step("AI 解析 (分離 + 音程 CREPE)", analyze)
        step("キー解析", lambda: __import__("key_analyzer").analyze_key(song)["short"])
        step("拍の解析", lambda: __import__("beat_analyzer").analyze(os.path.join(
            tmp, "cache", os.listdir(os.path.join(tmp, "cache"))[0], "accompaniment.wav"))["tempo"])

        def fx():
            import audio_fx

            x = (np.random.default_rng(0).normal(0, 3000, (44100 * 3, 2))).astype(np.int16)
            return audio_fx.process(x, 44100, 0.8, 2).shape

        step("速度・キー変更", fx)

        def gui():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            import pygame
            import singing_trainer as st

            pygame.init()
            pygame.display.set_mode((1280, 720))
            f = st.Fonts(pygame)
            s = f.render("ドレミ Pitch Luna — If Vocal", 20, (255, 255, 255))
            pygame.quit()
            return s.get_size()

        step("画面と日本語フォント", gui)
        out["ok"] = True
    except Exception:
        out["ok"] = False
    with open(os.path.join(base, "selftest_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("SELFTEST " + ("OK" if out["ok"] else "NG"), flush=True)
    return 0 if out["ok"] else 1


def main():
    multiprocessing.freeze_support()
    base = base_dir()
    if base not in sys.path:
        sys.path.insert(0, base)                 # exe の横の .py を読み込む
    os.chdir(base)
    args = sys.argv[1:]
    if args[:1] == ["--updater"]:
        _console()
        sys.argv = [os.path.join(base, "updater.py")] + args[1:]
        import updater

        updater.main()
        return 0
    if args[:1] == ["--selftest"]:
        _console()
        _log_to_file(base)
        return selftest(base)
    _log_to_file(base)
    sys.argv = [os.path.join(base, "singing_trainer.py")] + args
    try:
        runpy.run_path(os.path.join(base, "singing_trainer.py"), run_name="__main__")
    except SystemExit as e:
        return e.code or 0
    except Exception:
        traceback.print_exc()
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, "起動できませんでした。\n\n" + traceback.format_exc()[-1200:] +
                                             "\n\npitchluna_log.txt に詳しい記録があります。", "Pitch Luna — If Vocal", 0x10)
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
