# -*- coding: utf-8 -*-
"""
ソフト本体 (.py) が使っているライブラリを調べて、pitchluna_deps.py を作る (ビルド用)。

  本体の .py は exe の外に置くので、PyInstaller は中身を見ません。
  そこで「本体が import しているもの」を一覧にしたファイルを作り、exe に入れるライブラリを決めます。
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = ["singing_trainer.py", "audio_processor.py", "audio_fx.py", "beat_analyzer.py", "key_analyzer.py",
       "midi_loader.py", "library.py", "pitch_core.py", "ui_menu.py", "theme.py", "updater.py"]
OWN = {os.path.splitext(f)[0] for f in os.listdir(ROOT) if f.endswith(".py")}
# 本体からは直接 import していないが、中で使われるもの
EXTRA = ["tkinter.filedialog", "tkinter.simpledialog", "pygame._sdl2.audio", "soundfile", "audioread",
         "demucs.apply", "demucs.pretrained", "demucs.htdemucs", "demucs.hdemucs", "demucs.demucs",
         "demucs.states", "demucs.repo", "torchcrepe", "torchaudio", "julius", "openunmix", "einops",
         "librosa", "librosa.core", "librosa.feature", "numba", "concurrent.futures", "webbrowser",
         "http.client", "ssl", "email.utils"]


def main():
    mods = set()
    for f in APP:
        tree = ast.parse(open(os.path.join(ROOT, f), encoding="utf-8").read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    mods.add(a.name)
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods.add(n.module)
                for a in n.names:
                    mods.add(n.module + "." + a.name)        # from pygame._sdl2 import audio など
    mods |= set(EXTRA)
    mods = sorted(m for m in mods if m.split(".")[0] not in OWN and m != "__future__")
    lines = ["# 自動で作ったファイル (build/gen_deps.py)。ビルドに入れるライブラリの一覧です。", "# fmt: off"]
    for m in mods:
        lines.append(f"try:\n    import {m}  # noqa\nexcept Exception:\n    pass")
    open(os.path.join(ROOT, "pitchluna_deps.py"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"pitchluna_deps.py: {len(mods)} 個")


if __name__ == "__main__":
    sys.exit(main())
