# -*- coding: utf-8 -*-
"""
新しい版の version.json を作る (開発用)。

    python make_release.py 1.1.0 "変更点1" "変更点2" ...

  ・下の FILES のファイルの確認用の値 (SHA-256) と大きさを version.json に書きます。
  ・これまでの変更点 (history) は残し、新しい版を先頭に足します。
"""

import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# アップデートで配るファイル (ここに無いものは触らない)
FILES = [
    "singing_trainer.py", "audio_processor.py", "audio_fx.py", "beat_analyzer.py", "key_analyzer.py",
    "midi_loader.py", "library.py", "pitch_core.py", "ui_menu.py", "theme.py", "updater.py",
    "setup.bat", "setup_check.py", "test_mic_pitch.py", "取扱説明書.txt",
    "fonts/NotoSansJP-Regular.otf", "fonts/NotoSansJP-Bold.otf", "fonts/LICENSE-NotoSansJP.txt",
    "assets/icon.png", "assets/icon.ico", "launcher.py",
]

# 必要なライブラリ (入っていなければアップデートのときに pip で入れる)
REQUIREMENTS = [
    {"import": "pygame", "pip": "pygame"},
    {"import": "sounddevice", "pip": "sounddevice"},
    {"import": "numpy", "pip": "numpy"},
    {"import": "scipy", "pip": "scipy"},
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    version = sys.argv[1]
    notes = sys.argv[2:]
    path = os.path.join(HERE, "version.json")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    files = {}
    for rel in FILES:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            print(f"[注意] {rel} がありません (一覧から外します)")
            continue
        files[rel] = {"sha256": sha256(p), "size": os.path.getsize(p)}
    history = old.get("history", [])
    if old.get("version") and old.get("version") != version:
        history = [{"version": old["version"], "date": old.get("date", ""), "notes": old.get("notes", [])}] + history
    m = {"name": "Pitch Luna — If Vocal", "version": version, "date": time.strftime("%Y-%m-%d"),
         "notes": notes, "files": files, "requirements": REQUIREMENTS, "history": history[:30],
         "download_url": "https://github.com/shellbe1967/pitch-luna/actions/workflows/build.yml"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    print(f"version.json を作りました: {version}（ファイル {len(files)} 個）")


if __name__ == "__main__":
    main()
