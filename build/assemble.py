# -*- coding: utf-8 -*-
"""
ビルドした PitchLuna (exe + ライブラリ) に、ソフト本体の .py・フォント・取扱説明書などを入れて仕上げる。

    python build/assemble.py dist/PitchLuna  → dist/PitchLuna-<版番号>/
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    src = sys.argv[1]
    m = json.load(open(os.path.join(ROOT, "version.json"), encoding="utf-8"))
    ver = m["version"]
    for rel in m["files"]:
        dst = os.path.join(src, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)
    shutil.copy2(os.path.join(ROOT, "version.json"), os.path.join(src, "version.json"))
    info = {"build": ver}
    try:
        import torch

        info.update(torch=torch.__version__, cuda=torch.version.cuda)
    except Exception:
        pass
    json.dump(info, open(os.path.join(src, "build.json"), "w", encoding="utf-8"), ensure_ascii=False)
    out = os.path.join(os.path.dirname(src), f"PitchLuna-{ver}")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.rename(src, out)
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
    print(f"{out}  {size / 1e9:.2f} GB")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            f.write(f"dir={out}\nversion={ver}\n")


if __name__ == "__main__":
    main()
