# -*- coding: utf-8 -*-
"""
同梱の部品 (vendor フォルダの whl / zip) を、使うときだけ展開して読み込む。

  ・ビルド版 (PitchLuna.exe) を作り直さなくても、アップデートで部品を足せるようにするためのもの。
  ・初めて使うときに、ソフトのフォルダの _lib\\<部品の名前>\\ へ展開し、そこから import する。
  ・Python で動かしている場合など、すでにインストールされている部品があればそちらを使う。
"""

import importlib
import os
import shutil
import sys
import threading
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
LIB = os.path.join(HERE, "_lib")
_lock = threading.Lock()


def _platform_ok(fname):
    """whl のファイル名から、このパソコン・Python で使えるか。"""
    base = fname[:-4]
    parts = base.split("-")
    if len(parts) < 5:
        return True                                   # zip など: いつでも使える
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    if plat == "any":
        return True
    ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if py != ver and not py.startswith("py3"):
        return False
    if sys.platform == "win32":
        return plat in ("win_amd64",) if sys.maxsize > 2 ** 32 else plat == "win32"
    return False


def use(module, files):
    """module が import できるようにする。files: vendor の中のファイル名 (この順に展開して sys.path へ)。
    import できたら module を返す。できなければ例外。"""
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    with _lock:
        for fname in files:
            src = os.path.join(VENDOR, fname)
            if not os.path.exists(src) or not _platform_ok(fname):
                continue
            dst = os.path.join(LIB, os.path.splitext(fname)[0])
            ok_mark = os.path.join(dst, ".ok")
            if not os.path.exists(ok_mark):
                tmp = dst + ".tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                with zipfile.ZipFile(src) as z:
                    z.extractall(tmp)
                open(os.path.join(tmp, ".ok"), "w").close()
                shutil.rmtree(dst, ignore_errors=True)
                os.replace(tmp, dst)
            if dst not in sys.path:
                sys.path.insert(0, dst)
        importlib.invalidate_caches()
    return importlib.import_module(module)
