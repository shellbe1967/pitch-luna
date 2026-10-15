# -*- coding: utf-8 -*-
"""
Pitch Luna — If Vocal のアップデート

  GitHub (shellbe1967/pitch-luna) に置いた最新版と比べて、変わったファイルだけを入れ替えます。

  しくみ
    1. version.json (最新の版番号・変更点・ファイルの一覧と確認用の値) を読む
    2. 自分のファイルと確認用の値 (SHA-256) を比べ、違うファイルだけダウンロード
       → ダウンロードしたファイルの値が合っているか確かめる (壊れていたら使わない)
    3. ソフトを終了してから、このファイルが入れ替えを行う
         ・入れ替える前のファイルは backup\\版番号_日時\\ に残す
         ・途中で失敗したら、元に戻す
         ・新しく必要になったライブラリがあれば pip でインストール
         ・終わったらソフトを起動し直す
    設定 (settings.json)・解析した曲 (cache)・テイク・お手本の編集版 は触りません。

  コマンド
    python updater.py --check              最新版があるか調べる
    python updater.py --update             ダウンロードして入れ替える (ソフトを閉じてから)
    python updater.py --zip 更新.zip        ZIP から入れ替える (インターネットを使わない)
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "shellbe1967/pitch-luna"
BRANCH = "main"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
MANIFEST = "version.json"
STAGE_DIR = os.path.join(HERE, "_update")
BACKUP_DIR = os.path.join(HERE, "backup")
LOG = os.path.join(HERE, "update_log.txt")
FROZEN = bool(getattr(sys, "frozen", False))      # ビルド版 (PitchLuna.exe) で動いているか
# 入れ替えてはいけないもの (念のため。version.json の一覧にも入れない)
PROTECTED = ("settings.json", "cache/", "backup/", "_update/", "takes/", "setup_log.txt", "update_log.txt",
             "_internal/", "PitchLuna.exe", "build.json")


def build_version():
    """ビルド版の版番号 (exe と同梱ライブラリの版)。Python で動かしているときは None。"""
    if not FROZEN:
        return None
    try:
        with open(os.path.join(HERE, "build.json"), encoding="utf-8") as f:
            return json.load(f).get("build", "0.0.0")
    except Exception:
        return "0.0.0"


# ---------------------------------------------------------------------------
# 版番号
# ---------------------------------------------------------------------------
def parse_version(v):
    out = []
    for p in str(v).strip().lstrip("vV").split("."):
        num = "".join(c for c in p if c.isdigit())
        out.append(int(num) if num else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


def local_manifest():
    try:
        with open(os.path.join(HERE, MANIFEST), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": "0.0.0", "files": {}}


def local_version():
    return local_manifest().get("version", "0.0.0")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(rel):
    """一覧のファイル名が、ソフトのフォルダの外や守るべき場所を指していないか。"""
    rel = rel.replace("\\", "/")
    if rel.startswith("/") or ".." in rel.split("/") or ":" in rel:
        return False
    return not any(rel == p.rstrip("/") or rel.startswith(p) for p in PROTECTED)


# ---------------------------------------------------------------------------
# ダウンロード
# ---------------------------------------------------------------------------
def _url(rel):
    return RAW + urllib.parse.quote(rel.replace("\\", "/"))


def fetch(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "PitchLuna-Updater", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def remote_manifest(timeout=10):
    data = fetch(_url(MANIFEST) + f"?t={int(time.time())}", timeout)
    return json.loads(data.decode("utf-8"))


def check(timeout=10):
    """→ {"current", "latest", "available", "notes", "manifest", "error"}"""
    cur = local_version()
    try:
        m = remote_manifest(timeout)
    except Exception as e:
        return {"current": cur, "latest": None, "available": False, "notes": [], "manifest": None,
                "error": f"最新版を確かめられませんでした（インターネットの接続を確認してください）：{e}"}
    latest = m.get("version", "0.0.0")
    needs_build = False
    if FROZEN and m.get("min_build"):              # 中のライブラリが変わった版 → ZIP をダウンロードし直す
        needs_build = parse_version(build_version()) < parse_version(m["min_build"])
    return {"current": cur, "latest": latest, "available": parse_version(latest) > parse_version(cur),
            "notes": m.get("notes", []), "history": m.get("history", []), "manifest": m, "error": None,
            "needs_build": needs_build, "download_url": m.get("download_url", f"https://github.com/{REPO}/releases")}


def changed_files(manifest):
    """入れ替えが必要なファイル [(相対パス, sha256, size)]"""
    out = []
    for rel, info in sorted(manifest.get("files", {}).items()):
        if not _safe_rel(rel):
            continue
        p = os.path.join(HERE, rel)
        if os.path.exists(p) and os.path.getsize(p) == info.get("size") and sha256_file(p) == info["sha256"]:
            continue
        out.append((rel, info["sha256"], info.get("size")))
    return out


def download(manifest, progress=None):
    """変わったファイルを _update フォルダにダウンロードして、確認用の値を確かめる。→ 準備したフォルダ"""
    files = changed_files(manifest)
    stage = os.path.join(STAGE_DIR, manifest["version"])
    if os.path.isdir(stage):
        shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage, exist_ok=True)
    total = sum((s or 0) for _, _, s in files) or 1
    done = 0
    for rel, sha, size in files:
        data = fetch(_url(rel), timeout=60)
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"{rel} が正しくダウンロードできませんでした（確認用の値が違います）")
        dst = os.path.join(stage, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        done += size or len(data)
        if progress:
            progress(min(1.0, done / total), rel)
    with open(os.path.join(stage, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return stage


def stage_from_zip(zip_path):
    """ZIP (中に version.json と各ファイル) から入れ替えの準備をする。→ (準備したフォルダ, manifest)"""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        root = ""
        if MANIFEST not in names:                      # フォルダごと ZIP にした場合
            cand = [n for n in names if n.endswith("/" + MANIFEST)]
            if not cand:
                raise RuntimeError("ZIP の中に version.json がありません")
            root = cand[0][: -len(MANIFEST)]
        manifest = json.loads(z.read(root + MANIFEST).decode("utf-8"))
        stage = os.path.join(STAGE_DIR, manifest["version"])
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage, exist_ok=True)
        for rel, sha, size in changed_files(manifest):
            data = z.read(root + rel)
            if hashlib.sha256(data).hexdigest() != sha:
                raise RuntimeError(f"ZIP の中の {rel} が壊れています")
            dst = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(data)
        with open(os.path.join(stage, MANIFEST), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
    return stage, manifest


# ---------------------------------------------------------------------------
# 入れ替え (ソフトを閉じた後に行う)
# ---------------------------------------------------------------------------
def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(msg, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def wait_pid(pid, timeout=60):
    """ソフト (pid) が終わるまで待つ。"""
    if not pid:
        return
    t0 = time.time()
    if os.name == "nt":
        try:
            import ctypes

            SYNCHRONIZE = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
            if h:
                ctypes.windll.kernel32.WaitForSingleObject(h, int(timeout * 1000))
                ctypes.windll.kernel32.CloseHandle(h)
            time.sleep(0.5)
            return
        except Exception:
            pass
    while time.time() - t0 < timeout:
        try:
            os.kill(int(pid), 0)
        except Exception:
            return
        time.sleep(0.3)


def _install_requirements(manifest):
    """新しく必要になったライブラリを入れる (入っているものは何もしない)。"""
    if FROZEN:                                     # ビルド版はライブラリを exe の中に持っている
        import importlib.util

        miss = [r["pip"] for r in manifest.get("requirements", []) if r.get("import") and
                importlib.util.find_spec(r["import"]) is None]
        if miss:
            log("このビルドに入っていないライブラリがあります: " + ", ".join(miss) +
                " → 新しいビルド (ZIP) をダウンロードしてください")
        return
    for req in manifest.get("requirements", []):
        mod, pkg = req.get("import"), req.get("pip")
        if not mod or not pkg:
            continue
        r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True)
        if r.returncode == 0:
            continue
        log(f"ライブラリ {pkg} をインストールしています…")
        r = subprocess.run([sys.executable, "-m", "pip", "install", pkg], capture_output=True, text=True)
        log(f"  {'OK' if r.returncode == 0 else '失敗: ' + (r.stderr or '')[-300:]}")


def apply(stage, restart=False, pid=None):
    """準備したフォルダのファイルで入れ替える。失敗したら元に戻す。→ True / False"""
    wait_pid(pid)
    with open(os.path.join(stage, MANIFEST), encoding="utf-8") as f:
        manifest = json.load(f)
    old = local_version()
    new = manifest["version"]
    log(f"=== アップデート {old} → {new} ===")
    files = []
    for root, _, names in os.walk(stage):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), stage).replace("\\", "/")
            if rel != MANIFEST and _safe_rel(rel):
                files.append(rel)
    bdir = os.path.join(BACKUP_DIR, f"{old}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(bdir, exist_ok=True)
    done = []
    try:
        for rel in files:
            dst = os.path.join(HERE, rel)
            if os.path.exists(dst):
                b = os.path.join(bdir, rel)
                os.makedirs(os.path.dirname(b), exist_ok=True)
                shutil.copy2(dst, b)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for i in range(10):                          # 使用中なら少し待って再挑戦
                try:
                    shutil.copy2(os.path.join(stage, rel), dst)
                    break
                except PermissionError:
                    if i == 9:
                        raise
                    time.sleep(0.5)
            done.append(rel)
            log(f"  入れ替え: {rel}")
        shutil.copy2(os.path.join(HERE, MANIFEST), os.path.join(bdir, MANIFEST)) if os.path.exists(
            os.path.join(HERE, MANIFEST)) else None
        shutil.copy2(os.path.join(stage, MANIFEST), os.path.join(HERE, MANIFEST))
    except Exception as e:
        log(f"入れ替えに失敗しました: {e} → 元に戻します")
        for rel in done:
            b = os.path.join(bdir, rel)
            try:
                if os.path.exists(b):
                    shutil.copy2(b, os.path.join(HERE, rel))
                else:
                    os.remove(os.path.join(HERE, rel))
            except Exception:
                pass
        log("元に戻しました")
        if restart:
            _restart()
        return False
    try:
        _install_requirements(manifest)
    except Exception as e:
        log(f"ライブラリの確認でエラー: {e}")
    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    _prune_backups()
    log(f"アップデート完了: {new}（前の版は {bdir} に残しています）")
    if restart:
        _restart()
    return True


def _prune_backups(keep=5):
    try:
        ds = sorted((os.path.join(BACKUP_DIR, d) for d in os.listdir(BACKUP_DIR)), key=os.path.getmtime)
        for d in ds[:-keep]:
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


def _restart():
    log("ソフトを起動し直します…")
    kw = {"cwd": HERE}
    if FROZEN:
        subprocess.Popen([sys.executable], **kw)        # PitchLuna.exe
        return
    if os.name == "nt":
        kw["creationflags"] = 0x00000010                # CREATE_NEW_CONSOLE
    subprocess.Popen([sys.executable, os.path.join(HERE, "singing_trainer.py")], **kw)


def launch_apply(stage, restart=True):
    """ソフトから呼ぶ: 別のウィンドウで入れ替えを始める (ソフトはこの後すぐ終了する)。"""
    kw = {"cwd": HERE}
    if os.name == "nt":
        kw["creationflags"] = 0x00000010                # CREATE_NEW_CONSOLE (進み具合が見える)
    if FROZEN:                                         # PitchLuna.exe --updater --apply ...
        args = [sys.executable, "--updater", "--apply", stage, "--pid", str(os.getpid())]
    else:
        args = [sys.executable, os.path.abspath(__file__), "--apply", stage, "--pid", str(os.getpid())]
    if restart:
        args.append("--restart")
    subprocess.Popen(args, **kw)


def main():
    ap = argparse.ArgumentParser(description="Pitch Luna — If Vocal のアップデート")
    ap.add_argument("--check", action="store_true", help="最新版があるか調べる")
    ap.add_argument("--update", action="store_true", help="ダウンロードして入れ替える (ソフトを閉じてから)")
    ap.add_argument("--zip", help="ZIP ファイルから入れ替える")
    ap.add_argument("--apply", help=argparse.SUPPRESS)
    ap.add_argument("--pid", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--restart", action="store_true", help="終わったらソフトを起動し直す")
    a = ap.parse_args()
    if a.apply:
        ok = apply(a.apply, restart=a.restart, pid=a.pid)
        if not a.restart:
            input("Enter キーで閉じます")
        sys.exit(0 if ok else 1)
    if a.zip:
        stage, m = stage_from_zip(a.zip)
        apply(stage, restart=a.restart)
        return
    r = check()
    if r["error"]:
        print(r["error"])
        sys.exit(1)
    print(f"今の版: {r['current']}　最新版: {r['latest']}")
    if not r["available"]:
        print("最新版です。")
        return
    for n in r["notes"]:
        print("  ・" + n)
    if a.update:
        stage = download(r["manifest"], lambda f, rel: print(f"  {int(f * 100):3d}%  {rel}"))
        apply(stage, restart=a.restart)


if __name__ == "__main__":
    main()
