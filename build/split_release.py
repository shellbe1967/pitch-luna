# -*- coding: utf-8 -*-
"""
ビルドしたフォルダを、GitHub の Releases に置ける大きさ (1ファイル 2GB 未満) の ZIP に分ける。

    python build/split_release.py <ビルドのフォルダ> <版番号> <出力先>

  ・どの ZIP も中に「PitchLuna-<版番号>/」フォルダが入っていて、
    すべてを同じ場所に展開すると 1つのフォルダにまとまります (ファイルの重なりはありません)。
  ・Windows 標準の「すべて展開」で開ける、普通の ZIP です。
  ・どの ZIP にも「はじめにお読みください.txt」を入れます。
"""
import os
import sys
import zlib
import zipfile
from concurrent.futures import ThreadPoolExecutor

LIMIT = int(os.environ.get("SPLIT_LIMIT", "1900000000"))   # 1つの ZIP の上限 (GitHub は 2GiB 未満)
LEVEL = 6

README = """Pitch Luna — If Vocal {ver}

このソフトは、ZIP ファイル {n} 個に分かれています。
ダウンロードした {names} を、
すべて「同じ場所」に展開してください。

  1. どれか 1 つの ZIP を右クリック →「すべて展開」
  2. 展開先の欄の最後にある「\\PitchLuna-{ver}-part…」の部分を消して、
     例えば C:\\Users\\(あなたの名前)\\Downloads のようにする →「展開」
  3. 残りの ZIP も同じように、同じ展開先に展開する
  4. できた「PitchLuna-{ver}」フォルダの中の PitchLuna.exe をダブルクリック

ZIP がそろっていないと、ソフトは起動しません。
くわしい使い方は、フォルダの中の「取扱説明書.txt」をご覧ください。
"""


def deflated_size(path):
    c = zlib.compressobj(LEVEL, zlib.DEFLATED, -15)
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            n += len(c.compress(chunk))
    return n + len(c.flush())


def main():
    src, ver, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    files = []
    for r, _, fs in os.walk(src):
        for f in fs:
            p = os.path.join(r, f)
            files.append((os.path.relpath(p, src).replace(os.sep, "/"), p))
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        sizes = list(ex.map(lambda t: deflated_size(t[1]), files))
    total = sum(sizes)
    print(f"ファイル {len(files)} 個、圧縮後 およそ {total / 1e9:.2f} GB")

    # 大きい順に、いちばん空いている ZIP へ入れる (重さがそろうように)
    n = max(1, -(-int(total * 1.02) // LIMIT))
    while True:
        bins = [[] for _ in range(n)]
        load = [0] * n
        for (rel, p), s in sorted(zip(files, sizes), key=lambda t: -t[1]):
            i = min(range(n), key=lambda k: load[k])
            bins[i].append((rel, p))
            load[i] += s + 200 + len(rel.encode()) * 2
        if max(load) < LIMIT - 5_000_000:
            break
        n += 1
    # PitchLuna.exe・取扱説明書が入っている ZIP を 1 番目にする
    bins.sort(key=lambda b: 0 if any(rel == "PitchLuna.exe" for rel, _ in b) else 1)

    top = f"PitchLuna-{ver}"
    names = [f"PitchLuna-{ver}-part{i + 1}of{n}.zip" for i in range(n)]
    readme = README.format(ver=ver, n=n, names="・".join(names))
    readme = "\ufeff" + readme.replace("\n", "\r\n")
    for i, b in enumerate(bins):
        zp = os.path.join(out, names[i])
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=LEVEL, allowZip64=True) as z:
            z.writestr(f"{top}/はじめにお読みください（part{i + 1}）.txt", readme.encode("utf-8"))
            for rel, p in sorted(b):
                z.write(p, f"{top}/{rel}")
        sz = os.path.getsize(zp)
        print(f"{names[i]}  {sz / 1e9:.2f} GB  ファイル {len(b)} 個")
        if sz >= 2_000_000_000:
            raise SystemExit(f"{names[i]} が 2GB を超えました")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            f.write(f"parts={n}\n")


if __name__ == "__main__":
    main()
