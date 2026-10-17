# -*- coding: utf-8 -*-
"""
歌詞の表示のための処理 (画面の描画は singing_trainer.py)

  ・歌詞の文字は、使う人が入力 (貼り付け) したものをそのまま使う。
  ・漢字の読み (ふりがな) を調べ、1音 (モーラ) ずつに分ける。   例: 「風が」→ か・ぜ・が
  ・1音ずつ、お手本バーに順番に割り当てて「いつ歌うか」を決める。
  ・行の歌い出しの時刻 (タップ記録 / .lrc) があれば、それを手がかりにする。
  ・保存先: 曲のフォルダの lyrics.json
"""

import json
import os
import re
import sys
import threading
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
JANOME_WHL = os.path.join(HERE, "vendor", "Janome-0.5.0-py2.py3-none-any.whl")
JANOME_DIR = os.path.join(HERE, "_lib", "janome-0.5.0")

SMALL = set("ぁぃぅぇぉゃゅょゎゕゖァィゥェォャュョヮヵヶ")
LONG = set("ーｰ〜～")
SMALL_JOIN = SMALL | set("っッ")      # 前の音とまとめて 1音にする (「待っ」→ ま・っ ではなく まっ)


# ---------- 文字の種類 ----------
def kata_to_hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def is_kana(c):
    return ("ぁ" <= c <= "ゖ") or ("ァ" <= c <= "ヺ") or c in LONG or c in "ゝゞヽヾ"


def is_kanji(c):
    return ("一" <= c <= "鿿") or ("㐀" <= c <= "䶿") or c in "々〆ヵヶ〇"


def is_latin(c):
    return c.isascii() and c.isalnum() or c in "'’"


def split_morae(kana):
    """かな → 1音ずつ (小さい「ゃ」などは前の字とまとめる)。"""
    out = []
    for c in kana:
        if c in SMALL_JOIN and out:
            out[-1] += c
        else:
            out.append(c)
    return out


def latin_syllables(word):
    """英語の単語の、だいたいの音節の数 (母音のかたまりの数)。"""
    w = word.lower().strip("'’")
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


# ---------- 漢字の読み (Janome) ----------
_tok = None
_tok_lock = threading.Lock()
_tok_error = None


def _load_janome():
    """Janome (漢字の読みを調べる部品) を使えるようにする。初回は同梱の whl を展開する。"""
    global _tok, _tok_error
    with _tok_lock:
        if _tok is not None or _tok_error:
            return _tok
        try:
            try:
                from janome.tokenizer import Tokenizer
            except ImportError:
                if not os.path.isdir(os.path.join(JANOME_DIR, "janome")):
                    if not os.path.exists(JANOME_WHL):
                        raise ImportError("vendor/Janome の whl がありません")
                    import zipfile

                    tmp = JANOME_DIR + ".tmp"
                    with zipfile.ZipFile(JANOME_WHL) as z:
                        z.extractall(tmp)
                    if os.path.isdir(JANOME_DIR):
                        import shutil

                        shutil.rmtree(JANOME_DIR, ignore_errors=True)
                    os.replace(tmp, JANOME_DIR)
                if JANOME_DIR not in sys.path:
                    sys.path.insert(0, JANOME_DIR)
                from janome.tokenizer import Tokenizer
            _tok = Tokenizer()
        except Exception as e:                  # 読みが調べられなくても、歌詞は表示できるようにする
            _tok_error = f"{type(e).__name__}: {e}"
        return _tok


def reading_available():
    return _load_janome() is not None


def _align_ruby(surface, reading):
    """「振り返る」+「ふりかえる」→ [("振","ふ"),("り",""),("返","かえ"),("る","")]
    送りがなと読みを合わせて、漢字の部分だけにふりがなを付ける。"""
    runs = []
    for c in surface:
        k = "k" if is_kanji(c) else "a"
        if runs and runs[-1][0] == k:
            runs[-1][1] += c
        else:
            runs.append([k, c])
    if not any(k == "k" for k, _ in runs):
        return [(surface, "")]
    pat = "".join("(.+?)" if k == "k" else "(" + re.escape(kata_to_hira(s)) + ")" for k, s in runs)
    m = re.fullmatch(pat, kata_to_hira(reading))
    if not m:
        return [(surface, kata_to_hira(reading))]
    return [(s, m.group(i + 1) if k == "k" else "") for i, (k, s) in enumerate(runs)]


def make_units(text):
    """1行の歌詞 → 表示の単位のリスト。
    単位: {"s": 表示する文字, "r": ふりがな (漢字だけ), "m": [1音ずつ], "k": 漢字のあとに付く「っ」(あれば)}
    (記号・空白は m が空)"""
    units = []
    tok = _load_janome()
    pieces = []                                     # (表示, 読み) 読み=None は不明
    if tok is not None:
        try:
            for t in tok.tokenize(text):
                r = t.reading if t.reading and t.reading != "*" else None
                pieces.append((t.surface, r))
        except Exception:
            pieces = []
    if not pieces:
        pieces = [(text, None)]
    for surf, rd in pieces:
        if rd is not None and any(is_kanji(c) for c in surf):
            for s, r in _align_ruby(surf, rd):
                if r:
                    units.append({"s": s, "r": r, "m": split_morae(r)})
                else:
                    _add_plain(units, s)
        else:
            _add_plain(units, surf)
    return units


def _add_plain(units, s):
    """かな・英語・記号の部分を単位にする (読み不明の漢字は1字1音)。"""
    i = 0
    while i < len(s):
        c = s[i]
        if is_kana(c):
            j = i + 1
            while j < len(s) and s[j] in SMALL_JOIN:
                j += 1
            if c in SMALL_JOIN and units and units[-1]["m"]:
                p = units[-1]                          # 「っ」だけ: 前の音にくっつける (同じ 1音)
                if p["r"]:
                    p["k"] = p.get("k", "") + s[i:j]   # 漢字のあとの「っ」: ふりがなは漢字の上だけ
                else:
                    p["s"] += s[i:j]
                p["m"][-1] += kata_to_hira(s[i:j])
                i = j
                continue
            units.append({"s": s[i:j], "r": "", "m": [kata_to_hira(s[i:j])]})
            i = j
        elif is_latin(c):
            j = i
            while j < len(s) and is_latin(s[j]):
                j += 1
            w = s[i:j]
            n = latin_syllables(w) if any(ch.isalpha() for ch in w) else 1
            units.append({"s": w, "r": "", "m": [w] * n})
            i = j
        elif is_kanji(c):
            units.append({"s": c, "r": "", "m": [c]})
            i += 1
        else:
            j = i
            while j < len(s) and not (is_kana(s[j]) or is_latin(s[j]) or is_kanji(s[j])):
                j += 1
            units.append({"s": s[i:j], "r": "", "m": []})
            i = j


def set_ruby(unit, ruby):
    """ふりがなを直す (1音ずつの区切りも作り直す)。"""
    ruby = kata_to_hira(unicodedata.normalize("NFKC", ruby).strip())
    unit["r"] = ruby if ruby and ruby != unit["s"] else ""
    unit["m"] = split_morae(ruby) if ruby else ([unit["s"]] if unit["s"].strip() else [])


# ---------- 時間つき歌詞 (.lrc) ----------
_TAG = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")


def parse_lrc(text):
    """.lrc の中身 → [(時刻, 行の文字)]  (時刻の順)"""
    out = []
    for raw in text.splitlines():
        tags = list(_TAG.finditer(raw))
        if not tags:
            continue
        body = _TAG.sub("", raw)
        body = re.sub(r"<\d+:\d+(?:[.:]\d+)?>", "", body).strip()      # 1語ごとの時刻は使わない
        for m in tags:
            sec = int(m.group(1)) * 60 + float(m.group(2).replace(":", "."))
            out.append((round(sec, 3), body))
    out.sort(key=lambda x: x[0])
    return [(t, s) for t, s in out if s]


def fmt_lrc_time(t):
    t = max(0.0, t)
    m = int(t // 60)
    return f"{m:02d}:{t - m * 60:05.2f}"


# ---------- 歌詞のデータ ----------
class Lyrics:
    def __init__(self):
        self.lines = []          # [{"text": 行, "t": 歌い出し (秒) or None, "units": [...]}]
        self.times = []          # 1音ごとの (始まり, 終わり)。歌詞の最初の音から順番
        self.manual = False      # 手で直したか

    # --- 作る ---
    @classmethod
    def from_text(cls, text, times=None):
        """歌詞の文章 (改行で行を分ける) から作る。times: 行ごとの歌い出し (分かれば)"""
        ly = cls()
        rows = [r.strip() for r in text.replace("\r", "").split("\n")]
        rows = [r for r in rows if r]
        for i, r in enumerate(rows):
            t = times[i] if times and i < len(times) else None
            ly.lines.append({"text": r, "t": t, "units": make_units(r)})
        return ly

    @classmethod
    def from_lrc(cls, text):
        rows = parse_lrc(text)
        return cls.from_text("\n".join(s for _, s in rows), [t for t, _ in rows])

    def text(self):
        return "\n".join(l["text"] for l in self.lines)

    def to_lrc(self):
        out = []
        for i, l in enumerate(self.lines):
            t = l.get("t")
            if t is None:
                k = self.first_mora(i)
                t = self.times[k][0] if k is not None and k < len(self.times) else None
            if t is not None:
                out.append(f"[{fmt_lrc_time(t)}]{l['text']}")
        return "\n".join(out) + "\n"

    # --- 保存 ---
    def to_dict(self):
        return {"version": 1, "lines": self.lines, "times": [list(x) for x in self.times], "manual": self.manual}

    @classmethod
    def from_dict(cls, d):
        ly = cls()
        ly.lines = d.get("lines", [])
        ly.times = [tuple(x) for x in d.get("times", [])]
        ly.manual = bool(d.get("manual"))
        if len(ly.times) != ly.mora_count():
            ly.times = []
        return ly

    def save(self, path):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def copy(self):
        return Lyrics.from_dict(json.loads(json.dumps(self.to_dict())))

    # --- 1音ずつの番号 ---
    def mora_count(self):
        return sum(len(u["m"]) for l in self.lines for u in l["units"])

    def line_morae(self, i):
        return sum(len(u["m"]) for u in self.lines[i]["units"])

    def first_mora(self, i):
        """i 行目の最初の音の番号 (音が無い行は None)。"""
        k = sum(self.line_morae(j) for j in range(i))
        return k if self.line_morae(i) else None

    def iter_units(self):
        """(行の番号, 単位, 最初の音の番号) を順に。"""
        k = 0
        for i, l in enumerate(self.lines):
            for u in l["units"]:
                yield i, u, k
                k += len(u["m"])

    def line_of_mora(self, k):
        for i in range(len(self.lines)):
            n = self.line_morae(i)
            if k < n:
                return i
            k -= n
        return None

    def rebuild_units(self, i):
        self.lines[i]["units"] = make_units(self.lines[i]["text"])

    # ---------- お手本バーへの割り当て ----------
    def assign(self, notes, duration=None):
        """notes: お手本バー [{"start","end",...}] (時刻の順)。1音ずつの時刻 self.times を決める。"""
        notes = sorted(((float(n["start"]), float(n["end"])) for n in notes if n["end"] > n["start"]), key=lambda x: x[0])
        n_lines = len(self.lines)
        counts = [self.line_morae(i) for i in range(n_lines)]
        spans = [None] * n_lines                   # 行ごとに使うバーの範囲 (a, b) … notes[a:b]
        end_t = duration or (notes[-1][1] + 2 if notes else 0)
        anchors = [(i, l["t"]) for i, l in enumerate(self.lines) if l.get("t") is not None]
        if notes:
            starts = [s for s, _ in notes]
            import bisect

            def idx(t):
                return bisect.bisect_left(starts, t - 0.2)
            # 歌い出しが分かっている行の、バーの始まりの位置
            fixed = {i: idx(t) for i, t in anchors}
            bounds = sorted(fixed.items())
            # 歌い出しの分からない行を、まとまり (フレーズ) で割り当てる
            prev_line, prev_note = -1, 0
            for i, ni in bounds + [(n_lines, len(notes))]:
                group = list(range(prev_line + 1, i))
                if prev_line >= 0:
                    group = [prev_line] + group          # 前の目印の行から、次の目印の手前まで
                    lo = fixed[prev_line]
                else:
                    lo = 0
                hi = ni
                self._fit_lines(group, counts, notes, lo, hi, spans, keep_first=prev_line >= 0)
                prev_line, prev_note = i, ni
        times = []
        for i in range(n_lines):
            m = counts[i]
            if m == 0:
                continue
            sp = spans[i]
            if sp and sp[1] > sp[0]:
                times += _fit_morae(notes[sp[0]:sp[1]], m)
            else:                                   # バーが無い: 歌い出しから均等に並べる
                t0 = self.lines[i].get("t")
                if t0 is None:
                    t0 = times[-1][1] + 0.3 if times else 0.0
                nxt = next((self.lines[j]["t"] for j in range(i + 1, n_lines) if self.lines[j].get("t") is not None), None)
                t1 = min(nxt if nxt is not None else end_t, t0 + 0.35 * m)
                d = max(0.08, (t1 - t0) / m)
                times += [(t0 + k * d, t0 + (k + 1) * d) for k in range(m)]
        self.times = [(round(a, 3), round(b, 3)) for a, b in times]
        self.manual = False

    def _fit_lines(self, group, counts, notes, lo, hi, spans, keep_first):
        """group の行を notes[lo:hi] に割り当てる (フレーズ = 0.3秒以上の切れ目で区切ったバーのまとまり)。"""
        group = [i for i in group if counts[i] > 0]
        if not group or hi <= lo:
            return
        # フレーズに分ける
        ph = []
        a = lo
        for k in range(lo + 1, hi):
            if notes[k][0] - notes[k - 1][1] >= 0.3:
                ph.append((a, k))
                a = k
        ph.append((a, hi))
        L, P = len(group), len(ph)
        if L == 1:
            spans[group[0]] = (lo, hi)
            return
        INF = 1e18
        # f[i][j]: 最初の i 行を最初の j フレーズに割り当てたときの「ずれ」の合計
        f = [[INF] * (P + 1) for _ in range(L + 1)]
        back = [[None] * (P + 1) for _ in range(L + 1)]
        f[0][0] = 0.0
        for j in range(1, P + 1):                   # 行の前にあるフレーズは使わない (前奏の誤検出など)
            if not keep_first:
                n_skip = ph[j - 1][1] - ph[j - 1][0]
                c = f[0][j - 1] + 0.6 + 0.05 * n_skip
                if c < f[0][j]:
                    f[0][j], back[0][j] = c, ("skip", j - 1)
        for i in range(1, L + 1):
            m = counts[group[i - 1]]
            for j in range(1, P + 1):
                # i 行目が フレーズ k..j-1 を使う
                n = 0
                for k in range(j - 1, -1, -1):
                    n += ph[k][1] - ph[k][0]
                    if f[i - 1][k] >= INF:
                        continue
                    c = f[i - 1][k] + abs(m - n) / max(m, n) + 0.15 * (j - 1 - k)
                    if c < f[i][j]:
                        f[i][j], back[i][j] = c, ("line", k)
                    if n > 3 * m + 6:
                        break
                # 行のあとにフレーズを飛ばす (間奏の誤検出など)
                if i == L and j > 1 and f[i][j - 1] < INF:
                    n_skip = ph[j - 1][1] - ph[j - 1][0]
                    c = f[i][j - 1] + 0.6 + 0.05 * n_skip
                    if c < f[i][j]:
                        f[i][j], back[i][j] = c, ("skip", j - 1)
        if f[L][P] >= INF:                          # 割り当てられない: 均等に分ける
            per = max(1, (hi - lo) // L)
            for t, gi in enumerate(group):
                a = lo + t * per
                spans[gi] = (a, hi if t == L - 1 else min(hi, a + per))
            return
        i, j = L, P
        while i > 0 or j > 0:
            kind, k = back[i][j]
            if kind == "skip":
                j = k
            else:
                spans[group[i - 1]] = (ph[k][0], ph[j - 1][1])
                i, j = i - 1, k

    # ---------- 手で直す ----------
    def move_mora(self, k, t0):
        """k 番目の音の始まりを t0 にする (前後の音とは重ならない)。"""
        if not (0 <= k < len(self.times)):
            return
        a, b = self.times[k]
        lo = self.times[k - 1][0] + 0.03 if k > 0 else 0.0
        hi = b - 0.03
        t0 = min(max(t0, lo), hi)
        self.times[k] = (round(t0, 3), b)
        if k > 0 and self.times[k - 1][1] > t0 - 0.0005 or (k > 0 and abs(self.times[k - 1][1] - a) < 0.02):
            pa, _ = self.times[k - 1]
            self.times[k - 1] = (pa, round(t0, 3))
        self.manual = True

    def shift_from(self, k, d, to_line_end=True):
        """k 番目の音から (その行の終わりまで) を d 秒ずらす。"""
        i = self.line_of_mora(k)
        last = self.first_mora(i) + self.line_morae(i) if (to_line_end and i is not None) else len(self.times)
        lo = self.times[k - 1][0] + 0.03 if k > 0 else 0.0
        d = max(d, lo - self.times[k][0])
        if last < len(self.times):
            d = min(d, self.times[last][0] - 0.03 - self.times[last - 1][0])
        for j in range(k, last):
            a, b = self.times[j]
            self.times[j] = (round(a + d, 3), round(b + d, 3))
        if k > 0:
            pa, pb = self.times[k - 1]
            if pb > self.times[k][0] or abs(pb - (self.times[k][0] - d)) < 0.02:
                self.times[k - 1] = (pa, max(pa + 0.03, self.times[k][0]))
        self.manual = True

    def step_from(self, k, notes, step):
        """k 番目の音から行の終わりまでを、お手本バー1本ぶん前 (step=-1) / 後ろ (+1) へずらす。"""
        ns = sorted(((float(n["start"]), float(n["end"])) for n in notes), key=lambda x: x[0])
        if not ns or not (0 <= k < len(self.times)):
            return False
        starts = [s for s, _ in ns]
        import bisect

        def note_idx(t):
            i = bisect.bisect_right(starts, t + 0.02) - 1
            return max(0, i)
        i = self.line_of_mora(k)
        last = self.first_mora(i) + self.line_morae(i)
        # となりの行のバーには入らない (はみ出す分は、行の端のバーを分けて歌う)
        lo_i = note_idx(self.times[k - 1][0]) + 1 if k > 0 else 0
        hi_i = note_idx(self.times[last][0]) - 1 if last < len(self.times) else len(ns) - 1
        if last < len(self.times) and ns[min(len(ns) - 1, hi_i + 1)][0] > self.times[last][0] + 0.02:
            hi_i += 1
        if hi_i < lo_i:
            return False
        new = []
        for j in range(k, last):
            ni = min(hi_i, max(lo_i, note_idx(self.times[j][0]) + step))
            new.append(ni)
        if all(note_idx(self.times[j][0]) == ni for j, ni in zip(range(k, last), new)):
            return False
        # 同じバーに複数の音が入るときは、バーを分ける
        out = []
        p = 0
        while p < len(new):
            q = p
            while q < len(new) and new[q] == new[p]:
                q += 1
            s, e = ns[new[p]]
            d = (e - s) / (q - p)
            out += [(s + r * d, s + (r + 1) * d) for r in range(q - p)]
            p = q
        if k > 0 and out[0][0] < self.times[k - 1][0] + 0.03:
            return False
        if last < len(self.times) and out[-1][0] > self.times[last][0] - 0.03:
            return False
        for j, (a, b) in zip(range(k, last), out):
            self.times[j] = (round(a, 3), round(b, 3))
        if k > 0:
            pa, pb = self.times[k - 1]
            self.times[k - 1] = (pa, min(max(pb, pa + 0.03), self.times[k][0]) if pb > self.times[k][0] else pb)
        if last < len(self.times):
            a, b = self.times[last - 1]
            self.times[last - 1] = (a, min(b, self.times[last][0]))
        self.manual = True
        return True


def _fit_morae(notes, m):
    """バーの並び notes [(始まり, 終わり)] に m 個の音を割り当てる → 1音ごとの (始まり, 終わり)。"""
    segs = [[s, e] for s, e in notes]
    # とても短いバー (0.12秒未満: 音の移り変わりで出る細かいバー) は、先にとなりへまとめる
    while len(segs) > 1:
        short = [i for i in range(len(segs)) if segs[i][1] - segs[i][0] < 0.12]
        if not short:
            break
        i = min(short, key=lambda k: segs[k][1] - segs[k][0])
        gp = segs[i][0] - segs[i - 1][1] if i > 0 else 9.0
        gn = segs[i + 1][0] - segs[i][1] if i + 1 < len(segs) else 9.0
        if i > 0 and (gp <= gn or i + 1 >= len(segs)):
            segs[i - 1][1] = segs[i][1]
        else:
            segs[i + 1][0] = segs[i][0]
        del segs[i]
    # バーが多い: 短いバー (音の移り変わりの細かいバーなど) から、となりのバーにまとめる (1音を伸ばして歌う所)
    while len(segs) > m:
        def score(i):
            dur = segs[i][1] - segs[i][0]
            gp = segs[i][0] - segs[i - 1][1] if i > 0 else 9.0
            gn = segs[i + 1][0] - segs[i][1] if i + 1 < len(segs) else 9.0
            return dur + 0.5 * min(gp, gn)
        i = min(range(len(segs)), key=score)
        gp = segs[i][0] - segs[i - 1][1] if i > 0 else 9.0
        gn = segs[i + 1][0] - segs[i][1] if i + 1 < len(segs) else 9.0
        if i > 0 and (gp <= gn or i + 1 >= len(segs)):
            segs[i - 1][1] = segs[i][1]             # 前のバーとまとめる
        else:
            segs[i + 1][0] = segs[i][0]             # 後ろのバーとまとめる
        del segs[i]
    # 音が多い: 長いバーを半分ずつに分ける (1本のバーに 2音)
    while len(segs) < m:
        i = max(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])
        s, e = segs[i]
        mid = (s + e) / 2
        segs[i:i + 1] = [[s, mid], [mid, e]]
    return [(s, e) for s, e in segs]
