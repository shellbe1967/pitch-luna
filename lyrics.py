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
    try:
        import unicodedata

        ruby = unicodedata.normalize("NFKC", ruby)
    except Exception:
        pass
    ruby = kata_to_hira(ruby.strip())
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


_WTAG = re.compile(r"<(\d+):(\d+(?:[.:]\d+)?)>")


def _sec(m):
    return int(m.group(1)) * 60 + float(m.group(2).replace(":", "."))


def parse_lrc_words(text):
    """1語ごとの時刻つき .lrc (<00:12.34> を言葉の前に入れる形) も読む。
    → [(行の時刻, 行の文字, [(文字の位置, 時刻)])]"""
    out = []
    for raw in text.splitlines():
        tags = list(_TAG.finditer(raw))
        if not tags:
            continue
        rest = _TAG.sub("", raw)
        body, marks, pos = "", [], 0
        for m in _WTAG.finditer(rest):
            body += rest[pos:m.start()]
            marks.append((len(body), round(_sec(m), 3)))
            pos = m.end()
        body += rest[pos:]
        lead = len(body) - len(body.lstrip())
        body = body.strip()
        marks = [(max(0, i - lead), t) for i, t in marks if i - lead <= len(body)]
        for m in tags:
            out.append((round(_sec(m), 3), body, marks))
    out.sort(key=lambda x: x[0])
    return [x for x in out if x[1]]


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
        self.source = "user"     # "user" = 入力した歌詞 / "asr" = AI が聞き取った歌詞 (仮)

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
    def from_lrc(cls, text, notes=None):
        rows = parse_lrc_words(text)
        ly = cls.from_text("\n".join(s for _, s, _ in rows), [t for t, _, _ in rows])
        if any(marks for _, _, marks in rows) and len(ly.lines) == len(rows):
            ly._times_from_marks(rows, notes)
        return ly

    def _times_from_marks(self, rows, notes=None):
        """1語ごとの時刻 (文字の位置 → 時刻) から、1音ずつの時刻を作る。"""
        times = []
        ns = sorted(((float(n["start"]), float(n["end"])) for n in notes or [] if n["end"] > n["start"]), key=lambda x: x[0])
        for i, (lt, body, marks) in enumerate(rows):
            nxt = rows[i + 1][0] if i + 1 < len(rows) else None
            pts = sorted(set([(0, lt)] + marks))
            last_t = max(t for _, t in pts)
            end_t = min(nxt, last_t + 0.6) if nxt is not None and nxt > last_t else last_t + 0.6
            pts.append((len(body), end_t))
            pts = [(p, t) for p, t in sorted(pts)]

            def at(c):
                for (p0, t0), (p1, t1) in zip(pts, pts[1:]):
                    if p0 <= c <= p1:
                        return t0 if p1 == p0 else t0 + (t1 - t0) * (c - p0) / (p1 - p0)
                return pts[-1][1]
            pos = 0
            for u in self.lines[i]["units"]:
                ln = len(u["s"]) + len(u.get("k", ""))
                if u["m"]:
                    a, b = at(pos), at(pos + ln)
                    if b <= a:
                        b = a + 0.1 * len(u["m"])
                    n = len(u["m"])
                    inside = [(max(x, a), min(y, b)) for x, y in ns if x >= a - 0.03 and x < b - 0.03 and y > x] if ns else []
                    if n > 1 and inside:                # 2音以上の言葉: その中のお手本バーに合わせる
                        times += _fit_morae(inside, n)
                    else:
                        d = (b - a) / n
                        times += [(a + r * d, a + (r + 1) * d) for r in range(n)]
                pos += ln
        for k in range(1, len(times)):                  # 逆戻り・重なりを直す
            a, b = times[k]
            if a < times[k - 1][0] + 0.02:
                a = times[k - 1][0] + 0.02
                times[k] = (a, max(b, a + 0.03))
            if times[k - 1][1] > times[k][0]:
                times[k - 1] = (times[k - 1][0], times[k][0])
        if len(times) == self.mora_count():
            self.times = [(round(a, 3), round(b, 3)) for a, b in times]
            self.manual = True

    def text(self):
        return "\n".join(l["text"] for l in self.lines)

    def to_lrc(self, words=True, header=None):
        """.lrc の文字にする。words=True: 言葉ごとの時刻 <00:12.34> も入れる (カラオケの色変わりも残る)。"""
        out = list(header or [])
        k = 0
        for i, l in enumerate(self.lines):
            t = l.get("t")
            first = self.first_mora(i)
            if t is None and first is not None and first < len(self.times):
                t = self.times[first][0]
            body = l["text"]
            if words and self.times and "".join(u["s"] + u.get("k", "") for u in l["units"]) == l["text"]:
                parts = []
                kk = k
                for u in l["units"]:
                    txt = u["s"] + u.get("k", "")
                    if u["m"] and kk < len(self.times):
                        parts.append(f"<{fmt_lrc_time(self.times[kk][0])}>{txt}")
                    else:
                        parts.append(txt)
                    kk += len(u["m"])
                body = "".join(parts)
            k += self.line_morae(i)
            if t is not None:
                out.append(f"[{fmt_lrc_time(t)}]{body}")
        return "\n".join(out) + "\n"

    # --- 保存 ---
    def to_dict(self):
        return {"version": 1, "lines": self.lines, "times": [list(x) for x in self.times], "manual": self.manual,
                "source": self.source}

    @classmethod
    def from_dict(cls, d):
        ly = cls()
        ly.lines = d.get("lines", [])
        ly.times = [tuple(x) for x in d.get("times", [])]
        ly.manual = bool(d.get("manual"))
        ly.source = d.get("source", "user")
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

    def align_to_asr(self, asr, notes, duration=None):
        """AI の聞き取りの時刻を手がかりに、1音ずつの時刻を決める。
        聞き取りと同じ音 (2音以上続けて一致) はその時刻、残りはお手本バーで埋める。返り値: 一致した割合"""
        mine = [norm_mora(mm) for l in self.lines for u in l["units"] for mm in u["m"]]
        heard = asr_morae(asr)
        if not mine or not heard:
            self.assign(notes, duration)
            return 0.0
        pairs = _align(mine, [h[0] for h in heard])
        # ぽつんと 1音だけ一致したものは、たまたまの可能性があるので使わない
        keep = []
        for k, (i, j) in enumerate(pairs):
            nb = (k > 0 and pairs[k - 1] == (i - 1, j - 1)) or (k + 1 < len(pairs) and pairs[k + 1] == (i + 1, j + 1))
            if nb:
                keep.append((i, j))
        anchors = {}
        last_t = -1.0
        for i, j in keep:
            t0, t1 = heard[j][1], heard[j][2]
            if t0 >= last_t:                          # 時刻が逆戻りするものは使わない
                anchors[i] = (t0, t1)
                last_t = t0
        if not anchors:
            self.assign(notes, duration)
            return 0.0
        ns = sorted(((float(n["start"]), float(n["end"])) for n in notes if n["end"] > n["start"]), key=lambda x: x[0])
        starts = [a for a, _ in ns]
        import bisect

        def snap(t):                                  # 近くにお手本バーの頭があれば合わせる
            k = bisect.bisect_left(starts, t)
            best = None
            for c in (k - 1, k):
                if 0 <= c < len(starts) and abs(starts[c] - t) <= 0.12:
                    if best is None or abs(starts[c] - t) < abs(best - t):
                        best = starts[c]
            return best if best is not None else t
        N = len(mine)
        times = [None] * N
        for i, (a, b) in anchors.items():
            times[i] = [snap(a), b]
        # 一致しなかった所: 前後の一致した音のあいだを、お手本バーで埋める
        idx = sorted(anchors)
        gaps = []
        if idx[0] > 0:
            gaps.append((0, idx[0]))
        for p, q in zip(idx, idx[1:]):
            if q - p > 1:
                gaps.append((p + 1, q))
        if idx[-1] < N - 1:
            gaps.append((idx[-1] + 1, N))
        for g0, g1 in gaps:
            cnt = g1 - g0
            lo = times[g0 - 1][1] if g0 > 0 else None
            hi = times[g1][0] if g1 < N else None
            if lo is None:                            # 最初の一致より前: その直前のバーを使う
                cand = [n for n in ns if n[1] <= hi + 0.02]
                win = cand[-max(cnt, 1):] if cand else []
                lo = win[0][0] if win else max(0.0, hi - 0.4 * cnt)
            if hi is None:                            # 最後の一致より後: その直後のバーを使う
                cand = [n for n in ns if n[0] >= lo - 0.02]
                win = cand[:max(cnt, 1)] if cand else []
                hi = win[-1][1] if win else lo + 0.4 * cnt
            inside = [n for n in ns if n[0] >= lo - 0.03 and n[1] <= hi + 0.03 and n[1] > n[0]]
            if inside:
                fit = _fit_morae([(max(a, lo), min(b, hi)) for a, b in inside], cnt)
            else:
                d = max(0.05, (hi - lo) / cnt)
                fit = [(lo + r * d, lo + (r + 1) * d) for r in range(cnt)]
            for r, (a, b) in enumerate(fit):
                times[g0 + r] = [a, b]
        # 重なり・逆戻りを直す
        for i in range(N):
            a, b = times[i]
            if i > 0 and a < times[i - 1][0] + 0.02:
                a = times[i - 1][0] + 0.02
            times[i] = [a, max(b, a + 0.03)]
        for i in range(N - 1):
            if times[i][1] > times[i + 1][0]:
                times[i][1] = max(times[i][0] + 0.02, times[i + 1][0])
        self.times = [(round(a, 3), round(b, 3)) for a, b in times]
        self.manual = False
        return len(anchors) / N

    @classmethod
    def from_asr(cls, asr):
        """AI が聞き取った歌詞から作る (仮の歌詞)。"""
        rows = []
        for seg in asr.get("segments", []):
            t = "".join(w["w"] for w in seg.get("words") or []).strip()
            if t:
                rows.append(t)
        ly = cls.from_text("\n".join(rows))
        ly.source = "asr"
        return ly

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


# ---------- AI の聞き取り (asr.json) との照らし合わせ ----------
def norm_mora(m):
    """比べるための形: ひらがな・「っ」を除く・を→お など。"""
    m = kata_to_hira(m).replace("っ", "").replace("ッ", "")
    m = m.replace("を", "お").replace("ぢ", "じ").replace("づ", "ず")
    return m.lower() if m.isascii() else m


def asr_morae(asr):
    """聞き取り結果 → [(1音, 始まり, 終わり)]  (言葉の時刻を、その言葉の音の数で分ける)"""
    out = []
    for seg in asr.get("segments", []):
        words = seg.get("words") or []
        text = "".join(w["w"] for w in words)
        if not text:
            continue
        ct = []                                      # 1文字ずつの (始まり, 終わり)
        for w in words:
            n = max(1, len(w["w"]))
            d = (w["e"] - w["s"]) / n
            ct += [(w["s"] + i * d, w["s"] + (i + 1) * d) for i in range(len(w["w"]))]
        pos = 0
        for u in make_units(text):
            ln = len(u["s"]) + len(u.get("k", ""))
            span = ct[pos:pos + ln]
            pos += ln
            if not u["m"] or not span:
                continue
            a, b = span[0][0], span[-1][1]
            d = (b - a) / len(u["m"])
            for i, mm in enumerate(u["m"]):
                out.append((norm_mora(mm), a + i * d, a + (i + 1) * d))
    return out


def _align(xs, ys):
    """2つの並びを比べて、同じ音の組 [(xのi, yのj)] を返す (順番は変えない)。"""
    n, m = len(xs), len(ys)
    if n == 0 or m == 0:
        return []
    INF = 10 ** 9
    # 似ている音 (母音が同じなど) も少しだけ許す
    def cost(a, b):
        if a == b:
            return 0
        return 2
    prev = list(range(m + 1))
    back = [bytearray(m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        back[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        bi = back[i]
        bi[0] = 1
        xi = xs[i - 1]
        for j in range(1, m + 1):
            d = prev[j - 1] + cost(xi, ys[j - 1])
            u = prev[j] + 1
            l = cur[j - 1] + 1
            if d <= u and d <= l:
                cur[j], bi[j] = d, 0
            elif u <= l:
                cur[j], bi[j] = u, 1
            else:
                cur[j], bi[j] = l, 2
        prev = cur
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        b = back[i][j]
        if b == 0:
            if xs[i - 1] == ys[j - 1]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


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
