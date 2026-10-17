# -*- coding: utf-8 -*-
"""
歌詞の表示と操作 (Trainer に混ぜて使う)

  ・楽譜 (お手本バー) のすぐ下に「歌詞の段」を出す。文字はその音のバーの真下に並び、一緒に流れる。
  ・歌うところまで、カラオケのように左から色が変わる (音程が合っていたかで色を変えることもできる)。
  ・文字をドラッグ: その音のタイミングを直す (Shift+ドラッグ: その行の後ろも一緒に)
  ・右クリック: ふりがな・歌詞の修正、1音ずつ前 / 後ろのバーへ、割り当て直し
  ・行の歌い出しの記録: 再生しながら、各行の歌い出しで Enter
"""

import os
import time
import threading

import lyrics as LY
from theme import (BG_DEEP, PANEL, HAIR, HAIR_STRONG, OUTLINE, WHITE, DIM, FAINT, ACCENT, GOOD, USER, GOLD)
from ui_menu import MenuItem, SEPARATOR, PopupMenu

G = {}                       # singing_trainer の変数 (PX_PER_SEC・W・H など)。bind() で受け取る

SIZE_CHOICES = ((16, "小"), (20, "中"), (26, "大"))
SUNG = ACCENT                # 歌ったところの色
MISS = USER                  # 音程がずれていたところ (音程で色を変えるとき)
TODO = (226, 232, 246)       # これから歌うところ


def bind(globals_dict):
    G.clear()
    G.update({"_g": globals_dict})


def g(name):
    return G["_g"][name]


def lyric_lane_height(view, has_lyrics):
    if not has_lyrics or not view.get("lyrics", True):
        return 0
    fs = int(view.get("lyric_size", 20))
    rh = max(9, fs // 2) + 2 if view.get("ruby", True) else 0
    return fs + rh + 16


# ---------- 入力の小さな画面 (tkinter) ----------
def ask_multiline(title, prompt, default=""):
    """歌詞を貼り付ける画面。OK → 文字、キャンセル → None。"""
    try:
        import tkinter as tk
    except Exception:
        return None
    res = {"v": None}
    try:
        root = tk.Tk()
        root.title(title)
        root.attributes("-topmost", True)
        root.geometry("560x560")
        tk.Label(root, text=prompt, justify="left", anchor="w", wraplength=530).pack(fill="x", padx=12, pady=(12, 6))
        fr = tk.Frame(root)
        fr.pack(fill="both", expand=True, padx=12)
        sb = tk.Scrollbar(fr)
        sb.pack(side="right", fill="y")
        txt = tk.Text(fr, wrap="none", undo=True, font=("Meiryo", 12), yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.insert("1.0", default or "")
        txt.focus_set()
        bt = tk.Frame(root)
        bt.pack(fill="x", padx=12, pady=12)

        def ok():
            res["v"] = txt.get("1.0", "end").rstrip("\n")
            root.destroy()

        tk.Button(bt, text="キャンセル", width=12, command=root.destroy).pack(side="right")
        tk.Button(bt, text="OK", width=12, command=ok).pack(side="right", padx=8)
        root.mainloop()
    except Exception:
        return None
    return res["v"]


def ask_yesno(title, text):
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        v = messagebox.askyesno(title, text, parent=root)
        root.destroy()
        return bool(v)
    except Exception:
        return False


def choose_lrc(save=False, initial_dir=None, initial_name=None):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        types = [("時間つき歌詞 (.lrc)", "*.lrc"), ("テキスト", "*.txt"), ("すべて", "*.*")]
        if save:
            p = filedialog.asksaveasfilename(title=".lrc ファイルとして保存", initialdir=initial_dir,
                                             initialfile=initial_name, defaultextension=".lrc", filetypes=types)
        else:
            p = filedialog.askopenfilename(title="時間つき歌詞 (.lrc) を選んでください", initialdir=initial_dir, filetypes=types)
        root.destroy()
        return p or None
    except Exception:
        return None


def read_text_file(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp932", "utf-16"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", "replace")


class LyricsMixin:
    """Trainer に混ぜる歌詞の機能。"""

    def _lyrics_init(self):
        self.lyrics = None
        self.lyric_tap = None            # 歌い出しの記録中: 次に記録する行の番号
        self._lyric_job = None           # 読みを調べている途中 (別の処理で)
        self._lyric_rects = []           # [(Rect, 最初の音の番号, 単位)]
        self._lyric_hits_cache = {}

    # ---------- 保存・読み込み ----------
    def _lyrics_path(self):
        return os.path.join(self.song_dir, "lyrics.json") if self.song_dir else None

    def load_lyrics(self):
        self._lyrics_init()
        p = self._lyrics_path()
        if p and os.path.exists(p):
            try:
                self.lyrics = LY.Lyrics.load(p)
                if not self.lyrics.times and self.lyrics.mora_count():
                    self.lyrics.assign(self.guide.notes, self.pb.duration)
            except Exception:
                self.lyrics = None
        self._rebuild_background()

    def save_lyrics(self):
        p = self._lyrics_path()
        if p and self.lyrics is not None:
            try:
                self.lyrics.save(p)
            except Exception as e:
                self.show_toast(f"歌詞を保存できませんでした：{e}", 4.0)

    def _set_lyrics(self, ly, msg):
        if self.lyrics is not None:
            self._push_lyric_undo()
        self.lyrics = ly
        if ly.mora_count():
            ly.assign(self.guide.notes, self.pb.duration)
        self.save_lyrics()
        self.view["lyrics"] = True
        self._rebuild_background()
        self.show_toast(msg, 4.0)

    # ---------- 歌詞を入れる ----------
    def lyrics_input(self):
        if not self.song_dir:
            self.show_toast("先に曲を開いてください")
            return
        cur = self.lyrics.text() if self.lyrics else ""
        txt = ask_multiline("歌詞を入力・貼り付け",
                            "歌詞を入力するか、貼り付けてください。1行ずつ改行で区切ります。\n"
                            "（歌詞カードなどを見ながら。空の行は無視します）", cur)
        if txt is None:
            return
        if not txt.strip():
            self.lyrics_clear(ask=False)
            return
        times = None
        if self.lyrics and txt.strip() == cur.strip():      # 文字が同じ: 記録した歌い出しはそのまま
            times = [l.get("t") for l in self.lyrics.lines]
        self._start_lyric_job(lambda: LY.Lyrics.from_text(txt, times), "歌詞を入れました（お手本バーに自動で割り当てました）")

    def lyrics_open_lrc(self):
        if not self.song_dir:
            self.show_toast("先に曲を開いてください")
            return
        p = choose_lrc(initial_dir=os.path.dirname(self.song_path) if self.song_path else None)
        if not p:
            return
        try:
            text = read_text_file(p)
        except Exception as e:
            self.show_toast(f"読み込めませんでした：{e}", 4.0)
            return
        if not LY.parse_lrc(text):
            self.show_toast("時刻つきの行が見つかりませんでした（[00:12.34]歌詞 の形の .lrc を選んでください）", 5.0)
            return
        self._start_lyric_job(lambda: LY.Lyrics.from_lrc(text), "時間つき歌詞 (.lrc) を読み込みました")

    def lyrics_save_lrc(self):
        if not self.lyrics:
            return
        name = (os.path.splitext(os.path.basename(self.song_path))[0] if self.song_path else "lyrics") + ".lrc"
        p = choose_lrc(save=True, initial_dir=os.path.dirname(self.song_path) if self.song_path else None, initial_name=name)
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.lyrics.to_lrc())
            self.show_toast(f"保存しました：{os.path.basename(p)}", 3.0)
        except Exception as e:
            self.show_toast(f"保存できませんでした：{e}", 4.0)

    def lyrics_clear(self, ask=True):
        if not self.lyrics:
            return
        if ask and not ask_yesno("歌詞を消す", "この曲の歌詞を消しますか？\n（Ctrl+Z で元に戻せます）"):
            return
        self._push_lyric_undo()
        self.lyrics = None
        self.lyric_tap = None
        p = self._lyrics_path()
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        self._rebuild_background()
        self.show_toast("歌詞を消しました")

    def lyrics_reassign(self, ask=True):
        if not self.lyrics:
            return
        if ask and self.lyrics.manual and not ask_yesno(
                "自動で割り当て直す", "手で直したタイミングも、自動で割り当て直します。よろしいですか？\n（Ctrl+Z で元に戻せます）"):
            return
        self._push_lyric_undo()
        self.lyrics.assign(self.guide.notes, self.pb.duration)
        self.save_lyrics()
        self.show_toast("歌詞をお手本バーに割り当て直しました", 2.5)

    def _start_lyric_job(self, make, msg):
        """読みを調べる (初回は部品の準備に時間がかかるので別の処理で)。"""
        if self._lyric_job:
            return
        job = {"done": False, "result": None, "error": None, "msg": msg}

        def run():
            try:
                job["result"] = make()
            except Exception as e:
                job["error"] = f"{type(e).__name__}: {e}"
            job["done"] = True

        first = LY._tok is None and not LY._tok_error
        self.show_toast("歌詞の読み（ふりがな）を調べています…" + ("　初回は 30秒〜1分ほどかかります" if first else ""), 60.0)
        self._lyric_job = job
        threading.Thread(target=run, daemon=True).start()

    def _poll_lyric_job(self):
        job = self._lyric_job
        if not job or not job["done"]:
            return
        self._lyric_job = None
        if job["error"]:
            self.show_toast(f"歌詞を読み込めませんでした：{job['error']}", 5.0)
            return
        msg = job["msg"]
        if LY._tok_error:
            msg += "　※ふりがなを自動で調べられませんでした（右クリック → ふりがなを直す）"
        self._set_lyrics(job["result"], msg)

    # ---------- 行の歌い出しの記録 (タップ) ----------
    def lyrics_start_tap(self):
        if not self.lyrics or not self.lyrics.lines:
            self.show_toast("先に「歌詞を入力・貼り付け」で歌詞を入れてください", 3.0)
            return
        self._push_lyric_undo()
        self.lyric_tap = 0
        for l in self.lyrics.lines:
            l["t"] = None
        if self.pb.playing:
            self.pb.toggle_play()
        self.pb.seek(0)
        self.mic.clear()
        self.show_toast("曲を流しながら、各行の歌い出しで Enter を押してください（Space で再生）", 5.0)

    def _tap_now(self):
        i = self.lyric_tap
        if i is None or not self.lyrics:
            return
        t = max(0.0, self.pb.position() - 0.12)       # 押すのは少し遅れるので、その分を引く
        self.lyrics.lines[i]["t"] = round(t, 3)
        self.lyric_tap = i + 1
        if self.lyric_tap >= len(self.lyrics.lines):
            self.stop_lyric_tap()
        else:
            self.show_toast(f"{i + 1}行目 {g('fmt_time')(t)}　次：{self.lyrics.lines[self.lyric_tap]['text'][:24]}", 1.6)

    def _tap_back(self):
        if self.lyric_tap:
            self.lyric_tap -= 1
            t = self.lyrics.lines[self.lyric_tap]["t"]
            self.lyrics.lines[self.lyric_tap]["t"] = None
            if t is not None:
                self.pb.seek(max(0.0, t - 3.0))
                self.mic.clear()
            self.show_toast(f"{self.lyric_tap + 1}行目からやり直します（3秒前に戻りました）", 2.0)

    def stop_lyric_tap(self, cancel=False):
        if self.lyric_tap is None:
            return
        n = sum(1 for l in self.lyrics.lines if l.get("t") is not None)
        self.lyric_tap = None
        if cancel and n == 0:
            self.undo()
            return
        self.lyrics.assign(self.guide.notes, self.pb.duration)
        self.save_lyrics()
        self.show_toast(f"歌い出しを {n}行 記録して、歌詞を割り当て直しました", 3.0)

    # ---------- キー ----------
    def lyric_handle_key(self, k, mod):
        pg = self.pg
        K = lambda name: getattr(pg, name, -999)
        if self.lyric_tap is not None:
            if k in (K("K_RETURN"), K("K_KP_ENTER")):
                self._tap_now()
                return True
            if k == K("K_BACKSPACE"):
                self._tap_back()
                return True
        if k == K("K_k") and not mod:
            self.toggle_lyrics_view()
            return True
        return False

    def toggle_lyrics_view(self):
        if not self.lyrics:
            self.show_toast("歌詞はまだありません（メニューの「歌詞」→「歌詞を入力・貼り付け」）", 3.0)
            return
        self.view["lyrics"] = not self.view.get("lyrics", True)
        self._view_changed_lyrics()
        self.show_toast("歌詞：表示" if self.view["lyrics"] else "歌詞：非表示")

    def _set_view_opt(self, key, val):
        self.view[key] = val
        self._view_changed_lyrics()

    def _view_changed_lyrics(self):
        self._rebuild_background()
        if self.on_view_change:
            self.on_view_change()

    # ---------- メニュー ----------
    def _lyrics_items(self):
        has = lambda: self.lyrics is not None
        song = lambda: self.song_dir is not None
        items = [
            MenuItem("歌詞を入力・貼り付け…", self.lyrics_input, enabled=song),
            MenuItem("時間つき歌詞 (.lrc) を読み込む…", self.lyrics_open_lrc, enabled=song),
            SEPARATOR,
            MenuItem(lambda: "歌い出しの記録をやめる" if self.lyric_tap is not None else "行の歌い出しを記録する（再生しながら Enter）",
                     lambda: self.stop_lyric_tap() if self.lyric_tap is not None else self.lyrics_start_tap(), enabled=has),
            MenuItem("自動で割り当て直す", self.lyrics_reassign, enabled=has),
            SEPARATOR,
            MenuItem("歌詞を表示", self.toggle_lyrics_view, "K", checked=lambda: has() and self.view.get("lyrics", True),
                     enabled=has),
            MenuItem("ふりがなを表示", lambda: self._set_view_opt("ruby", not self.view.get("ruby", True)),
                     checked=lambda: self.view.get("ruby", True)),
            MenuItem("音程が合ったかで色を変える", lambda: self._set_view_opt("lyric_color", not self.view.get("lyric_color", True)),
                     checked=lambda: self.view.get("lyric_color", True)),
        ] + [MenuItem(f"文字の大きさ：{name}", (lambda s=s: self._set_view_opt("lyric_size", s)),
                      checked=(lambda s=s: int(self.view.get("lyric_size", 20)) == s)) for s, name in SIZE_CHOICES] + [
            SEPARATOR,
            MenuItem(".lrc ファイルとして保存…", self.lyrics_save_lrc, enabled=has),
            MenuItem("歌詞を消す…", self.lyrics_clear, enabled=has),
        ]
        return items

    # ---------- 段の位置 ----------
    def _lyric_lane_rect(self):
        L = self.layout
        h = lyric_lane_height(self.view, self.lyrics is not None)
        return self.pg.Rect(0, L.bottom, g("W"), h)

    def _unit_at(self, pos):
        for r, k, u in reversed(self._lyric_rects):
            if r.inflate(4, 6).collidepoint(pos):
                return k, u
        return None

    # ---------- マウス ----------
    def lyric_mouse_down(self, pos, button, double):
        """歌詞の段の上なら処理して True。"""
        if self.lyrics is None or not self.view.get("lyrics", True):
            return False
        lane = self._lyric_lane_rect()
        if not lane.collidepoint(pos) or pos[0] < self.layout.left:
            return False
        hit = self._unit_at(pos)
        if button == 3:
            self._open_lyric_popup(pos, hit)
            return True
        if button != 1:
            return True
        if hit and double:
            k, u = hit
            self._edit_ruby(k, u) if u.get("r") or any(LY.is_kanji(c) for c in u["s"]) else self._edit_line(self.lyrics.line_of_mora(k))
            return True
        if hit and self.lyrics.times:
            k, u = hit
            self.drag = {"kind": "lyric", "k": k, "x0": pos[0], "a0": self.lyrics.times[k][0],
                         "shift": self._shift_held(), "moved": False, "snap": None}
            return True
        # 空いた所: その時刻へ移動
        self._seek_to(self._time_at_x(pos[0]))
        return True

    def lyric_mouse_move(self, pos):
        d = self.drag
        if not d or d["kind"] != "lyric":
            return False
        dx = pos[0] - d["x0"]
        if abs(dx) < 3 and not d["moved"]:
            return True
        if not d["moved"]:
            self._push_lyric_undo()
            d["moved"] = True
            d["snap"] = self.lyrics.copy()
        # ドラッグを始めた時点の状態から計算し直す (行き過ぎて戻しても崩れない)
        self.lyrics.times = list(d["snap"].times)
        t = d["a0"] + dx / g("PX_PER_SEC")
        if d["shift"]:
            self.lyrics.shift_from(d["k"], t - d["a0"])
        else:
            self.lyrics.move_mora(d["k"], t)
        return True

    def lyric_mouse_up(self, d):
        if d and d.get("kind") == "lyric":
            if d["moved"]:
                self.lyrics.manual = True
                self.save_lyrics()
            return True
        return False

    def _open_lyric_popup(self, pos, hit):
        ly = self.lyrics
        notes = self.guide.notes
        if hit:
            k, u = hit
            i = ly.line_of_mora(k)
            title = f"「{u['s']}{u.get('k', '')}」" + (f"（{u['r']}）" if u.get("r") else "") + \
                    (f"　{g('fmt_time')(ly.times[k][0])}" if k < len(ly.times) else "")
            items = [
                MenuItem("ふりがなを直す…", lambda: self._edit_ruby(k, u), "ダブルクリック"),
                MenuItem("この行の歌詞を直す…", lambda: self._edit_line(i)),
                SEPARATOR,
                MenuItem("ここから行の終わりまでを 1つ前のバーへ", lambda: self._lyric_step(k, -1), enabled=lambda: bool(notes)),
                MenuItem("ここから行の終わりまでを 1つ後ろのバーへ", lambda: self._lyric_step(k, 1), enabled=lambda: bool(notes)),
                MenuItem("この行の歌い出しを、今の再生位置に", lambda: self._line_to_now(i)),
                MenuItem("この行だけ自動で割り当て直す", lambda: self._reassign_line(i), enabled=lambda: bool(notes)),
                SEPARATOR,
                MenuItem("元に戻す", self.undo, "Ctrl+Z", enabled=lambda: bool(self._undo)),
                MenuItem("歌詞全体を自動で割り当て直す…", self.lyrics_reassign),
            ]
        else:
            title = "歌詞"
            items = [MenuItem("歌詞を入力・貼り付け…", self.lyrics_input),
                     MenuItem("行の歌い出しを記録する（再生しながら Enter）", self.lyrics_start_tap),
                     MenuItem("歌詞全体を自動で割り当て直す…", self.lyrics_reassign),
                     MenuItem("元に戻す", self.undo, "Ctrl+Z", enabled=lambda: bool(self._undo))]
        self.popup = PopupMenu(self.pg, self.fonts, items, pos, (0, g("MENU_H"), g("W"), g("H") - g("MENU_H")),
                               font_size=14, title=title)

    def _edit_ruby(self, k, u):
        v = g("ask_text")("ふりがなを直す", f"「{u['s']}」の読み（ひらがな）を入力してください\n"
                                            "（1音ずつに分けて、お手本バーに割り当て直します）", u.get("r") or u["s"])
        if v is None or not v.strip():
            return
        self._push_lyric_undo()
        i = self.lyrics.line_of_mora(k)
        LY.set_ruby(u, v)
        self.lyrics.times = []           # 音の数が変わることがあるので、この行を割り当て直す
        self._reassign_after_edit(i)
        self.show_toast(f"ふりがな：{u['s']}（{u.get('r') or v}）", 2.0)

    def _edit_line(self, i):
        if i is None:
            return
        l = self.lyrics.lines[i]
        v = g("ask_text")("この行の歌詞を直す", f"{i + 1}行目の歌詞", l["text"])
        if v is None or not v.strip() or v.strip() == l["text"]:
            return
        self._push_lyric_undo()
        old = self.lyrics.copy()
        l["text"] = v.strip()
        self.lyrics.rebuild_units(i)
        self.lyrics.times = []
        self._reassign_after_edit(i, old)
        self.show_toast(f"{i + 1}行目を直しました", 2.0)

    def _reassign_after_edit(self, i, old=None):
        """i 行目だけ音の数が変わったとき: ほかの行のタイミングは残し、i 行目だけ割り当て直す。"""
        old = old or (self._undo[-1][1] if self._undo and self._undo[-1][0] == "lyrics" else None)
        ly = self.lyrics
        if not old or not old.times:
            ly.assign(self.guide.notes, self.pb.duration)
        else:
            # 古いタイミングを行ごとに取り出す
            per_line = []
            k = 0
            for j in range(len(old.lines)):
                n = old.line_morae(j)
                per_line.append(old.times[k:k + n])
                k += n
            span = per_line[i]
            if span:
                a, b = span[0][0], span[-1][1]
            else:
                a = per_line[i - 1][-1][1] if i > 0 and per_line[i - 1] else 0.0
                b = a + 2.0
            m = ly.line_morae(i)
            ns = [(n["start"], n["end"]) for n in self.guide.notes if n["end"] > a + 0.02 and n["start"] < b - 0.02]
            if m == 0:
                new = []
            elif ns:
                new = LY._fit_morae(ns, m)
            else:
                d = (b - a) / m
                new = [(a + r * d, a + (r + 1) * d) for r in range(m)]
            per_line[i] = [(round(x, 3), round(y, 3)) for x, y in new]
            ly.times = [t for s in per_line for t in s]
            ly.manual = True
        self.save_lyrics()

    def _lyric_step(self, k, step):
        self._push_lyric_undo()
        if self.lyrics.step_from(k, self.guide.notes, step):
            self.save_lyrics()
        else:
            self._undo.pop()
            self.show_toast("それ以上ずらせません（となりの行と重なります）", 2.0)

    def _line_to_now(self, i):
        k = self.lyrics.first_mora(i)
        if k is None or k >= len(self.lyrics.times):
            return
        self._push_lyric_undo()
        self.lyrics.lines[i]["t"] = round(self.pb.position(), 3)
        self.lyrics.shift_from(k, self.pb.position() - self.lyrics.times[k][0])
        self.save_lyrics()

    def _reassign_line(self, i):
        k = self.lyrics.first_mora(i)
        if k is None:
            return
        self._push_lyric_undo()
        old = self.lyrics.copy()
        self._reassign_after_edit(i, old)
        self.show_toast(f"{i + 1}行目を割り当て直しました", 2.0)

    # ---------- 元に戻す ----------
    def _push_lyric_undo(self):
        if self.lyrics is None:
            self._undo.append(("lyrics", None, None))
        else:
            self._undo.append(("lyrics", self.lyrics.copy(), None))
        if len(self._undo) > 60:
            self._undo.pop(0)
        self._redo.clear()

    def _lyric_state(self):
        return ("lyrics", self.lyrics.copy() if self.lyrics else None, None)

    def _restore_lyrics(self, ly):
        self.lyrics = ly
        if ly is None:
            p = self._lyrics_path()
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        else:
            self.save_lyrics()
        self._rebuild_background()

    # ---------- 描画 ----------
    def _unit_hit_ratio(self, a, b):
        """a〜b 秒のあいだ、お手本と音程が合っていた割合 (歌っていなければ None)。"""
        total = got = 0.0
        gkey = id(self.guide)
        for n in self.guide.visible(a, b):
            if n["start"] >= b:
                break
            ov = min(b, n["end"]) - max(a, n["start"])
            if ov <= 0:
                continue
            total += ov
            for h0, h1 in self.hits.get((gkey, n["start"]), ()):
                got += max(0.0, min(b, h1) - max(a, h0))
        if total <= 0:
            return None
        return got / total

    def _draw_lyrics(self, L, now, bar_lines):
        pg, scr, F = self.pg, self.screen, self.fonts
        self._poll_lyric_job()
        self._lyric_rects = []
        if self.lyrics is None or not self.view.get("lyrics", True):
            return
        lane = self._lyric_lane_rect()
        if lane.height <= 0:
            return
        W = g("W")
        pg.draw.rect(scr, BG_DEEP, lane)
        pg.draw.line(scr, HAIR, (0, lane.y), (W, lane.y), 1)
        lab = F.render("歌詞", 12, FAINT, bold=True)
        scr.blit(lab, (L.left - lab.get_width() - 10, lane.centery - lab.get_height() // 2))
        clip = pg.Rect(L.left, lane.y + 1, W - L.left, lane.height - 1)
        scr.set_clip(clip)
        if bar_lines is not None and self.view["line"] > 0:          # 小節線を歌詞の段まで伸ばす
            for tt, is_bar, bar, beat in bar_lines:
                if is_bar:
                    x = round(L.x_of(tt, now))
                    pg.draw.line(scr, self._lc(HAIR_STRONG), (x, lane.y), (x, lane.bottom), 1)
        fs = int(self.view.get("lyric_size", 20))
        ruby = self.view.get("ruby", True)
        rfs = max(9, fs // 2)
        y_txt = lane.y + 7 + ((rfs + 2) if ruby else 0)
        t0w, t1w = L.time_window(now)
        ly = self.lyrics
        times = ly.times
        color_mode = self.view.get("lyric_color", True)
        last_right, last_line = -1e9, None
        if times:
            for i, u, k in ly.iter_units():
                n = len(u["m"])
                if n == 0 or k + n > len(times):
                    continue
                a, b = times[k][0], times[k + n - 1][1]
                if b < t0w - 1 or a > t1w + 1:
                    continue
                x = L.x_of(a, now)
                if i == last_line:
                    x = max(x, last_right + 1)
                surf_main = F.render(u["s"] + u.get("k", ""), fs, TODO, bold=True)
                w = surf_main.get_width()
                # 歌ったところまでの割合 (1音ずつの時刻で)
                if now <= a:
                    frac = 0.0
                elif now >= b:
                    frac = 1.0
                else:
                    j = 0
                    while j < n - 1 and now >= times[k + j + 1][0]:
                        j += 1
                    ma, mb = times[k + j]
                    frac = (j + min(1.0, max(0.0, (now - ma) / max(1e-3, mb - ma)))) / n
                if frac > 0:
                    col = SUNG
                    if color_mode and self.score_total > 0:
                        r = self._unit_hit_ratio(a, min(b, now))
                        if r is not None and (now - a) > 0.08:
                            col = SUNG if r >= 0.45 else MISS
                    surf_sung = F.render(u["s"] + u.get("k", ""), fs, col, bold=True)
                rect = pg.Rect(round(x), y_txt, w, surf_main.get_height())
                scr.blit(surf_main, (rect.x, rect.y))
                if frac > 0:
                    cw = int(round(w * frac))
                    if cw > 0:
                        scr.blit(surf_sung, (rect.x, rect.y), (0, 0, cw, rect.height))
                if ruby and u.get("r"):
                    rs = F.render(u["r"], rfs, DIM if frac < 1 else SUNG)
                    sw = F.render(u["s"], fs, TODO, bold=True).get_width() if u.get("k") else w
                    scr.blit(rs, (round(x + sw / 2 - rs.get_width() / 2), y_txt - rfs - 3))
                if self.drag and self.drag.get("kind") == "lyric" and self.drag["k"] == k:
                    pg.draw.rect(scr, GOLD, rect.inflate(4, 4), 1, border_radius=3)
                self._lyric_rects.append((rect, k, u))
                last_right, last_line = rect.right, i
        # 歌い出しの記録中
        if self.lyric_tap is not None:
            for l in ly.lines:
                if l.get("t") is not None:
                    x = round(L.x_of(l["t"], now))
                    pg.draw.polygon(scr, GOLD, [(x, lane.y + 2), (x + 6, lane.y + 2), (x, lane.y + 10)])
        scr.set_clip(None)
        pg.draw.line(scr, (200, 212, 236), (round(L.playhead_x), lane.y), (round(L.playhead_x), lane.bottom), 1)
        if self.lyric_tap is not None:
            i = self.lyric_tap
            nxt = ly.lines[i]["text"] if i < len(ly.lines) else ""
            msg = f"歌い出しを記録中 {i + 1}/{len(ly.lines)}　次の行：{nxt}"
            s1 = F.render(msg, 15, GOLD, bold=True)
            s2 = F.render("歌い出しで Enter　Backspace：1行戻す　Esc：終わる", 12, DIM)
            bw = max(s1.get_width(), s2.get_width()) + 28
            box = pg.Rect(L.left + 12, lane.y - 62, bw, 54)
            pg.draw.rect(scr, PANEL, box, border_radius=8)
            pg.draw.rect(scr, GOLD, box, 1, border_radius=8)
            scr.blit(s1, (box.x + 14, box.y + 7))
            scr.blit(s2, (box.x + 14, box.y + 31))
