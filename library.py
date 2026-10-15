# -*- coding: utf-8 -*-
"""
曲ライブラリ: フォルダの中の曲を一覧表示し、選んだ曲を解析して練習画面へ

  - フォルダを選ぶ (最近使ったフォルダは settings.json に記憶)
  - サブフォルダも含めて探すかを切り替え
  - 各曲の状態: 解析済みか / 同じ名前の MIDI があるか / キー
    (状態はバックグラウンドで調べるので、曲が多くても画面は止まらない)
"""

import json
import os
import threading
import time

from theme import ACCENT, BG_DEEP, FAINT, HAIR, HAIR_STRONG, OUTLINE, PANEL, PANEL_HI

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma")
MIDI_EXTS = (".mid", ".midi", ".kar")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.path.join(HERE, "cache")
SETTINGS_PATH = os.path.join(HERE, "settings.json")
INDEX_PATH = os.path.join(CACHE_ROOT, "library_index.json")


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        s = {}
    s.setdefault("folder", None)
    s.setdefault("recent_folders", [])
    s.setdefault("recursive", True)
    return s


def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def remember_folder(settings, folder):
    folder = os.path.abspath(folder)
    settings["folder"] = folder
    rec = [f for f in settings.get("recent_folders", []) if os.path.normcase(f) != os.path.normcase(folder)]
    settings["recent_folders"] = [folder] + rec[:7]
    save_settings(settings)


def default_folder():
    for cand in (os.path.join(os.path.expanduser("~"), "Music"), os.path.join(os.path.expanduser("~"), "Downloads"), HERE):
        if os.path.isdir(cand):
            return cand
    return HERE


def choose_folder(initial=None):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="曲が入っているフォルダを選んでください", initialdir=initial)
        root.destroy()
        return path or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 曲の一覧
# ---------------------------------------------------------------------------
def scan_folder(folder, recursive=True, limit=3000):
    """フォルダの中の曲 → [{"path", "name", "rel", "size", "mtime", "midi"}]"""
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    skip_dirs = {os.path.normcase(CACHE_ROOT)}
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and os.path.normcase(os.path.join(root, d)) not in skip_dirs)
        lower = {f.lower(): f for f in files}
        for f in sorted(files, key=str.lower):
            if not f.lower().endswith(AUDIO_EXTS):
                continue
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            stem = os.path.splitext(f)[0].lower()
            midi = next((os.path.join(root, lower[stem + e]) for e in MIDI_EXTS if stem + e in lower), None)
            out.append({"path": p, "name": os.path.splitext(f)[0], "rel": os.path.relpath(root, folder),
                        "size": st.st_size, "mtime": st.st_mtime, "midi": midi})
            if len(out) >= limit:
                return out
        if not recursive:
            break
    return out


class StatusChecker:
    """各曲が解析済みか・キーは何か を裏で調べる。結果は cache/library_index.json に覚えておく。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.status = {}          # path -> {"analyzed": bool, "key": str|None, "dir": str}
        self.queue = []
        self._index = self._load_index()
        self._thread = None

    @staticmethod
    def _load_index():
        try:
            with open(INDEX_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self):
        try:
            os.makedirs(CACHE_ROOT, exist_ok=True)
            with open(INDEX_PATH, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False)
        except Exception:
            pass

    def request(self, songs):
        with self.lock:
            self.queue = [s for s in songs]
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, daemon=True)
            self._thread.start()

    def refresh(self, path):
        """解析が終わった曲の状態を更新。"""
        with self.lock:
            self.status.pop(path, None)
            self.queue.insert(0, {"path": path})
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, daemon=True)
            self._thread.start()

    def get(self, path):
        with self.lock:
            return self.status.get(path)

    def _cache_dir(self, path):
        from audio_processor import cache_dir_for

        st = os.stat(path)
        key = os.path.normcase(os.path.abspath(path))
        ent = self._index.get(key)
        if ent and ent.get("size") == st.st_size and abs(ent.get("mtime", 0) - st.st_mtime) < 1:
            return ent["dir"]
        d = cache_dir_for(path, CACHE_ROOT)        # ファイル全体のハッシュ (少し時間がかかる)
        self._index[key] = {"size": st.st_size, "mtime": st.st_mtime, "dir": os.path.basename(d)}
        return os.path.basename(d)

    def _work(self):
        changed = 0
        while True:
            with self.lock:
                if not self.queue:
                    break
                s = self.queue.pop(0)
                if s["path"] in self.status:
                    continue
            try:
                d = os.path.join(CACHE_ROOT, self._cache_dir(s["path"]))
                analyzed = os.path.exists(os.path.join(d, "pitch.json")) and os.path.exists(os.path.join(d, "vocals.wav"))
                key = None
                kp = os.path.join(d, "key.json")
                if os.path.exists(kp):
                    with open(kp, encoding="utf-8") as f:
                        key = json.load(f).get("short")
                res = {"analyzed": analyzed, "key": key, "dir": d}
                changed += 1
            except Exception:
                res = {"analyzed": False, "key": None, "dir": None}
            with self.lock:
                self.status[s["path"]] = res
            if changed and changed % 20 == 0:
                self._save_index()
        if changed:
            self._save_index()


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------
class LibraryView:
    """曲の一覧画面。選んだ曲は on_open(path, midi) で知らせる。"""

    ROW_H = 40

    def __init__(self, pg, fonts, settings, on_open, menu_factory, requests=None):
        self.pg, self.fonts, self.settings = pg, fonts, settings
        self.on_open = on_open
        self.requests = requests if requests is not None else []
        self.menu = menu_factory(self)
        self.checker = StatusChecker()
        self.songs, self.shown = [], []
        self.selected = 0
        self.scroll = 0
        self.mouse = (-1, -1)
        self.buttons = []
        self.filter_text = ""
        self.message = ""
        self._last_click = (0.0, -1)
        self.sort_key = "name"
        self.folder = settings.get("folder") or default_folder()
        self.reload()

    # ----- 一覧 -----
    def reload(self):
        self.songs = scan_folder(self.folder, self.settings.get("recursive", True))
        self.apply_filter()
        self.checker.request(self.songs)

    def set_folder(self, folder):
        if folder and os.path.isdir(folder):
            self.folder = os.path.abspath(folder)
            remember_folder(self.settings, self.folder)
            self.selected, self.scroll, self.filter_text = 0, 0, ""
            self.reload()

    def toggle_recursive(self):
        self.settings["recursive"] = not self.settings.get("recursive", True)
        save_settings(self.settings)
        self.reload()

    def set_sort(self, key):
        self.sort_key = key
        self.apply_filter()

    def apply_filter(self):
        q = self.filter_text.lower()
        shown = [s for s in self.songs if q in s["name"].lower() or q in s["rel"].lower()]
        if self.sort_key == "new":
            shown.sort(key=lambda s: -s["mtime"])
        elif self.sort_key == "done":
            shown.sort(key=lambda s: (not (self.checker.get(s["path"]) or {}).get("analyzed", False), s["name"].lower()))
        else:
            shown.sort(key=lambda s: (s["rel"].lower(), s["name"].lower()))
        self.shown = shown
        self.selected = min(self.selected, max(0, len(shown) - 1))

    def current(self):
        return self.shown[self.selected] if 0 <= self.selected < len(self.shown) else None

    def open_selected(self):
        s = self.current()
        if s:
            self.on_open(s["path"], s["midi"])

    def select_path(self, path):
        for i, s in enumerate(self.shown):
            if os.path.normcase(s["path"]) == os.path.normcase(os.path.abspath(path)):
                self.selected = i
                self._ensure_visible()
                return True
        return False

    # ----- 配置 -----
    def _list_rect(self, W, H):
        top = self.menu.height + 128
        return self.pg.Rect(16, top, W - 32, H - top - 60)

    def _visible_rows(self, W, H):
        return max(1, self._list_rect(W, H).height // self.ROW_H - 1)

    def _ensure_visible(self, W=1280, H=720):
        n = self._visible_rows(W, H)
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + n:
            self.scroll = self.selected - n + 1

    # ----- 入力 -----
    def handle_key(self, ev, W, H):
        pg = self.pg
        k = ev.key
        if k in (pg.K_DOWN, pg.K_UP):
            self.selected = max(0, min(len(self.shown) - 1, self.selected + (1 if k == pg.K_DOWN else -1)))
            self._ensure_visible(W, H)
        elif k in (pg.K_PAGEDOWN, pg.K_PAGEUP):
            n = self._visible_rows(W, H)
            self.selected = max(0, min(len(self.shown) - 1, self.selected + (n if k == pg.K_PAGEDOWN else -n)))
            self._ensure_visible(W, H)
        elif k in (pg.K_RETURN, pg.K_KP_ENTER):
            self.open_selected()
        elif k == pg.K_BACKSPACE:
            self.filter_text = self.filter_text[:-1]
            self.apply_filter()
        elif k == pg.K_F5:
            self.reload()

    def handle_text(self, text):
        """文字を打つと曲名で絞り込み。"""
        if text and text.isprintable():
            self.filter_text += text
            self.selected, self.scroll = 0, 0
            self.apply_filter()

    def handle_mouse_down(self, pos, W, H, button=1):
        self.mouse = pos
        if button != 1 or self.menu.on_mouse_down(pos):
            return
        for b in self.buttons:
            if b.rect.collidepoint(pos):
                if b.enabled:
                    b.action()
                return
        lr = self._list_rect(W, H)
        if lr.collidepoint(pos) and pos[1] > lr.y + self.ROW_H:
            i = self.scroll + (pos[1] - lr.y - self.ROW_H) // self.ROW_H
            if 0 <= i < len(self.shown):
                now = time.time()
                if self._last_click[1] == i and now - self._last_click[0] < 0.45:
                    self.selected = i
                    self.open_selected()          # ダブルクリックで開く
                    self._last_click = (0.0, -1)
                else:
                    self.selected = i
                    self._last_click = (now, i)

    def handle_mouse_move(self, pos):
        self.mouse = pos
        self.menu.on_mouse_move(pos)

    def handle_wheel(self, dy, W, H):
        if self.menu.is_open:
            return
        n = self._visible_rows(W, H)
        self.scroll = max(0, min(max(0, len(self.shown) - n), self.scroll - dy * 3))

    def cursor_kind(self, W, H):
        pos = self.mouse
        if self.menu.is_clickable(pos) or any(b.rect.collidepoint(pos) and b.enabled for b in self.buttons):
            return "hand"
        lr = self._list_rect(W, H)
        if lr.collidepoint(pos) and pos[1] > lr.y + self.ROW_H:
            i = self.scroll + (pos[1] - lr.y - self.ROW_H) // self.ROW_H
            if 0 <= i < len(self.shown):
                return "hand"
        return "arrow"

    # ----- 描画 -----
    def draw(self, scr, W, H, colors):
        from ui_menu import Button, draw_tooltip

        pg, F = self.pg, self.fonts
        BG, WHITE, DIM, GOLD, GOOD, NEAR = (colors[k] for k in ("BG", "WHITE", "DIM", "GOLD", "GOOD", "NEAR"))
        scr.fill(BG)
        oy = self.menu.height
        # 見出し
        scr.blit(F.render("曲を選んでください", 24, WHITE, bold=True), (20, oy + 14))
        folder_txt = self.folder if len(self.folder) < 90 else "…" + self.folder[-88:]
        scr.blit(F.render("フォルダ", 12, FAINT), (22, oy + 54))
        scr.blit(F.render(folder_txt, 14, DIM), (78, oy + 52))
        dt = getattr(self, "device_text", "")
        if dt:
            dcol = GOOD if dt.startswith("解析：グラボ") else NEAR
            ds = F.render(dt if len(dt) < 70 else dt[:68] + "…", 13, dcol)
            scr.blit(ds, (W - ds.get_width() - 20, oy + 20))
        # ボタン行
        self.buttons = []
        x, y = 20, oy + 80
        rec = self.settings.get("recursive", True)
        specs = [
            ("← 楽譜に戻る", lambda: self.request("score"), "曲を選ばずに楽譜の画面へ戻る (Esc)", False, True, None),
            ("フォルダを選ぶ…", lambda: self.request("open_folder"), "曲が入っているフォルダを選ぶ (Ctrl+F)", False, True, None),
            ("曲ファイルを直接開く…", lambda: self.request("open_song"), "1曲だけ選んで開く (Ctrl+O)", False, True, None),
            (("✓ " if rec else "") + "サブフォルダも探す", self.toggle_recursive, "フォルダの中のフォルダも探す", rec, True, None),
            ("一覧を更新", self.reload, "一覧を読み直す (F5)", False, True, None),
        ]
        for label, act, tip, active, en, acc in specs:
            w = F.render(label, 15, WHITE, bold=True).get_width() + 28
            b = Button(pg.Rect(x, y, w, 34), label, act, tip, active, en, acc)
            b.draw(pg, scr, F, b.rect.collidepoint(self.mouse) and not self.menu.is_open)
            self.buttons.append(b)
            x += w + 8
        # 絞り込み欄
        fx = max(x + 12, W - 360)
        frect = pg.Rect(fx, y, W - 20 - fx, 34)
        pg.draw.rect(scr, BG_DEEP, frect, border_radius=6)
        pg.draw.rect(scr, OUTLINE, frect, 1, border_radius=6)
        ft = self.filter_text or "文字を入力すると曲名で絞り込み"
        scr.blit(F.render("検索", 12, FAINT), (frect.x + 12, frect.y + 10))
        scr.blit(F.render(ft[-30:], 14, WHITE if self.filter_text else FAINT), (frect.x + 52, frect.y + 8))

        # 一覧
        lr = self._list_rect(W, H)
        pg.draw.rect(scr, BG_DEEP, lr, border_radius=8)
        pg.draw.rect(scr, HAIR, lr, 1, border_radius=8)
        cols = [("曲名", 0.50), ("状態", 0.17), ("キー", 0.21), ("MIDI", 0.12)]
        cx = lr.x + 14
        col_x = []
        for name, frac in cols:
            col_x.append(cx)
            scr.blit(F.render(name, 12, FAINT, bold=True), (cx, lr.y + 13))
            cx += int((lr.width - 28) * frac)
        pg.draw.line(scr, HAIR, (lr.x + 8, lr.y + self.ROW_H - 2), (lr.right - 8, lr.y + self.ROW_H - 2))
        n = self._visible_rows(W, H)
        self.scroll = max(0, min(self.scroll, max(0, len(self.shown) - n)))
        if not self.shown:
            msg = "このフォルダには曲がありません。「フォルダを選ぶ…」で曲の入ったフォルダを選んでください" \
                if not self.filter_text else f"「{self.filter_text}」に合う曲がありません（Backspace で消す）"
            scr.blit(F.render(msg, 16, DIM), (lr.x + 20, lr.y + self.ROW_H + 20))
        for row, s in enumerate(self.shown[self.scroll : self.scroll + n]):
            i = self.scroll + row
            ry = lr.y + self.ROW_H + row * self.ROW_H
            r = pg.Rect(lr.x + 4, ry, lr.width - 8, self.ROW_H - 2)
            hover = r.collidepoint(self.mouse) and not self.menu.is_open
            if i == self.selected:
                pg.draw.rect(scr, PANEL_HI, r, border_radius=6)
                pg.draw.rect(scr, ACCENT, (r.x, r.y + 8, 3, r.height - 16), border_radius=2)   # 左に細い水色の印
            elif hover:
                pg.draw.rect(scr, PANEL, r, border_radius=6)
            else:
                pg.draw.line(scr, (20, 27, 46), (r.x + 8, r.bottom), (r.right - 8, r.bottom), 1)
            name = s["name"] if s["rel"] in (".", "") else f"{s['rel']}{os.sep}{s['name']}"
            maxw = col_x[1] - col_x[0] - 16
            label = name
            while F.render(label, 16, WHITE).get_width() > maxw and len(label) > 4:
                label = label[:-2]
            if label != name:
                label = label[:-1] + "…"
            scr.blit(F.render(label, 16, WHITE), (col_x[0], ry + 9))
            st = self.checker.get(s["path"])
            if st is None:
                scr.blit(F.render("確認中…", 14, DIM), (col_x[1], ry + 11))
            elif st["analyzed"]:
                scr.blit(F.render("✓ 解析済み", 15, GOOD, bold=True), (col_x[1], ry + 10))
            else:
                scr.blit(F.render("未解析", 15, NEAR), (col_x[1], ry + 10))
            if st and st.get("key"):
                scr.blit(F.render(st["key"], 14, WHITE), (col_x[2], ry + 11))
            if s["midi"]:
                scr.blit(F.render("あり", 14, ACCENT, bold=True), (col_x[3], ry + 11))
        # スクロールバー
        if len(self.shown) > n:
            track = pg.Rect(lr.right - 8, lr.y + self.ROW_H, 4, lr.height - self.ROW_H - 6)
            pg.draw.rect(scr, HAIR, track, border_radius=2)
            th = max(24, track.height * n // len(self.shown))
            ty = track.y + (track.height - th) * self.scroll // max(1, len(self.shown) - n)
            pg.draw.rect(scr, HAIR_STRONG, (track.x, ty, 4, th), border_radius=2)

        # 下の行: 件数 と 開くボタン
        by = H - 50
        cur = self.current()
        done = sum(1 for s in self.songs if (self.checker.get(s["path"]) or {}).get("analyzed"))
        info = f"{len(self.shown)} 曲" + (f"（全 {len(self.songs)} 曲中）" if len(self.shown) != len(self.songs) else "") + f"　解析済み {done} 曲"
        scr.blit(F.render(info, 14, DIM), (20, by + 4))
        hint = "ダブルクリック または Enter で開きます。未解析の曲は解析してから開きます（初回は数分〜十数分）"
        scr.blit(F.render(hint, 12, FAINT), (20, by + 25))
        st = self.checker.get(cur["path"]) if cur else None
        label = ("▶ この曲で練習" if st and st["analyzed"] else "▶ 解析して練習") if cur else "▶ 練習"
        bw = F.render(label, 17, WHITE, bold=True).get_width() + 40
        b = Button(pg.Rect(W - bw - 20, by, bw, 38), label, self.open_selected,
                   (cur["name"] if cur else ""), True, cur is not None, GOOD)
        b.draw(pg, scr, F, b.rect.collidepoint(self.mouse) and not self.menu.is_open, 17)
        self.buttons.append(b)

        if self.message:
            scr.blit(F.render(self.message, 15, NEAR, bold=True), (W // 2 - 200, by - 26))

        self.menu.draw(scr, self.mouse)
        hover = next((b for b in self.buttons if b.rect.collidepoint(self.mouse)), None)
        if hover and hover.tooltip and not self.menu.is_open:
            draw_tooltip(pg, scr, F, hover.tooltip, hover.rect.midtop)

    def request(self, what):
        self.requests.append(what)
