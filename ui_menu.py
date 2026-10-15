# -*- coding: utf-8 -*-
"""
pygame 用のメニューバーとボタン (マウス操作の部品)

  MenuBar … 画面上部の「ファイル / 再生 / 音声 / お手本 / 表示 / ヘルプ」
            クリックで開き、開いたまま横に動かすと隣のメニューに切り替わる
  Button  … ツールバーのボタン (クリックで実行、マウスを乗せると説明を表示)
"""

from theme import ACCENT, BG_DEEP, DIM, FAINT, HAIR, OUTLINE, PANEL, PANEL_HI, TEXT

MENU_BG = BG_DEEP
MENU_HOVER = PANEL_HI
DROP_BG = PANEL
DROP_BORDER = OUTLINE
TEXT_DIM = FAINT


class MenuItem:
    def __init__(self, label, action, shortcut="", checked=None, enabled=None):
        self.label = label            # 文字列、または「その時の状態で変わる文字列」を返す関数
        self.action = action
        self.shortcut = shortcut
        self.checked = checked        # 関数 → True/False でチェック印 (None なら印なし)
        self.enabled = enabled        # 関数 → False なら灰色で押せない

    def text(self):
        return self.label() if callable(self.label) else self.label

    def is_enabled(self):
        return self.enabled is None or bool(self.enabled())

    def is_checked(self):
        return self.checked is not None and bool(self.checked())


SEPARATOR = None


class MenuBar:
    def __init__(self, pg, fonts, menus, height=28, font_size=15):
        self.pg, self.fonts = pg, fonts
        self.menus = menus             # [(タイトル, [MenuItem or SEPARATOR, ...]), ...]
        self.height = height
        self.fs = font_size
        self.open_index = None         # 開いているメニュー
        self.hover_item = None
        self.title_rects = []
        self.item_rects = []           # 開いているメニューの (rect, item)
        self.drop_rect = None
        self.width = 0
        self.screen_h = 720

    # ----- 配置 -----
    def _layout_titles(self, width):
        self.width = width
        x = 6
        self.title_rects = []
        for title, _ in self.menus:
            w = self.fonts.render(title, self.fs, TEXT).get_width()
            r = self.pg.Rect(x, 0, w + 22, self.height)
            self.title_rects.append(r)
            x += r.width

    def _layout_drop(self):
        self.item_rects = []
        self.drop_rect = None
        if self.open_index is None:
            return
        _, items = self.menus[self.open_index]
        if callable(items):                # その時の状態で中身が変わるメニュー (機器の一覧など)
            items = items()
        F = self.fonts
        label_w = max((F.render(i.text(), self.fs, TEXT).get_width() for i in items if i), default=100)
        short_w = max((F.render(i.shortcut, self.fs - 2, TEXT_DIM).get_width() for i in items if i and i.shortcut), default=0)
        w = 30 + label_w + (36 + short_w if short_w else 20) + 12
        x = self.title_rects[self.open_index].x
        x = min(x, max(0, self.width - w - 4))
        y = self.height
        row_h, sep_h = self.fs + 15, 9
        n_rows = sum(1 for it in items if it is not SEPARATOR)
        total = row_h * n_rows + sep_h * (len(items) - n_rows) + 4
        room = self.screen_h - y - 6
        if total > room:                   # 画面に入りきらなければ行を詰める
            row_h = max(self.fs + 4, (room - 4 - sep_h * (len(items) - n_rows)) // max(1, n_rows))
        for it in items:
            h = sep_h if it is SEPARATOR else row_h
            self.item_rects.append((self.pg.Rect(x, y, w, h), it))
            y += h
        self.drop_rect = self.pg.Rect(x, self.height, w, y - self.height + 4)

    # ----- 状態 -----
    @property
    def is_open(self):
        return self.open_index is not None

    def close(self):
        self.open_index = None
        self.hover_item = None
        self._layout_drop()

    def open(self, index):
        self.open_index = index
        self._layout_drop()

    def contains(self, pos):
        if pos[1] < self.height:
            return True
        return self.drop_rect is not None and self.drop_rect.collidepoint(pos)

    def is_clickable(self, pos):
        """マウスカーソルを「指」の形にする場所か。"""
        if any(r.collidepoint(pos) for r in self.title_rects):
            return True
        for r, it in self.item_rects:
            if it is not SEPARATOR and r.collidepoint(pos) and it.is_enabled():
                return True
        return False

    # ----- マウス -----
    def on_mouse_down(self, pos):
        """クリックを処理したら True (下の画面には渡さない)。"""
        for i, r in enumerate(self.title_rects):
            if r.collidepoint(pos):
                if self.open_index == i:
                    self.close()
                else:
                    self.open(i)
                return True
        if self.is_open:
            for r, it in self.item_rects:
                if r.collidepoint(pos):
                    if it is not SEPARATOR and it.is_enabled():
                        self.close()
                        it.action()
                    return True
            self.close()            # メニューの外をクリック → 閉じるだけ
            return True
        return pos[1] < self.height

    def on_mouse_move(self, pos):
        self.hover_item = None
        if self.is_open:
            for i, r in enumerate(self.title_rects):   # 開いたまま横に動かすと切り替え
                if r.collidepoint(pos) and i != self.open_index:
                    self.open(i)
            for r, it in self.item_rects:
                if it is not SEPARATOR and r.collidepoint(pos):
                    self.hover_item = it

    # ----- 描画 -----
    def draw(self, surf, mouse_pos=None):
        pg, F = self.pg, self.fonts
        w = surf.get_width()
        if w != self.width or not self.title_rects or surf.get_height() != self.screen_h:
            self.screen_h = surf.get_height()
            self._layout_titles(w)
            self._layout_drop()
        pg.draw.rect(surf, MENU_BG, (0, 0, w, self.height))
        pg.draw.line(surf, HAIR, (0, self.height - 1), (w, self.height - 1))
        for i, ((title, _), r) in enumerate(zip(self.menus, self.title_rects)):
            hot = i == self.open_index or (mouse_pos is not None and r.collidepoint(mouse_pos))
            if hot:
                pg.draw.rect(surf, MENU_HOVER, r)
            s = F.render(title, self.fs - 1, TEXT if hot else DIM)
            surf.blit(s, (r.x + 11, r.y + (r.height - s.get_height()) // 2))

        if self.drop_rect is None:
            return
        shadow = self.drop_rect.move(0, 4)
        pg.draw.rect(surf, (6, 9, 18), shadow, border_radius=8)
        pg.draw.rect(surf, DROP_BG, self.drop_rect, border_radius=8)
        pg.draw.rect(surf, DROP_BORDER, self.drop_rect, 1, border_radius=8)
        for r, it in self.item_rects:
            if it is SEPARATOR:
                y = r.y + r.height // 2
                pg.draw.line(surf, DROP_BORDER, (r.x + 10, y), (r.right - 10, y))
                continue
            enabled = it.is_enabled()
            if it is self.hover_item and enabled:
                pg.draw.rect(surf, MENU_HOVER, r.inflate(-6, -2), border_radius=4)
            col = TEXT if enabled else TEXT_DIM
            if it.is_checked():
                c = F.render("✓", self.fs, ACCENT, bold=True)
                surf.blit(c, (r.x + 9, r.y + (r.height - c.get_height()) // 2))
            s = F.render(it.text(), self.fs, col)
            surf.blit(s, (r.x + 30, r.y + (r.height - s.get_height()) // 2))
            if it.shortcut:
                k = F.render(it.shortcut, self.fs - 2, TEXT_DIM)
                surf.blit(k, (r.right - k.get_width() - 14, r.y + (r.height - k.get_height()) // 2))


class PopupMenu:
    """右クリックで出るメニュー (マウスの位置に開く)。"""

    def __init__(self, pg, fonts, items, pos, area, font_size=14, title=None):
        self.pg, self.fonts, self.items, self.fs, self.title = pg, fonts, items, font_size, title
        self.hover_item = None
        self.closed = False
        F = fonts
        label_w = max((F.render(i.text(), font_size, TEXT).get_width() for i in items if i), default=100)
        short_w = max((F.render(i.shortcut, font_size - 2, TEXT_DIM).get_width() for i in items if i and i.shortcut),
                      default=0)
        w = 30 + label_w + (30 + short_w if short_w else 16) + 12
        ax, ay, aw, ah = area                      # 開いてよい範囲 (画面)
        row_h, sep_h = font_size + 15, 9
        head = (font_size + 14) if title else 0
        total = head + sum(sep_h if it is SEPARATOR else row_h for it in items) + 8
        if total > ah:                             # 入りきらなければ行を詰める
            n_rows = sum(1 for it in items if it is not SEPARATOR)
            row_h = max(font_size + 6, (ah - head - 8 - sep_h * (len(items) - n_rows)) // max(1, n_rows))
            total = head + sum(sep_h if it is SEPARATOR else row_h for it in items) + 8
        x = min(max(ax, pos[0] + 2), ax + aw - w - 2)
        y = min(max(ay, pos[1] + 2), ay + ah - total)
        self.rect = pg.Rect(x, y, w, total)
        self.head_h = head
        self.item_rects = []
        yy = y + 4 + head
        for it in items:
            h = sep_h if it is SEPARATOR else row_h
            self.item_rects.append((pg.Rect(x, yy, w, h), it))
            yy += h

    def on_mouse_down(self, pos):
        """クリックを処理 (いつも True)。項目を押したら実行して閉じる。外を押したら閉じるだけ。"""
        if self.rect.collidepoint(pos):
            for r, it in self.item_rects:
                if r.collidepoint(pos) and it is not SEPARATOR and it.is_enabled():
                    self.closed = True
                    it.action()
                    return True
            return True
        self.closed = True
        return True

    def on_mouse_move(self, pos):
        self.hover_item = None
        for r, it in self.item_rects:
            if it is not SEPARATOR and r.collidepoint(pos):
                self.hover_item = it

    def is_clickable(self, pos):
        return any(it is not SEPARATOR and r.collidepoint(pos) and it.is_enabled() for r, it in self.item_rects)

    def draw(self, surf):
        pg, F = self.pg, self.fonts
        pg.draw.rect(surf, (6, 9, 18), self.rect.move(0, 4), border_radius=8)
        pg.draw.rect(surf, DROP_BG, self.rect, border_radius=8)
        pg.draw.rect(surf, DROP_BORDER, self.rect, 1, border_radius=8)
        if self.title:
            s = F.render(self.title, self.fs - 2, TEXT_DIM)
            surf.blit(s, (self.rect.x + 12, self.rect.y + 9))
            pg.draw.line(surf, DROP_BORDER, (self.rect.x + 8, self.rect.y + self.head_h + 2),
                         (self.rect.right - 8, self.rect.y + self.head_h + 2))
        for r, it in self.item_rects:
            if it is SEPARATOR:
                y = r.y + r.height // 2
                pg.draw.line(surf, DROP_BORDER, (r.x + 10, y), (r.right - 10, y))
                continue
            enabled = it.is_enabled()
            if it is self.hover_item and enabled:
                pg.draw.rect(surf, MENU_HOVER, r.inflate(-6, -2), border_radius=4)
            if it.is_checked():
                c = F.render("✓", self.fs, ACCENT, bold=True)
                surf.blit(c, (r.x + 9, r.y + (r.height - c.get_height()) // 2))
            s = F.render(it.text(), self.fs, TEXT if enabled else TEXT_DIM)
            surf.blit(s, (r.x + 30, r.y + (r.height - s.get_height()) // 2))
            if it.shortcut:
                k = F.render(it.shortcut, self.fs - 2, TEXT_DIM)
                surf.blit(k, (r.right - k.get_width() - 12, r.y + (r.height - k.get_height()) // 2))


class Button:
    def __init__(self, rect, label, action, tooltip="", active=False, enabled=True, accent=None,
                 repeat=False, static=False, wheel=None):
        self.rect, self.label, self.action = rect, label, action
        self.tooltip, self.active, self.enabled, self.accent = tooltip, active, enabled, accent
        self.repeat = repeat      # 押しっぱなしで繰り返す
        self.static = static      # 表示だけ (押せないが色は付ける)
        self.wheel = wheel        # ホイールで呼ぶ関数 (dy) / None
        if static:
            self.enabled = False

    def draw(self, pg, surf, fonts, hover, size=15):
        if self.static:                           # 表示だけ: 面も枠もなし
            pass
        elif hover and self.enabled:
            pg.draw.rect(surf, PANEL_HI, self.rect, border_radius=6)
        else:
            pg.draw.rect(surf, PANEL, self.rect, border_radius=6)
        if self.active:
            pg.draw.rect(surf, self.accent or ACCENT, self.rect, 1, border_radius=6)
        elif not self.static:
            pg.draw.rect(surf, OUTLINE, self.rect, 1, border_radius=6)
        col = (self.accent or TEXT) if (self.enabled or self.static) else TEXT_DIM
        s = fonts.render(self.label, size, col, bold=True)
        surf.blit(s, (self.rect.centerx - s.get_width() // 2, self.rect.centery - s.get_height() // 2))


def draw_tooltip(pg, surf, fonts, text, pos):
    s = fonts.render(text, 13, TEXT)
    r = pg.Rect(0, 0, s.get_width() + 14, s.get_height() + 8)
    r.midbottom = (pos[0], pos[1] - 10)
    r.x = max(4, min(r.x, surf.get_width() - r.width - 4))
    r.y = max(4, r.y)
    pg.draw.rect(surf, (8, 11, 22), r, border_radius=5)
    pg.draw.rect(surf, DROP_BORDER, r, 1, border_radius=5)
    surf.blit(s, (r.x + 7, r.y + 4))
