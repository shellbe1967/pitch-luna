#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI ファイルからお手本バーを作る

  1. MIDI を読み込み (外部ライブラリ不要)、トラック/チャンネルごとにメロディ候補を作る
  2. AI で解析した実際の歌声と照らし合わせて
       - どのパートがボーカルのメロディか (自動選択)
       - 曲の録音とのタイミングのずれ (自動で合わせる)
       - テンポのわずかな違い (±4% まで自動補正)
       - オクターブの違い
     を決める
  3. お手本バー (audio_processor と同じ形式) を返す

使い方 (単体で確認):
    python midi_loader.py melody.mid                 # MIDI の中身 (パート一覧) を表示
    python midi_loader.py melody.mid song.mp3        # 曲と合わせた結果を表示
"""

import hashlib
import json
import os
import struct
import sys

import numpy as np

from pitch_core import midi_to_doremi, midi_to_english

HOP = 0.01  # 10ms


# ---------------------------------------------------------------------------
# 標準 MIDI ファイルの読み込み
# ---------------------------------------------------------------------------
def _varlen(data, pos):
    value = 0
    while True:
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos


def parse_midi(path):
    """MIDI → {"tracks": [{"name", "notes": [(start_tick, end_tick, note, vel, ch)]}], "tempos": [(tick, us_per_beat)], ...}"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"MThd":
        # RIFF でくるまれた MIDI (.rmi)
        k = data.find(b"MThd")
        if k < 0:
            raise ValueError("MIDI ファイルではありません")
        data = data[k:]
    hlen = struct.unpack(">I", data[4:8])[0]
    fmt, ntrk, division = struct.unpack(">HHH", data[8:14])
    pos = 8 + hlen
    smpte = None
    if division & 0x8000:  # SMPTE (まれ)
        fps = 256 - (division >> 8)
        smpte = fps * (division & 0xFF)

    tracks, tempos, keysigs = [], [], []
    for _ in range(ntrk):
        while pos + 8 <= len(data) and data[pos : pos + 4] != b"MTrk":  # 未知のチャンクを飛ばす
            pos += 8 + struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        if pos + 8 > len(data):
            break
        length = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        p, end = pos + 8, pos + 8 + length
        pos = end
        tick, status = 0, 0
        name = ""
        active = {}  # (ch, note) -> [(start, vel), ...]
        notes = []
        programs = set()
        while p < end:
            delta, p = _varlen(data, p)
            tick += delta
            b = data[p]
            if b & 0x80:
                status = b
                p += 1
            elif status == 0:
                break  # 壊れたデータ
            hi = status & 0xF0
            if status == 0xFF:  # メタイベント
                mtype = data[p]
                ln, p = _varlen(data, p + 1)
                body = data[p : p + ln]
                p += ln
                if mtype == 0x51 and ln == 3:
                    tempos.append((tick, int.from_bytes(body, "big")))
                elif mtype in (0x03, 0x04) and not name:
                    for enc in ("utf-8", "shift_jis", "latin-1"):
                        try:
                            name = body.decode(enc).strip("\x00 ")
                            break
                        except UnicodeDecodeError:
                            continue
                elif mtype == 0x59 and ln == 2:
                    keysigs.append((tick, struct.unpack("b", body[:1])[0], body[1]))
                elif mtype == 0x2F:
                    break
                status = 0
            elif status in (0xF0, 0xF7):  # SysEx
                ln, p = _varlen(data, p)
                p += ln
                status = 0
            else:
                ch = status & 0x0F
                if hi in (0x80, 0x90):
                    note, vel = data[p], data[p + 1]
                    p += 2
                    key = (ch, note)
                    if hi == 0x90 and vel > 0:
                        active.setdefault(key, []).append((tick, vel))
                    elif active.get(key):
                        st, v = active[key].pop(0)
                        if tick > st:
                            notes.append((st, tick, note, v, ch))
                elif hi in (0xA0, 0xB0, 0xE0):
                    p += 2
                elif hi in (0xC0, 0xD0):
                    if hi == 0xC0:
                        programs.add((ch, data[p]))
                    p += 1
                else:
                    break
        for (ch, note), lst in active.items():  # 終わりの無いノートは曲の最後まで
            for st, v in lst:
                if tick > st:
                    notes.append((st, tick, note, v, ch))
        tracks.append({"name": name, "notes": sorted(notes), "programs": sorted(programs)})

    return {"format": fmt, "division": division, "smpte": smpte, "tracks": tracks,
            "tempos": sorted(tempos) or [(0, 500000)], "keysigs": keysigs}


def tick_to_sec_fn(midi):
    """テンポ変化を考慮した「tick → 秒」の変換関数。"""
    if midi["smpte"]:
        return lambda t: t / midi["smpte"]
    div = midi["division"]
    tempos = midi["tempos"]
    if tempos[0][0] != 0:
        tempos = [(0, 500000)] + tempos
    ticks = np.array([t for t, _ in tempos], dtype=np.float64)
    us = np.array([u for _, u in tempos], dtype=np.float64)
    secs = np.concatenate([[0.0], np.cumsum(np.diff(ticks) * us[:-1] / div / 1e6)])

    def conv(t):
        i = np.searchsorted(ticks, t, side="right") - 1
        return secs[i] + (t - ticks[i]) * us[i] / div / 1e6

    return conv


# ---------------------------------------------------------------------------
# メロディ候補
# ---------------------------------------------------------------------------
def melody_candidates(midi):
    """トラック × チャンネルごとに、単音のメロディ (同時に鳴る音は一番高い音) を作る。"""
    conv = tick_to_sec_fn(midi)
    cands = []
    for ti, tr in enumerate(midi["tracks"]):
        by_ch = {}
        for st, en, note, vel, ch in tr["notes"]:
            if ch == 9:  # ドラム
                continue
            by_ch.setdefault(ch, []).append((conv(st), conv(en), note))
        for ch, notes in by_ch.items():
            if len(notes) < 8:
                continue
            mono = skyline(notes)
            if not mono:
                continue
            poly = _polyphony(notes)
            cands.append({
                "id": f"{ti}:{ch}", "track": ti, "channel": ch,
                "name": tr["name"] or f"トラック{ti + 1}",
                "notes": mono, "count": len(notes), "polyphony": poly,
                "low": min(n[2] for n in mono), "high": max(n[2] for n in mono),
            })
    return cands


def _polyphony(notes):
    """同時に鳴っている音の平均数 (和音パートほど大きい)。"""
    ev = sorted([(s, 1) for s, e, n in notes] + [(e, -1) for s, e, n in notes])
    cur, last, area, total = 0, None, 0.0, 0.0
    for t, d in ev:
        if last is not None and cur > 0:
            area += cur * (t - last)
            total += t - last
        cur += d
        last = t
    return area / total if total else 1.0


def skyline(notes):
    """重なった音は一番高い音を残して単音化。戻り値: [(start, end, note)]"""
    notes = sorted(notes, key=lambda x: (x[0], -x[2]))
    out = []
    for s, e, n in notes:
        if out and s < out[-1][1]:
            ps, pe, pn = out[-1]
            if n > pn:          # 新しい音のほうが高い → 前の音をここで切る
                if s - ps < 0.03:
                    out.pop()
                else:
                    out[-1] = (ps, s, pn)
                out.append((s, e, n))
            elif e > pe:        # 低い音が前の音より長く続く → 前の音の後ろだけ残す
                if e - pe >= 0.05:
                    out.append((pe, e, n))
            continue
        out.append((s, e, n))
    return [x for x in out if x[1] - x[0] >= 0.02]


def notes_to_roll(notes, n_frames, offset=0.0, stretch=1.0, hop=HOP):
    """メロディ → hop 秒ごとの MIDI 番号 (無音は nan)。"""
    roll = np.full(n_frames, np.nan)
    for s, e, n in notes:
        a = int(round((s * stretch + offset) / hop))
        b = int(round((e * stretch + offset) / hop))
        a, b = max(0, a), min(n_frames, b)
        if b > a:
            roll[a:b] = n
    return roll


def _best_offset(vocal, notes, stretch, offsets, hop):
    """offsets (秒の配列) の中で、歌声と音名が一致するフレーム数が最大のもの。"""
    v_ok = ~np.isnan(vocal)
    n = len(vocal)
    pad = int(np.ceil(np.max(np.abs(offsets)) / hop)) + 1
    roll = notes_to_roll(notes, n + 2 * pad, pad * hop, stretch, hop)  # 前後に余白をつけた MIDI
    best, best_o = -1, 0.0
    for o in offsets:
        k = pad - int(round(o / hop))          # MIDI を o 秒後ろへずらす = 参照位置を前へ
        r = roll[k : k + n]
        both = v_ok & ~np.isnan(r)
        if not both.any():
            continue
        d = vocal[both] - r[both]
        d = d - 12 * np.round(d / 12)          # オクターブ違いは同じ音として数える
        sc = np.count_nonzero(np.abs(d) <= 0.6)
        if sc > best:
            best, best_o = sc, float(o)
    return best, best_o


def stable_frames(vocal, width=4, tol=0.35):
    """前後 width フレームとの差が tol 半音以内のフレームだけ残す。"""
    out = vocal.copy()
    n = len(vocal)
    for d in (-width, width):
        sh = np.full(n, np.nan)
        if d > 0:
            sh[:-d] = vocal[d:]
        else:
            sh[-d:] = vocal[:d]
        with np.errstate(invalid="ignore"):
            out[~(np.abs(vocal - sh) <= tol)] = np.nan
    return out


def align(cands, vocal_midi, max_offset=40.0, stretches=None):
    """歌声 (10ms ごとの MIDI) に一番合う候補・タイミング・テンポ・オクターブを探す。"""
    if stretches is None:
        stretches = np.round(np.arange(0.96, 1.0401, 0.005), 3)
    vocal = np.array([np.nan if v is None else v for v in vocal_midi], dtype=np.float64)
    # 音程が安定している部分だけで照合する (音の変わり目はしゃくりや残響で遅れて見えるため)
    vocal = stable_frames(vocal)
    # 粗い探索は 50ms 単位 (5 フレームの中央値) で速く
    k = 5
    m = len(vocal) // k * k
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vc = np.nanmedian(vocal[:m].reshape(-1, k), axis=1)
    hop_c = HOP * k
    voiced_c = max(1, np.count_nonzero(~np.isnan(vc)))
    coarse = np.arange(-max_offset, max_offset + 1e-9, 0.1)

    # 1) 全パートをテンポ 1.0 で比べる
    results = []
    for c in cands:
        sc, off = _best_offset(vc, c["notes"], 1.0, coarse, hop_c)
        results.append([sc / voiced_c, c, off, 1.0])
    results.sort(key=lambda r: -r[0])

    # 2) 上位 3 パートはテンポのわずかな違いも探す
    for r in results[:3]:
        for st in stretches:
            sc, off = _best_offset(vc, r[1]["notes"], st, coarse, hop_c)
            if sc / voiced_c > r[0]:
                r[0], r[2], r[3] = sc / voiced_c, off, float(st)
    results.sort(key=lambda r: -r[0])
    score, c, off, st = results[0]

    # 3) 10ms 単位で微調整
    voiced = max(1, np.count_nonzero(~np.isnan(vocal)))
    fine = np.round(np.arange(off - 0.12, off + 0.1201, HOP), 3)
    sc, off = _best_offset(vocal, c["notes"], st, fine, HOP)
    score = sc / voiced

    # オクターブ: 歌声と MIDI の高さの差 (12 の倍数) の多数決
    roll = notes_to_roll(c["notes"], len(vocal), off, st)
    both = ~np.isnan(vocal) & ~np.isnan(roll)
    octave = int(np.median(np.round((vocal[both] - roll[both]) / 12))) if both.any() else 0
    return {
        "candidate": c, "offset": float(off), "stretch": float(st), "octave": octave, "score": float(score),
        "ranking": [(r[1]["id"], r[1]["name"], round(float(r[0]), 3)) for r in results[:6]],
    }


def guide_notes_from_midi(cand, offset, stretch, octave):
    """候補のメロディ → お手本バー (audio_processor と同じ形式)。"""
    out = []
    for s, e, n in cand["notes"]:
        m = n + 12 * octave
        st, en = s * stretch + offset, e * stretch + offset
        if en <= 0:
            continue
        out.append({
            "start": round(max(0.0, st), 3), "end": round(en, 3), "midi": int(m), "midi_f": float(m),
            "doremi": midi_to_doremi(m), "name": midi_to_english(m),
        })
    return out


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:10]


def build_guide(midi_path, vocal_data, cache_dir=None, track=None, offset=None, log=print):
    """MIDI → 歌声に合わせたお手本データ。結果は cache_dir/midi_<hash>.json に保存・再利用。

    track: "トラック:チャンネル" (例 "2:0") で手動指定 / offset: 秒で手動指定
    """
    cache_path = os.path.join(cache_dir, f"midi_{file_hash(midi_path)}.json") if cache_dir else None
    saved = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            saved = json.load(f)

    midi = parse_midi(midi_path)
    cands = melody_candidates(midi)
    if not cands:
        raise ValueError("MIDI にメロディになりそうなパートがありません")

    want = track or saved.get("track")
    if want:
        chosen = [c for c in cands if c["id"] == want]
        if not chosen:
            raise ValueError(f"指定のパート {want} がありません。候補: " + ", ".join(c["id"] for c in cands))
        cands_for_align = chosen
    else:
        cands_for_align = cands

    if saved and saved.get("track") in [c["id"] for c in cands_for_align] and offset is None and not track:
        c = next(c for c in cands if c["id"] == saved["track"])
        info = {"candidate": c, "offset": saved["offset"], "stretch": saved["stretch"], "octave": saved["octave"],
                "score": saved.get("score", 0), "ranking": saved.get("ranking", [])}
        log(f"  MIDI: 前回の合わせ方を再利用します (パート {c['id']} {c['name']})")
    else:
        log(f"  MIDI: {len(cands)} パートの中から、歌声に一番合うパートとタイミングを探しています…")
        # 和音パートや極端に低いパートは後回し (計算を軽くする)
        order = sorted(cands_for_align, key=lambda c: (c["polyphony"] > 1.6, c["high"] < 55, -c["count"]))
        info = align(order, vocal_data["pitch"])
        if offset is not None:
            info["offset"] = float(offset)
        c = info["candidate"]
        log(f"  MIDI: パート {c['id']}「{c['name']}」を採用 (一致率 {info['score']:.0%})、"
            f"ずれ {info['offset']:+.2f}秒、テンポ ×{info['stretch']:.3f}、オクターブ {info['octave']:+d}")

    notes = guide_notes_from_midi(info["candidate"], info["offset"], info["stretch"], info["octave"])
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"track": info["candidate"]["id"], "offset": info["offset"], "stretch": info["stretch"],
                       "octave": info["octave"], "score": info["score"], "ranking": info["ranking"],
                       "midi": os.path.basename(midi_path)}, f, ensure_ascii=False, indent=1)

    mids = [n["midi"] for n in notes] or [60, 72]
    return {
        "notes": notes,
        "range": {"min_midi": int(min(mids)), "max_midi": int(max(mids))},
        "info": {"track": info["candidate"]["id"], "name": info["candidate"]["name"], "offset": info["offset"],
                 "stretch": info["stretch"], "octave": info["octave"], "score": info["score"],
                 "ranking": info["ranking"], "cache_path": cache_path},
    }


def save_offset(cache_path, offset):
    """画面で微調整したずれを保存 (次回から使う)。"""
    if not cache_path or not os.path.exists(cache_path):
        return
    with open(cache_path, encoding="utf-8") as f:
        d = json.load(f)
    d["offset"] = round(float(offset), 3)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def find_midi_for(mp3_path):
    """曲と同じ名前の .mid / .midi が隣にあれば返す。"""
    stem = os.path.splitext(mp3_path)[0]
    for ext in (".mid", ".midi", ".MID", ".MIDI", ".kar"):
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    midi = parse_midi(sys.argv[1])
    cands = melody_candidates(midi)
    print(f"MIDI: {os.path.basename(sys.argv[1])} / パート {len(cands)} 個")
    for c in cands:
        print(f"  [{c['id']:>5}] {c['name'][:24]:<24} 音数 {c['count']:4d}  音域 {midi_to_doremi(c['low'])}〜{midi_to_doremi(c['high'])}"
              f"  和音度 {c['polyphony']:.1f}")
    if len(sys.argv) >= 3:
        import audio_processor

        res = audio_processor.process(sys.argv[2])
        g = build_guide(sys.argv[1], res["data"], res["dir"])
        print("\n一致率の高いパート:")
        for pid, name, sc in g["info"]["ranking"]:
            print(f"  [{pid:>5}] {name[:24]:<24} {sc:.0%}")
        print(f"\n最初のバー: " + "  ".join(f"{n['start']:.2f}秒 {n['doremi']}" for n in g["notes"][:6]))


if __name__ == "__main__":
    main()
