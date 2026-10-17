"""Player names for track pieces, typed while tagging passes (notebooks/tag_passes_v2.ipynb).

A piece is one joined player (`joined_id`, tools/join_pieces.py) when the run has it, else one tracker
id (`raw_tracker_id`; `track_id` for older runs). On the long hand-unified
windows pieces are 87-93% one player, but no automatic join between pieces was reliable (even joins
over a 0.5 s gap were 83% right at best: the tracker loses players where they cross), so pieces are
joined only by name: every piece named "A10" is the same player.

A name is stored as (piece, from_frame, name) and holds from that frame until the piece's next
entry; the piece's first entry also covers the frames before it. So a wrong name - the tracker
swapped two players inside the piece - is fixed by renaming at the frame where it is wrong, without
touching the earlier part.

Names: a bare shirt number gets its team letter (team_id 0 -> "A", 1 -> "B", unknown -> "?"), so
"10" on a team-1 player is "B10"; anything else typed (e.g. "B7", "GK", "REF") is kept as typed,
upper-case.
"""
from __future__ import annotations

import os

import pandas as pd

PEOPLE = (1, 2)  # goalkeeper, player
TEAM_LETTER = {0: "A", 1: "B"}
COLUMNS = ["piece", "from_frame", "name"]


def piece_column(tracks: pd.DataFrame) -> str:
    if "joined_id" in tracks.columns and (tracks["joined_id"] > 0).any():
        return "joined_id"
    if "raw_tracker_id" in tracks.columns and (tracks["raw_tracker_id"] >= 0).any():
        return "raw_tracker_id"
    return "track_id"


def label(text: str, team_id) -> str | None:
    """What the tagger typed -> stored name ("" -> None)."""
    t = str(text or "").strip().upper().replace(" ", "")
    if not t:
        return None
    if t.isdigit():
        team = int(team_id) if team_id is not None and pd.notna(team_id) else -1
        return f"{TEAM_LETTER.get(team, '?')}{int(t)}"
    return t


class NameBook:
    def __init__(self, path: str | None = None):
        self.path = path
        self.entries: dict[int, list[tuple[int, str]]] = {}
        if path and os.path.exists(path):
            for r in pd.read_csv(path).itertuples():
                self.entries.setdefault(int(r.piece), []).append((int(r.from_frame), str(r.name)))
            for v in self.entries.values():
                v.sort()

    def name_at(self, piece, frame: int) -> str | None:
        e = self.entries.get(int(piece)) if piece is not None and pd.notna(piece) else None
        if not e:
            return None
        name = e[0][1]
        for f, n in e:
            if f <= frame:
                name = n
        return name

    def set(self, piece: int, frame: int, name: str | None) -> None:
        """Name the piece from `frame` on. Re-entering the name it already has there changes nothing."""
        if name is None or self.name_at(piece, frame) == name:
            return
        e = [x for x in self.entries.get(int(piece), []) if x[0] != int(frame)]
        e.append((int(frame), name))
        self.entries[int(piece)] = sorted(e)

    def save(self, path: str | None = None) -> None:
        path = path or self.path
        rows = [(p, f, n) for p, v in self.entries.items() for f, n in v]
        pd.DataFrame(rows, columns=COLUMNS).sort_values(["piece", "from_frame"]).to_csv(path, index=False)

    def __len__(self) -> int:
        return len(self.entries)


def apply_names(tracks: pd.DataFrame, book: NameBook) -> pd.DataFrame:
    """Adds `player_name` (people rows of named pieces; empty otherwise)."""
    out = tracks.copy()
    out["player_name"] = pd.NA
    col = piece_column(out)
    people = out["class_id"].isin(PEOPLE)
    for piece, entries in book.entries.items():
        m = people & (out[col] == piece)
        if not m.any():
            continue
        frames = out.loc[m, "frame"]
        names = pd.Series(entries[0][1], index=frames.index, dtype=object)
        for f, n in entries[1:]:
            names[frames >= f] = n
        out.loc[m, "player_name"] = names
    return out


def coverage(tracks: pd.DataFrame, book: NameBook) -> dict:
    """Share of player rows that carry a name, and how many distinct names."""
    named = apply_names(tracks, book)
    p = named[named["class_id"].isin(PEOPLE)]
    return {"named_rows": float(p["player_name"].notna().mean()) if len(p) else 0.0,
            "names": int(p["player_name"].nunique()), "pieces_named": len(book)}
