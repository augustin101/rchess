#!/usr/bin/env python3
"""
match.py — rchess NNUE vs static eval match runner.

Compiles rchess-uci, spawns two variants (NNUE / static eval), plays a series
of games with a live board display, and records all games to PGN + JSON.

Usage (from project root):
    python training/match.py
    python training/match.py --games 20
    python training/match.py --out matches/run1
"""

import argparse
import json
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ── ANSI helpers ──────────────────────────────────────────────────────────────

R   = '\033[0m'
B   = '\033[1m'
DIM = '\033[2m'
GRN = '\033[32m'
YLW = '\033[33m'
CYN = '\033[36m'
RED = '\033[31m'
WHT = '\033[97m'

_alt_screen = False

def _enter_alt():
    global _alt_screen
    sys.stdout.write('\033[?1049h\033[H')
    sys.stdout.flush()
    _alt_screen = True

def _leave_alt():
    global _alt_screen
    if _alt_screen:
        sys.stdout.write('\033[?1049l')
        sys.stdout.flush()
        _alt_screen = False

def _sigint(sig, frame):
    _leave_alt()
    print('\nInterrupted.')
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint)

_display_lock = threading.Lock()

def repaint(lines: list[str]):
    with _display_lock:
        sys.stdout.write('\033[H')
        sys.stdout.write('\n'.join(lines))
        sys.stdout.write('\033[J')
        sys.stdout.flush()


# ── Board ─────────────────────────────────────────────────────────────────────

class Board:
    """Minimal board tracker — enough for display, repetition and 50-move rule."""

    def __init__(self, fen: str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        self.sq = ['.'] * 64   # index = rank*8 + file  (rank 0 = rank 1)
        self.stm = 'w'
        self.hmclock = 0
        self._history: list[str] = []
        self._parse(fen)
        self._history.append(self._key())

    def _parse(self, fen: str):
        parts = fen.split()
        rank, file = 7, 0
        for ch in parts[0]:
            if ch == '/':
                rank -= 1; file = 0
            elif ch.isdigit():
                file += int(ch)
            else:
                self.sq[rank * 8 + file] = ch; file += 1
        self.stm    = parts[1] if len(parts) > 1 else 'w'
        self.hmclock = int(parts[4]) if len(parts) > 4 else 0

    def _key(self) -> str:
        return ''.join(self.sq) + self.stm

    def apply(self, move: str):
        fc = ord(move[0]) - ord('a')
        fr = int(move[1]) - 1
        tc = ord(move[2]) - ord('a')
        tr = int(move[3]) - 1
        promo   = move[4].upper() if len(move) > 4 else None
        from_sq = fr * 8 + fc
        to_sq   = tr * 8 + tc
        piece   = self.sq[from_sq]
        cap     = self.sq[to_sq]

        self.hmclock = 0 if (piece.lower() == 'p' or cap != '.') else self.hmclock + 1

        # En passant
        if piece.lower() == 'p' and fc != tc and cap == '.':
            self.sq[fr * 8 + tc] = '.'

        placed = (promo if piece.isupper() else promo.lower()) if promo else piece
        self.sq[to_sq]   = placed
        self.sq[from_sq] = '.'

        # Castling: move rook
        if piece.lower() == 'k' and abs(fc - tc) == 2:
            if tc > fc:
                self.sq[fr * 8 + 5] = self.sq[fr * 8 + 7]; self.sq[fr * 8 + 7] = '.'
            else:
                self.sq[fr * 8 + 3] = self.sq[fr * 8 + 0]; self.sq[fr * 8 + 0] = '.'

        self.stm = 'b' if self.stm == 'w' else 'w'
        self._history.append(self._key())

    def draw_repetition(self) -> bool:
        return self._history.count(self._key()) >= 3

    def draw_50(self) -> bool:
        return self.hmclock >= 100

    def render(self) -> list[str]:
        out = [f'  {DIM}a b c d e f g h{R}']
        for rank in range(7, -1, -1):
            row = f'{DIM}{rank + 1}{R} '
            for file in range(8):
                p = self.sq[rank * 8 + file]
                if p == '.':
                    row += f'{DIM}.{R} '
                elif p.isupper():
                    row += f'{B}{WHT}{p}{R} '
                else:
                    row += f'{YLW}{p}{R} '
            out.append(row)
        return out


# ── UCI engine wrapper ────────────────────────────────────────────────────────

class Engine:
    def __init__(self, binary: str, extra_args: list[str], label: str):
        self.label = label
        self.name  = label
        self._p = subprocess.Popen(
            [binary] + extra_args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self._p.stdout:
            self._q.put(line.rstrip('\n'))

    def send(self, cmd: str):
        self._p.stdin.write(cmd + '\n')
        self._p.stdin.flush()

    def _collect(self, until: str, timeout: float) -> list[str]:
        lines, deadline = [], time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self._q.get(timeout=min(0.05, deadline - time.monotonic()))
                lines.append(line)
                if line.startswith(until):
                    return lines
            except queue.Empty:
                pass
        return lines

    def init(self) -> str:
        self.send('uci')
        for l in self._collect('uciok', 5.0):
            if l.startswith('id name'):
                self.name = l[len('id name'):].strip()
        self.send('isready')
        self._collect('readyok', 5.0)
        return self.name

    def new_game(self):
        self.send('ucinewgame')
        self.send('isready')
        self._collect('readyok', 5.0)

    def go(self, pos: str, go_cmd: str, timeout: float) -> tuple[str, str]:
        """Returns (bestmove, last info line with score)."""
        self.send(pos)
        self.send(go_cmd)
        last_info = ''
        for l in self._collect('bestmove', timeout):
            if l.startswith('info') and 'score' in l:
                last_info = l
            if l.startswith('bestmove'):
                parts = l.split()
                return (parts[1] if len(parts) > 1 else '0000'), last_info
        return '0000', last_info

    def quit(self):
        try:
            self.send('quit')
            self._p.wait(timeout=3.0)
        except Exception:
            self._p.kill()


# ── Openings (for variation) ─────────────────────────────────────────────────

OPENINGS: list[list[str]] = [
    # --- Open games ---
    ['e2e4', 'e7e5'],
    ['e2e4', 'e7e5', 'g1f3', 'b8c6'],
    ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4'],           # Italian
    ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5'],            # Ruy Lopez
    ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6'],   # Morphy Defence
    ['e2e4', 'e7e5', 'f2f4'],                             # King's Gambit
    # --- Semi-open ---
    ['e2e4', 'c7c5'],                                     # Sicilian
    ['e2e4', 'c7c5', 'g1f3', 'd7d6'],
    ['e2e4', 'c7c5', 'g1f3', 'b8c6'],
    ['e2e4', 'e7e6'],                                     # French
    ['e2e4', 'e7e6', 'd2d4', 'd7d5'],
    ['e2e4', 'e7e6', 'd2d4', 'd7d5', 'b1c3'],
    ['e2e4', 'c7c6'],                                     # Caro-Kann
    ['e2e4', 'd7d5'],                                     # Scandinavian
    # --- Closed / 1.d4 ---
    ['d2d4', 'd7d5'],
    ['d2d4', 'd7d5', 'c2c4'],                             # Queen's Gambit
    ['d2d4', 'd7d5', 'c2c4', 'e7e6'],                    # QGD
    ['d2d4', 'd7d5', 'c2c4', 'c7c6'],                    # Slav
    ['d2d4', 'g8f6'],
    ['d2d4', 'g8f6', 'c2c4', 'g7g6'],                    # King's Indian
    ['d2d4', 'g8f6', 'c2c4', 'e7e6'],                    # Nimzo / QID setup
    ['d2d4', 'g8f6', 'c2c4', 'e7e6', 'b1c3', 'f8b4'],   # Nimzo-Indian
    # --- Flank / others ---
    ['c2c4'],                                             # English
    ['c2c4', 'e7e5'],
    ['c2c4', 'c7c5'],
    ['g1f3', 'd7d5'],
    ['g1f3', 'g8f6'],
    ['g1f3', 'g8f6', 'c2c4'],
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_ms(ms: int) -> str:
    ms = max(ms, 0)
    mins, secs = divmod(ms // 1000, 60)
    return f'{mins}:{secs:02d}.{(ms % 1000) // 100}'

def parse_score(info: str) -> str:
    parts = info.split()
    for i, tok in enumerate(parts):
        if tok == 'cp'   and i + 1 < len(parts): return f'{int(parts[i+1]):+d} cp'
        if tok == 'mate' and i + 1 < len(parts): return f'mate {parts[i+1]}'
    return ''

# ── SAN conversion ───────────────────────────────────────────────────────────
#
# Converts UCI moves to Standard Algebraic Notation by replaying the position.
# Handles: piece disambiguation, captures, castling, promotions, check (+/#).
# Limitations: en-passant and castling are not generated in the legal-move
# enumerator used for checkmate detection (# vs +), so very rare positions may
# get '+' instead of '#'. All other SAN fields are correct.

def _piece_attacks(sq_list: list, from_sq: int, piece: str) -> list[int]:
    """Squares attacked by `piece` (e.g. 'N', 'n') sitting on from_sq."""
    f, r  = from_sq % 8, from_sq // 8
    pu    = piece.upper()
    hits: list[int] = []

    def _slide(dirs):
        for df, dr in dirs:
            cf, cr = f + df, r + dr
            while 0 <= cf < 8 and 0 <= cr < 8:
                s = cr * 8 + cf
                hits.append(s)
                if sq_list[s] != '.':
                    break
                cf += df; cr += dr

    if pu == 'N':
        for df, dr in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
            nf, nr = f+df, r+dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                hits.append(nr*8+nf)
    if pu in ('B', 'Q'):
        _slide([(1,1),(1,-1),(-1,1),(-1,-1)])
    if pu in ('R', 'Q'):
        _slide([(1,0),(-1,0),(0,1),(0,-1)])
    if pu == 'K':
        for df, dr in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nf, nr = f+df, r+dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                hits.append(nr*8+nf)
    if pu == 'P':
        dr = 1 if piece.isupper() else -1
        for df in (-1, 1):
            nf, nr = f+df, r+dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                hits.append(nr*8+nf)
    return hits


def _is_attacked(sq_list: list, target: int, by_white: bool) -> bool:
    """Is `target` square attacked by any piece belonging to `by_white`?"""
    for s in range(64):
        p = sq_list[s]
        if p == '.' or p.isupper() != by_white:
            continue
        if target in _piece_attacks(sq_list, s, p):
            return True
    return False


def _apply_sq(sq_list: list, uci: str) -> list:
    """Return a NEW sq_list with the UCI move applied (non-destructive)."""
    sl = list(sq_list)
    fc = ord(uci[0])-ord('a'); fr = int(uci[1])-1
    tc = ord(uci[2])-ord('a'); tr = int(uci[3])-1
    promo = uci[4].upper() if len(uci) > 4 else None
    fs = fr*8+fc; ts = tr*8+tc
    piece = sl[fs]; cap = sl[ts]
    # en passant
    if piece.upper() == 'P' and fc != tc and cap == '.':
        sl[fr*8+tc] = '.'
    placed = (promo if piece.isupper() else promo.lower()) if promo else piece
    sl[ts] = placed; sl[fs] = '.'
    # castling: move rook
    if piece.upper() == 'K' and abs(fc-tc) == 2:
        if tc > fc:
            sl[fr*8+5] = sl[fr*8+7]; sl[fr*8+7] = '.'
        else:
            sl[fr*8+3] = sl[fr*8+0]; sl[fr*8+0] = '.'
    return sl


def _king_sq(sq_list: list, white: bool) -> int:
    k = 'K' if white else 'k'
    for s in range(64):
        if sq_list[s] == k:
            return s
    return -1


def _in_check(sq_list: list, white: bool) -> bool:
    ks = _king_sq(sq_list, white)
    return ks >= 0 and _is_attacked(sq_list, ks, not white)


def _pseudo_moves(sq_list: list, white: bool) -> list[str]:
    """Pseudo-legal moves for `white`'s pieces (enough for checkmate detection)."""
    moves: list[str] = []
    for fs in range(64):
        p = sq_list[fs]
        if p == '.' or p.isupper() != white:
            continue
        f, r = fs % 8, fs // 8
        pu = p.upper()
        fa = chr(ord('a')+f); ra = str(r+1)

        if pu == 'P':
            dr = 1 if white else -1
            # single push
            nr = r + dr
            if 0 <= nr < 8:
                ts = nr*8+f
                if sq_list[ts] == '.':
                    nra = str(nr+1)
                    if nr in (0, 7):  # promotion rank
                        for pp in ('q','r','b','n'):
                            moves.append(f'{fa}{ra}{fa}{nra}{pp}')
                    else:
                        moves.append(f'{fa}{ra}{fa}{nra}')
                    # double push from home rank
                    if (white and r == 1) or (not white and r == 6):
                        ts2 = (nr+dr)*8+f
                        if sq_list[ts2] == '.':
                            moves.append(f'{fa}{ra}{fa}{nr+dr+1}')
            # pawn captures
            for df in (-1, 1):
                nf, nrr = f+df, r+dr
                if 0 <= nf < 8 and 0 <= nrr < 8:
                    ts = nrr*8+nf
                    cap = sq_list[ts]
                    if cap != '.' and cap.isupper() != white:
                        nra = str(nrr+1); nfa = chr(ord('a')+nf)
                        if nrr in (0, 7):
                            for pp in ('q','r','b','n'):
                                moves.append(f'{fa}{ra}{nfa}{nra}{pp}')
                        else:
                            moves.append(f'{fa}{ra}{nfa}{nra}')
        else:
            for ts in _piece_attacks(sq_list, fs, p):
                cap = sq_list[ts]
                if cap == '.' or cap.isupper() != white:
                    tf = chr(ord('a')+ts%8); tr2 = str(ts//8+1)
                    moves.append(f'{fa}{ra}{tf}{tr2}')
    return moves


def _has_legal_move(sq_list: list, white: bool) -> bool:
    """Does `white` have at least one legal move (king not left in check)?"""
    for uci in _pseudo_moves(sq_list, white):
        if not _in_check(_apply_sq(sq_list, uci), white):
            return True
    return False


def san_from_uci(board: 'Board', uci: str) -> str:
    """Convert a UCI move string to SAN given the *current* board position."""
    fc = ord(uci[0])-ord('a'); fr = int(uci[1])-1
    tc = ord(uci[2])-ord('a'); tr = int(uci[3])-1
    promo = uci[4].upper() if len(uci) > 4 else None
    fs = fr*8+fc; ts = tr*8+tc
    piece    = board.sq[fs]
    pu       = piece.upper()
    is_white = piece.isupper()
    cap      = board.sq[ts]
    dest     = f'{chr(ord("a")+tc)}{tr+1}'

    # ── Castling ──────────────────────────────────────────────────────────────
    if pu == 'K' and abs(fc-tc) == 2:
        san  = 'O-O' if tc > fc else 'O-O-O'
        sl2  = _apply_sq(board.sq, uci)
        opp  = not is_white
        if _in_check(sl2, opp):
            san += '#' if not _has_legal_move(sl2, opp) else '+'
        return san

    is_ep  = pu == 'P' and fc != tc and cap == '.'
    is_cap = cap != '.' or is_ep

    # ── Pawn ──────────────────────────────────────────────────────────────────
    if pu == 'P':
        san = (f'{chr(ord("a")+fc)}x{dest}' if is_cap else dest)
        if promo:
            san += f'={promo}'

    # ── Piece (N/B/R/Q/K) ────────────────────────────────────────────────────
    else:
        # Disambiguation: find other pieces of same type that can also reach ts
        # legally (i.e., without leaving their own king in check).
        ambig: list[int] = []
        for s in range(64):
            if s == fs:
                continue
            p2 = board.sq[s]
            if p2 == '.' or p2.upper() != pu or p2.isupper() != is_white:
                continue
            if ts not in _piece_attacks(board.sq, s, p2):
                continue
            # Check legality of that alternative move
            uci2 = f'{chr(ord("a")+s%8)}{s//8+1}{dest}'
            if not _in_check(_apply_sq(board.sq, uci2), is_white):
                ambig.append(s)

        disambig = ''
        if ambig:
            other_files = {s % 8 for s in ambig}
            other_ranks = {s // 8 for s in ambig}
            if fc not in other_files:
                disambig = chr(ord('a')+fc)           # file sufficient
            elif fr not in other_ranks:
                disambig = str(fr+1)                   # rank sufficient
            else:
                disambig = f'{chr(ord("a")+fc)}{fr+1}' # full square

        san = f'{pu}{disambig}{"x" if is_cap else ""}{dest}'

    # ── Check / checkmate suffix ──────────────────────────────────────────────
    sl2 = _apply_sq(board.sq, uci)
    opp = not is_white
    if _in_check(sl2, opp):
        san += '#' if not _has_legal_move(sl2, opp) else '+'

    return san


def _moves_to_san_pgn(moves: list[str]) -> str:
    """Replay a game from the start and produce a PGN move-text in SAN."""
    board = Board()
    parts: list[str] = []
    for i, uci in enumerate(moves):
        if i % 2 == 0:
            parts.append(f'{i // 2 + 1}.')
        parts.append(san_from_uci(board, uci))
        board.apply(uci)
    return ' '.join(parts)


# ── Display ───────────────────────────────────────────────────────────────────

SEP = '─' * 46

def render(
    game_no: int, total: int,
    ew: Engine, eb: Engine,
    sw: float, sb: float,
    board: Board,
    moves: list[str],
    wtime: int, btime: int,
    last_mv: str, last_score: str,
    status: str, thinking_s: float,
) -> list[str]:
    lines = [
        f'{B}{CYN}  rchess match  —  game {game_no} / {total}{R}',
        SEP,
        f'  {B}White{R}  {ew.name:<32}  {sw:>4.1f} pts',
        f'  {B}Black{R}  {eb.name:<32}  {sb:>4.1f} pts',
        SEP,
        '',
    ]
    lines += ['  ' + l for l in board.render()]
    lines += ['']

    to_move_engine = ew if board.stm == 'w' else eb
    to_move_side   = 'White' if board.stm == 'w' else 'Black'

    lines.append(f'  Clocks   {B}W{R} {fmt_ms(wtime)}   {B}B{R} {fmt_ms(btime)}')

    if thinking_s > 0:
        lines.append(f'  {DIM}Thinking ({to_move_side} / {to_move_engine.name}) … {thinking_s:.1f}s{R}')
    elif last_mv:
        moved_side = 'Black' if board.stm == 'w' else 'White'
        sc = f'  {DIM}{last_score}{R}' if last_score else ''
        lines.append(f'  Last     {B}{last_mv}{R}  ({moved_side}){sc}')

    if moves:
        tail = ' '.join(moves[-10:])
        lines.append(f'  Moves    {DIM}{tail}{R}')

    lines.append(SEP)
    if status:
        lines.append(f'  {B}{GRN}{status}{R}')
    else:
        lines.append('')
    return lines


# ── Build ─────────────────────────────────────────────────────────────────────

def build(root: Path) -> Path:
    print(f'{CYN}Building rchess-uci --release …{R}')
    r = subprocess.run(
        ['cargo', 'build', '--release', '--bin', 'rchess-uci'],
        cwd=root,
    )
    if r.returncode != 0:
        print(f'{RED}Build failed.{R}')
        sys.exit(1)
    binary = root / 'target' / 'release' / 'rchess-uci'
    print(f'{GRN}OK → {binary}{R}')
    return binary


# ── Game ─────────────────────────────────────────────────────────────────────

INC_MS  = 1_000    # 1-second increment
BASE_MS = 60_000   # 1-minute base time

def play_game(
    ew: Engine, eb: Engine,
    game_no: int, total: int,
    sw: float, sb: float,
    opening: list[str],
) -> tuple[str, str, list[str]]:
    """
    Returns (result '1-0'|'0-1'|'1/2-1/2', reason, full_move_list).
    """
    ew.new_game()
    eb.new_game()

    board     = Board()
    moves:    list[str] = []
    wtime_ms  = BASE_MS
    btime_ms  = BASE_MS
    last_mv   = ''
    last_sc   = ''

    # Apply opening as the starting position (both engines see it via position cmd)
    for mv in opening:
        board.apply(mv)
        moves.append(mv)

    thinking_s = 0.0

    def show(status: str = '', thinking: float = 0.0):
        repaint(render(
            game_no, total, ew, eb, sw, sb,
            board, moves, wtime_ms, btime_ms,
            last_mv, last_sc, status, thinking,
        ))

    show()

    while True:
        # ── Draw / adjudication checks ──────────────────────────────────────
        if len(moves) >= 400:
            show('Draw — move limit (200 moves)')
            time.sleep(1.5)
            return '1/2-1/2', 'move limit', moves

        if board.draw_50():
            show('Draw — 50-move rule')
            time.sleep(1.5)
            return '1/2-1/2', '50-move rule', moves

        if board.draw_repetition():
            show('Draw — threefold repetition')
            time.sleep(1.5)
            return '1/2-1/2', 'threefold repetition', moves

        # ── Send position + go ───────────────────────────────────────────────
        is_white = board.stm == 'w'
        engine   = ew if is_white else eb

        pos_cmd = 'position startpos' + (f' moves {" ".join(moves)}' if moves else '')
        go_cmd  = f'go wtime {wtime_ms} btime {btime_ms} winc {INC_MS} binc {INC_MS}'

        # ── Live thinking timer (background thread) ──────────────────────────
        t0        = time.monotonic()
        _thinking = {'on': True}

        def _timer():
            while _thinking['on']:
                elapsed = time.monotonic() - t0
                show(thinking=elapsed)
                time.sleep(0.4)

        timer_thr = threading.Thread(target=_timer, daemon=True)
        timer_thr.start()

        mv, info = engine.go(pos_cmd, go_cmd, timeout=max(wtime_ms, btime_ms) / 1000 + 60)

        _thinking['on'] = False
        timer_thr.join(timeout=1.0)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # ── Update clocks ────────────────────────────────────────────────────
        if is_white:
            wtime_ms = max(0, wtime_ms - elapsed_ms + INC_MS)
        else:
            btime_ms = max(0, btime_ms - elapsed_ms + INC_MS)

        # ── Handle special responses ─────────────────────────────────────────
        if mv in ('0000', '(none)', 'none', ''):
            result = '0-1' if is_white else '1-0'
            show(f'No legal moves — {result}')
            time.sleep(2.0)
            return result, 'checkmate or stalemate', moves

        if (is_white and wtime_ms <= 0) or (not is_white and btime_ms <= 0):
            result = '0-1' if is_white else '1-0'
            show(f'Time forfeit — {result}')
            time.sleep(2.0)
            return result, 'time forfeit', moves

        last_mv = mv
        last_sc = parse_score(info)
        board.apply(mv)
        moves.append(mv)
        show()
        time.sleep(0.08)


# ── PGN / JSON ────────────────────────────────────────────────────────────────

def append_pgn(
    path: Path,
    event: str, round_no: int, date: str,
    white: str, black: str,
    result: str, reason: str, moves: list[str],
):
    res = {'1-0': '1-0', '0-1': '0-1', '1/2-1/2': '1/2-1/2'}.get(result, '*')
    with open(path, 'a') as f:
        f.write(
            f'[Event "{event}"]\n'
            f'[Site "localhost"]\n'
            f'[Date "{date}"]\n'
            f'[Round "{round_no}"]\n'
            f'[White "{white}"]\n'
            f'[Black "{black}"]\n'
            f'[Result "{res}"]\n'
            f'[TimeControl "60+1"]\n'
            f'\n'
            f'{_moves_to_san_pgn(moves)} {{{reason}}} {res}\n\n'
        )


# ── Match runner ──────────────────────────────────────────────────────────────

def run_match(args: argparse.Namespace):
    root   = Path(__file__).resolve().parent.parent
    binary = build(root)

    # Git hash for version tagging
    try:
        git_hash = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=root, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        git_hash = 'unknown'

    nnue_path = root / 'networks' / 'nnue.bin'
    if not nnue_path.exists():
        print(f'{RED}Warning: {nnue_path} not found — NNUE engine will fall back to static eval.{R}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    pgn_path  = out_dir / f'match_{ts}.pgn'
    json_path = out_dir / f'match_{ts}.json'
    date_str  = datetime.now().strftime('%Y.%m.%d')
    event     = f'rchess NNUE vs Static ({git_hash})'

    # Spawn engines
    eng_nnue   = Engine(str(binary), [],            'NNUE')
    eng_static = Engine(str(binary), ['--no-nnue'], 'Static')
    name_nnue   = eng_nnue.init()
    name_static = eng_static.init()

    print(f'\n{B}Engine A (NNUE):{R}   {name_nnue}')
    print(f'{B}Engine B (Static):{R} {name_static}')
    print(f'{B}Games:{R}             {args.games}')
    print(f'{B}Output:{R}            {out_dir}')
    print(f'{B}Git:{R}               {git_hash}')
    time.sleep(1.5)

    score_nnue   = 0.0
    score_static = 0.0
    game_log     = []
    used_openings: list[list[str]] = []

    _enter_alt()
    try:
        for game_no in range(1, args.games + 1):
            # Alternate colors; NNUE is White in odd-numbered games
            nnue_white = (game_no % 2 == 1)
            ew, eb = (eng_nnue, eng_static) if nnue_white else (eng_static, eng_nnue)
            sw = score_nnue if nnue_white else score_static
            sb = score_static if nnue_white else score_nnue

            # Pick an opening not used recently (avoid consecutive repeats)
            candidates = [o for o in OPENINGS if o not in used_openings[-4:]]
            if not candidates:
                candidates = OPENINGS
            opening = random.choice(candidates)
            used_openings.append(opening)

            result, reason, all_moves = play_game(
                ew, eb, game_no, args.games, sw, sb, opening,
            )

            # Translate result to NNUE's perspective
            if result == '1-0':
                nnue_pts = 1.0 if nnue_white else 0.0
            elif result == '0-1':
                nnue_pts = 0.0 if nnue_white else 1.0
            else:
                nnue_pts = 0.5

            score_nnue   += nnue_pts
            score_static += 1.0 - nnue_pts

            append_pgn(
                pgn_path,
                event=event, round_no=game_no, date=date_str,
                white=ew.name, black=eb.name,
                result=result, reason=reason, moves=all_moves,
            )

            game_log.append({
                'game':        game_no,
                'white':       ew.name,
                'black':       eb.name,
                'nnue_white':  nnue_white,
                'result':      result,
                'reason':      reason,
                'opening':     opening,
                'n_moves':     len(all_moves),
            })
    finally:
        _leave_alt()

    eng_nnue.quit()
    eng_static.quit()

    # ── Final summary ─────────────────────────────────────────────────────────
    wins   = sum(1 for g in game_log if
                 (g['nnue_white'] and g['result'] == '1-0') or
                 (not g['nnue_white'] and g['result'] == '0-1'))
    losses = sum(1 for g in game_log if
                 (g['nnue_white'] and g['result'] == '0-1') or
                 (not g['nnue_white'] and g['result'] == '1-0'))
    draws  = args.games - wins - losses

    print(f'\n{B}{CYN}══ Match complete ══{R}')
    print(f'  {name_nnue:<38}  {score_nnue:>5.1f} / {args.games}  ({wins}W {losses}L {draws}D)')
    print(f'  {name_static:<38}  {score_static:>5.1f} / {args.games}')
    print()
    for g in game_log:
        sym = {'1-0': '1-0', '0-1': '0-1', '1/2-1/2': '½-½'}.get(g['result'], '?')
        opening_str = ' '.join(g['opening'])
        print(f"  Game {g['game']:>2}  {g['white']:<32} vs {g['black']:<32}  {B}{sym}{R}  "
              f"{g['n_moves']} moves  ({g['reason']})  opening: {opening_str}")

    summary = {
        'event':          event,
        'git_hash':       git_hash,
        'date':           date_str,
        'engine_nnue':    name_nnue,
        'engine_static':  name_static,
        'nnue_binary':    str(nnue_path),
        'n_games':        args.games,
        'score_nnue':     score_nnue,
        'score_static':   score_static,
        'nnue_wins':      wins,
        'nnue_losses':    losses,
        'draws':          draws,
        'games':          game_log,
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print(f'\n  {B}PGN{R}  → {pgn_path}')
    print(f'  {B}JSON{R} → {json_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='rchess NNUE vs static eval match runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--games', type=int, default=10,   help='Number of games to play')
    p.add_argument('--out',   default='matches',       help='Output directory for PGN and JSON')
    run_match(p.parse_args())
