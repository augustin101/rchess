#!/usr/bin/env python3
"""
match.py — rchess NNUE vs static eval match runner.

Compiles rchess-uci (with embedded NNUE), spawns two variants, plays a series
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

import chess
import chess.pgn

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


# ── Board rendering ───────────────────────────────────────────────────────────

def render_board(board: chess.Board) -> list[str]:
    out = [f'  {DIM}a b c d e f g h{R}']
    for rank in range(7, -1, -1):
        row = f'{DIM}{rank + 1}{R} '
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                row += f'{DIM}.{R} '
            elif piece.color == chess.WHITE:
                row += f'{B}{WHT}{piece.symbol().upper()}{R} '
            else:
                row += f'{YLW}{piece.symbol().lower()}{R} '
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
        """Returns (bestmove_uci, last info line containing score)."""
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


# ── Display ───────────────────────────────────────────────────────────────────

SEP = '─' * 46

def render(
    game_no: int, total: int,
    ew: Engine, eb: Engine,
    sw: float, sb: float,
    board: chess.Board,
    san_moves: list[str],
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
    lines += ['  ' + l for l in render_board(board)]
    lines += ['']

    to_move_engine = ew if board.turn == chess.WHITE else eb
    to_move_side   = 'White' if board.turn == chess.WHITE else 'Black'

    lines.append(f'  Clocks   {B}W{R} {fmt_ms(wtime)}   {B}B{R} {fmt_ms(btime)}')

    if thinking_s > 0:
        lines.append(f'  {DIM}Thinking ({to_move_side} / {to_move_engine.name}) … {thinking_s:.1f}s{R}')
    elif last_mv:
        moved_side = 'Black' if board.turn == chess.WHITE else 'White'
        sc = f'  {DIM}{last_score}{R}' if last_score else ''
        lines.append(f'  Last     {B}{last_mv}{R}  ({moved_side}){sc}')

    if san_moves:
        # Format as "1. e4 e5 2. Nf3 …" for the last few moves
        start = max(0, len(san_moves) - 10)
        first_move_no = start // 2 + 1
        tail_parts: list[str] = []
        for i, mv in enumerate(san_moves[start:], start=start):
            if i % 2 == 0:
                tail_parts.append(f'{i // 2 + 1}.')
            tail_parts.append(mv)
        lines.append(f'  Moves    {DIM}{" ".join(tail_parts)}{R}')

    lines.append(SEP)
    lines.append(f'  {B}{GRN}{status}{R}' if status else '')
    return lines


# ── Build ─────────────────────────────────────────────────────────────────────

def build(root: Path) -> Path:
    try:
        cfg = (root / 'engine.toml').read_text()
        nnue_line = next((l for l in cfg.splitlines() if l.strip().startswith('nnue')), '')
        nnue_file = nnue_line.split('=', 1)[-1].strip().strip('"\'') if nnue_line else '?'
    except Exception:
        nnue_file = '?'

    print(f'{CYN}Building rchess-uci --release --features embed-nnue{R}')
    print(f'{DIM}  embedding: {nnue_file}{R}')
    r = subprocess.run(
        ['cargo', 'build', '--release', '--bin', 'rchess-uci', '--features', 'embed-nnue'],
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
    """Returns (result, reason, uci_move_list)."""
    ew.new_game()
    eb.new_game()

    board     = chess.Board()
    moves:     list[str] = []   # UCI — sent to the engines via position command
    san_moves: list[str] = []   # SAN — used for display and PGN
    wtime_ms  = BASE_MS
    btime_ms  = BASE_MS
    last_mv   = ''
    last_sc   = ''

    for uci in opening:
        move = chess.Move.from_uci(uci)
        san_moves.append(board.san(move))
        board.push(move)
        moves.append(uci)

    def show(status: str = '', thinking: float = 0.0):
        repaint(render(
            game_no, total, ew, eb, sw, sb,
            board, san_moves, wtime_ms, btime_ms,
            last_mv, last_sc, status, thinking,
        ))

    show()

    while True:
        # ── Draw / adjudication ──────────────────────────────────────────────
        if len(moves) >= 400:
            show('Draw — move limit')
            time.sleep(1.5)
            return '1/2-1/2', 'move limit', moves

        if board.is_fifty_moves():
            show('Draw — 50-move rule')
            time.sleep(1.5)
            return '1/2-1/2', '50-move rule', moves

        if board.is_repetition(3):
            show('Draw — threefold repetition')
            time.sleep(1.5)
            return '1/2-1/2', 'threefold repetition', moves

        if board.is_insufficient_material():
            show('Draw — insufficient material')
            time.sleep(1.5)
            return '1/2-1/2', 'insufficient material', moves

        # ── Ask the engine ───────────────────────────────────────────────────
        is_white = board.turn == chess.WHITE
        engine   = ew if is_white else eb

        pos_cmd = 'position startpos' + (f' moves {" ".join(moves)}' if moves else '')
        go_cmd  = f'go wtime {wtime_ms} btime {btime_ms} winc {INC_MS} binc {INC_MS}'

        t0        = time.monotonic()
        _thinking = {'on': True}

        def _timer():
            while _thinking['on']:
                show(thinking=time.monotonic() - t0)
                time.sleep(0.4)

        timer_thr = threading.Thread(target=_timer, daemon=True)
        timer_thr.start()

        uci_mv, info = engine.go(pos_cmd, go_cmd, timeout=max(wtime_ms, btime_ms) / 1000 + 60)

        _thinking['on'] = False
        timer_thr.join(timeout=1.0)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if is_white:
            wtime_ms = max(0, wtime_ms - elapsed_ms + INC_MS)
        else:
            btime_ms = max(0, btime_ms - elapsed_ms + INC_MS)

        # ── Handle terminal responses ─────────────────────────────────────────
        if uci_mv in ('0000', '(none)', 'none', ''):
            if board.is_checkmate():
                result = '0-1' if is_white else '1-0'
                reason = 'checkmate'
            else:
                result, reason = '1/2-1/2', 'stalemate'
            show(f'{reason.capitalize()} — {result}')
            time.sleep(2.0)
            return result, reason, moves

        if (is_white and wtime_ms <= 0) or (not is_white and btime_ms <= 0):
            result = '0-1' if is_white else '1-0'
            show(f'Time forfeit — {result}')
            time.sleep(2.0)
            return result, 'time forfeit', moves

        move = chess.Move.from_uci(uci_mv)
        last_mv = board.san(move)     # compute SAN before pushing
        last_sc = parse_score(info)
        board.push(move)
        moves.append(uci_mv)
        san_moves.append(last_mv)
        show()
        time.sleep(0.08)


# ── PGN / JSON ────────────────────────────────────────────────────────────────

def write_pgn(
    path: Path, *,
    game_no: int, event: str, date: str,
    white: str, black: str,
    result: str, reason: str, moves: list[str],
):
    game = chess.pgn.Game()
    game.headers.update({
        'Event': event, 'Site': 'localhost', 'Date': date,
        'Round': str(game_no), 'White': white, 'Black': black,
        'Result': result, 'TimeControl': '60+1',
    })
    node = game
    for uci in moves:
        node = node.add_variation(chess.Move.from_uci(uci))
    node.comment = reason
    with open(path, 'a') as f:
        print(game, file=f)
        print(file=f)


# ── Match runner ──────────────────────────────────────────────────────────────

def run_match(args: argparse.Namespace):
    root   = Path(__file__).resolve().parent.parent
    binary = build(root)

    try:
        git_hash = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=root, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        git_hash = 'unknown'

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    pgn_path  = out_dir / f'match_{ts}.pgn'
    json_path = out_dir / f'match_{ts}.json'
    date_str  = datetime.now().strftime('%Y.%m.%d')
    event     = f'rchess NNUE vs Static ({git_hash})'

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
    game_log: list[dict] = []
    used_openings: list[list[str]] = []

    _enter_alt()
    try:
        for game_no in range(1, args.games + 1):
            nnue_white = (game_no % 2 == 1)
            ew, eb = (eng_nnue, eng_static) if nnue_white else (eng_static, eng_nnue)
            sw = score_nnue if nnue_white else score_static
            sb = score_static if nnue_white else score_nnue

            candidates = [o for o in OPENINGS if o not in used_openings[-4:]]
            opening = random.choice(candidates or OPENINGS)
            used_openings.append(opening)

            result, reason, all_moves = play_game(
                ew, eb, game_no, args.games, sw, sb, opening,
            )

            nnue_pts     = (1.0 if result == '1-0' else 0.0 if result == '0-1' else 0.5) if nnue_white \
                      else (0.0 if result == '1-0' else 1.0 if result == '0-1' else 0.5)
            score_nnue   += nnue_pts
            score_static += 1.0 - nnue_pts

            write_pgn(
                pgn_path,
                game_no=game_no, event=event, date=date_str,
                white=ew.name, black=eb.name,
                result=result, reason=reason, moves=all_moves,
            )
            game_log.append({
                'game': game_no, 'white': ew.name, 'black': eb.name,
                'nnue_white': nnue_white, 'result': result,
                'reason': reason, 'opening': opening, 'n_moves': len(all_moves),
            })
    finally:
        _leave_alt()

    eng_nnue.quit()
    eng_static.quit()

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
        print(f"  Game {g['game']:>2}  {g['white']:<32} vs {g['black']:<32}  "
              f"{B}{sym}{R}  {g['n_moves']} moves  ({g['reason']})")

    json_path.write_text(json.dumps({
        'event': event, 'git_hash': git_hash, 'date': date_str,
        'engine_nnue': name_nnue, 'engine_static': name_static,
        'n_games': args.games,
        'score_nnue': score_nnue, 'score_static': score_static,
        'nnue_wins': wins, 'nnue_losses': losses, 'draws': draws,
        'games': game_log,
    }, indent=2))

    print(f'\n  {B}PGN{R}  → {pgn_path}')
    print(f'  {B}JSON{R} → {json_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='rchess NNUE vs static eval match runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--games', type=int, default=10,  help='Number of games to play')
    p.add_argument('--out',   default='matches',      help='Output directory for PGN and JSON')
    run_match(p.parse_args())
