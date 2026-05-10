# Alpha-Beta Engine

The engine performs iterative-deepening negamax search with alpha-beta pruning and several standard heuristics.  Each optimisation below is described with its key assumption and the trade-off it accepts.

---

## Iterative Deepening

The engine searches depth 1, 2, … N before returning the best move at the requested depth.

**Assumption**: the best move at depth *d* is a good first move to try at depth *d+1*, improving pruning efficiency.

**Trade-off**: doing redundant shallower searches.  The overhead is small in practice because the tree grows exponentially and the deeper iteration dominates total work.

---

## Alpha-Beta Pruning

The core of the search.  `alpha` is the best score the maximising side can guarantee so far; `beta` is the best the minimising side can guarantee.  A branch is cut when `alpha >= beta` — no move in that subtree can affect the final result.

**Assumption**: correct — pruning is mathematically exact; no good moves are skipped.

**Trade-off**: none in correctness.  In the best case (perfect move ordering) pruning reduces the effective branching factor from *b* to *√b*.  In the worst case (reversed order) it provides no benefit.

---

## Transposition Table (TT)

A fixed-size (1 M entry, ≈ 24 MB) hash map indexed by the Zobrist hash of the position.  Each entry stores the depth, best move, score, and bound type (exact / lower / upper).

**Assumption**: the Zobrist hash is collision-free for all positions encountered in practice.  Collisions are possible but extremely rare.

**Trade-offs**:
- **Always-replace** strategy: a new entry overwrites an old one only if it is at the same position or at equal-or-greater depth.  This keeps deep results, but wastes space on stale shallow results for positions never re-visited.
- The table is not cleared between moves, so results persist across the iterative-deepening loop and between turns.  History in the table can occasionally cause subtle horizon artefacts, but the benefit to re-use vastly outweighs this.

---

## Move Ordering

The quality of alpha-beta pruning depends entirely on how quickly the best move is tried.  Moves are ordered (highest to lowest priority):

| Priority       | What                          | Score          |
|----------------|-------------------------------|----------------|
| 1 (highest)    | TT / hash move                | 30 000 000     |
| 2              | Promotions (by promoted piece)| 9 000 000+     |
| 3              | Captures — MVV-LVA            | 1 000 000+     |
| 4              | Killer moves                  | 900 000        |
| 5              | History heuristic             | 0 – 32 000     |

**MVV-LVA** (Most Valuable Victim — Least Valuable Aggressor): score = victim value × 10 − attacker value.  Prioritises winning captures (capturing a queen with a pawn) over losing ones (capturing a pawn with a queen).

**Assumption**: the TT move, captures, and killers tend to be the best moves.

**Trade-off**: none — ordering is done with an incremental selection sort (O(*n*) per selected move), which is cheaper than sorting all moves upfront for lists that are often cut short.

---

## Killer Moves

Two quiet moves per ply that caused a beta cut-off somewhere else in the tree at the same depth.  They are tried just below captures.

**Assumption**: a move that refuted a sibling position is likely to refute similar positions at the same tree depth.

**Trade-off**: killers are position-independent (they don't check legality in the new position).  Illegal killers are simply discarded at make-move time.

---

## History Heuristic

A 2 × 64 × 64 table accumulating `depth²` points each time a quiet move from square A to square B causes a beta cut-off.  Scores are halved ("aged") at the start of each turn.

**Assumption**: moves that have historically been strong are likely to continue to be strong.

**Trade-off**: the table is not position-specific, so it can misorder moves when the same from-to pair has different effects in different positions.  The ageing prevents stale data from dominating.

---

## Null Move Pruning

Before searching our moves, we try "passing" (making no move).  If the resulting score after the opponent replies still fails high (≥ beta), our position is probably too good for the opponent — we prune.

**Assumption**: having the initiative is worth at least something; if passing is still winning, we can prune.

**Trade-offs**:
- **Disabled in check** — the side to move must get out of check; a null move would be illegal.
- **Disabled in pawn/king endings** — zugzwang positions exist where passing is catastrophically bad; the `has_non_pawn_material` guard mitigates this.
- Reduction: `R = 3` at depth ≥ 6, `R = 2` otherwise.  Larger reductions are faster but can miss narrow refutations.

---

## Futility Pruning

At depth 1 and 2, compute the static evaluation.  If it is below alpha by more than a fixed margin (100 cp at depth 1, 300 cp at depth 2), skip quiet moves — they are unlikely to close the gap.

**Assumption**: a single quiet move changes the evaluation by at most the margin.

**Trade-off**: this is unsound — a quiet move can occasionally trigger a tactical change that dramatically improves the score.  The heuristic is disabled in check to avoid missing forced defensive resources.

---

## Late Move Reductions (LMR)

Quiet moves later in the ordered list (after the first 3 quiet moves have been tried) are searched at a reduced depth.  If the reduced-depth search beats alpha, a full-depth re-search is done.

**Reduction formula**: `R = 1 + quiet_index / 6`

Conditions for LMR:
- Move is quiet (no capture, no promotion)
- Move does not give check
- No extension applied to this move
- At least `LMR_MIN_QUIET = 3` quiet moves already tried
- Depth ≥ `LMR_MIN_DEPTH = 3`

**Assumption**: late moves in a well-ordered list are rarely the best; the reduced search identifies them cheaply, and the re-search is triggered rarely.

**Trade-off**: occasionally a late quiet move turns out to be the best move, requiring two searches.  The overall saving is large: LMR typically halves search time at depths ≥ 5.

---

## Check Extension

When a move delivers check, the search is extended by one extra ply (capped at `MAX_PLY - 1 = 63`).

**Assumption**: positions where the opponent is in check are tactically sharp; the horizon should not fall there.

**Trade-off**: extensions can stack with other extensions (e.g. promotions) in principle, though currently only check is extended.  Uncapped extension chains could make the search tree unbounded; the `MAX_PLY` cap prevents this.

---

## Quiescence Search

When depth reaches 0, the search continues with only captures (and all moves when in check) until a "quiet" position is reached, then returns the static evaluation.

**Assumption**: quiet positions are safe to evaluate statically; tactical sequences must be resolved first.

**Trade-offs**:
- **Stand-pat**: if the side to move's static score already beats beta without playing any capture, return immediately.  This assumes the side to move always has the *option* of playing a quiet move (i.e. they are not in check).
- Only captures and promotions are searched outside check.  Checks and quiet threats are not followed, so the quiescence search still has a horizon, just much narrower than the main search.
- No depth limit in quiescence.  Very long capture chains are fully resolved, but infinite loops are impossible because captures reduce material monotonically.
