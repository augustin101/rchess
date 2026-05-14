Search
======

The engine searches with iterative-deepening alpha-beta (``src/engine/alpha_beta.rs``).
All parameters referenced below are ``const`` values in that file.

----

Iterative Deepening
-------------------

The root loop runs depth *d* = 1, 2, … up to ``MAX_SEARCH_DEPTH`` (20) or until the
soft time limit fires between depths::

    for d in 1..=self.depth {
        if d > 1 && tm.soft_expired() { break; }
        // run one aspiration-window search at depth d
    }

The best move from the last *fully completed* depth is returned.  A depth that is
aborted mid-tree is discarded entirely so a partial result never corrupts the choice.

----

Aspiration Windows
------------------

From depth ``ASPIRATION_MIN_DEPTH`` = 4 onward the search is bounded by a narrow
window around the previous score instead of the full ``[-∞, +∞]`` window::

    lo = prev_score - ASPIRATION_DELTA   (50 cp)
    hi = prev_score + ASPIRATION_DELTA

If the score falls outside ``[lo, hi]``:

* **Fail-low** (``score ≤ lo``): widen down — ``lo -= delta``, double ``delta``, extend time by 20 %.
* **Fail-high** (``score ≥ hi``): widen up — ``hi += delta``, double ``delta``.

``delta`` is doubled on each failure until it exceeds ``ASPIRATION_MAX_DELTA`` = 1500 cp,
at which point the full ``[-∞, +∞]`` window is used.

----

Transposition Table
-------------------

A flat array of 2 :sup:`20` = 1 048 576 entries (~24 MB).  Each entry stores:

.. code-block:: text

    TtEntry { hash: u64, score: i32, best_move: Move, depth: u8, bound: Bound }

    Bound ∈ { Exact, Lower, Upper }

Probe: an entry hits when ``entry.hash == board.hash`` and ``entry.depth >= current_depth``.

* **Exact** — return the stored score immediately.
* **Lower** (beta cutoff was stored) — raise ``alpha`` if ``score > alpha``; cut if ``alpha ≥ beta``.
* **Upper** (all moves failed low) — lower ``beta`` if ``score < beta``; cut if ``alpha ≥ beta``.

Replacement policy: replace if the slot is empty, matches the same hash, or the new
depth is at least as deep.

----

Move Ordering
-------------

Moves are sorted by a priority score before searching.  Higher score = searched first.

.. list-table::
   :header-rows: 1

   * - Priority bucket
     - Score
     - Condition
   * - TT / hash move
     - 30 000 000
     - move matches the TT best move for this position
   * - Promotion
     - 9 000 000 + piece value
     - move has promo flag
   * - Capture (MVV-LVA)
     - 1 000 000 + victim×10 − attacker
     - target square occupied or en passant
   * - Killer move
     - 900 000
     - quiet move stored in killer table at this ply
   * - History heuristic
     - 0 … 32 000
     - bonus accumulated from past beta cutoffs
   * - Futility skip
     - −1 (sentinel)
     - sorted to back, then skipped

**MVV-LVA** (Most Valuable Victim – Least Valuable Attacker):

.. code-block:: text

    capture_score = 1_000_000 + victim_value × 10 − attacker_value

    piece values (cp): P=100, N=320, B=330, R=500, Q=900, K=10 000

Sorting uses *incremental selection sort* (swap the maximum remaining element to the
front on each iteration) so the best move is available immediately without sorting
the whole list — useful when a cutoff is found early.

----

Killer Moves
------------

A table of two quiet moves per ply (``KillerTable([[Move; 2]; 64])``) that previously
caused a beta cutoff at that ply.  When a new killer arrives the older one is shifted
out.  Killers are searched just below captures but above plain history moves.

----

History Heuristic
-----------------

A ``[color][from][to]`` array of bonus scores.  When a quiet move causes a beta
cutoff at depth *d*, its history score increases by *d* :sup:`2`::

    score = min(score + d², HISTORY_MAX)    HISTORY_MAX = 32 000

At the start of each search the table is *aged* (halved) to prevent stale scores
from dominating.

----

Null-Move Pruning
-----------------

Before expanding all moves at an interior node the engine tries passing (null move).
If the resulting score still exceeds ``beta`` (opponent's bound) the position is
likely strong enough that we can prune without searching fully.

Conditions to apply:

* Not in check.
* Depth ≥ ``NULL_MIN_DEPTH`` = 3.
* Side to move has at least one non-pawn, non-king piece (avoids zugzwang).

Reduction *R*:

.. code-block:: text

    R = NULL_R_PARTIAL (2)   if depth < NULL_FULL_DEPTH (6)
    R = NULL_R_FULL    (3)   if depth ≥ NULL_FULL_DEPTH

The null move is searched at ``depth − 1 − R`` with ``allow_null = false`` to prevent
two consecutive null moves.  If ``null_score ≥ beta`` the node returns ``beta``
immediately.

----

Futility Pruning
----------------

Applied at shallow depth (1–2) before making quiet moves.  If the static evaluation
plus a margin cannot reach ``alpha``, the move is unlikely to recover and is skipped.

.. code-block:: text

    futility_prunable = !in_check
                      ∧ depth ≤ FUTILITY_MAX_DEPTH (2)
                      ∧ static_eval + margin < alpha

    margin at depth 1 = FUTILITY_MARGIN_1 = 100 cp
    margin at depth 2 = FUTILITY_MARGIN_2 = 300 cp

Only quiet moves (non-capture, non-promotion) are skipped; captures and promotions
are always searched.

----

Late Move Reductions (LMR)
--------------------------

After the first few quiet moves have been searched at full depth, later moves are
searched at a reduced depth.  If the reduced search beats ``alpha`` a full re-search
is done.

Conditions to reduce:

* Move is quiet (non-capture, non-promotion).
* Move does not give check.
* No check extension on this move.
* At least ``LMR_MIN_QUIET`` = 3 quiet moves already searched this node.
* Depth ≥ ``LMR_MIN_DEPTH`` = 3.

Reduction formula::

    reduction = 1 + quiet_count / 6

The LMR path does a null-window scout first (``[-alpha-1, -alpha]``); only on a
scout beat does it re-search with the full window.

----

Check Extensions
----------------

When a move gives check the search depth is extended by 1 ply::

    extension = 1 if gives_check ∧ ply < MAX_PLY − 1 else 0

``MAX_PLY`` = 64 caps the total depth including extensions.

----

Quiescence Search
-----------------

At depth 0 the engine does not return the static evaluation directly.  Instead it
enters *quiescence search* — a reduced search that continues with captures and
(when in check) all legal moves until the position is quiet.

Algorithm:

1. If not in check, compute ``stand_pat = static_eval``.
   If ``stand_pat ≥ beta`` return ``stand_pat`` (beta cutoff).
   Raise ``alpha`` to ``max(alpha, stand_pat)``.
2. Score captures by MVV-LVA; score quiet moves as ``SCORE_QUIET_SKIP`` (−1) unless
   in check (all moves tried when in check).
3. Once the sorted list reaches a quiet move and we are not in check, stop.
4. If in check and no legal move was found, return ``−CHECKMATE_SCORE + ply``.

Quiescence has no depth limit; it terminates naturally when captures/checks run out.

----

Draw Detection Inside Search
-----------------------------

Two draw conditions are checked at every interior node before searching moves:

**50-move rule** — ``board.half_move_clock ≥ 100`` → return 0.

**Threefold repetition** — the engine maintains a ``position_history`` vector
(hashes of all positions on the path from the game start to the current node).
It is split at ``game_history_len`` into the *game* portion and the *search* portion::

    game_reps   = count(board.hash in position_history[..game_history_len])
    search_reps = count(board.hash in position_history[game_history_len..])

    if game_reps ≥ 2 or search_reps ≥ 1: return 0

``game_reps ≥ 2`` catches true threefold repetition (position already appeared twice
before the search root; making this move would be the third occurrence).
``search_reps ≥ 1`` prevents the engine from cycling within its own search tree.

----

Time Management
---------------

Managed by ``TimeManager`` (``src/engine/time_manager.rs``).

**Soft limit** — checked between ID depths; if expired, no new depth is started.

**Hard limit** — checked every ``check_interval`` nodes inside ``negamax``::

    check_interval = NORMAL_CHECK_INTERVAL (2048)   normally
                   = PANIC_CHECK_INTERVAL  (256)    when panic mode is active

Panic mode activates when the remaining clock time is below a threshold, tightening
the check interval to react faster to a low clock.

**Time allocation** — for an incremental time control the engine uses approximately::

    allocated ≈ remaining_time / 30 + increment × 0.75

adjusted for the move overhead.  The soft limit is set to about half of allocated
time; the hard limit is the full allocated time.

**Time extension** — when an aspiration window fails low, the hard deadline is
extended by 20 % (``tm.extend(0.20)``) because the position is volatile and
requires deeper search for a stable score.
