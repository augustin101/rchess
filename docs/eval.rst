Static Evaluation
=================

The static evaluator is in ``src/engine/eval.rs`` and ``src/engine/pst.rs``.
All scores are in centipawns (cp) from White's perspective
(positive = White is better, negative = Black is better).

When NNUE weights are available the engine uses the neural evaluator instead;
the static evaluator is the fallback and is also used to build training targets
through the Lichess dataset.

----

Tapered Evaluation
------------------

The score is a linear interpolation between a middlegame (MG) and endgame (EG)
value based on the amount of material remaining on the board.

**Game phase**

Each piece type contributes a weight to the *phase counter*:

.. list-table::
   :header-rows: 1

   * - Piece
     - Weight
   * - Queen
     - 4
   * - Rook
     - 2
   * - Bishop
     - 1
   * - Knight
     - 1

The phase counter is the sum over all pieces of both colors, clamped to
``PHASE_MAX`` = 24 (the value for a fully-loaded starting position)::

    phase = min(PHASE_MAX,
                sum over all pieces of phase_weight[piece_type])

**Interpolation formula**

.. code-block:: text

    score = (mg_score × phase + eg_score × (PHASE_MAX − phase)) / PHASE_MAX

At ``phase = 24`` (opening/middlegame) the result equals ``mg_score``.
At ``phase = 0`` (bare-king endgame) the result equals ``eg_score``.

Every term below produces an ``(mg, eg)`` pair that is passed through this formula.

----

Material + Piece-Square Tables
-------------------------------

**Material values (cp)**:

.. list-table::
   :header-rows: 1

   * - Piece
     - MG
     - EG
   * - Pawn
     - 100
     - 110
   * - Knight
     - 320
     - 300
   * - Bishop
     - 330
     - 340
   * - Rook
     - 500
     - 530
   * - Queen
     - 900
     - 940

Pawns are slightly more valuable in the endgame (closer to promotion).
Knights are less valuable in open endgames; bishops gain.

**Piece-square tables (PSTs)**

Each piece type has a 64-entry MG and EG table stored in ``src/engine/pst.rs``.
All tables are from White's perspective (a1=0, h8=63); for Black, the square
index is vertically mirrored (``sq ^ 56``).

The per-square contribution of one piece::

    contribution = material_value + pst[mirrored_sq]

Sign: +1 for White pieces, −1 for Black pieces.

Representative PST entries (pawn MG, cp adjustment on top of base value):

.. code-block:: text

    rank 7 (about to promote):  +50 everywhere
    rank 6 (advanced):          +10 … +30 (central bonus)
    rank 5:                      +5 … +25
    center files at rank 4:     +20 … +25
    starting rank:               −20 … +10

----

Pawn Structure
--------------

All terms are evaluated for each side and differenced.

**Doubled pawns** — extra penalty for each pawn beyond the first on the same file::

    mg += (count_on_file − 1) × DOUBLED_MG    (−12 cp per extra pawn)
    eg += (count_on_file − 1) × DOUBLED_EG    (−24 cp per extra pawn)

**Isolated pawns** — penalty when a pawn has no friendly pawn on either adjacent
file::

    mg += count_isolated × ISOLATED_MG    (−15 cp each)
    eg += count_isolated × ISOLATED_EG    (−25 cp each)

**Pawn islands** — penalty per extra island beyond the first.  An island is a
maximal contiguous group of files that contain at least one pawn::

    mg += (islands − 1) × ISLAND_MG    (−8 cp per extra island)
    eg += (islands − 1) × ISLAND_EG    (−12 cp per extra island)

**Passed pawns** — bonus for a pawn with no enemy pawn blocking or guarding the
promotion path (same file and adjacent files ahead).  Bonus scales with how far
the pawn has advanced (0 = starting rank, 5 = one step from promotion):

.. list-table::
   :header-rows: 1

   * - Advancement
     - MG
     - EG
   * - 0
     - 0
     - 0
   * - 1
     - +10
     - +20
   * - 2
     - +20
     - +40
   * - 3
     - +35
     - +65
   * - 4
     - +55
     - +105
   * - 5 (one step from promotion)
     - +75
     - +150

The endgame bonus is roughly double the middlegame bonus, reflecting the increased
danger of passed pawns when there are few pieces left to stop them.

----

Rook Bonuses
------------

**Open file** — a file with no pawns of either color::

    mg += ROOK_OPEN_MG    (+22 cp)
    eg += ROOK_OPEN_EG    (+28 cp)

**Semi-open file** — a file with an enemy pawn but no friendly pawn::

    mg += ROOK_SEMI_MG    (+10 cp)
    eg += ROOK_SEMI_EG    (+16 cp)

**7th rank** — rook on the 7th rank (2nd rank for Black), where it can threaten
the enemy king and back pawns::

    mg += ROOK_SEVENTH_MG    (+16 cp)
    eg += ROOK_SEVENTH_EG    (+28 cp)

----

Bishop Pair
-----------

Owning both bishops is a long-term strategic advantage, especially in open positions
and endgames::

    bonus per side with both bishops:
        mg += BISHOP_PAIR_MG    (+25 cp)
        eg += BISHOP_PAIR_EG    (+50 cp)

The endgame bonus is double the middlegame bonus because the absence of pawns and
minor pieces leaves long diagonals open for the bishops.

----

King Safety
-----------

King safety contributes only to the middlegame component and fades linearly to zero
as the phase drops toward ``PHASE_MAX / 4``.  In the endgame, king activity is
captured by the king PST instead.

**Fade formula**::

    king_safety_mg = raw_mg
                   × max(0, phase − PHASE_MAX/4) / (3 × PHASE_MAX/4)

**Pawn shield** — counts friendly pawns in the two rows immediately in front of
and flanking the king::

    shield = {squares one and two steps forward of the king, ±1 file}
    score += count(own_pawns ∩ shield) × PAWN_SHIELD_MG    (+15 cp per pawn)

**Open-file exposure** — when the king sits on a file with no pawns of either
color it is exposed to rooks and queens::

    score += KING_OPEN_FILE_MG    (−40 cp)

----

Mobility
--------

For each major piece, the mobility bonus equals the number of squares it attacks
that are not occupied by a friendly piece, multiplied by a type-specific weight::

    bonus = attacked_squares_not_occupied_by_own × MOB_weight[piece_type]

    MOB_KNIGHT = 7 cp/square
    MOB_BISHOP = 5 cp/square
    MOB_ROOK   = 3 cp/square
    MOB_QUEEN  = 2 cp/square

Mobility is a middlegame-only bonus (EG weight = 0 in the taper).

----

Check Penalty
-------------

A flat penalty is applied when the side to move is in check::

    CHECK_PENALTY = 60 cp

This slightly biases search toward positions where the opponent has been checked,
complementing the positional value of check that the search already captures.

----

Full Evaluation Pipeline
-------------------------

.. code-block:: text

    phase = game_phase(board)

    score = material_pst(board, phase)
          + pawn_structure(board, phase)
          + rook_eval(board, phase)
          + bishop_pair(board, phase)
          + king_safety(board, phase)
          + mobility(board, phase)
          + check_penalty(board)

The result is from White's perspective.  The search converts to side-to-move
perspective by negating when it is Black's turn.
