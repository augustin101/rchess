//! Dual-limit time management for iterative deepening.
//!
//! Formula:  T_target = time_left / (moves_remaining + 20)  +  increment × 0.8
//!
//! Soft limit (0.6 × T_target) — checked between depths; if expired, skip new depth.
//! Hard limit (1.5 × T_target) — checked inside the search every N nodes via an
//!   AtomicBool abort flag; triggers immediate return from the current search.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MOVES_TO_GO_K:   u64 = 20;   // denominator constant in the formula
const INC_FRACTION:    f64 = 0.8;  // fraction of increment to use
const SOFT_FRACTION:   f64 = 0.6;  // soft = 0.6 × T_target
const HARD_FRACTION:   f64 = 1.2;  // hard = 1.2 × T_target

pub const PANIC_MS: u64 = 500;     // switch to panic mode below this (ms remaining)

pub struct TimeManager {
    pub start:         Instant,
    pub soft_deadline: Instant,
    pub hard_deadline: Instant,
    /// True when the engine has very little time left.
    pub panic_mode:    bool,
    /// Shared with SearchContext — set true to abort the ongoing search immediately.
    pub abort:         Arc<AtomicBool>,
}

impl TimeManager {
    /// Standard incremental time control.
    /// `time_left_ms` should already have the UCI move-overhead subtracted.
    pub fn new(time_left_ms: u64, increment_ms: u64, moves_to_go: Option<u32>) -> Self {
        let abort      = Arc::new(AtomicBool::new(false));
        let start      = Instant::now();
        let safe_ms    = time_left_ms.max(1);
        let panic_mode = safe_ms < PANIC_MS;

        let t_target_ms = if panic_mode {
            safe_ms / 2          // survive the next move at minimum
        } else {
            let n   = moves_to_go.unwrap_or(40) as u64;
            let inc = (increment_ms as f64 * INC_FRACTION) as u64;
            safe_ms / (n + MOVES_TO_GO_K) + inc
        };

        let soft_ms = ((t_target_ms as f64) * SOFT_FRACTION) as u64;
        let hard_ms = ((t_target_ms as f64) * HARD_FRACTION) as u64;

        Self {
            start,
            soft_deadline: start + Duration::from_millis(soft_ms),
            hard_deadline: start + Duration::from_millis(hard_ms),
            panic_mode,
            abort,
        }
    }

    /// Fixed move-time (`go movetime X`).
    /// `movetime_ms` should already have overhead subtracted.
    pub fn from_movetime(movetime_ms: u64) -> Self {
        let abort  = Arc::new(AtomicBool::new(false));
        let start  = Instant::now();
        let hard   = Duration::from_millis(movetime_ms.max(1));
        // Soft at 95% — avoids starting a new depth when time is nearly exhausted.
        let soft   = Duration::from_millis((movetime_ms.max(1) as f64 * 0.95) as u64);
        Self {
            start,
            soft_deadline: start + soft,
            hard_deadline: start + hard,
            panic_mode:    movetime_ms < PANIC_MS,
            abort,
        }
    }

    /// Unlimited time (fixed-depth searches, `go infinite`).
    pub fn infinite() -> Self {
        let start = Instant::now();
        let far   = start + Duration::from_secs(86_400);
        Self {
            start,
            soft_deadline: far,
            hard_deadline: far,
            panic_mode:    false,
            abort:         Arc::new(AtomicBool::new(false)),
        }
    }

    /// True if no new depth should be started (between-depth check).
    #[inline]
    pub fn soft_expired(&self) -> bool {
        Instant::now() >= self.soft_deadline
    }

    /// True if the search must abort immediately (inside-search check).
    #[inline]
    pub fn hard_expired(&self) -> bool {
        Instant::now() >= self.hard_deadline
    }

    /// Extend both deadlines by `frac` of the hard time still remaining.
    /// Called when aspiration windows fail (volatile position — needs more time).
    pub fn extend(&mut self, frac: f64) {
        let now       = Instant::now();
        let remaining = self.hard_deadline.saturating_duration_since(now);
        let ext       = Duration::from_millis((remaining.as_millis() as f64 * frac) as u64);
        self.soft_deadline += ext;
        self.hard_deadline += ext;
    }

    /// Wall-clock time spent since the search started.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}
