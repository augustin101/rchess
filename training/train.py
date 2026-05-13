"""
NNUE training — dual-perspective 768 → 256 → 512 → 32 → 32 → 1.

Reads pre-built binpack files (build_binpack.py + shuffle_split.py).

Usage:
    python training/train.py --splits data/binpack/splits.json
    python training/train.py --splits data/binpack/splits.json --epochs 10 --batch 8192
"""

import argparse
import math
import os
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from model import NNUE, SCALE_CP
from dataset import make_loaders, unpack_batch
from validate import run_validation

CHECKPOINT_DIR = Path('checkpoints')
SPARK_CHARS    = ' ▁▂▃▄▅▆▇█'


# ── Terminal sparkline ────────────────────────────────────────────────────────

def sparkline(values: list[float], width: int = 24) -> str:
    v = list(values)[-width:]
    if len(v) < 2:
        return ' ' * width
    lo, hi = min(v), max(v)
    span = hi - lo or 1e-9
    return ''.join(SPARK_CHARS[int((x - lo) / span * (len(SPARK_CHARS) - 1))] for x in v)


# ── Rich live display ─────────────────────────────────────────────────────────

def build_display(
    *,
    epoch:        int,
    total_epochs: int,
    step:         int,
    approx_steps: int,
    recent_losses: deque,
    pps_history:  deque,
    last_val:     float,
    best_val:     float,
    lr:           float,
    epoch_elapsed: float,
    epoch_progress: Progress,
    total_progress: Progress,
):
    smooth_loss = float(np.mean(recent_losses)) if recent_losses else float('nan')
    last_loss   = recent_losses[-1] if recent_losses else float('nan')
    avg_pps     = float(np.mean(pps_history)) if pps_history else 0.0
    iter_ms     = (1000.0 / avg_pps * (approx_steps / max(step, 1))) if avg_pps > 0 else 0.0
    # Rough ETA for current epoch
    done_frac   = step / max(approx_steps, 1)
    eta_epoch   = (epoch_elapsed / done_frac - epoch_elapsed) if done_frac > 0.001 else 0.0

    # ── Metrics table ─────────────────────────────────────────────────────────
    m = Table.grid(padding=(0, 2))
    m.add_column(style='bold cyan',  no_wrap=True, min_width=20)
    m.add_column(no_wrap=True)

    m.add_row('Loss (batch)',    f'[yellow]{last_loss:.5f}[/]')
    m.add_row('Loss (smooth)',   f'[yellow]{smooth_loss:.5f}[/]  [dim]{sparkline(list(recent_losses))}[/]')
    m.add_row('Val loss',
              f'[green]{last_val:.5f}[/]  best [bold green]{best_val:.5f}[/]'
              if not np.isnan(last_val) else '[dim]—[/]')
    if not np.isnan(last_val):
        gap = last_val - smooth_loss
        m.add_row('Train/Val gap',
                  f'[red]{gap:+.5f}[/]' if gap > 0.005 else f'[green]{gap:+.5f}[/]')
    m.add_row('',                '')
    m.add_row('Positions / sec', f'[cyan]{avg_pps:,.0f}[/]')
    m.add_row('ms / batch',      f'{1000.0 / avg_pps:.2f}' if avg_pps > 0 else '—')
    m.add_row('ETA epoch',       f'{int(eta_epoch // 60):02d}:{int(eta_epoch % 60):02d}')
    m.add_row('Epoch elapsed',   f'{int(epoch_elapsed // 60):02d}:{int(epoch_elapsed % 60):02d}')
    m.add_row('',                '')
    m.add_row('Learning rate',   f'{lr:.2e}')

    # ── Assemble panel ────────────────────────────────────────────────────────
    prog_group = Group(
        Text(f'Epoch {epoch}/{total_epochs}', style='bold'),
        epoch_progress,
        total_progress,
    )
    body = Columns([prog_group, m], padding=(0, 4))
    return Panel(body, title='[bold blue]rchess NNUE Training[/]', border_style='blue')


# ── Matplotlib plots ──────────────────────────────────────────────────────────

def _smooth(values: list[float], window: int = 100) -> list[float]:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid').tolist()


def save_plots(
    step_losses:  list[float],
    epoch_train:  list[float],
    epoch_val:    list[float],
    lrs:          list[float],
    plot_dir:     Path,
    label:        str,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'rchess NNUE — {label}', fontsize=13)

    ax = axes[0]
    if step_losses:
        w  = min(200, max(10, len(step_losses) // 30))
        ax.plot(_smooth(step_losses, w), linewidth=0.8, color='steelblue', label='smoothed')
        ax.set_xlabel('Step')
        ax.set_ylabel('BCE loss')
        ax.set_title('Train loss (smoothed)')
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if epoch_train:
        ep = list(range(1, len(epoch_train) + 1))
        ax.plot(ep, epoch_train, marker='o', label='train', color='steelblue')
        ax.plot(ep, epoch_val,   marker='s', label='val',   color='tomato')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BCE loss')
        ax.set_title('Train vs Validation loss')
        ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if lrs:
        ax.plot(list(range(1, len(lrs) + 1)), lrs, marker='o', color='seagreen')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning rate')
        ax.set_title('LR schedule')
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / 'latest.png', dpi=120, bbox_inches='tight')
    if 'epoch' in label:
        plt.savefig(plot_dir / f'{label}.png', dpi=120, bbox_inches='tight')
    plt.close()


# ── Loss function ─────────────────────────────────────────────────────────────

class MixedLoss(nn.Module):
    """
    (1 - α) · BCE(logit, sigmoid(cp/SCALE)) + α · MSE(logit, cp/SCALE)

    BCE handles "who is winning" (strategy).
    MSE on raw logits handles "by how much" (precision).
    Start with a small α (e.g. 0.01).
    """
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self._bce  = nn.BCEWithLogitsLoss()
        self._mse  = nn.MSELoss()

    def forward(self, logits: torch.Tensor, cp: torch.Tensor) -> torch.Tensor:
        cp_norm = (cp / SCALE_CP).unsqueeze(1)
        bce = self._bce(logits, torch.sigmoid(cp_norm))
        mse = self._mse(logits, cp_norm)
        return (1.0 - self.alpha) * bce + self.alpha * mse


# ── Training helpers ──────────────────────────────────────────────────────────

def _train_step(
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    loss_fn:    nn.Module,
    batch:      tuple,
    device:     torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """One forward+backward pass. Returns (loss_value, positions_per_sec)."""
    wbits, bbits, stm, cp = batch
    t0 = time.perf_counter()
    wf, bf, stm, cp = unpack_batch(wbits, bbits, stm, cp, device)
    optimizer.zero_grad()
    pred = model(wf, bf, stm)
    loss = loss_fn(pred, cp)
    loss.backward()
    optimizer.step()
    return loss.item(), batch_size / (time.perf_counter() - t0)


def _run_validation(model: nn.Module, loader, device: torch.device) -> float:
    """Evaluate on loader, restore train mode, return BCE loss."""
    val_loss = run_validation(model, loader, device)['bce_loss']
    model.train()
    return val_loss


def _print_epoch_summary(
    console:      Console,
    epoch:        int,
    total_epochs: int,
    avg_train:    float,
    last_val:     float,
    lr_now:       float,
    elapsed:      float,
    is_best:      bool,
) -> None:
    best_marker  = '  [bold green]★ best[/]' if is_best else ''
    gap          = last_val - avg_train
    gap_color    = 'red' if gap > 0.005 else 'green'
    overfit_warn = '  [bold red]⚠ overfit[/]' if gap > 0.01 else ''
    console.print(
        f'[bold]Epoch {epoch}/{total_epochs}[/]  '
        f'train=[yellow]{avg_train:.5f}[/]  '
        f'val=[green]{last_val:.5f}[/]  '
        f'Δ=[{gap_color}]{gap:+.5f}[/]  '
        f'lr={lr_now:.2e}  '
        f'[dim]{elapsed:.0f}s[/]'
        f'{best_marker}{overfit_warn}'
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    plot_dir = CHECKPOINT_DIR / 'plots'
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, _ = make_loaders(
        splits_json = args.splits,
        batch_size  = args.batch,
        num_workers = args.workers,
    )
    approx_steps = max(1, len(train_loader.dataset) // args.batch)

    model     = NNUE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn   = MixedLoss(alpha=args.loss_alpha)

    # Hold LR flat for lr_flat_frac of training, then cosine-decay to 0.
    flat_epochs  = int(args.lr_flat_frac * args.epochs)
    decay_epochs = max(1, args.epochs - flat_epochs)
    def _lr_lambda(epoch: int) -> float:
        if epoch < flat_epochs:
            return 1.0
        t = (epoch - flat_epochs) / decay_epochs
        return 0.5 * (1.0 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    recent_losses: deque[float] = deque(maxlen=300)
    pps_history:   deque[float] = deque(maxlen=30)
    step_losses: list[float] = []
    epoch_train: list[float] = []
    epoch_val:   list[float] = []
    lrs:         list[float] = []
    last_val = float('nan')
    best_val = float('inf')
    lr_now   = args.lr

    epoch_prog = Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=False),
    )
    total_prog = Progress(
        TextColumn('[bold blue]{task.description}'),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        console=Console(stderr=False),
    )
    epoch_task = epoch_prog.add_task('Steps', total=approx_steps)
    total_task = total_prog.add_task('Epochs', total=args.epochs)

    console = Console()
    console.print(
        f'[bold]rchess NNUE[/]  device=[cyan]{device}[/]  '
        f'params=[cyan]{sum(p.numel() for p in model.parameters()):,}[/]  '
        f'batch=[cyan]{args.batch:,}[/]  '
        f'lr=[cyan]{args.lr}[/]  '
        f'epochs=[cyan]{args.epochs}[/]'
    )
    console.print(f'Plots → [dim]{plot_dir}/latest.png[/]  (updated every {args.plot_steps} steps)\n')

    with Live(console=console, refresh_per_second=4, transient=False) as live:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss  = 0.0
            n_batches   = 0
            epoch_start = time.perf_counter()
            epoch_prog.reset(epoch_task, total=approx_steps,
                             description=f'Epoch {epoch}/{args.epochs}')

            def refresh(step: int) -> None:
                live.update(build_display(
                    epoch=epoch, total_epochs=args.epochs,
                    step=step, approx_steps=approx_steps,
                    recent_losses=recent_losses, pps_history=pps_history,
                    last_val=last_val, best_val=best_val, lr=lr_now,
                    epoch_elapsed=time.perf_counter() - epoch_start,
                    epoch_progress=epoch_prog, total_progress=total_prog,
                ))

            for step, batch in enumerate(train_loader, 1):
                loss_val, pps = _train_step(model, optimizer, loss_fn, batch, device, args.batch)
                recent_losses.append(loss_val)
                pps_history.append(pps)
                step_losses.append(loss_val)
                epoch_loss += loss_val
                n_batches  += 1
                epoch_prog.advance(epoch_task)

                if args.val_steps > 0 and step % args.val_steps == 0:
                    last_val = _run_validation(model, val_loader, device)
                    if last_val < best_val:
                        best_val = last_val
                        torch.save(dict(epoch=epoch, model=model.state_dict(),
                                        train_loss=float(np.mean(recent_losses)),
                                        val_loss=last_val),
                                   CHECKPOINT_DIR / 'best.pt')
                    refresh(step)

                if step % args.display_steps == 0:
                    refresh(step)

                if step % args.plot_steps == 0:
                    save_plots(step_losses, epoch_train, epoch_val, lrs, plot_dir, f'step_{step}')

            # ── End of epoch ─────────────────────────────────────────────────
            scheduler.step()
            avg_train = epoch_loss / max(1, n_batches)
            lr_now    = float(scheduler.get_last_lr()[0])
            elapsed   = time.perf_counter() - epoch_start

            last_val = _run_validation(model, val_loader, device)
            is_best  = last_val < best_val
            if is_best:
                best_val = last_val

            epoch_train.append(avg_train)
            epoch_val.append(last_val)
            lrs.append(lr_now)
            total_prog.advance(total_task)

            ckpt = dict(epoch=epoch, model=model.state_dict(),
                        train_loss=avg_train, val_loss=last_val)
            torch.save(ckpt, CHECKPOINT_DIR / f'epoch_{epoch:02d}.pt')
            if is_best:
                torch.save(ckpt, CHECKPOINT_DIR / 'best.pt')

            refresh(approx_steps)
            save_plots(step_losses, epoch_train, epoch_val, lrs, plot_dir, f'epoch_{epoch:02d}')
            _print_epoch_summary(console, epoch, args.epochs, avg_train, last_val, lr_now, elapsed, is_best)

    torch.save(model.state_dict(), CHECKPOINT_DIR / 'nnue_final.pt')
    console.print(f'\n[bold green]Done.[/]  Checkpoints in [dim]{CHECKPOINT_DIR}/[/]')
    console.print(f'Best val BCE={best_val:.5f} → [dim]{CHECKPOINT_DIR}/best.pt[/]')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--splits',        default='data/binpack/splits.json')
    p.add_argument('--epochs',        type=int,   default=5)
    p.add_argument('--batch',         type=int,   default=4096)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--lr_flat_frac',  type=float, default=0.0,
                   help='Fraction of epochs to hold LR flat before cosine decay (0=decay immediately)')
    p.add_argument('--loss_alpha',    type=float, default=0.01,
                   help='MSE weight in mixed loss — 0=pure BCE, 1=pure MSE')
    p.add_argument('--workers',       type=int,   default=min(8, os.cpu_count() or 4))
    p.add_argument('--display_steps', type=int,   default=10,
                   help='Refresh live display every N steps')
    p.add_argument('--plot_steps',    type=int,   default=500,
                   help='Save plots every N steps')
    p.add_argument('--val_steps',     type=int,   default=0,
                   help='Validate every N steps mid-epoch (0 = epoch end only)')
    train(p.parse_args())
