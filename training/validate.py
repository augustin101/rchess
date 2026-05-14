"""
Validation script — evaluates a checkpoint against the fixed validation split.

The validation set is never seen during training. Use this script after training
to evaluate checkpoints, or to find the best epoch across all saved checkpoints.

Usage (from project root):
    python training/validate.py                          # evaluate best.pt on val
    python training/validate.py --ckpt checkpoints/epoch_03.pt
    python training/validate.py --split test            # final held-out test set
    python training/validate.py --find_best             # scan all epoch_*.pt, pick best
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader

from dataset import BinpackDataset, collate_fn, unpack_batch
from model import NNUE, SCALE_CP

console = Console()


# ── Metrics ───────────────────────────────────────────────────────────────────

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@torch.no_grad()
def run_validation(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
) -> dict:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    all_logits  = []   # raw model outputs
    all_targets = []   # win-probability targets in (0,1)
    all_cp      = []   # original centipawn values

    total_loss = 0.0
    n_batches  = 0
    t0         = time.perf_counter()

    for wbits, bbits, stm, cp in loader:
        wf, bf, stm_t, cp_t = unpack_batch(wbits, bbits, stm, cp, device)
        target  = sigmoid(cp_t / SCALE_CP).unsqueeze(1)
        logits  = model(wf, bf, stm_t)

        total_loss += loss_fn(logits, target).item()
        n_batches  += 1

        all_logits.append(logits.squeeze(1).cpu())
        all_targets.append(target.squeeze(1).cpu())
        all_cp.append(cp_t.cpu())

    elapsed = time.perf_counter() - t0

    logits  = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    cp_vals = torch.cat(all_cp)

    n = len(logits)

    # Win-probability predictions (after sigmoid)
    probs = sigmoid(logits)

    # Centipawn predictions: invert the sigmoid → logit * SCALE_CP
    cp_pred = logits * SCALE_CP
    cp_err  = cp_pred - cp_vals

    # Metrics
    bce_loss   = total_loss / max(1, n_batches)
    mse_cp     = float(cp_err.pow(2).mean())
    mae_cp     = float(cp_err.abs().mean())
    # Accuracy: does the sign of cp_pred match the sign of cp_true?
    sign_acc   = float(((cp_pred * cp_vals) >= 0).float().mean()) * 100.0

    # Sign accuracy excluding near-equal positions (|cp_true| <= 50 cp)
    decisive_mask = cp_vals.abs() > 50
    n_decisive    = int(decisive_mask.sum())
    if n_decisive > 0:
        sign_acc_decisive = float(
            ((cp_pred[decisive_mask] * cp_vals[decisive_mask]) >= 0).float().mean()
        ) * 100.0
    else:
        sign_acc_decisive = float('nan')

    # Calibration: bucket probs [0,1] into 10 bins, compare mean pred vs mean target
    bins    = torch.linspace(0, 1, 11)
    cal_err = []
    for i in range(10):
        lo, hi  = bins[i].item(), bins[i + 1].item()
        mask    = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            cal_err.append(float((probs[mask] - targets[mask]).abs().mean()))
    calib_mae = float(np.mean(cal_err)) if cal_err else float('nan')

    return {
        'n_positions':       n,
        'bce_loss':          bce_loss,
        'mse_cp':            mse_cp,
        'mae_cp':            mae_cp,
        'rmse_cp':           math.sqrt(mse_cp),
        'sign_acc':          sign_acc,
        'sign_acc_decisive': sign_acc_decisive,
        'n_decisive':        n_decisive,
        'calib_mae':         calib_mae,
        'elapsed_s':   elapsed,
        'pos_per_sec': n / elapsed,
        # Raw tensors for plotting
        '_logits':  logits,
        '_targets': targets,
        '_cp_vals': cp_vals,
        '_cp_pred': cp_pred,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_validation_plots(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cp_vals = results['_cp_vals'].numpy()
    cp_pred = results['_cp_pred'].numpy()
    targets = results['_targets'].numpy()
    logits  = results['_logits'].numpy()
    probs   = 1 / (1 + np.exp(-logits))   # sigmoid

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Validation analysis', fontsize=13)

    # 1. Predicted cp vs true cp (2D density)
    ax = axes[0]
    lim = 900
    mask = (np.abs(cp_vals) < lim) & (np.abs(cp_pred) < lim)
    ax.hexbin(cp_vals[mask], cp_pred[mask], gridsize=80,
              cmap='plasma', mincnt=1, linewidths=0.1)
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1, label='perfect')
    ax.set_xlabel('True centipawns')
    ax.set_ylabel('Predicted centipawns')
    ax.set_title(f'Predicted vs True cp  (MAE={results["mae_cp"]:.1f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 2. Predicted win-prob vs target win-prob (calibration)
    ax = axes[1]
    bins  = np.linspace(0, 1, 21)
    bmid  = (bins[:-1] + bins[1:]) / 2
    mean_pred   = []
    mean_target = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() > 0:
            mean_pred.append(probs[m].mean())
            mean_target.append(targets[m].mean())
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='perfect calibration')
    ax.plot(mean_target, mean_pred, marker='o', markersize=4,
            color='steelblue', label='model')
    ax.set_xlabel('Target win probability')
    ax.set_ylabel('Predicted win probability')
    ax.set_title(f'Calibration  (MAE={results["calib_mae"]:.4f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Error distribution (cp_pred - cp_true)
    ax = axes[2]
    errors = cp_pred - cp_vals
    clipped = errors[np.abs(errors) < 1000]
    ax.hist(clipped, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax.axvline(0, color='red', linewidth=1, linestyle='--')
    ax.set_xlabel('Prediction error (cp)')
    ax.set_ylabel('Count')
    ax.set_title(f'Error distribution  (RMSE={results["rmse_cp"]:.1f} cp)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    console.print(f'Plot saved → [dim]{out_path}[/]')


# ── Rich summary table ────────────────────────────────────────────────────────

def print_summary(results: dict, ckpt_path: Path, split_name: str) -> None:
    t = Table(title='Validation results', show_header=True,
              header_style='bold cyan', border_style='blue')
    t.add_column('Metric',    style='bold cyan', no_wrap=True)
    t.add_column('Value',     justify='right')
    t.add_column('Notes',     style='dim')

    t.add_row('Checkpoint',    str(ckpt_path),                   '')
    t.add_row('Split',         split_name,                       '')
    t.add_row('Positions',     f'{results["n_positions"]:,}',     '')
    t.add_row('', '', '')
    t.add_row('BCE loss',      f'{results["bce_loss"]:.5f}',      'lower = better  (random ≈ 0.693)')
    t.add_row('MAE cp',        f'{results["mae_cp"]:.1f}',        'mean abs error in centipawns')
    t.add_row('RMSE cp',       f'{results["rmse_cp"]:.1f}',       'root mean squared error')
    t.add_row('Sign accuracy', f'{results["sign_acc"]:.2f}%',     'correct who is better')
    decisive_str = (
        f'{results["sign_acc_decisive"]:.2f}%  ({results["n_decisive"]:,} pos)'
        if not math.isnan(results["sign_acc_decisive"]) else '—'
    )
    t.add_row('Sign acc (|cp|>10)', decisive_str,                 'excl. near-equal positions')
    t.add_row('Calib MAE',     f'{results["calib_mae"]:.4f}',     'win-prob calibration error')
    t.add_row('', '', '')
    t.add_row('Positions/sec', f'{results["pos_per_sec"]:,.0f}', '')
    t.add_row('Elapsed',       f'{results["elapsed_s"]:.1f}s',   '')

    console.print(Panel(t, border_style='blue'))


# ── Main ──────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> tuple[NNUE, dict]:
    data  = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model = NNUE()
    model.load_state_dict(data['model'] if 'model' in data else data)
    model.to(device)
    return model, data


def make_loader(data_path: str, batch: int) -> DataLoader:
    dataset = BinpackDataset(data_path)
    return DataLoader(
        dataset,
        batch_size  = batch,
        num_workers = min(4, os.cpu_count() or 1),
        pin_memory  = True,
        collate_fn  = collate_fn,
        shuffle     = False,
    )


def find_best(args: argparse.Namespace) -> None:
    """Scan all epoch_*.pt checkpoints, evaluate each on val, copy the best to best.pt."""
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir  = Path(args.ckpt).parent
    splits    = json.loads(Path(args.splits).read_text())
    loader    = make_loader(splits['val_path'], args.batch)

    epoch_ckpts = sorted(ckpt_dir.glob('epoch_*.pt'))
    if not epoch_ckpts:
        console.print(f'[red]No epoch_*.pt files found in {ckpt_dir}[/]')
        raise SystemExit(1)

    console.print(f'Scanning [cyan]{len(epoch_ckpts)}[/] checkpoints on val split …\n')

    results_table = Table(show_header=True, header_style='bold cyan', border_style='blue')
    results_table.add_column('Checkpoint', no_wrap=True)
    results_table.add_column('Epoch', justify='right')
    results_table.add_column('Train loss', justify='right')
    results_table.add_column('Val BCE', justify='right')
    results_table.add_column('MAE cp', justify='right')
    results_table.add_column('Sign acc', justify='right')
    results_table.add_column('Sign acc (|cp|>10)', justify='right')

    best_bce   = float('inf')
    best_ckpt  = epoch_ckpts[0]
    all_scores = []

    for ckpt_path in epoch_ckpts:
        model, meta = load_model(ckpt_path, device)
        r = run_validation(model, loader, device)
        epoch      = meta.get('epoch', '?')
        train_loss = meta.get('train_loss', float('nan'))

        marker = ''
        if r['bce_loss'] < best_bce:
            best_bce  = r['bce_loss']
            best_ckpt = ckpt_path
            marker    = ' ◀'

        decisive_str = (
            f'{r["sign_acc_decisive"]:.2f}%'
            if not math.isnan(r["sign_acc_decisive"]) else '—'
        )
        results_table.add_row(
            ckpt_path.name + marker,
            str(epoch),
            f'{train_loss:.5f}',
            f'[bold green]{r["bce_loss"]:.5f}[/]' if marker else f'{r["bce_loss"]:.5f}',
            f'{r["mae_cp"]:.1f}',
            f'{r["sign_acc"]:.2f}%',
            decisive_str,
        )
        all_scores.append((ckpt_path, r))

    console.print(Panel(results_table, title='[bold]Checkpoint comparison[/]', border_style='blue'))

    # Copy best checkpoint
    import shutil
    best_out = ckpt_dir / 'best.pt'
    shutil.copy2(best_ckpt, best_out)
    console.print(f'\n[bold green]Best:[/] [cyan]{best_ckpt.name}[/]  val BCE={best_bce:.5f}')
    console.print(f'Copied → [dim]{best_out}[/]')

    # Save plots for the best checkpoint
    _, best_results = next(s for s in all_scores if s[0] == best_ckpt)
    save_validation_plots(best_results, Path(args.plot_out))


def main(args: argparse.Namespace) -> None:
    if args.find_best:
        find_best(args)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        console.print(f'[red]Checkpoint not found:[/] {ckpt_path}')
        raise SystemExit(1)

    console.print(f'Loading [cyan]{ckpt_path}[/] …')
    model, meta = load_model(ckpt_path, device)
    ckpt_info   = f"  epoch {meta['epoch']}" if 'epoch' in meta else ''
    console.print(f'Model loaded{ckpt_info}')

    splits = json.loads(Path(args.splits).read_text())
    if args.split == 'val':
        data_path = splits['val_path']
    elif args.split == 'test':
        data_path = splits['test_path']
    else:
        console.print(f'[red]Unknown split:[/] {args.split}  (choose val or test)')
        raise SystemExit(1)

    console.print(f'Split: [cyan]{args.split}[/] → {data_path}')
    loader = make_loader(data_path, args.batch)
    console.print(f'Positions: [cyan]{len(loader.dataset):,}[/]\n')  # type: ignore[arg-type]

    results = run_validation(model, loader, device)
    print_summary(results, ckpt_path, args.split)
    save_validation_plots(results, Path(args.plot_out))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      default='checkpoints/best.pt',
                   help='Checkpoint to evaluate')
    p.add_argument('--splits',    default='data/binpack/splits.json')
    p.add_argument('--split',     default='test', choices=['val', 'test'],
                   help='Which split to evaluate (val or test)')
    p.add_argument('--find_best', action='store_true',
                   help='Scan all epoch_*.pt, evaluate on val, copy best to best.pt')
    p.add_argument('--batch',     type=int, default=4096)
    p.add_argument('--plot_out',  default='checkpoints/plots/validation.png')
    main(p.parse_args())
