import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from utils.args import (
    add_resume_checkpoints,
    add_speed_optimizer_args,
    add_wandb_args,
)
from utils.distribution import init_distributed_mode
from utils.lr_scheduler import get_cosine_schedule_with_warmup
from utils.load_model import count_parameters, get_model_arch
from utils.wandb_vis import grid_to_pil
from src.ARC_loader import IGNORE_INDEX, build_dataloaders
# from ARC_LoopViT_v1 import LoopARCViT

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def _format_eta(seconds: float) -> str:
    total_seconds = int(max(seconds, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Loop-augmented ARC ViT")
    add_resume_checkpoints(parser)
    add_wandb_args(parser)
    add_speed_optimizer_args(parser)

    parser.add_argument("--data-root", type=str, default="raw_data/ARC-AGI")
    parser.add_argument("--train-split", type=str, default="training")
    parser.add_argument("--eval-split", type=str, default="training")
    parser.add_argument("--eval-subset", type=str, choices=("train", "test"), default="test")
    
    parser.add_argument("--architecture", type=str, default="loop_vit", 
                        choices=["loop_vit", "loop_vit1", "loop_vit2"],
                        help="Architecture variant to use.")

    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-colors", type=int, default=12)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--best-save-path", type=str, default=None)
    parser.add_argument("--lr-scheduler", type=str, choices=("none", "cosine"), default="cosine")
    parser.add_argument("--vis-every", type=int, default=25)

    # Loop-specific knobs
    parser.add_argument("--loop-core-depth", type=int, default=2, help="Number of shared transformer layers per loop iteration.")
    parser.add_argument("--max-loop-steps", type=int, default=8, help="Maximum loop iterations during training.")
    parser.add_argument("--min-loop-steps", type=int, default=2, help="Minimum steps before exit gate may trigger.")
    parser.add_argument("--disable-exit-gate", action="store_true", help="Disable the exit gate head and always run fixed loops.")
    parser.add_argument("--exit-gate-threshold", type=float, default=0.6, help="Gate probability threshold for early exit.")
    parser.add_argument("--gate-entropy-weight", type=float, default=1e-3, help="Weight for the negative-entropy regularizer on gate probabilities.")
    parser.add_argument("--loop-penalty-weight", type=float, default=5e-4, help="Weight encouraging fewer loop steps (normalized).")
    parser.add_argument("--train-dynamic-exit", action="store_true", help="Allow early exit decisions during training (default is teacher forcing).")
    parser.add_argument("--eval-dynamic-exit", action="store_true", help="Enable dynamic exit during evaluation.")
    parser.add_argument("--no-step-embedding", action="store_true", help="Remove loop step embeddings.")

    parser.add_argument("--include-rearc", action="store_true", help="Include Re-ARC tasks as additional training data.")
    parser.add_argument("--rearc-path", type=str, default="raw_data/re_arc")
    parser.add_argument("--rearc-limit", type=int, default=-1)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-task-tokens", type=int, default=1)
    parser.add_argument("--fix-scale-factor", type=int, default=2)
    parser.add_argument("--disable-translation", action="store_true")
    parser.add_argument("--disable-resolution-augmentation", action="store_true")

    return parser


def _build_model(args, train_dataset, device, distributed: bool, rank: int, local_rank: int):
    model = get_model_arch(args, train_dataset)
    model.to(device)

    if not args.no_compile and hasattr(torch, "compile"):
        # Disable compile for now to avoid CUDAGraphs issues with dynamic shapes/loops
        # if (not distributed) or rank == 0:
        #     print("Applying torch.compile for loop model...")
        # model = torch.compile(model, mode=args.compile_mode)
        pass

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    model_for_eval = model.module if distributed else model
    return model, model_for_eval


def _resume_if_needed(model, optimizer, scaler, scheduler, args, device):
    start_epoch = 1
    if not args.resume_checkpoint:
        return start_epoch

    ckpt_path = Path(args.resume_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state", {})
    
    # Handle torch.compile prefix (_orig_mod.)
    new_state_dict = {}
    for key, value in state_dict.items():
        # If checkpoint has _orig_mod but current model doesn't (or vice versa), handle it
        # But here we are loading into a DDP model which might wrap a compiled model
        # The error shows the model expects keys starting with "module._orig_mod."
        # but the checkpoint likely has keys starting with "module." or just the parameter name.
        
        # Case 1: Checkpoint has "module." prefix (from DDP)
        if key.startswith("module."):
            # If the current model is DDP wrapping a compiled model, it expects "module._orig_mod."
            # We try to adapt the key.
            if "_orig_mod." not in key:
                 new_key = key.replace("module.", "module._orig_mod.", 1)
                 new_state_dict[new_key] = value
            else:
                 new_state_dict[key] = value
        else:
            # Case 2: Checkpoint is raw model state
            # If current model is DDP + Compile, we need "module._orig_mod." prefix
            new_key = "module._orig_mod." + key
            new_state_dict[new_key] = value
            
            # Also keep the original key just in case
            new_state_dict[key] = value

    # Fallback: if the above heuristic fails, we can try to be smarter.
    # Let's just strip everything down to base names and match them to the current model.
    model_keys = set(model.state_dict().keys())
    final_state_dict = {}
    
    for k, v in state_dict.items():
        # [NEW] Skip task token embeddings if requested
        if args.resume_skip_task_token and "task_token_embed" in k:
            continue

        # Normalize checkpoint key: remove module. and _orig_mod.
        norm_k = k.replace("module.", "").replace("_orig_mod.", "")
        
        # Try to find matching key in current model
        # Current model keys might look like "module._orig_mod.layer..." or "module.layer..."
        found = False
        for mk in model_keys:
            norm_mk = mk.replace("module.", "").replace("_orig_mod.", "")
            if norm_k == norm_mk:
                final_state_dict[mk] = v
                found = True
                break
        
        if not found:
            # If not found, just keep original key (might be extra buffer)
            final_state_dict[k] = v

    model.load_state_dict(final_state_dict, strict=False)
    if not args.resume_reset_optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if not args.resume_reset_optimizer and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if not args.resume_reset_optimizer and "scheduler_state" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if not args.resume_reset_epoch and "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"] + 1
    return start_epoch


def _compute_gate_regularizers(metadata, args) -> Tuple[torch.Tensor, torch.Tensor]:
    if not metadata.gate_probs or args.disable_exit_gate:
        zero = torch.tensor(0.0, device=metadata.exit_steps.device if isinstance(metadata.exit_steps, torch.Tensor) else "cpu")
        return zero, zero
    gate_stack = torch.stack(metadata.gate_probs, dim=1)  # (B, steps)
    eps = 1e-6
    entropy = -(gate_stack * torch.log(gate_stack + eps) + (1 - gate_stack) * torch.log(1 - gate_stack + eps))
    neg_entropy = -entropy.mean()  # maximize entropy
    normalized_steps = metadata.exit_steps.float() / metadata.max_steps
    step_penalty = normalized_steps.mean()
    return neg_entropy, step_penalty


def evaluate(model, loader: DataLoader, device: torch.device, args, *, distributed: bool = False) -> Tuple[float, float, float, Dict[int, Any]]:
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    total_exact = 0
    total_examples = 0
    avg_steps_accumulator = 0.0
    visualizations: Dict[int, Any] = {}

    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        dataset.disable_translation()
        dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)

    with torch.no_grad():
        for batch in loader:
            task_ids_cpu = batch["task_ids"]
            inputs = batch["inputs"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            task_ids = task_ids_cpu.to(device)

            logits, metadata = model(
                inputs,
                task_ids,
                attention_mask=attention_mask,
                dynamic_exit=args.eval_dynamic_exit,
                gate_threshold=args.exit_gate_threshold,
            )

            num_colors = logits.size(1)
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
            loss = F.cross_entropy(
                logits_flat,
                targets.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            total_loss += loss.item()
            total_pixels += (targets != IGNORE_INDEX).sum().item()

            predictions = logits.argmax(dim=1)
            batch_size = predictions.size(0)
            avg_steps_accumulator += metadata.exit_steps.float().sum().item()
            for idx in range(batch_size):
                target = targets[idx]
                prediction = predictions[idx]
                valid = target != IGNORE_INDEX
                is_exact = bool(torch.equal(prediction[valid], target[valid])) if valid.any() else False
                total_exact += int(is_exact)
                total_examples += 1

                input_grid = inputs[idx]
                mask = attention_mask[idx]
                visualizations[task_ids_cpu[idx].item()] = grid_to_pil(mask, input_grid, target, prediction, IGNORE_INDEX=IGNORE_INDEX)

    if distributed and dist.is_initialized():
        totals = torch.tensor(
            [total_loss, total_pixels, total_exact, total_examples, avg_steps_accumulator],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_pixels, total_exact, total_examples, avg_steps_accumulator = totals.tolist()

    avg_loss = total_loss / max(total_pixels, 1)
    accuracy = total_exact / max(total_examples, 1)
    mean_steps = avg_steps_accumulator / max(total_examples, 1)

    if dataset is not None:
        if not args.disable_translation:
            dataset.enable_translation()
        if not args.disable_resolution_augmentation:
            dataset.enable_resolution_augmentation()

    return avg_loss, accuracy, mean_steps, visualizations


def train(args: argparse.Namespace) -> None:
    parser_args = args
    distributed, rank, world_size, local_rank, device = init_distributed_mode(parser_args)
    set_seed(args.seed + (rank if distributed else 0))

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    if args.disable_translation:
        train_dataset.disable_translation()
        if eval_dataset is not None:
            eval_dataset.disable_translation()
    else:
        train_dataset.enable_translation()
        if eval_dataset is not None:
            eval_dataset.enable_translation()

    if args.disable_resolution_augmentation:
        train_dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)
        if eval_dataset is not None:
            eval_dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)
    else:
        train_dataset.enable_resolution_augmentation()
        if eval_dataset is not None:
            eval_dataset.enable_resolution_augmentation()

    if (not distributed) or rank == 0:
        print(f"Total training examples: {len(train_dataset)}")

    model, model_for_eval = _build_model(args, train_dataset, device, distributed, rank, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))
    scheduler = None
    if args.lr_scheduler == "cosine":
        steps_per_epoch = max(len(train_loader), 1)
        total_training_steps = steps_per_epoch * max(args.epochs, 1)
        warmup_epochs = min(args.epochs // 5, 10)
        warmup_steps = max(warmup_epochs * steps_per_epoch, 1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

    start_epoch = _resume_if_needed(model, optimizer, scaler, scheduler, args, device)
    if (not distributed) or rank == 0:
        print(f"Parameter count: {count_parameters(model_for_eval) / 1_000_000:.2f}M (excluding task tokens)")
        if scaler.is_enabled():
            print("Using AMP training")
        else:
            print("AMP disabled")

    wandb_run = None
    is_main_process = (not distributed) or rank == 0
    if args.use_wandb and is_main_process:
        if wandb is None:
            raise RuntimeError("Weights & Biases is not installed but --use-wandb was set.")
        wandb_config = dict(vars(args))
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name or None, config=wandb_config)
        wandb.watch(model_for_eval, log=None)

    best_eval_acc = float("-inf")
    global_start = time.time()
    previous_total_steps = 0

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            sample_count = 0
            train_exact = 0
            train_examples = 0
            avg_steps_accumulator = 0.0
            epoch_start = time.time()

            for step, batch in enumerate(train_loader, 1):
                inputs = batch["inputs"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                task_ids = batch["task_ids"].to(device)

                optimizer.zero_grad(set_to_none=True)

                autocast_device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cuda"
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    logits, metadata = model(
                        inputs,
                        task_ids,
                        attention_mask=attention_mask,
                        dynamic_exit=args.train_dynamic_exit,
                        gate_threshold=args.exit_gate_threshold,
                    )
                    num_colors = logits.size(1)
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
                    loss = F.cross_entropy(
                        logits_flat,
                        targets.view(-1),
                        ignore_index=IGNORE_INDEX,
                    )

                    gate_entropy_loss, loop_penalty = _compute_gate_regularizers(metadata, args)
                    if args.gate_entropy_weight > 0:
                        loss = loss + args.gate_entropy_weight * gate_entropy_loss
                    if args.loop_penalty_weight > 0:
                        loss = loss + args.loop_penalty_weight * loop_penalty

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                sample_count += inputs.size(0)
                avg_steps_accumulator += metadata.exit_steps.float().sum().item()

                predictions = logits.argmax(dim=1)
                for idx in range(inputs.size(0)):
                    target = targets[idx]
                    prediction = predictions[idx]
                    valid = target != IGNORE_INDEX
                    is_exact = bool(torch.equal(prediction[valid], target[valid])) if valid.any() else False
                    train_exact += int(is_exact)
                    train_examples += 1

                total_batches = len(train_loader)
                if total_batches > 0 and is_main_process and step % 10 == 0:
                    elapsed = time.time() - epoch_start
                    avg_step_time = elapsed / step
                    steps_completed = previous_total_steps + step
                    all_steps = len(train_loader) * args.epochs
                    remaining_steps = all_steps - steps_completed
                    elapsed_global = time.time() - global_start
                    avg_time_per_step_global = elapsed_global / max(steps_completed, 1)
                    eta = remaining_steps * avg_time_per_step_global
                    bar_length = 30
                    progress_ratio = steps_completed / all_steps if all_steps else 0
                    filled = int(bar_length * progress_ratio)
                    bar = "#" * filled + "-" * (bar_length - filled)
                    progress = 100.0 * progress_ratio
                    sys.stdout.write(f"\rEpoch {epoch} [{bar}] {progress:5.1f}% ETA {_format_eta(eta)}")
                    sys.stdout.flush()

            previous_total_steps += len(train_loader)
            if is_main_process:
                sys.stdout.write("\n")

            epoch_duration = time.time() - epoch_start if len(train_loader) > 0 else 0.0
            if distributed and dist.is_initialized():
                train_totals = torch.tensor(
                    [running_loss, sample_count, train_exact, train_examples, avg_steps_accumulator],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
                running_loss, sample_count, train_exact, train_examples, avg_steps_accumulator = train_totals.tolist()

            avg_train_loss = running_loss / max(sample_count, 1)
            train_acc = train_exact / max(train_examples, 1)
            avg_train_steps = avg_steps_accumulator / max(train_examples, 1)

            log_parts = [
                f"epoch={epoch}",
                f"train_loss={avg_train_loss:.4f}",
                f"train_acc={train_acc:.4f}",
                f"avg_train_steps={avg_train_steps:.2f}",
                f"epoch_time={epoch_duration:.1f}s",
            ]

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.learning_rate
            log_parts.append(f"lr={current_lr:.6f}")

            eval_loss = None
            eval_acc = None
            eval_steps = None
            visualizations = {}
            save_best_checkpoint = False
            if eval_loader is not None:
                eval_loss, eval_acc, eval_steps, visualizations = evaluate(
                    model,
                    eval_loader,
                    device,
                    args,
                    distributed=distributed,
                )
                if is_main_process:
                    log_parts.append(f"eval_loss={eval_loss:.4f}")
                    log_parts.append(f"eval_acc={eval_acc:.4f}")
                    log_parts.append(f"eval_steps={eval_steps:.2f}")
                if eval_acc is not None and eval_acc > best_eval_acc and args.best_save_path and is_main_process:
                    best_eval_acc = eval_acc
                    best_payload: Dict[str, Any] = {
                        "model_state": model_for_eval.state_dict(),
                        "config": vars(args),
                        "epoch": epoch,
                        "best_eval_accuracy": best_eval_acc,
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                    }
                    if scheduler is not None:
                        best_payload["scheduler_state"] = scheduler.state_dict()
                    best_path = Path(args.best_save_path)
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_payload, best_path)
                    save_best_checkpoint = True

            if distributed and dist.is_initialized():
                save_flag = torch.tensor(int(save_best_checkpoint), device=device)
                dist.broadcast(save_flag, src=0)
                save_best_checkpoint = bool(save_flag.item())

                best_acc_tensor = torch.tensor(best_eval_acc if is_main_process else 0.0, device=device)
                dist.broadcast(best_acc_tensor, src=0)
                if not is_main_process:
                    best_eval_acc = best_acc_tensor.item()

                if save_best_checkpoint:
                    dist.barrier()

            if is_main_process:
                print(" | ".join(log_parts))

            if wandb_run is not None and is_main_process:
                metrics = {
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "train/accuracy": train_acc,
                    "train/avg_steps": avg_train_steps,
                    "train/lr": current_lr,
                }
                if eval_loss is not None:
                    metrics.update(
                        {
                            "eval/loss": eval_loss,
                            "eval/accuracy": eval_acc,
                            "eval/avg_steps": eval_steps,
                        }
                    )
                    if (
                        ((epoch % args.vis_every) == 0 or epoch == args.epochs)
                        and visualizations
                        and eval_dataset is not None
                    ):
                        reverse_lookup = {v: k for k, v in eval_dataset.task_lookup.items()}
                        metrics["visualizations/eval"] = [
                            wandb.Image(v, mode="RGBA", caption=f"task {reverse_lookup.get(task_id, task_id)}")
                            for task_id, v in visualizations.items()
                        ]
                wandb.log(metrics, step=epoch)

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if distributed and dist.is_initialized():
            dist.barrier()

    if args.save_path and ((not distributed) or rank == 0):
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": model_for_eval.state_dict(),
            "config": vars(args),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        }
        torch.save(payload, save_path)

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
