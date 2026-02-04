import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List
from PIL import Image

# Reuse existing analysis tool
from utils.analyze_prediction import analyze_data
# Reuse existing wandb vis for color mapping
from utils.wandb_vis import COLORS_UINT8, PIL_IMG_MAG, BORDER_SIZE

IGNORE_INDEX = 10
PAD_INDEX = 11

def _to_numpy(tensor_like):
    if tensor_like is None:
        return None
    if isinstance(tensor_like, np.ndarray):
        return tensor_like
    if hasattr(tensor_like, "detach"):
        tensor_like = tensor_like.detach()
    if hasattr(tensor_like, "cpu"):
        tensor_like = tensor_like.cpu()
    if hasattr(tensor_like, "numpy"):
        tensor_like = tensor_like.numpy()
    return np.array(tensor_like)

def _identity_transform(grid):
    return grid

def _ensure_list(grid):
    if grid is None:
        return None
    if isinstance(grid, list):
        return grid
    return grid.tolist()

def _build_task_file_lookup(dataset_root: Optional[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    if dataset_root is None:
        return lookup
    for json_path in dataset_root.glob("*.json"):
        lookup[json_path.stem] = json_path
    return lookup

def _resolve_color_inverse_map(
    task_name: str,
    task_file_lookup: Dict[str, Path],
    cache: Dict[str, Optional[Dict[int, int]]],
) -> Optional[Dict[int, int]]:
    if task_name in cache:
        return cache[task_name]

    task_path = task_file_lookup.get(task_name)
    if task_path is None or not task_path.exists():
        cache[task_name] = None
        return None

    try:
        with task_path.open("r") as fh:
            payload = json.load(fh)
    except Exception:
        cache[task_name] = None
        return None

    color_map = payload.get("augmentation", {}).get("color_map")
    if not color_map:
        cache[task_name] = None
        return None

    normalized = {int(k): int(v) for k, v in color_map.items()}
    inverse_map = {v: k for k, v in normalized.items()}
    cache[task_name] = inverse_map
    return inverse_map

def _apply_color_map_to_grid(grid, inverse_color_map: Optional[Dict[int, int]]):
    if grid is None or not inverse_color_map:
        return grid

    if isinstance(grid, np.ndarray):
        iterable = grid.tolist()
    else:
        iterable = grid

    return [
        [inverse_color_map.get(value, value) for value in row]
        for row in iterable
    ]

def _undo_eval_rot_grid(grid, suffix: str):
    if grid is None or not suffix:
        return grid

    array = np.asarray(grid)
    if array.ndim < 2 or array.size == 0:
        return _ensure_list(array)

    if "rotate_90_" in suffix:
        transformed = np.rot90(array, k=3)
    elif "rotate_180_" in suffix:
        transformed = np.rot90(array, k=2)
    elif "rotate_270_" in suffix:
        transformed = np.rot90(array, k=1)
    elif "flip_0_" in suffix:
        transformed = np.flipud(array)
    elif "flip_1_" in suffix:
        transformed = np.fliplr(array)
    else:
        transformed = array
    return transformed.tolist() if isinstance(transformed, np.ndarray) else transformed

def get_eval_rot_transform_resolver() -> Callable[[str], Tuple[str, Callable]]:
    def resolver(task_name: str) -> Tuple[str, Callable]:
        if "_" not in task_name:
            return task_name, _identity_transform
        base, suffix = task_name.split("_", 1)

        def undo_fn(grid):
            return _undo_eval_rot_grid(grid, suffix)

        return base, undo_fn
    return resolver

def grid_to_image_array(grid, max_h=32, max_w=32):
    """Convert a 2D grid to a colored RGB image array."""
    grid = _to_numpy(grid)
    if grid is None or grid.size == 0:
         return np.zeros((max_h * PIL_IMG_MAG, max_w * PIL_IMG_MAG, 3), dtype=np.uint8)
    
    grid = grid.astype(int, copy=False)
    
    h, w = grid.shape
    mult_factor = min((max_h*PIL_IMG_MAG)//max(h, 1), (max_w*PIL_IMG_MAG)//max(w, 1))
    mult_factor = max(mult_factor, 1)

    grid = np.repeat(np.repeat(grid, mult_factor, axis=0), mult_factor, axis=1)
    safe_indices = np.where((grid < 0) | (grid >= COLORS_UINT8.shape[0]), 0, grid)
    
    # Map to colors
    color_grid = COLORS_UINT8[safe_indices, :].copy() 

    # Add grid lines
    line_color = np.array([50, 50, 50], dtype=np.uint8)
    color_grid[::mult_factor, :, :] = line_color
    color_grid[mult_factor-1::mult_factor, :, :] = line_color
    color_grid[:, ::mult_factor, :] = line_color
    color_grid[:, mult_factor-1::mult_factor, :] = line_color

    # Pad to max size for uniform strip
    final_h, final_w = max_h * PIL_IMG_MAG, max_w * PIL_IMG_MAG
    padded = np.full((final_h, final_w, 3), 0, dtype=np.uint8)
    
    # Center it
    ph, pw, _ = color_grid.shape
    start_h = (final_h - ph) // 2
    start_w = (final_w - pw) // 2
    
    ph = min(ph, final_h)
    pw = min(pw, final_w)
    
    padded[start_h:start_h+ph, start_w:start_w+pw, :] = color_grid[:ph, :pw, :]
    
    return padded

def prob_diff_to_image_array(prob_diff_map, max_h=32, max_w=32):
    """
    Visualizes probability difference as a heatmap.
    prob_diff_map: 2D array of floats (diff magnitudes per pixel).
    """
    if prob_diff_map is None:
        return np.full((max_h * PIL_IMG_MAG, max_w * PIL_IMG_MAG, 3), 128, dtype=np.uint8) 

    # Normalize to 0-255 for heatmap
    # Range is technically [0, 1] or [0, sqrt(2)]. 
    # Let's fix scale [0, 1] to make comparable across steps
    
    heatmap = np.clip(prob_diff_map * 255.0 * 2.0, 0, 255).astype(np.uint8) # Amplify small diffs by 2x
    
    # White (0) -> Red (High)
    r_ch = np.full_like(heatmap, 255)
    gb_ch = 255 - heatmap
    rgb_grid = np.stack([r_ch, gb_ch, gb_ch], axis=-1)
    
    h, w = prob_diff_map.shape
    mult_factor = min((max_h*PIL_IMG_MAG)//max(h, 1), (max_w*PIL_IMG_MAG)//max(w, 1))
    mult_factor = max(mult_factor, 1)
    
    rgb_grid = np.repeat(np.repeat(rgb_grid, mult_factor, axis=0), mult_factor, axis=1)
    
    # Pad
    final_h, final_w = max_h * PIL_IMG_MAG, max_w * PIL_IMG_MAG
    padded = np.full((final_h, final_w, 3), 255, dtype=np.uint8) # White background
    
    ph, pw, _ = rgb_grid.shape
    start_h = (final_h - ph) // 2
    start_w = (final_w - pw) // 2
    ph = min(ph, final_h)
    pw = min(pw, final_w)
    
    padded[start_h:start_h+ph, start_w:start_w+pw, :] = rgb_grid[:ph, :pw, :]
    
    return padded

def compute_prob_diff(logits_prev, logits_curr):
    """
    Computes difference between Softmax Probabilities.
    logits: (C, H, W)
    Returns: 2D map of differences (H, W).
    """
    # Softmax
    probs_p = F.softmax(logits_prev.float(), dim=0) # (C, H, W)
    probs_c = F.softmax(logits_curr.float(), dim=0) # (C, H, W)
    
    # Diff L2 Norm per pixel
    # sum((p_c - p_p)^2) per pixel
    diff = probs_c - probs_p
    diff_norms = torch.norm(diff, p=2, dim=0) # (H, W)
    
    diff_map = diff_norms.cpu().numpy()
    mean_diff = diff_norms.mean().item()
    
    return diff_map, mean_diff


def create_filmstrip(input_grid, intermediates, final_grid, target_grid, save_path, diff_infos=None, padding_width=10):
    """
    Stitches grids into a horizontal filmstrip. 
    Row 2: Probability Difference Heatmaps.
    """
    # ROW 1: Input -> Steps -> Target
    row1_images = []
    
    row1_images.append(grid_to_image_array(input_grid))
    for grid in intermediates:
        row1_images.append(grid_to_image_array(grid))
    
    if target_grid is not None:
         row1_images.append(grid_to_image_array(target_grid))
    
    # Row 2: Empty -> Diffs -> Empty
    row2_images = []
    if diff_infos:
        h, w, c = row1_images[0].shape
        empty_block = np.full((h, w, c), 255, dtype=np.uint8) # White
        row2_images.append(empty_block) # Under Input
        
        for d_map, d_val in diff_infos:
            if d_map is None:
                row2_images.append(empty_block)
            else:
                row2_images.append(prob_diff_to_image_array(d_map))
        
        if target_grid is not None:
             row2_images.append(empty_block) 

    def pad_row(img_list):
        if not img_list: return None
        h, w, c = img_list[0].shape
        spacer = np.full((h, padding_width, c), 255, dtype=np.uint8)
        padded = []
        for i, img in enumerate(img_list):
            padded.append(img)
            if i < len(img_list) - 1:
                padded.append(spacer)
        return np.concatenate(padded, axis=1)

    strip_row1 = pad_row(row1_images)
    
    if row2_images:
        strip_row2 = pad_row(row2_images)
        h1, w1, c1 = strip_row1.shape
        v_spacer = np.full((padding_width, w1, c1), 255, dtype=np.uint8)
        
        if strip_row2.shape[1] != w1:
             if strip_row2.shape[1] < w1:
                 pad_w = w1 - strip_row2.shape[1]
                 strip_row2 = np.pad(strip_row2, ((0,0),(0,pad_w),(0,0)), constant_values=255)
             else:
                 strip_row2 = strip_row2[:, :w1, :]

        filmstrip = np.concatenate([strip_row1, v_spacer, strip_row2], axis=0)
    else:
        filmstrip = strip_row1
    
    Image.fromarray(filmstrip).save(save_path)


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
    eval_split: str,
    attempt_nums: int = 10,
    task_transform_resolver: Optional[Callable[[str], Tuple[str, Callable]]] = None,
    border_size: int = 1,
    fix_scale_factor: int = 1,
    disable_translation: bool = False,
    if_fix_scale: bool = False,
    save_name = "ttt_eval",
    task_type: str = "ARC-AGI",
    forward_fn: Optional[Callable] = None,
    visualize_loop_steps: bool = False,
) -> None:
    model.eval()
    
    answer_sets = {} 
    convergence_stats = {}

    transform_cache: Dict[str, Tuple[str, Callable]] = {}

    dataset = getattr(loader, "dataset", None)
    task_file_lookup: Dict[str, Path] = {}
    color_inverse_cache: Dict[str, Optional[Dict[int, int]]] = {}
    if dataset is not None:
        dataset.enable_translation()
        if disable_translation:
            dataset.disable_translation()
            attempt_nums = 1
        if if_fix_scale:
            dataset.disable_resolution_augmentation(fix_scale_factor=fix_scale_factor)
        else:
            dataset.enable_resolution_augmentation()

        existing_lookup = getattr(dataset, "_task_file_lookup", None)
        if existing_lookup is None:
            dataset_root = getattr(dataset, "root", None)
            dataset_root = Path.joinpath(dataset_root, "data")
            tasks_path = Path.joinpath(dataset_root, eval_split)
            if not tasks_path.exists():
                pass 
            else:
                existing_lookup = _build_task_file_lookup(tasks_path)
            setattr(dataset, "_task_file_lookup", existing_lookup)
        task_file_lookup = existing_lookup or {}
    else:
        if disable_translation:
            attempt_nums = 1

    save_path = Path(save_name)
    if not save_path.is_absolute():
        save_path = Path("outputs") / save_path

    viz_dir = save_path / "visualization"
    if visualize_loop_steps:
        viz_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(attempt_nums):
        for batch in tqdm(loader):
            inputs = batch["inputs"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            task_ids = batch["task_ids"].to(device)
            offsets = batch['offset'].to(device)
            scale_factors = batch['scale_factors'].to(device)
            if "targets" in batch:
                targets = batch["targets"].to(device)
            else:
                targets = None

            intermediates_logits = []
            metadata = None
            
            if forward_fn is not None:
                ret = forward_fn(model, inputs, task_ids, attention_mask)
            else:
                ret = model(inputs, task_ids, attention_mask=attention_mask)
            
            if isinstance(ret, tuple):
                if len(ret) == 3:
                     # intermediates_list = [logits_t1, logits_t2 ...]
                     logits, metadata, intermediates_list = ret 
                else:
                     logits, metadata = ret
                     intermediates_list = []
            else:
                logits = ret
                metadata = None
                intermediates_list = []

            batch_size = logits.size(0)
            preds = logits.argmax(dim=1).cpu()  # Final predictions
            
            all_steps_preds_raw = [] 
            all_steps_logits = []
            
            if intermediates_list:
                step_logits_list = intermediates_list # List of tensors
                
                acc_steps = [l.argmax(dim=1).cpu() for l in step_logits_list]
                
                if metadata is not None and hasattr(metadata, "exit_steps"):
                    exit_steps = metadata.exit_steps.cpu() 
                else:
                    exit_steps = torch.full((batch_size,), len(step_logits_list), dtype=torch.long)
                
                for b in range(batch_size):
                    sample_steps = []
                    sample_logits = []
                    exited_at_step_val = exit_steps[b].item()
                    
                    num_avail_steps = len(acc_steps)
                    final_frozen_pred = preds[b]
                    
                    for s_idx in range(num_avail_steps):
                        current_step_val = s_idx + 1
                        if current_step_val <= exited_at_step_val:
                            sample_steps.append(acc_steps[s_idx][b])
                            sample_logits.append(step_logits_list[s_idx][b])
                        else:
                            # Freeze
                            sample_steps.append(final_frozen_pred)
                            # Logits freeze to last valid
                            last_valid_idx = max(0, exited_at_step_val - 1)
                            sample_logits.append(step_logits_list[last_valid_idx][b])
                    
                    all_steps_preds_raw.append(sample_steps)
                    all_steps_logits.append(sample_logits)
            else:
                for b in range(batch_size):
                    all_steps_preds_raw.append([preds[b]])
                    all_steps_logits.append([None]) 

            example_indices = batch["example_indices"].cpu()

            for idx, task_name in enumerate(batch["task_names"]):
                scale_factor = scale_factors[idx].item()
                if task_transform_resolver:
                    if task_name not in transform_cache:
                        transform_cache[task_name] = task_transform_resolver(task_name)
                    base_task_name, undo_fn = transform_cache[task_name]
                else:
                    base_task_name, undo_fn = task_name, _identity_transform

                cur_index = example_indices[idx].item()
                color_inverse_map = _resolve_color_inverse_map(
                    task_name, task_file_lookup, color_inverse_cache
                )
                
                def process_grid(raw_pred):
                    offset_x, offset_y = offsets[idx]
                    np_predict = np.array(raw_pred).reshape(img_size, img_size)
                    np_predict_grid = np_predict[offset_y:, offset_x:]
                    len_x, len_y = 0, 0
                    while len_x < np_predict_grid.shape[1] and np_predict_grid[0][len_x] != PAD_INDEX:
                        len_x += 1
                    while len_y < np_predict_grid.shape[0] and np_predict_grid[len_y][0] != PAD_INDEX:
                        len_y += 1
                    predict_grid = np_predict_grid[:len_y, :len_x].tolist()
                    predict_grid = undo_fn(predict_grid)
                    if scale_factor > 1:
                        pass
                    predict_grid = _apply_color_map_to_grid(predict_grid, color_inverse_map)
                    return predict_grid

                try:
                    # 1. Final Grid
                    final_grid = process_grid(preds[idx])
                    task_predictions = answer_sets.setdefault("final", {}).setdefault(base_task_name, {})
                    if cur_index not in task_predictions: task_predictions[cur_index] = []
                    task_predictions[cur_index].append(final_grid)
                    
                    # 2. Steps & Prob Diff
                    if not all_steps_logits[idx] or all_steps_logits[idx][0] is None:
                         continue
                    
                    sample_steps_raw = all_steps_preds_raw[idx]
                    sample_logits = all_steps_logits[idx]
                    
                    processed_steps = []
                    diff_infos = [] 
                    sample_stats = []
                    
                    for s_i, raw_s in enumerate(sample_steps_raw):
                        step_grid = process_grid(raw_s)
                        processed_steps.append(step_grid)
                        
                        step_key = f"step_{s_i+1}"
                        task_step_preds = answer_sets.setdefault(step_key, {}).setdefault(base_task_name, {})
                        if cur_index not in task_step_preds: task_step_preds[cur_index] = []
                        task_step_preds[cur_index].append(step_grid)

                        # Output Probability Diff
                        curr_logits = sample_logits[s_i]
                        d_val = 0.0
                        d_map = None
                        
                        if s_i == 0:
                            d_val = -1.0 # First step, no prev
                        else:
                            prev_logits = sample_logits[s_i-1]
                            d_map, d_val = compute_prob_diff(prev_logits, curr_logits)
                        
                        diff_infos.append((d_map, d_val))
                        sample_stats.append({"step": s_i+1, "prob_diff": d_val})

                    # Record Stats: Just key by task name for now in memory
                    task_stats = convergence_stats.setdefault(base_task_name, {})
                    if cur_index not in task_stats: task_stats[cur_index] = []
                    task_stats[cur_index] = sample_stats 

                    # 3. Visualizations
                    if visualize_loop_steps:
                        input_grid = process_grid(inputs[idx].cpu())
                        target_grid = None
                        if targets is not None:
                             target_grid = process_grid(targets[idx].cpu())
                        
                        filename = f"{base_task_name}_ex{cur_index}.png"
                        create_filmstrip(input_grid, processed_steps, final_grid, target_grid, viz_dir / filename, diff_infos=diff_infos)

                except Exception as e:
                    print(f"Error processing {task_name} ex {cur_index}: {e}")

    # Save Predictions (standard)
    for set_name, ans_set in answer_sets.items():
        if not ans_set: continue
        out_folder = save_path / set_name
        out_folder.mkdir(parents=True, exist_ok=True)
        for t_name, t_data in ans_set.items():
             with open(out_folder / f'{t_name}_predictions.json', 'w') as f:
                 json.dump(t_data, f)
    
    # [MODIFIED] Save Convergence Stats SEPARATELY per task
    if convergence_stats:
        stats_dir = save_path / "convergence_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        for t_name, t_stats in convergence_stats.items():
            # t_stats is {ex_index: stats}
            # Save as json
            with open(stats_dir / f"{t_name}_stats.json", "w") as f:
                json.dump(t_stats, f, indent=2)
