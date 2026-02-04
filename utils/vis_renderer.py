
import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser("Visualize Attention & Logits Grid from Dumps")
    p.add_argument("--dump-path", type=str, required=True, help="Path to .pt dump file")
    p.add_argument("--out-dir", type=str, default="visualization_assets", help="Directory to save images")
    p.add_argument("--alpha", type=float, default=0.6, help="Heatmap opacity")
    p.add_argument("--only-heatmap", action="store_true", help="Draw only heatmap without image scaling")
    # Selection
    p.add_argument("--step", type=int, default=None, help="If set, only visualize this step index")
    return p.parse_args()

def get_valid_range(mask):
    """
    Extract valid range from attention mask (H, W).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return 0, mask.shape[0], 0, mask.shape[1]

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax + 1, cmin, cmax + 1

    plt.tight_layout()
    out_path = Path(out_dir) / f"{Path(dump_path).stem}_attn_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved attention grid to {out_path}")

def render_features_pca(all_steps_hidden, out_dir, dump_path, num_task_tokens, image_size, patch_size, rmin, rmax, cmin, cmax):
    """
    Render PCA of hidden states across steps.
    """
    try:
        from sklearn.decomposition import PCA
        use_sklearn = True
    except ImportError:
        print("sklearn not installed, falling back to torch.linalg.svd for PCA.")
        use_sklearn = False

    if not all_steps_hidden:
        return

    # Gather all tokens from all steps to fit PCA globally
    # hidden: (1, L, D) -> need spatial tokens
    
    all_spatial_tokens = []
    
    # meta
    ph = image_size // patch_size
    pw = image_size // patch_size
    
    for h_step in all_steps_hidden:
        # h_step: (1, L, D)
        spatial = h_step[0, num_task_tokens:, :] # (N, D)
        all_spatial_tokens.append(spatial)
        
    stacked = torch.cat(all_spatial_tokens, dim=0) # (Steps*N, D)
    
    # Fit PCA -> 3 components for RGB
    if use_sklearn:
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(stacked.numpy()) # (Steps*N, 3)
    else:
        # Torch Fallback
        X = stacked.float()
        mean = torch.mean(X, dim=0)
        X_centered = X - mean
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        V = Vh.T
        components = V[:, :3]
        transformed = (X_centered @ components).numpy()
    
    # Normalize to 0-1
    
    # Normalize to 0-1
    t_min = transformed.min(axis=0)
    t_max = transformed.max(axis=0)
    transformed = (transformed - t_min) / (t_max - t_min + 1e-8)
    
    # Reshape back to grids
    num_steps = len(all_steps_hidden)
    n_tokens = ph * pw
    
    fig, axes = plt.subplots(1, num_steps, figsize=(3 * num_steps, 3))
    if num_steps == 1: axes = [axes]
    
    for s_i in range(num_steps):
        # Extract this step's tokens
        step_pca = transformed[s_i*n_tokens : (s_i+1)*n_tokens, :] # (N, 3)
        
        # Reshape to (H, W, 3)
        grid_pca = step_pca.reshape(ph, pw, 3)
        
        # Upsample to image size
        grid_pca_tensor = torch.tensor(grid_pca).permute(2, 0, 1).unsqueeze(0) # (1, 3, ph, pw)
        grid_pca_up = F.interpolate(grid_pca_tensor, size=(image_size, image_size), mode='nearest')[0].permute(1, 2, 0).numpy()
        
        # Crop
        cropped = grid_pca_up[rmin:rmax, cmin:cmax]
        
        ax = axes[s_i]
        ax.imshow(cropped)
        ax.set_title(f"Step {s_i+1}")
        ax.axis('off')
        
    out_path = Path(out_dir) / f"{Path(dump_path).stem}_features_pca.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved PCA features to {out_path}")

def render_logit_entropy(all_steps_logits, out_dir, dump_path, image_size, rmin, rmax, cmin, cmax):
    """
    Render Entropy of logits per step.
    """
    if not all_steps_logits:
        return

    num_steps = len(all_steps_logits)
    fig, axes = plt.subplots(1, num_steps, figsize=(3 * num_steps, 3))
    if num_steps == 1: axes = [axes]

    for s_i in range(num_steps):
        # logits: (1, C, H, W)
        logits = all_steps_logits[s_i]
        
        # Probabilities
        probs = F.softmax(logits, dim=1)
        
        # Entropy = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1) # (1, H, W)
        
        heatmap = entropy[0].numpy() # (H, W)
        
        # Normalize per step or globally? Locally for contrast
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Upscale if needed? Logits are usually same reso as image in ARC_LoopViT if projected
        # Model output is (1, C, img_size, img_size).
        
        cropped = heatmap[rmin:rmax, cmin:cmax]
        
        ax = axes[s_i]
        im = ax.imshow(cropped, cmap='magma')
        ax.set_title(f"Entropy S{s_i+1}")
        ax.axis('off')

    out_path = Path(out_dir) / f"{Path(dump_path).stem}_logit_entropy.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Logit Entropy to {out_path}")

def render_dump(dump_path, out_dir, alpha=0.6, only_heatmap=False, target_step=None):
    print(f"Loading dump: {dump_path}")
    try:
        dump = torch.load(dump_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load dump {dump_path}: {e}")
        return
    
    # Expected structure:
    # {
    #   "inputs": (1, C, H, W),
    #   "attention_mask": (1, H, W),
    #   "all_steps_attn": List[List[Dict]], # [step][layer] -> {'scores': (1, Heads, N, N), ...}
    #   "all_steps_hidden": List[Tensor], # [step] -> (1, L, D)
    #   "all_steps_logits": List[Tensor], # [step] -> (1, C, H, W)
    #   "logits": (1, C, H, W),
    #   "meta": {...}
    # }
    
    inputs = dump["inputs"]
    attention_mask = dump.get("attention_mask")
    all_steps_attn = dump.get("all_steps_attn", [])
    all_steps_hidden = dump.get("all_steps_hidden", [])
    all_steps_logits = dump.get("all_steps_logits", [])
    meta = dump.get("meta", {})
    
    image_size = meta.get("image_size", 64)
    patch_size = meta.get("patch_size", 2)
    num_task_tokens = meta.get("num_task_tokens", 1)
    
    if attention_mask is not None:
        rmin, rmax, cmin, cmax = get_valid_range(attention_mask[0])
    else:
        rmin, rmax, cmin, cmax = 0, image_size, 0, image_size
        
    print(f"Valid Region: [{rmin}:{rmax}, {cmin}:{cmax}]")
    
    # 1. Attention Grid
    if all_steps_attn:
        try:
             # Target Query Pixel: Center of valid region
            center_r = (rmin + rmax) // 2
            center_c = (cmin + cmax) // 2
            
            ph = image_size // patch_size
            pw = image_size // patch_size
            patch_r = center_r // patch_size
            patch_c = center_c // patch_size
            query_index_patch = patch_r * pw + patch_c
            query_index_seq = num_task_tokens + query_index_patch
            
            print(f"Querying Attention for Pixel ({center_r}, {center_c}) -> Seq Idx {query_index_seq}")

            if inputs.ndim == 3: # (B, H, W)
                 input_img = inputs[0].float().numpy()
                 input_img = input_img / (input_img.max() + 1e-8)
                 input_img = np.stack([input_img]*3, axis=-1)
            else:
                 input_img = np.zeros((image_size, image_size, 3))

            input_img_cropped = input_img[rmin:rmax, cmin:cmax]

            num_steps = len(all_steps_attn)
            num_layers = len(all_steps_attn[0])
            
            if target_step is not None:
                steps_to_viz = [target_step]
                n_rows = 1
            else:
                steps_to_viz = list(range(num_steps))
                n_rows = num_steps
                
            n_cols = num_layers
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
            if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
            elif n_rows == 1: axes = axes.reshape(1, -1)
            elif n_cols == 1: axes = axes.reshape(-1, 1)
            
            for row_idx, step_idx in enumerate(steps_to_viz):
                step_data = all_steps_attn[step_idx] # List of layer dicts
                
                for col_idx, layer_data in enumerate(step_data):
                    # layer_data = {'scores': (1, H, N, N), ...}
                    if "scores" not in layer_data: continue
                    scores = layer_data["scores"][0] # (Heads, N, N)
                    
                    # Mean over heads
                    attn_map = scores.mean(dim=0) # (N, N)
                    
                    # Query row
                    if query_index_seq >= attn_map.shape[0]:
                         print(f"Warning: Query index {query_index_seq} out of bounds {attn_map.shape}")
                         continue

                    row_attn = attn_map[query_index_seq, :] # (N,)
                    
                    # Extract pixel tokens
                    pixel_attn = row_attn[num_task_tokens:]
                    
                    # Reshape to patch grid
                    if pixel_attn.numel() != ph * pw:
                         # Mismatch?
                         continue
                         
                    patch_attn_grid = pixel_attn.view(ph, pw)
                    
                    # Upsample
                    patch_attn_grid = patch_attn_grid.unsqueeze(0).unsqueeze(0)
                    heatmap = F.interpolate(patch_attn_grid, size=(image_size, image_size), mode='nearest')[0,0]
                    
                    # Crop
                    heatmap_cropped = heatmap[rmin:rmax, cmin:cmax]
                    
                    # Normalize
                    heatmap_np = heatmap_cropped.numpy()
                    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
                    
                    ax = axes[row_idx, col_idx]
                    
                    if not only_heatmap:
                         ax.imshow(input_img_cropped, alpha=1.0)
                    
                    ax.imshow(heatmap_np, cmap='jet', alpha=alpha)
                    
                    if row_idx == 0:
                        ax.set_title(f"Layer {col_idx}")
                    if col_idx == 0:
                        ax.set_ylabel(f"Step {step_idx+1}")
                    
                    ax.axis('off')
        
            plt.tight_layout()
            out_path = Path(out_dir) / f"{Path(dump_path).stem}_attn_grid.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved attention grid to {out_path}")
        except Exception as e:
            print(f"Error rendering attention: {e}")

    # 2. Features PCA
    if all_steps_hidden:
        try:
            render_features_pca(all_steps_hidden, out_dir, dump_path, num_task_tokens, image_size, patch_size, rmin, rmax, cmin, cmax)
        except Exception as e:
            print(f"Error rendering PCA: {e}")

    # 3. Logit Entropy
    if all_steps_logits:
        try:
            render_logit_entropy(all_steps_logits, out_dir, dump_path, image_size, rmin, rmax, cmin, cmax)
        except Exception as e:
            print(f"Error rendering Logit Entropy: {e}")

def main():
    args = parse_args()
    if os.path.isdir(args.dump_path):
        # Process all .pt in dir
        files = list(Path(args.dump_path).glob("*.pt"))
        for f in files:
            render_dump(f, args.out_dir, args.alpha, args.only_heatmap, args.step)
    else:
        render_dump(args.dump_path, args.out_dir, args.alpha, args.only_heatmap, args.step)

if __name__ == "__main__":
    main()
