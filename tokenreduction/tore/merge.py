import torch
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union, Callable
import math


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
):
    
    if len(metric.shape) == 2:
        metric = metric[None,...]

    # We can only reduce by a maximum of 50% tokens
    # T = metric.shape[1]
    
    t = metric.shape[1]
    r = min(r, t//2)
    
    if r <= 0:
        raise ValueError("The number of tokens to be merged is non-positive.")

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True, stable=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if len(x.shape) == 2:
            x.unsqueeze_(0)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        
        
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # Not deterministic since scatter_reduce is not deterministic (not commutative)
        # dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        # Deterministic version
        num_dst_tokens = dst.shape[1]
        idx_flat = dst_idx.squeeze(-1)
        assignment_mat = F.one_hot(idx_flat, num_classes=num_dst_tokens).to(dtype=src.dtype)
        merged_sum = torch.bmm(assignment_mat.transpose(1, 2), src)
        if mode == "sum": # 
            dst = dst + merged_sum
        elif mode == "mean":
            counts = assignment_mat.transpose(1, 2).sum(dim=-1, keepdim=True)
            dst = (dst + merged_sum) / (1 + counts)
        elif mode == "amax": # commutative
             dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="amax")
        
        return torch.cat([unm, dst], dim=1)
    
    
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


# distinctive_anchor_merging
def distinctive_anchor_merging(
    metric: torch.Tensor,
    n_merged: int,
    class_token: bool = False,
):
    """
    DPC-style merging.
    """
    B, N, C = metric.shape
    n_src = N - n_merged # r
    
    if n_merged <= 0:
         raise ValueError("Too many tokens to reduce.")

    with torch.no_grad():
        metric_norm = metric / metric.norm(dim=-1, keepdim=True)
        
        sim_matrix = metric_norm @ metric_norm.transpose(-1, -2) # (B, N, N)

        density = sim_matrix.sum(dim=-1) # (B, N)
        better_mask = density.unsqueeze(1) < density.unsqueeze(2) 
        
        masked_sim = sim_matrix.clone()
        masked_sim[~better_mask] = -1.0 
        
        max_sim_to_better, _ = masked_sim.max(dim=-1)
        
        is_global_leader = (max_sim_to_better == -1.0)
        max_sim_to_better[is_global_leader] = 0.0 
        diversity_factor = 1.0 - max_sim_to_better
        final_score = density * diversity_factor

        if class_token:
            final_score[..., 0] = float('inf') 
        _, dst_idx = final_score.topk(k=n_merged, dim=-1)
        dst_idx, _ = dst_idx.sort(dim=-1) # (B, K)


        mask = torch.ones(B, N, dtype=torch.bool, device=metric.device)
        mask.scatter_(1, dst_idx, False) 
        
        all_indices = torch.arange(N, device=metric.device).expand(B, N)
        src_idx = all_indices[mask].view(B, n_src)
        
        src_sims = sim_matrix.gather(1, src_idx.unsqueeze(-1).expand(B, n_src, N))
        src_to_dst_sims = src_sims.gather(2, dst_idx.unsqueeze(1).expand(B, n_src, n_merged))
        
        best_match_rel_idx = src_to_dst_sims.argmax(dim=-1) # (B, r)
        assignment_idx = best_match_rel_idx.unsqueeze(-1)


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        n, _, c = x.shape
        dst = x.gather(1, dst_idx.unsqueeze(-1).expand(n, n_merged, c))
        src = x.gather(1, src_idx.unsqueeze(-1).expand(n, n_src, c))
        
        assignment_mat = F.one_hot(assignment_idx.squeeze(-1), num_classes=n_merged).to(dtype=x.dtype)
        merged_sum = torch.bmm(assignment_mat.transpose(1, 2), src) # (B, K, C)
        
        if mode == "sum":
            dst = dst + merged_sum
        elif mode == "mean":
            counts = assignment_mat.transpose(1, 2).sum(dim=1, keepdim=True)
            dst = (dst + merged_sum) / (1 + counts)
        elif mode == "amax":
            dst = dst + merged_sum
            dst = dst.clamp(max=1.0) 
            
        return dst 

    return merge, None


def uniform_grid_clustering(
    metric: torch.Tensor, # (B, T, D)
    n_merged: int,               # Target Number of tokens (e.g., 32)
    class_token: bool = False,
):
    """
    Args:
        metric: Similarity metric tensor (usually keys or features)
        r: The final number of tokens to remain (Target Count)
        class_token: Whether the first token is a class token (preserved)
    """
    if len(metric.shape) == 2:
        metric = metric[None, ...]
    
    B, T, C = metric.shape

    if class_token:
        protected = 1
        metric = metric[:, 1:, :]
    
    # Ensure r is valid (cannot have more targets than available tokens)
    num_clusters = min(n_merged, T)
    if num_clusters <= 0:
        raise ValueError("Target number of tokens must be positive.")

    with torch.no_grad():
        # 2. Select Target Indices (Uniform Sampling / Grid)
        # VisionZip logic: stride based selection
        step = max(1, T // num_clusters)
        
        # Indices relative to metric
        target_indices = torch.arange(0, T, step, device=metric.device)[:num_clusters]
        
        # Indices relative to metric that are NOT targets (to be merged)
        # Using boolean mask is efficient
        is_target = torch.zeros(T, dtype=torch.bool, device=metric.device)
        is_target[target_indices] = True
        
        # (src -> target) assignment
        # Normalize for Cosine Similarity
        metric_norm = metric / metric.norm(dim=-1, keepdim=True)
        
        target_feats = metric_norm[:, target_indices, :]  # (B, num_clusters, C)
        src_feats = metric_norm[:, ~is_target, :]         # (B, num_src, C)
        
        # Calculate Similarity: (B, num_src, num_clusters)
        sim = src_feats @ target_feats.transpose(-1, -2)
        
        # For each source token, find the index of the best matching target token (0 ~ num_clusters-1)
        # assign_idx shape: (B, num_src)
        assign_idx = sim.argmax(dim=-1)

        # Pre-calculate counts for Mean pooling
        # We need to know how many src tokens are assigned to each target
        # (B, num_src, num_clusters)
        num_src = src_feats.shape[1]
        assign_one_hot = F.one_hot(assign_idx, num_classes=num_clusters).to(dtype=metric.dtype)
        
        # (B, num_clusters) - How many sources merged into each target
        src_counts = assign_one_hot.sum(dim=1) 
        
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        if len(x.shape) == 2:
            x.unsqueeze_(0)
            
        # Separate Class Token if needed
        if class_token:
            cls_token = x[:, :1, :]
            x_content = x[:, 1:, :]
        else:
            x_content = x

        B, _, C = x_content.shape
        
        # Split content into Targets and Sources
        x_target = x_content[:, target_indices, :] # (B, num_clusters, C)
        x_src = x_content[:, ~is_target, :]        # (B, num_src, C)

        # Aggregate Sources into Targets
        # merged_sum: (B, num_clusters, C)
        merged_sum = torch.bmm(assign_one_hot.transpose(1, 2).to(dtype=x_src.dtype), x_src)

        if mode == "sum":
            out = x_target + merged_sum
        elif mode == "mean":
            # True Mean: (Target_Value + Sum_of_Sources) / (1 + Num_Sources)
            # The target itself counts as 1
            total_counts = 1 + src_counts.unsqueeze(-1) # (B, num_clusters, 1)
            out = (x_target + merged_sum) / total_counts
        else:
            # Fallback to sum or implement amax if needed
            out = x_target + merged_sum

        # Concatenate Class Token back if it existed
        if class_token:
            out = torch.cat([cls_token, out], dim=1)
            
        return out

    return merge, None
    


def merge_wavg(
    merge: Callable, 
    x: torch.Tensor, 
    size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, size 

def merge_source(
    merge: Callable, 
    x: torch.Tensor, 
    source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source



def sort_rows_by_first_source(merging_pattern: torch.Tensor):
    """
    merging_pattern: (T, 576)  # row = destination, col = source
    - Each column is one-hot vector
    - The columns that have 1 in the same row are merged

    Returns:
        merged_sorted: (T, 576)  # 
        perm: (T,)               # new = old[perm]
    """
    
    first_idx = merging_pattern.argmax(dim=1)  # (T,)
    perm = torch.argsort(first_idx, stable=True)  # (T,)
    merged_sorted = merging_pattern[perm] 
    
    return merged_sorted, perm, first_idx