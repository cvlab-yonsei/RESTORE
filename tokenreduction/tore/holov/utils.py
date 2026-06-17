import torch


def HoloV(
    image_tokens, 
    attention, 
    num_patches=16, 
    new_image_token_num=64, 
    esp=1e-6
    ):
    
    B, N, D = image_tokens.shape
    device = image_tokens.device
    alpha = 0.09
    pruned_image_tokens_list = []

    for b in range(B):
        image_token = image_tokens[b]  # [N, D]
        image_attention = attention[b]  # [N]

        # Calculate dynamic patch size - handle uneven divisions
        patch_size = N // num_patches # 36 for 576 vis tokens
        remainder = N % num_patches # 0 for 576 vis tokens

        # Create patches with potentially uneven sizes
        image_tokens_patches = []
        attention_patches = []
        start_idx = 0

        for p in range(num_patches): # for each token in each grid
            # Last few patches get an extra token if there's a remainder
            current_patch_size = patch_size + (1 if p < remainder else 0)
            end_idx = start_idx + current_patch_size

            if current_patch_size > 0:  # Skip empty patches
                image_tokens_patches.append(image_token[start_idx:end_idx]) # Not 2D grid - inconsistent to the paper Fig.7 (middle)
                attention_patches.append(image_attention[start_idx:end_idx]) 

            start_idx = end_idx

        # Process each patch separately
        patch_scores = []
        all_patches = []

        for p in range(len(image_tokens_patches)): # for each patch
            patch_tokens = image_tokens_patches[p]  # [current_patch_size, D]: (16, D)
            patch_attn = attention_patches[p]  # [current_patch_size]: (16,)
            current_patch_size = len(patch_tokens)

            if current_patch_size <= 1:
                # If patch has only one token or is empty, handle specially
                patch_scores.append(patch_attn.mean() if len(patch_attn) > 0 else torch.tensor(0.0, device=device))
                all_patches.append(patch_tokens)
                continue

            with torch.no_grad():
                # Normalize patch tokens
                F_normalized = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + esp) # (16, D)

                # Compute similarity matrix
                S = torch.mm(F_normalized, F_normalized.transpose(0, 1)) # (16, 16)

                # Create eye mask of appropriate size
                eye_mask = 1 - torch.eye(current_patch_size, device=device) # shape: (16, 16)
                S_masked = S * eye_mask # ()

                # Compute mean and variance
                valid_entries = current_patch_size - 1 # 15
                mean_sim = S_masked.sum(dim=1) / valid_entries # (16,)
                var_sim = ((S_masked - mean_sim.unsqueeze(1))**2).sum(dim=1) / valid_entries # (16,)

                # Scale attention
                patch_attn_scaled = patch_attn * 1e3 # No explanation in the paper (Eqn. 3)

                # Scale variance
                var_scaling = (torch.mean(torch.abs(patch_attn_scaled)) / 
                              (torch.mean(torch.abs(var_sim)) + esp))
                var_sim_scaled = var_sim * var_scaling

                # Calculate token scores: Eqn. 3
                token_scores =  patch_attn_scaled + alpha * var_sim_scaled # (16,)

                # Compute patch score
                patch_score = token_scores.mean() # (1,)
                patch_scores.append(patch_score)
                all_patches.append(patch_tokens)

        # Convert to tensor
        patch_scores = torch.stack(patch_scores) if patch_scores else torch.zeros(0, device=device) # (16,)

        # Allocate new tokens based on scores
        if len(patch_scores) > 0:
            weights = (patch_scores ) / ((patch_scores).sum() + esp) # (16,)
            allocated = (weights * new_image_token_num).floor().long() # (16,)

            # Distribute remaining tokens
            remaining = new_image_token_num - allocated.sum() # scalar
            if remaining > 0 and len(weights) > 0:
                _, indices = torch.topk(weights, k=min(remaining.item(), len(weights)))
                for idx in indices[:remaining]:
                    allocated[idx] += 1

            # Handle token overflow
            new_patches = []
            for i, (patch, alloc) in enumerate(zip(all_patches, allocated)):
                patch_size = len(patch)
                if alloc <= 0:
                    continue
                elif alloc >= patch_size:
                    # Keep all tokens in this patch
                    new_patches.append(patch)
                else:
                    # Sample tokens based on attention scores
                    patch_attn = attention_patches[i]
                    _, top_indices = torch.topk(patch_attn, k=min(alloc.item(), patch_size))
                    new_patches.append(patch[top_indices])

            # Combine all selected tokens
            if new_patches:
                new_image_tokens = torch.cat(new_patches, dim=0) # (64, D)
            else:
                new_image_tokens = torch.zeros((0, D), device=device)
        else:
            # No patches to process
            new_image_tokens = torch.zeros((0, D), device=device) 

        # Pad or truncate to match expected new_image_token_num
        actual_tokens = new_image_tokens.size(0) 
        if actual_tokens < new_image_token_num:
            # Pad with zeros if we don't have enough tokens
            padding = torch.zeros((new_image_token_num - actual_tokens, D), device=device)
            new_image_tokens = torch.cat([new_image_tokens, padding], dim=0)
        elif actual_tokens > new_image_token_num:
            # Truncate if we have too many tokens
            new_image_tokens = new_image_tokens[:new_image_token_num]

        pruned_image_tokens_list.append(new_image_tokens)

    # Stack batches
    return torch.stack(pruned_image_tokens_list, dim=0).to(image_tokens.dtype) 



def HoloV_with_Indices(
    image_tokens, 
    attention, 
    num_patches=16, 
    new_image_token_num=64, 
    esp=1e-6
    ):
    """
    HoloV Implementation with Index Tracking and Budget Redistribution.
    
    Args:
        image_tokens: [B, N, D] - Vision Encoder output tokens
        attention: [B, N] - Attention scores (e.g., from CLS token)
        num_patches: int - Number of spatial crops (default: 16)
        new_image_token_num: int - Target number of tokens to keep (default: 64)
        esp: float - Small value for numerical stability
        
    Returns:
        final_retained_tokens: [B, new_image_token_num, D] - Selected tokens
        final_retained_indices: [B, new_image_token_num] - Original indices of selected tokens
        final_pruned_indices_ranked: [B, N - new_image_token_num] - Indices of dropped tokens, sorted by importance
    """
    
    B, N, D = image_tokens.shape
    device = image_tokens.device
    alpha = 0.09
    
    # Lists to collect results
    retained_tokens_list = []
    retained_indices_list = []
    pruned_indices_ranked_list = []

    # Create original indices from 0 to N-1
    original_indices = torch.arange(N, device=device)

    for b in range(B):
        image_token = image_tokens[b]       # [N, D]
        image_attention = attention[b]      # [N]
        image_indices = original_indices    # [N]

        # ---------------------------------------------------------
        # 1. Patch splitting (Dynamic Patching)
        # ---------------------------------------------------------
        patch_size = N // num_patches
        remainder = N % num_patches

        image_tokens_patches = []
        attention_patches = []
        indices_patches = []
        
        start_idx = 0
        for p in range(num_patches):
            # If there is a remainder, leading patches each take one extra token
            current_patch_size = patch_size + (1 if p < remainder else 0)
            end_idx = start_idx + current_patch_size

            if current_patch_size > 0:
                image_tokens_patches.append(image_token[start_idx:end_idx])
                attention_patches.append(image_attention[start_idx:end_idx])
                indices_patches.append(image_indices[start_idx:end_idx])

            start_idx = end_idx

        # ---------------------------------------------------------
        # 2. Per-patch scoring (Variance-Modulated Scoring)
        # ---------------------------------------------------------
        patch_scores = []
        
        # Keep patch data as lists for later use
        all_patches_token = []
        all_patches_indices = []
        all_patches_attn = [] # used in the Selection step

        for p in range(len(image_tokens_patches)):
            patch_tokens = image_tokens_patches[p]
            patch_attn = attention_patches[p]
            patch_idxs = indices_patches[p]
            current_patch_size = len(patch_tokens)

            # Store
            all_patches_token.append(patch_tokens)
            all_patches_indices.append(patch_idxs)
            all_patches_attn.append(patch_attn)

            # If patch size <= 1, variance cannot be computed -> use attention mean only
            if current_patch_size <= 1:
                score = patch_attn.mean() if len(patch_attn) > 0 else torch.tensor(0.0, device=device)
                patch_scores.append(score)
                continue

            with torch.no_grad():
                # (1) Normalize
                F_normalized = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + esp)
                
                # (2) Similarity Matrix & Masking
                S = torch.mm(F_normalized, F_normalized.transpose(0, 1))
                eye_mask = 1 - torch.eye(current_patch_size, device=device)
                S_masked = S * eye_mask

                # (3) Variance Calculation
                valid_entries = current_patch_size - 1
                mean_sim = S_masked.sum(dim=1) / valid_entries
                var_sim = ((S_masked - mean_sim.unsqueeze(1))**2).sum(dim=1) / valid_entries

                # (4) Scaling
                patch_attn_scaled = patch_attn * 1e3
                var_scaling = (torch.mean(torch.abs(patch_attn_scaled)) / 
                              (torch.mean(torch.abs(var_sim)) + esp))
                var_sim_scaled = var_sim * var_scaling

                # (5) Final Token Score & Patch Score
                token_scores =  patch_attn_scaled + alpha * var_sim_scaled
                patch_score = token_scores.mean()
                
                patch_scores.append(patch_score)

        # ---------------------------------------------------------
        # 3. Token allocation and redistribution (Allocation & Redistribution)
        # ---------------------------------------------------------
        patch_scores = torch.stack(patch_scores) if patch_scores else torch.zeros(0, device=device)
        patch_sizes = torch.tensor([len(p) for p in all_patches_token], device=device)
        
        # Initialize base allocation to 0
        allocated = torch.zeros(len(patch_scores), dtype=torch.long, device=device)

        if len(patch_scores) > 0:
            # (1) Initial weight-based allocation
            weights = (patch_scores) / ((patch_scores).sum() + esp)
            allocated = (weights * new_image_token_num).floor().long()

            # (2) First pass: distribute the leftover (remaining after floor)
            remaining_total = new_image_token_num - allocated.sum()
            if remaining_total > 0:
                _, indices = torch.topk(weights, k=min(remaining_total.item(), len(weights)))
                for idx in indices[:remaining_total]:
                    allocated[idx] += 1
            
            # (3) [core] Deficit-based redistribution (Iterative Redistribution)
            # Reclaim the excess from patches allocated more than they hold and give it to patches with room
            for _ in range(new_image_token_num):
                # Clamp the excess down to the actual capacity
                allocated = torch.min(allocated, patch_sizes)

                # Compute the deficit relative to the target
                deficit = new_image_token_num - allocated.sum().item()
                if deficit <= 0:
                    break # target reached

                # Find patches that can receive more
                can_receive = (allocated < patch_sizes)
                if can_receive.sum() == 0:
                    break # all patches are full

                # Among patches with room, redistribute in order of highest weight
                valid_weights = weights.clone()
                valid_weights[~can_receive] = -float('inf')
                k = min(deficit, can_receive.sum().item())
                _, top_idx = torch.topk(valid_weights, k=k)
                allocated[top_idx] += 1

        # ---------------------------------------------------------
        # 4. Selection and collection of dropped tokens
        # ---------------------------------------------------------
        kept_tokens_batch = []
        kept_indices_batch = []
        
        pruned_candidates_indices = []
        pruned_candidates_scores = []

        for i, (patch_tk, patch_alloc, patch_idx) in enumerate(zip(all_patches_token, allocated, all_patches_indices)):
            alloc_num = patch_alloc.item()
            curr_attn = all_patches_attn[i]
            patch_size = len(patch_tk)

            # Thanks to redistribution, alloc_num <= patch_size is guaranteed
            if alloc_num == 0:
                # all dropped
                pruned_candidates_indices.append(patch_idx)
                pruned_candidates_scores.append(curr_attn)
            
            elif alloc_num == patch_size:
                # all kept
                kept_tokens_batch.append(patch_tk)
                kept_indices_batch.append(patch_idx)
            
            else:
                # partially kept (Top-k by local attention)
                top_vals, top_indices = torch.topk(curr_attn, k=alloc_num)
                
                # store kept tokens
                kept_tokens_batch.append(patch_tk[top_indices])
                kept_indices_batch.append(patch_idx[top_indices])
                
                # store dropped tokens (Masking)
                mask = torch.ones(patch_size, dtype=torch.bool, device=device)
                mask[top_indices] = False
                
                pruned_candidates_indices.append(patch_idx[mask])
                pruned_candidates_scores.append(curr_attn[mask])

        # ---------------------------------------------------------
        # 5. Finalize results (merge and padding)
        # ---------------------------------------------------------
        
        # (1) Handle kept tokens
        if kept_tokens_batch:
            final_tokens = torch.cat(kept_tokens_batch, dim=0)
            final_indices = torch.cat(kept_indices_batch, dim=0)
        else:
            final_tokens = torch.zeros((0, D), device=device)
            final_indices = torch.zeros(0, dtype=torch.long, device=device)

        # Safety: if the count does not match, pad/truncate
        curr_len = final_tokens.size(0)
        if curr_len < new_image_token_num:
            pad_len = new_image_token_num - curr_len
            final_tokens = torch.cat([final_tokens, torch.zeros((pad_len, D), device=device)], dim=0)
            final_indices = torch.cat([final_indices, torch.full((pad_len,), -1, device=device)], dim=0)
        elif curr_len > new_image_token_num:
            final_tokens = final_tokens[:new_image_token_num]
            final_indices = final_indices[:new_image_token_num]

        retained_tokens_list.append(final_tokens)
        retained_indices_list.append(final_indices)

        # (2) Rank dropped tokens (Global Ranking)
        if pruned_candidates_indices:
            all_pruned_idxs = torch.cat(pruned_candidates_indices, dim=0)
            all_pruned_scores = torch.cat(pruned_candidates_scores, dim=0)
            
            sorted_scores, sort_idx = torch.sort(all_pruned_scores, descending=True)
            sorted_pruned_indices = all_pruned_idxs[sort_idx]
            
            pruned_indices_ranked_list.append(sorted_pruned_indices)
        else:
            pruned_indices_ranked_list.append(torch.zeros(0, dtype=torch.long, device=device))

    # Stack batches
    final_retained_tokens = torch.stack(retained_tokens_list, dim=0)
    final_retained_indices = torch.stack(retained_indices_list, dim=0)
    
    # The number of pruned tokens is constant (N - new_image_token_num), so stacking works
    # If N differs per batch, return a list instead (here we assume a fixed N)
    try:
        final_pruned_indices_ranked = torch.stack(pruned_indices_ranked_list, dim=0)
    except:
        final_pruned_indices_ranked = pruned_indices_ranked_list

    return final_retained_tokens, final_retained_indices, final_pruned_indices_ranked