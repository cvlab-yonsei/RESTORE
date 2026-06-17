import types

from .llava_arch_restore import (
    encode_images,
    prepare_inputs_labels_for_multimodal, 
)

def vispruner(
    model,
    n_vis,
    is_restore,
):
    if is_restore:
        print(f'========== Base reduction method: VisPruner | n_vis: {n_vis} | restore: {is_restore} ==========')
    else:
        print(f'========== Base reduction method: VisPruner | n_vis: {n_vis} | restore: {is_restore} ==========')

    model.n_vis = n_vis
    model.restore = is_restore

    # Two arguments in the original paper
    # visual_token_num: n_vis * retain_ratio -> n_retain  (restore --> pruning rate (gamma) 0.5)
    # important ratio: 0.5 (fixed) # No explanation in the paper?
    model.n_retain = int(n_vis * 0.5) if is_restore else n_vis
    model.important_ratio = 0.5

    # Bind new methods to the model instance
    model.encode_images = types.MethodType(encode_images, model)
    model.prepare_inputs_labels_for_multimodal = types.MethodType(prepare_inputs_labels_for_multimodal, model)

    return model
