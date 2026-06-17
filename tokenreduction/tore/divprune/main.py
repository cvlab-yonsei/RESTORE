import types

from .llava_arch_restore import (
    prepare_inputs_labels_for_multimodal, 
)


def divprune(
    model,
    n_vis,
    is_restore,
):
    if is_restore:
        print(f'========== Base reduction method: DivPrune | n_vis: {n_vis} | restore: {is_restore} ==========')
    else:
        print(f'========== Base reduction method: DivPrune | n_vis: {n_vis} | restore: {is_restore} ==========')

    model.n_vis = n_vis
    model.restore = is_restore

    # Bind new methods to the model instance
    model.prepare_inputs_labels_for_multimodal = types.MethodType(prepare_inputs_labels_for_multimodal, model)

    return model
