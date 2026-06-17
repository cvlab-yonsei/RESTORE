def load_reduction_base(
    model,
    base,
    n_vis,
    is_restore,
):

    if base == 'DivPrune':
        from tore.divprune import divprune
        model = divprune(model, n_vis, is_restore)
    elif base == 'VisPruner':
        from tore.vispruner import vispruner
        model = vispruner(model, n_vis, is_restore)
    elif base == 'HoloV':
        from tore.holov import holov
        model = holov(model, n_vis, is_restore)
    else:
        raise ValueError(f'Undefined base reduction method: {base}')

    return model
