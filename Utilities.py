def assign_linear_comb(m, tm, tau):
    import tensorflow as tf
    from keras import backend as K
    '''Sets the value of a tensor variable,
    from a Numpy array.
    '''
    assign_op = tm.assign(m.value() * tau + (1-tau) * tm.value())
    return assign_op

def update_target_graph(target_model, model, tau):
    var_assign_ops = []
    for idxLayer in range(len(model.layers)):
        model_layer = model.layers[idxLayer]
        target_model_layer = target_model.layers[idxLayer]
        for idxWeight in range(len(model_layer.weights)):
            var_assign_ops.append(
                assign_linear_comb(model_layer.weights[idxWeight], target_model_layer.weights[idxWeight], tau)
            )
    return var_assign_ops

def update_target(var_assign_ops):
    from keras import backend as K
    for var_assign_op in var_assign_ops:
        K.get_session().run(var_assign_op)
