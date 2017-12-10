import model

def score_diff_predictor(input_features):
    layer_sizes = [1024, 768, 512, 384, 256, 192, 128, 64, 1]
    layer_vars = []
    nn = model.base.model_head(None, input_features, layer_sizes, layer_vars, True)
    return nn, layer_vars