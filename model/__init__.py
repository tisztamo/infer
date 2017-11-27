import model.base as base

NUM_LABELS = 1972

def feature_extractor(data):
    return base.feature_extractor(data)

def policy_model(data, feature_tensor = None):
    return base.policy_model(data, feature_tensor)

def result_model(data, feature_tensor = None):
    return base.result_model(data, feature_tensor)
  