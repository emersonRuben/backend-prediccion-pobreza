import numpy as np

def extract_probabilities(estimator, matrix):
    import sklearn.calibration as calib
    
    original_get_response = calib._get_response_values
    
    def patched_get_response(est, X, response_method, pos_label=None, return_response_method_used=False):
        try:
            return original_get_response(est, X, response_method, pos_label, return_response_method_used)
        except ValueError as e:
            if "Got a regressor" in str(e):
                y_pred = est.predict_proba(X)
                if y_pred.shape[1] == 2:
                    y_pred = y_pred[:, 1]
                if return_response_method_used:
                    return y_pred, None, "predict_proba"
                return y_pred, None
            raise e
            
    try:
        calib._get_response_values = patched_get_response
        probabilities = estimator.predict_proba(matrix)
        return np.asarray(probabilities)[:, 1]
    finally:
        calib._get_response_values = original_get_response
