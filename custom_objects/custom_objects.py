# custom_objects.py
import tensorflow as tf
from tensorflow.keras import backend as K

def true_positives(y_true, y_pred):
    # Convert predicted probabilities to binary (threshold = 0.5)
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)

    # True Positives (TP)
    tp = tf.reduce_sum(y_true * y_pred_binary)

    return tp

def all_positives(y_true, y_pred):
    # All positives
    ap = tf.reduce_sum(y_true)
    
    return ap

def recall_mul_prediction(y_true, y_pred):
    # Convert predicted probabilities to binary (threshold = 0.5)
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)

    # True Positives (TP)
    tp = tf.reduce_sum(y_true * y_pred_binary)

    # Precision = TP / (TP + FP)
    precision = tp / (tf.reduce_sum(y_pred_binary) + tf.keras.backend.epsilon())

    # Recall = TP / (TP + FN)
    recall = tp / (tf.reduce_sum(y_true) + tf.keras.backend.epsilon())

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = precision * recall
    return f1

def f1_score_metric(y_true, y_pred):
    # Convert predicted probabilities to binary (threshold = 0.5)
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)

    # True Positives (TP)
    tp = tf.reduce_sum(y_true * y_pred_binary)

    # Precision = TP / (TP + FP)
    precision = tp / (tf.reduce_sum(y_pred_binary) + tf.keras.backend.epsilon())

    # Recall = TP / (TP + FN)
    recall = tp / (tf.reduce_sum(y_true) + tf.keras.backend.epsilon())

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

def precision_with_threshold(threshold=0.7):
    def precision_with_threshold_fixed(y_true, y_pred):
        # Apply the custom threshold
        y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
        true_positives = tf.reduce_sum(tf.cast(y_true * y_pred_binary, tf.float32))
        predicted_positives = tf.reduce_sum(y_pred_binary)
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision
    return precision_with_threshold_fixed

def recall_with_threshold(threshold=0.7):
    def recall_with_threshold_fixed(y_true, y_pred):
        # Apply the custom threshold
        y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
        true_positives = tf.reduce_sum(tf.cast(y_true * y_pred_binary, tf.float32))
        actual_positives = tf.reduce_sum(y_true)
        recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
        return recall
    return recall_with_threshold_fixed

def focal_loss(alpha=0.25, gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = -alpha_t * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return focal_loss_fixed
    
def cubic_loss(power=3):
    def cubic_loss_fixed(y_true, y_pred):
        return tf.reduce_mean((tf.abs(y_true - y_pred) ** 3) * 100)
    return cubic_loss_fixed
    
def gap_based_loss(target, max, power=2):
    def gap_based_loss_fixed(y_true, y_pred):
        condition = tf.logical_or(tf.logical_and(y_true < target, y_pred < target), tf.logical_and(y_true > max, y_pred > max))
        loss = tf.where(condition,
                        0.0,  # Loss when any condition is True
                        tf.abs(y_true - y_pred) ** power)
                    
        return tf.reduce_mean(loss)
    return gap_based_loss_fixed
    
def weighted_bce(ratio):
    def weighted_bce_fixed(y_true, y_pred, sample_weight=None):
        # Class weights based on ratio
        weight_for_0 = (ratio + 1) / (2 * ratio)
        weight_for_1 = weight_for_0 * ratio  # Minority class gets higher weight
        
        # Apply class weights
        class_weights = (y_true * weight_for_1) + ((1 - y_true) * weight_for_0)
        
        # Calculate BCE
        bce = K.binary_crossentropy(y_true, y_pred)
        
        # Apply both class weights and sample weights
        if sample_weight is not None:
            # Element-wise multiplication of both weight types
            weights = class_weights * sample_weight
        else:
            weights = class_weights
            
        return K.mean(bce * weights)
    return weighted_bce_fixed
    
def multi_weighted_bce(ratios):
    """
    Create a weighted binary cross-entropy loss function for multiple binary outputs.
    
    Parameters:
    ratios (list or dict): Class imbalance ratios for each output.
                          If list: Each element is the imbalance ratio for the corresponding output.
                          If dict: Keys are output indices, values are the imbalance ratios.
    
    Returns:
    Function that computes the multi-output weighted BCE loss.
    """
    def multi_weighted_bce_fixed(y_true, y_pred, sample_weight=None):
        # Ensure types are float
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        
        # Clip y_pred to avoid log(0)
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        
        # Get the number of outputs (should be the last dimension)
        num_outputs = K.int_shape(y_pred)[-1]
        
        # Initialize the loss
        total_loss = 0.0
        
        # Process each output separately
        for i in range(num_outputs):
            # Get the appropriate ratio for this output
            if isinstance(ratios, list):
                ratio = ratios[i] if i < len(ratios) else 1.0
            elif isinstance(ratios, dict):
                ratio = ratios.get(i, 1.0)
            else:
                ratio = ratios  # Assume it's a single value to be used for all outputs
            
            # Extract current output
            y_true_i = y_true[..., i:i+1]
            y_pred_i = y_pred[..., i:i+1]
            
            # Compute class weights for this output
            weight_for_0 = (ratio + 1) / (2 * ratio)
            weight_for_1 = weight_for_0 * ratio
            class_weights = (y_true_i * weight_for_1) + ((1 - y_true_i) * weight_for_0)
            
            # Binary crossentropy for this output
            bce = K.binary_crossentropy(y_true_i, y_pred_i)
            
            # Apply sample weights if present
            if sample_weight is not None:
                sample_weight = K.cast(sample_weight, 'float32')
                if len(K.int_shape(sample_weight)) == 1:
                    sample_weight = K.expand_dims(sample_weight, axis=-1)
                weights = class_weights * sample_weight
            else:
                weights = class_weights
            
            # Sum up the weighted loss for this output
            total_loss += K.mean(bce * weights)
        
        # Return total loss
        return total_loss
    
    return multi_weighted_bce_fixed

    
def rifat_loss(ratio):
    def rifat_loss_fixed(y_true, y_pred):
        weights = 1 + y_pred
        bce = tf.math.squared_difference(y_true, y_pred)
        res = bce * weights
        return tf.reduce_mean(res)
    return rifat_loss_fixed

# Define a Loss Function for Multiple Quantiles
def multi_quantile_loss(quantiles):
    quantiles = tf.constant(quantiles, dtype=tf.float32)  # Convert to Tensor
    def multi_quantile_loss_fixed(y_true, y_pred):
        y_true = tf.expand_dims(y_true, -1)  # Expand dimensions to match y_pred shape
        error = y_true - y_pred  # Compute error term
        
        quantiles_expanded = tf.reshape(quantiles, [1, -1])  # Ensure correct shape
        quantiles_expanded = tf.tile(quantiles_expanded, [tf.shape(y_true)[0], 1])  # Repeat along batch dimension
        
        loss = tf.maximum(quantiles_expanded * error, (quantiles_expanded - 1) * error)
        return tf.reduce_mean(tf.abs(loss))  # Ensure non-negative output
    return multi_quantile_loss_fixed
    
def multi_expectile_loss(expectiles):
    expectiles = tf.constant(expectiles, dtype=tf.float32)  # Convert to Tensor
    
    def multi_expectile_loss_fixed(y_true, y_pred):
        y_true = tf.expand_dims(y_true, -1)  # Expand dimensions to match y_pred shape
        error = y_true - y_pred  # Compute error term
        
        expectiles_expanded = tf.reshape(expectiles, [1, -1])  # Ensure correct shape
        expectiles_expanded = tf.tile(expectiles_expanded, [tf.shape(y_true)[0], 1])  # Repeat along batch dimension
        
        loss = tf.where(error >= 0, expectiles_expanded * tf.square(error), (1 - expectiles_expanded) * tf.square(error))
        return tf.reduce_mean(loss)  # Mean loss across all expectiles
    
    return multi_expectile_loss_fixed

# Register the custom object globally
def register_custom_objects():
    tf.keras.utils.get_custom_objects().update({"focal_loss": focal_loss})
    tf.keras.utils.get_custom_objects().update({"focal_loss_fixed": focal_loss})
    tf.keras.utils.get_custom_objects().update({"weighted_bce": weighted_bce})
    tf.keras.utils.get_custom_objects().update({"weighted_bce_fixed": weighted_bce})
    tf.keras.utils.get_custom_objects().update({"multi_weighted_bce": multi_weighted_bce})
    tf.keras.utils.get_custom_objects().update({"multi_weighted_bce_fixed": multi_weighted_bce})
    tf.keras.utils.get_custom_objects().update({"multi_quantile_loss": multi_quantile_loss})
    tf.keras.utils.get_custom_objects().update({"multi_quantile_loss_fixed": multi_quantile_loss})
    tf.keras.utils.get_custom_objects().update({"multi_expectile_loss": multi_expectile_loss})
    tf.keras.utils.get_custom_objects().update({"multi_expectile_loss_fixed": multi_expectile_loss})
    tf.keras.utils.get_custom_objects().update({"rifat_loss": rifat_loss})
    tf.keras.utils.get_custom_objects().update({"rifat_loss_fixed": rifat_loss})
    tf.keras.utils.get_custom_objects().update({"cubic_loss": cubic_loss})
    tf.keras.utils.get_custom_objects().update({"cubic_loss_fixed": cubic_loss})
    tf.keras.utils.get_custom_objects().update({"gap_based_loss": gap_based_loss})
    tf.keras.utils.get_custom_objects().update({"gap_based_loss_fixed": gap_based_loss})
    tf.keras.utils.get_custom_objects().update({"all_positives": all_positives})
    tf.keras.utils.get_custom_objects().update({"true_positives": true_positives})
    tf.keras.utils.get_custom_objects().update({"f1_score_metric": f1_score_metric})
    tf.keras.utils.get_custom_objects().update({"recall_with_threshold": recall_with_threshold})
    tf.keras.utils.get_custom_objects().update({"recall_with_threshold_fixed": recall_with_threshold})
    tf.keras.utils.get_custom_objects().update({"precision_with_threshold": precision_with_threshold})
    tf.keras.utils.get_custom_objects().update({"precision_with_threshold_fixed": precision_with_threshold})
    tf.keras.utils.get_custom_objects().update({"recall_mul_prediction": recall_mul_prediction})
