def custom_accuracy_score(y_true, y_pred):
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

def custom_precision_score(y_true, y_pred):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    false_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    return precision

def custom_recall_score(y_true, y_pred):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    false_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)

    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    return recall

def custom_f1_score(y_true, y_pred):
    precision = custom_precision_score(y_true, y_pred)
    recall = custom_recall_score(y_true, y_pred)

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1