# Function to compute accuracy
def get_accuracy(predictions, true_labels):
    correct_predictions = sum(predictions == true_labels)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    return accuracy