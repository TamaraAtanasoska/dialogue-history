import torch
from torch.autograd import Variable


def calculate_accuracy(predictions, targets):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    predicted_classes = predictions.topk(1)[1].squeeze(1)
    accuracy = torch.eq(predicted_classes, targets).sum().item()/float(targets.size(0))
    return accuracy


def calculate_accuracy_all(predictions, targets):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    accuracies = []
    predicted_classes = predictions.topk(1)[1].squeeze(1)
    for accuracy in torch.eq(predicted_classes, targets):
        accuracies.append(accuracy.item())
    return accuracies


def calculate_accuracy_verbose(predictions, targets):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    predicted_classes_probs, predicted_classes = predictions.topk(1)
    predicted_classes = predicted_classes.squeeze()
    predicted_classes_probs = predicted_classes_probs.squeeze()
    guesses = torch.eq(predicted_classes, targets)
    accuracy = guesses.sum().item()/float(targets.size(0))
    return accuracy, guesses, predicted_classes_probs
