import torch


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path="checkpoint.pt", mode="min"):
        """
        Args:
            patience: how many epochs to wait before stopping
            delta: minimum change in the value to consider as an improvement
            path: path to save the model checkpoint
            mode: 'min' or 'max' (lower or higher is better)
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric, model):
        score = -metric if self.mode == "min" else metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
