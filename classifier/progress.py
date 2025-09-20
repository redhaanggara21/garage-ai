_progress = {"epoch": 0, "loss": 0, "accuracy": 0, "val_loss": 0, "val_accuracy": 0}

def update_progress(data):
    global _progress
    _progress.update(data)

def get_progress():
    return _progress
