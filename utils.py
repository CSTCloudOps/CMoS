import numpy as np

class EarlyStoppingBySlope:
    def __init__(self, window_size=5, slope_threshold=-0.001, patience=5, verbose=True):
        self.window_size = window_size
        self.slope_threshold = slope_threshold
        self.patience = patience
        self.verbose = verbose

        self.train_losses = []
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss):
        self.train_losses.append(current_loss)

        if len(self.train_losses) < self.window_size:
            return  
        
        y = np.array(self.train_losses[-self.window_size:])
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        print(f"Slope: {slope:.6f}\n")

        if self.verbose:
            print(f"[EarlyStopping] Slope: {slope:.6f}")

        if slope > self.slope_threshold:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No significant loss drop. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.counter = 0  # reset counter if slope is still sufficiently negative

