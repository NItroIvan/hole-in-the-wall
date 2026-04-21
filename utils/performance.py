class PerformanceTracker:
    def __init__(self):
        self.total_time = 0
        self.frames = 0

    def update(self, elapsed):
        self.total_time +=elapsed
        self.frames += 1

    def average(self):
        if self.frames == 0:
            return 0
        return self.total_time / self.frames