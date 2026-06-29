import os

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        # Additional initialization code
        # os.makedirs(log_dir, exist_ok=True)
        self.count = 0
        self.success_count = 0
        self.collision_count = 0
        self.iteration_logs = []
    
