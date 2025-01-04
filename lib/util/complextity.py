class Complexity_Analysiser:
    def __init__(self):
        self.count_time = 0
        self.count_FLOPS = 0
        self.count_Params = 0

        self.FPS = 0

        self.total_time = 0
        self.total_FLOPS = 0
        self.total_Params = 0

    def update(self, time, FLOPS, Params):
        if time is not None:
            self.count_time = self.count_time + 1
            self.total_time = self.total_time + time
            self.avg_time = self.total_time / self.count_time
            self.FPS = 1 / self.avg_time

        if FLOPS is not None:
            self.count_FLOPS = self.count_FLOPS + 1
            self.total_FLOPS = self.total_FLOPS + FLOPS
            self.avg_FLOPS = self.total_FLOPS / self.count_FLOPS

        if Params is not None:
            self.count_Params = self.count_Params + 1
            self.total_Params = self.total_Params + Params
            self.avg_Params = self.total_Params / self.count_Params