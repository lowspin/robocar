MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.int_val = self.last_int_val = self.last_error = 0.

    def reset(self):
        self.int_val = 0.0
        self.last_int_val = 0.0

    def step(self, error, sample_time):

        ###################################e####################
        # TO-DO:
        # 1. Calculate and apply Proportional control gain
        # 2. Calculate and apply Derivative Control gain
        # 3. Calculate and apply Integral Control gain
        # 4. Modify y below to include P, I, and D control
        #######################################################
        y = error;

        val = max(self.min, min(y, self.max))

        return val
