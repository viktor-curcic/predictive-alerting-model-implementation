import pandas as pd

class RuleBasedBaseline:
    
    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self.name = "Static Rules"
        
    def predict(self, window: pd.DataFrame) -> float:
        cpu_high = window['cpu_utilization'].iloc[-1] > 85
        
        error_high = window['error_rate'].iloc[-1] > 3
        
        cpu_recent = window['cpu_utilization'].iloc[-5:].values
        cpu_rising = (cpu_recent[-1] - cpu_recent[0]) > 10 if len(cpu_recent) > 1 else False
        
        mem_high = window['memory_usage'].iloc[-1] > 85
        mem_recent = window['memory_usage'].iloc[-5:].values
        mem_rising = (mem_recent[-1] - mem_recent[0]) > 5 if len(mem_recent) > 1 else False
        
        score = 0.0
        if cpu_high:
            score += 0.4
        if error_high:
            score += 0.5
        if cpu_rising:
            score += 0.3
        if mem_high and mem_rising:
            score += 0.3
        
        return min(score, 1.0)