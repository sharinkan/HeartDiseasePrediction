"""
This file is made to segment audio files for models
- overlapping window
- same duration
"""
from typing import Literal

def intervals(lower : int, upper : int, stop : int,  step : int):
    def _validator():
        assert lower < upper, "invalid bounds"
        assert stop >= upper, "upper bound greater than stop"
        assert (stop - upper) % step == 0, "step size not divisible"
        
    _validator()    
    
    while upper < stop:
        yield (lower, upper)
        lower += step
        upper += step
    yield (lower, upper)
    return


def round_bounds(lower : int, upper : int, stop : int,  step : int, drop : Literal["upper","lower"] = "upper"):
    # to deal with "step size not divisible"
    drop_rate = (stop - upper) % step
    if drop == "upper":
        return (lower, upper, stop - drop_rate, step)
    
    return (lower + drop_rate, upper + drop_rate, stop, step)

print(round_bounds(3, 10, 100, 4, "lower"))
print(list(intervals( *round_bounds(3, 10, 100, 4, "lower") )))
# print(list(intervals(3, 10, 100, 3)))