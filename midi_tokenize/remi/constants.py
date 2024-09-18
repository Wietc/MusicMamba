import numpy as np

DEFAULT_POS_PER_QUARTER = 4 # split each beat into 4 subbeats
DEFAULT_PITCH_BINS = np.arange(start=1,stop=128)
DEFAULT_DURATION_BINS = np.arange(start=1,stop=DEFAULT_POS_PER_QUARTER*16+1)
DEFAULT_VELOCITY_BINS = np.arange(start=1,stop=127)
# DEFAULT_TEMPO_BINS = np.arange(start=17,stop=250,step=3) ## you can use it and decide the min and the max of the tempo depend on your dataset
DEFAULT_TEMPO_BINS = np.linspace(0, 240, 32+1, dtype=int)
DEFAULT_POSITION_BINS = np.arange(start=0,stop=16)
MODE_PATTERN = [
    [2, 2, 3, 2], 
    [2, 3, 2, 3], 
    [3, 2, 3, 2], 
    [2, 3, 2, 2], 
    [3, 2, 2, 3], 
    [2, 2, 1, 2, 2, 2], 
    [2, 1, 2, 2, 2, 1], 
    [1, 2, 2, 2, 1, 2], 
    [2, 2, 1, 2, 2, 1], 
    [2, 2, 2, 1, 2, 2], 
    [2, 1, 2, 2, 1, 2], 
    [1, 2, 2, 1, 2, 2], 
]