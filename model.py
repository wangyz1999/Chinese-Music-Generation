import torch

import numpy as np
import os
import time
import functools

from music_handler import *

song = """X:496
T: Ai erwa
N: C1915
O: China, Hebei Guyuan
S: IV, 30]
R: Xiaodiao]
M: 2/4
L: 1/16
K: F
c4d2A2 | c6G2 | c2GAG2D2 | F2D2F2G2 | c2GAG2d2 | F8 |
G3Ac4 | d3AG4 | c2G2G2D2 | C8 |
F2G2G2D2 | F2G2G2D2 | C2CCF2D2 | C8"""

play_song(song)