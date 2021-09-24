import sys, os
from datetime import datetime
import time

# arrays and data packages
import numpy as np
import pandas as pd
import re

# speed improvement
import numba as nb  # jit, njit, vectorize
from concurrent.futures import ThreadPoolExecutor

# symbolic integration
import sympy as sympy

# scipy packages
from scipy.integrate import solve_ivp, quad

# plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
