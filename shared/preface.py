import sys, os
from datetime import datetime
import time

# arrays and data packages
import numpy as np
import pandas as pd
import re

# astrophysics
from astropy import units as unit
from astropy import constants as const

# speed improvement
import numba as nb  # jit, njit, vectorize
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# gpu packages


# symbolic integration
import sympy as sympy

# scipy packages
from scipy.integrate import solve_ivp, quad, simpson
from scipy.special import expit

# plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rc('axes', labelsize=11)
plt.rc('axes', titlesize=13)