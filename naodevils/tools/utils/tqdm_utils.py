from utils import *
from tqdm import tqdm

TQDM_UPDATE_INTERVAL = 0.25
TQDM_SMOOTHING = 0.9
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n:.1f}/{total_fmt} [{elapsed_s:.2f}s<{remaining_s:.2f}s, {rate_noinv_fmt}]"