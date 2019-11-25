from tqdm import tqdm
import os
import sys
import time

pbar = tqdm(total=100)

for i in range(100):
    time.sleep(0.02)
    pbar.update(1)

# pbar.clear()
pbar.refresh()

for i in range(100):
    time.sleep(0.02)
    pbar.update(1)

pbar.close()
