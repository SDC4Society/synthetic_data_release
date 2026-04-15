import joblib
from tqdm import tqdm
import time

def my_func(x):
    time.sleep(0.5)
    return x * 2

tasks = [1, 2, 3, 4]

with tqdm(total=len(tasks)) as pbar:
    results = joblib.Parallel(n_jobs=2, return_as='generator')(
        joblib.delayed(my_func)(t) for t in tasks
    )
    final = []
    for r in results:
        final.append(r)
        pbar.update(1)
print(final)
