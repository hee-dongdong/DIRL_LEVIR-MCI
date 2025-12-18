import os
import numpy as np
from PIL import Image
from tqdm import tqdm

label_dir = '/data/inseong/skrr/DIRL_Capstone/data_loader/LEVIR-MCI-dataset/images/train/label_rgb'
files = sorted(os.listdir(label_dir))[:100]

unique_values = set()

print(f"Checking {len(files)} files...")
for f in tqdm(files):
    path = os.path.join(label_dir, f)
    # Load as RGB to find unique colors
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    # Reshape to (N, 3)
    pixels = arr.reshape(-1, 3)
    # Find unique rows
    uniques = np.unique(pixels, axis=0)
    for u in uniques:
        unique_values.add(tuple(u))

print(f"Unique RGB values found: {sorted(list(unique_values))}")
