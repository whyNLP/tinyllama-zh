from pathlib import Path
import os
import json
import random
from tqdm import tqdm, trange
import zstandard as zstd

path_to_source = Path("./WuDaoCorpus2.0_base_200G").absolute()
path_to_target = Path("./WuDaoCorpus2.0_base_200G_shuffled").absolute()

os.makedirs(path_to_target, exist_ok=True)
files = sorted((path_to_source).glob("*.json"))

all_data = []
for file in tqdm(files):
    with open(file, "r", encoding="utf-8") as fin:
        data = json.load(fin)
        for item in data:
            item['filename'] = file.name
    all_data.extend(data)

# Shuffle data
print("Shuffling data...")
random.shuffle(all_data)

# Save shuffled data into 1000 shards
print("Saving data into 1000 shards...")
shard_size = 1000
for shard in trange(shard_size):
    with zstd.open(path_to_target / f"part-{shard:04d}.json.zst", "wt", encoding="utf-8") as fout:
        fout.write(json.dumps(all_data[shard::shard_size], ensure_ascii=False, indent=4))
