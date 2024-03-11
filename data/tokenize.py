from typing import List
from pathlib import Path
import json
import os
from itertools import chain, islice
from transformers import AutoTokenizer
import numpy as np
import time
from tqdm import tqdm
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor as Pool
import nltk
import random
import logging
logging.getLogger().setLevel(logging.ERROR)


path_to_source = Path("./WuDaoCorpus2.0_base_200G_shuffled").absolute()
path_to_target = Path("./WuDaoCorpus2.0_base_200G_tokenized").absolute()
tokenizer_path = "THUDM/chatglm3-6b"
batch_size = 1000 # 1000 samples per process
long_sentence_threshold = 1000000

def batched(it, n):
    return iter(lambda: list(islice(it, n)), ())

def filter_long_sentence(sentence: str):
    if len(sentence) > long_sentence_threshold:
        groups = nltk.sent_tokenize(sentence)
        for group in groups:
            yield group
    else:
        yield sentence

def worker(file: Path):
    target_file = path_to_target / file.relative_to(path_to_source).parent / (file.with_suffix("").with_suffix(".npz").name)

    if target_file.exists():
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    outputs = []

    def reader():
        with zstd.open(file, "rt", encoding="utf-8") as fin:
            data = json.load(fin)
            for item in data:
                for sentence in filter_long_sentence(item['content']):
                    yield sentence

    for batch in batched(reader(), batch_size):
        if not batch:
            break
        output = tokenizer(batch)
        outputs.extend([np.array(item, dtype=np.uint16) for item in output['input_ids']])
    outputs = np.concatenate(outputs, axis=0)
    
    np.savez_compressed(target_file, input_ids=outputs)

def run(max_workers: int):
    os.makedirs(path_to_target, exist_ok=True)
    files = sorted((path_to_source).glob("*.json.zst"))

    pool = Pool(max_workers=max_workers)
    futures = [pool.submit(worker, file) for file in files]

    with tqdm(total=len(futures), leave=False, desc="Tokenize") as t:
        for future in futures:
            future.add_done_callback(lambda x: t.update())
        for future in futures:
            future.result()

if __name__ == '__main__':
    import sys
    run(max_workers=int(sys.argv[1]))
    # python tokenize.py 20
    # ps -ef | grep tokenize.py | grep -v grep | awk '{print $2}' | xargs kill
