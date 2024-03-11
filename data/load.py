import datasets
from datasets import IterableDataset, DatasetDict
import numpy as np
from pathlib import Path

class IterableDatasetWithLength(IterableDataset):
    def __len__(self):
        return self.length
    
    def set_length(self, length: int):
        self.length = length
        return self

datasets.builder.IterableDataset = IterableDatasetWithLength

def get_length(split: str, block_size: int):
    tokens = {
        "train": 45*1024*1024*1024,
        "validation": 10*1024*1024,
        "test": 45*1024*1024,
    }
    return tokens[split] // block_size

def get_files(path: Path, split: str):
    if split == "train":
        files = sorted(path.rglob("*.npz"))
        files.remove(path / "part-0999.npz")
    else:
        files = [path / "part-0999.npz"]
    return files

def data_iterator(split: str, path: Path, block_size: int):
    counter = 0
    for file in get_files(path, split):
        data = np.load(file)['input_ids'].astype(np.int64)
        for i in range(len(data) // block_size):

            if counter >= get_length(split, block_size):
                return
            counter += 1

            input_ids = data[i*block_size:(i+1)*block_size]
            attention_mask = np.ones_like(input_ids)
            yield {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}

def get_datasets(path: str, block_size: int):
    datasets = DatasetDict({
        'train': IterableDataset.from_generator(data_iterator, gen_kwargs={"split": "train", "path": Path(path), "block_size": block_size}).set_length(get_length("train", block_size)),
        'validation': IterableDataset.from_generator(data_iterator, gen_kwargs={"split": "validation", "path": Path(path), "block_size": block_size}).set_length(get_length("validation", block_size)),
        'test': IterableDataset.from_generator(data_iterator, gen_kwargs={"split": "test", "path": Path(path), "block_size": block_size}).set_length(get_length("test", block_size)),
    })

    return datasets

if __name__ == "__main__":
    path_to_wudao_tokenized = Path("./WuDaoCorpus2.0_base_200G_tokenized").absolute()
    block_size = 2048

    dataset = get_datasets(path_to_wudao_tokenized, block_size)
    breakpoint()
