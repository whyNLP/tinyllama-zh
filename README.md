# Chinese TinyLlama

[HuggingFace Model](https://huggingface.co/whynlp/tinyllama-zh) | [Wandb](https://wandb.ai/whynlp/tinyllama-zh/reports/tinyllama-zh--Vmlldzo3MTA0NDY1) | [ModelScope Dataset](https://www.modelscope.cn/datasets/whynlp/WuDaoCorpus-200G-shuffled)

A demo project that pretrains a tinyllama on Chinese corpora, with minimal modification to the huggingface transformers code. It serves as a use case to demonstrate how to use the huggingface version [TinyLlama](https://github.com/whyNLP/tinyllama) to pretrain a model on a large corpus.

## Installation
The installation follows that of [TinyLlama](https://github.com/whyNLP/tinyllama). The following is my installation process on a machine with CUDA 12.1.

```sh
conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Dataset Preparation
Compared to fine-tuning, pretraining requires a larger corpus. So usually I pre-tokenize the corpus and save it to the disk manually -- Huggingface's example code is not suitable for this task.

In this project, I use the [WuDaoCorpora Text](https://www.scidb.cn/en/detail?dataSetId=c6a3fe684227415a9db8e21bac4a15ab) as the pretraining corpus. It is constructed by Beijing Academy of Artificial Intelligence(BAAI) and has 200GB open data, mostly collected before 2022. You may also visit the dataset from huggingface [datasets](https://huggingface.co/datasets/p208p2002/wudao).

### Download
You may download the dataset from the official website. Since it is a .rar file, you may need to install `unrar` to extract the data:
```sh
unrar x data.rar
```

Now you should get a folder named `WuDaoCorpus2.0_base_200G` with many `.json` files.

### Shuffling
This is very important! The original dataset is sorted by category or something else, which may lead to a biased pretraining. Since this dataset is not very large, I shuffle the dataset by reading all files in memory and split it into smaller files to make the pretraining process easier.

```sh
python data/shuffle.py
```

Now you should get a folder named `WuDaoCorpus2.0_base_200G_shuffled` with many `.json.zst` files, which are compressed with zstd. If you do not have a large enough RAM, you may download the shuffled version [here](https://www.modelscope.cn/datasets/whynlp/WuDaoCorpus-200G-shuffled).

### Tokenization
According to some previous work [1][2], the original vocabulary is not suitable for Chinese. Here I simply use the `THUDM/chatglm3-6b` tokenizer from huggingface. To tokenize the dataset, you may run the following command:

```sh
python data/tokenize.py 20
```

where the argument `20` is the number of processes to use. You may adjust it according to your machine's capability. Now you should get a folder named `WuDaoCorpus2.0_base_200G_tokenized` with many `.npz` files, each one contains a long concatenated tensor of tokenized text.

If you do not have enough CPU resources, you may download the tokenized version [here](https://www.modelscope.cn/datasets/whynlp/WuDaoCorpus-200G-ChatGLM-tokenized).


[1] https://github.com/DLLXW/baby-llama2-chinese  
[2] https://github.com/ymcui/Chinese-LLaMA-Alpaca

### Build the Dataset
To make the dataset compatible with the huggingface's trainer, I build an iterable dataset with the `datasets.IterableDataset` class. I set the last file as the validation/test set (I only use the validation set). The number of training tokens is set to 45B and the number of validation tokens is set to 10M (I do not want to load all the data into memory, but it must have a dataset size for training). The codes can be found in `data/load.py`.

## Training
The training script follows the [huggingface example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py), with the tokenization process removed.

Since I replaced the tokenizer, the vocabulary size is different from the original one. So I create a new config file `configs/tinyllama-zh.json` with a different vocabulary size.

I use the TinyLlama's parameters to initialize the model. To avoid conflict when loading the model, run the following command to remove the embeddings and prediction head:

```sh
python convert_to_no_embedding.py --input TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T --output outputs/tinyllama-wo-embedding
```

Feel free to change the input to `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` if you want.

Now everything is ready. You may run the following command to start the training:

```sh
srun bash run_clm_large_zh.sh
```

This script uses the accelerate launcher that supports multi-node training on SLURM. It trains the model on 45B tokens for 2 epochs. The hyperparameters follow that of TinyLlama. You may modify it according to your own environment.

## Results

The training takes about 6 days on 8 A100 GPUs. The final perplexity is 11.5 on the validation set. You may visit the dev loss curve on [wandb](https://wandb.ai/whynlp/tinyllama-zh/reports/tinyllama-zh--Vmlldzo3MTA0NDY1).

You may use huggingface [example script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py) to do generation. I provide a copy in this repo:

```sh
bash run_generation.sh
```

Here are some examples:
```
Prompt: 床前明月光，疑是地上霜。举头望明月，
Model: 床前明月光，疑是地上霜。举头望明月，低头思故乡。 28. 平林漠漠烟如织，寒山一带伤心碧。暝色入高楼，有人楼上愁。 29. 吟鞭东指即天涯，芳草无情，更在斜阳外。 30. 风飘飘，雨潇潇，红叶黄花秋意晚。 31. 野火烧不尽，春风吹又生。 

Prompt: 北京大学是一所
Model: 北京大学是一所极富国际性、高水平研究型大学。享有“学府北辰”的美誉，同时作为世界上建校历史第二长的大学，其历史更久。作为亚洲最古老、最杰出的大学之一，北京大学自创建以来，始终秉承“允公允能、日新月异”的校训精神，秉持“独立之精神、自由之思想”的校风和“实事求是”的学风，为国家培养了大批杰出人才。

Prompt: 花自语，鸟有语，生活处处有语言。生命也可以用语言来解读，雕塑、基因……都可以用语言来传递。语言丰富生活，语言诠释生命，语言传承文明。请根据所给材料作文，自己拟题，问题不限，诗歌除外，不少于800字。
Model: 花自语，鸟有语，生活处处有语言。生命也可以用语言来解读，雕塑、基因……都可以用语言来传递。语言丰富生活，语言诠释生命，语言传承文明。请根据所给材料作文，自己拟题，问题不限，诗歌除外，不少于800字。 注意：立意自定，角度自选，除诗歌外，文体不限，不得抄袭。 【审题立意】 ①情感+说理：语言+生活、文化、精神 ②关键词语转化：风雅颂变语言 生活因了语言而精彩；文化因了语言而厚重。语言是社会文明的发动机。语言是一种“社会生长性”的存在，它与语言共生，同时又致力于语言“成熟性”的呈现。也就是说，语言不仅是对社会存在的一种表达，更是对语言自身的一种构造，为了获得这种语言的“社会生长性”，语言又在不断地向“成熟性”的转化。这里所说的语言的“成熟性”究竟是什么，显而易见，在汉语或英语里，它就是“语言演进形式”。
```

The CMMLU result is slightly above 25. Again, this project only serves as a demonstration of how to pretrain a TinyLlama on a large corpus. For better performance, one may use better corpus (e.g. [wanjuan](https://opendatalab.org.cn/OpenDataLab/WanJuan1_dot_0)).

For researchers who want to do pretraining on SlimPajama, it follows the same procedure where you even do not need to change the tokenizer. A pre-tokenized version of SlimPajama is available [here](https://www.modelscope.cn/datasets/whynlp/SlimPajama-627B-Llama-tokenized).
