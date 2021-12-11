# 11785 Intro to Deep Learning
### Making Recommender Systems More Knowledgeable:A Framework to Incorporate Side Information
#### Team15: Yukun Jiang, Xinyi Chen, Leo Guo, Jing Xi Liu

This is the repository of final project for 11-785 Intro to Deep Learning at CMU fall 2021

Our code implementation is heavily dependent on the source code released by original authors of the paper. The links could be found as follows:

+ RepeatNet: (https://github.com/PengjieRen/RepeatNet-pytorch)
+ SRGNN: (https://github.com/CRIPAC-DIG/SR-GNN)

The original dataset is too big to be pushed up here, we provide the following links for reference:

+ Last.FM : (http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html)
+ MovieLens: (https://grouplens.org/datasets/movielens/)
+ Digi: (https://competitions.codalab.org/competitions/11161#learn_the_details-data2)
+ TaFeng: (https://www.kaggle.com/chiranjivdas09/ta-feng-grocery-dataset)
--- 

### Run RepeatNet/RepeatNet_Sideinfo

+ Author: Yukun J, Leo G
+ Init: Fri. Oct 29 15:12:45 PM
+ LastUpdate: Sat. Oct 30 11:54:20 AM
+ Source1: [https://github.com/PengjieRen/RepeatNet]
+ Source2: [https://github.com/PengjieRen/RepeatNet-pytorch]

#### Caution:

The original code is for distributed training. To be able to run it on our poor little stingy colab, it has gone quite a few modifications. Please read through this document and don't look back to the source links above. And there is a few newly added stuff. When in question, reach out to us!

#### How to run

Train command

```
!python -m torch.distributed.launch --nproc_per_node=1 ./RepeatNet/Run.py --mode='train'
```
or
```
!python -m torch.distributed.launch --nproc_per_node=1 ./RepeatNet_Sideinfo/Run.py --mode='train'
```
to incorporate side information into training process

Validation + Test command
```
!python -m torch.distributed.launch --nproc_per_node=1 ./RepeatNet/Run.py --mode='infer'
```

There is currently no easy way to do training while doing validationt to see the performance on the fly. Since now we could only do training on **GPU**, yet the validation and testing must be done on **CPU** due to memory storage issue.

A typical workflow would be like this:

+ 1. You decide on some hyperparamters and statically type them
+ 2. On **GPU**, do training from beginning to end, save the model file
+ 3. Switch to **CPU**, load the model and do Inference, store the result
+ 4. Do metric calculation offline using scripts

The way how this works is that, you will upload this whole folder to your google drive, you in colab mount your google drive, `cd` command to this folder and run one of the above commands.

The structure of this folder looks like this:

```python
-Preprocessing
    -process_lastfm.py
    -split_lastfm.py
    -data_aug_lastfm.py
-Common
    -BilinearAttention.py
    -CumulativeTrainer.py
    -EMA.py
-RepeatNet
    -Dataset.py
    -Model.py
    -Run.py
-output
    -RepeatNet
        -lastest_repeatnet.pt
        -repeatnet.pt
-datasets
    -demo
        -lastfm_valid.artist.txt
        -lastfm_test.artist.txt
        -lastfm_train.artist.txt
-metrics.py
-README.md
```

A few things to notice in pipeline workflow:

1. The two files that you might frequently need to change is the `Run.py` and `CumulativeTrainer.py`

2. prepare your data and put your data under `datasets/demo/"your_some_data.txt"`. Notice your item labelling should begin at `1`, because `0` is saved for empty padding. In the case of `Lastfm`, you run the pipeline `process_lastfm.py` -> `split_lastfm.py` -> `data_aug_lastfm.py`. You can draw the same idea to your dataset.

3. go to `Run.py`, change 3 lines. In function `train(args)` The file name in `train_dataset`, In function `infer(args)` The file name in `valid_dataset`, `test_dataset`

4. decide your `batch_size` and change it in the `train(args)` and `infer(args)`. In the case of `Lastfm` dataset, the max is `256` so that the poor colab doesn't blow up, though the original paper use `2048`, surely they have superior machines that I cannot afford.

5. change the `training epoch number`, `embedding_size`, `hidden_size` and `item_vocab_size` in the `Run.py`. In the paper they use `100` for both `embedding_size`, `hidden_size`. The `item_vocab_size` is the highest item id you give plus 2 for 0-indexed and padding. 

6. change the `optimizer` paramters in the function `train(args)`. By default we are using Adam with default parameters. Currently I am using hand scheduler, every 3 epoches I halve the learning rate, this is implemented in the `train(args)` as well.

7. You should probably now can do training. Run the command given above

8. After finish training, the model should be stored as `output/RepeatNet/lastest_repeatnet.pt`, now rename it to `repeatnet.pt`.

9. You should probably now can do Inference. Run the command given above (Using **CPU**!!!)

10. You should get in the `output/RepeatNet/...` two files called `valid_result.txt` and `test_result.txt`. Put the `test_result.txt` with the `metrics.py`. 

11. Run the `metrics.py` and get the metric evaulation about `Recall@10`, `Recall@20`, `MRR@10`, `MRR@20`.

12. So far I have successfully replicate the result from the original RepeatNet paper on the `LastFM` dataset as follows:


+ LASTFM

| Metrics   | Original | Reproduce |
|-----------|----------|-----------|
| Recall@10 | 24.18    | 23.85     |
| Recall@20 | 32.38    | 31.83     |
| MRR@10    | 11.46    | 11.53     |
| MRR@20    | 12.03    | 12.08     |


+ DIGINETICA

| Metrics   | Original | Reproduce |
|-----------|----------|-----------|
| Recall@10 | 36.86    | 37.79     |
| Recall@20 | 47.79    | 49.99     |
| MRR@10    | 16.90    | 16.69     |
| MRR@20    | 17.66    | 17.54     |

+ MOVIELENS

| Metrics   | Reproduce |
|-----------|-----------|
| Recall@10 | 8.64      |
| Recall@20 | 13.21     |
| MRR@10    | 3.51      |
| MRR@20    | 3.82      |


