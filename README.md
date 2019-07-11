# ETM

This is code that accompanies the paper titled "Topic Modeling in Embedding Spaces" by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. (Arxiv link: )

ETM defines words and topics in the same embedding space. The likelihood of a word under ETM is a Categorical whose natural parameter is given by the dot product between the word embedding and its assigned topic's embedding. ETM is a document model that learns interpretable topics and word embeddings and is robust to large vocabularies that include rare words and stop words.

## Dependencies

+ python 3.6
+ pytorch 1.1.0

## To Run

To learn interpretable embeddings and topics using ETM on the 20NewsGroup dataset, run
```
python main.py --mode train --dataset 20ng --data_path data/20ng --num_topics 50 --train_embeddings 1 --epochs 1000
```

To evaluate perplexity on document completion, topic coherence, topic diversity, and visualize the topics/embeddings run
```
python main.py --mode eval --dataset 20ng --data_path data/20ng --num_topics 50 --train_embeddings 1 --tc 1 --td 1 --load_from CKPT_PATH
```

To learn interpretable topics using ETM with pre-fitted word embeddings (called Labelled-ETM in the paper) on the 20NewsGroup dataset:

+ first fit the word embeddings. For example to use simple skipgram you can run
```
python skipgram.py --data_file PATH_TO_DATA --emb_file PATH_TO_EMBEDDINGS --dim_rho 300 --iters 50 --window_size 4 
```

+ then run the following 
```
python main.py --mode train --dataset 20ng --data_path data/20ng --emb_path PATH_TO_EMBEDDINGS --num_topics 50 --train_embeddings 0 --epochs 1000
```

## Citation

```
@article{dieng2019topic,
  title={Topic modeling in embedding spaces},
  author={Dieng, Adji B and Ruiz, Francisco J R and Blei, David M},
  journal={arXiv preprint arXiv:},
  year={2019}
}
```

