# OEILP - Ontology Enhance Inductive Link Prediction 

This is the code for EMNLP2023 "Inductive Relation Inference of Knowledge Graph Enhanced by Ontology Information".

## Requiremetns

You can use the following command to create a environment and enter it.

```shell
conda create OEILP
source activate OEILP
```

In order to use the Cuda version of DGL, the following command needs to be run

```shell
conda install -c dglteam/label/cu113 dgl
```

All other required packages can be installed by running 

```shell
pip install -r requirements.txt`
```

## Inductive relation prediction experiments

The full inductive datasets used in these experiments can be found in the `data` folder.

### Training
To start training a model, run the following commands: 

```shell
python train.py -d yago -e yago
python train.py -d db -e db
```

### Evaluation 

To test model, run the following commands:

```shell
python test_ranking.py -d yago -e yago
python test_ranking.py -d db -e db
```

The trained model and the logs are stored in `experiments` folder. Negative triples sampled during the evaluation of OEILP are stored in the `data\{dataset}` folder, with {dataset} being the dataset used for testing.

### Type prediction

To test type prediction, run the following commands:

```shell
python test_typing.py -d yago -e yago
python test_typing.py -d db -e db
```

# Citation

If this code is useful for you, we appreciate if you cite the following:

```
@inproceedings{zhou-etal-2023-inductive,
    title = "Inductive Relation Inference of Knowledge Graph Enhanced by Ontology Information",
    author = "Zhou, Wentao  and
      Zhao, Jun  and
      Gui, Tao  and
      Zhang, Qi  and
      Huang, Xuanjing",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.431",
    pages = "6491--6502",
}
```
