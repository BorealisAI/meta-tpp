# Meta Temporal Point Processes
[[Paper](https://openreview.net/pdf?id=QZfdDpTX1uM)][[Poster](https://iclr.cc/media/PosterPDFs/ICLR%202023/11395.png?t=1682361273.0520558)][[OpenReview](https://openreview.net/forum?id=QZfdDpTX1uM)]

## Datasets
We provide the compressed datasets: Stack Overflow, Mooc, Reddit, Wiki, Sin, Uber, NYC Taxi, in this [link](https://drive.google.com/file/d/1pL1wDG1elgtUa0CPv4GP21xGII-Ymk0x/view?usp=drive_link).
Unzip the compressed file and locate it in the `$ROOT` directory.


## Setup
Setup the pipeline by installing dependencies using the following command.
pretrained models and utils.
```bash
pip install -r requirements.txt
```
For nfe pacakge, install the package in [neural flows repo](https://github.com/mbilos/neural-flows-experiments) using
```bash
pip install -e .
```


## Pre-trained models
We also provide the checkpoints for Intensity free, THP+ and Attentive TPP on all the datasets.
Please download the compress file in this [link](https://drive.google.com/file/d/1frnaUoToJIMh9BnQaqz4zy3HNtaoKe35/view?usp=drive_link), unzip it and locate it in the `$ROOT` directory.



## Train
A model can be trained using the following command.
```bash
python src/train.py data/datasets=$DATASET model=$MODEL
```
`$DATASET` can be chosen from `{so_fold1, mooc, reddit, wiki, sin, uber_drop, taxi_times_jan_feb}` and `$MODEL` can be chosen from `{intensity_free,thp_mix,attn_lnp}`.
Other configurations can be also easily modified using hydra syntax. Please refer to [hydra](https://hydra.cc/docs/intro/) for further details.


## Eval
A model can be evaluated on test datasets using the following command.
```bash
python src/eval.py data/datasets=$DATASET model=$MODEL
```
Here, the default checkpoint paths are set to the ones in `checkpoints` directory we provided above.
To use different checkpoints, please chagne `ckpt_path` argument in `configs/eval.yaml`.


## Modifications
We made some modifications during code refactorization after ICLR 2023.
For the NLL metric, we took out L2 norm of model params, which we had to include for implementation purpose using the internal pipeline.
Note that since our proposed model contains stricly more number of parameters, this change is on favor of our method.
For the RMSE metric, we divide MSE by the number events for which we previously included the first events but not anymore.
We made this change because we do not make predictions on the first events.
Note that it is applied the same way for every model.
These changes do not change the rank of models nor the narrative of the paper.

Furthermore, instead of using boostrapped confidence interval (parentheses in Table 1 and 2), we encourage researchers to use confidence interval using different random seeds.
We provide a simple script for this as follows,
```bash
bash run_seeds.sh -d mooc -l 0.0001 -w 0.00001
```


## Citation
If you use this code or model for your research, please cite:

    @inproceedings{bae2023meta,
      title = {Meta Temporal Point Processes},
      author = {Bae, Wonho and Ahmed, Mohamed Osama and Tung, Frederick and Oliveira, Gabriel L},
      booktitle={The International Conference on Learning Representations (ICLR)},
      year={2023}
    }


## Acknowledgment
The pipeline is built on [PyTorch-Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template).
Intensity free is based on [the original implementation](https://github.com/shchur/ifl-tpp) and THP+ is based on (the corrected version of THP](https://github.com/yangalan123/anhp-andtt).



