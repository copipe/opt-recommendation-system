# SIGNATE SOTA Challenge
This is the code for the SIGNATE recommendation competition. (Click [here](https://signate.jp/competitions/268) for the competition page.) I entered the [SOTA Challenge](https://signate.jp/features/state-of-the-art-challenge), not the original competition. The rankings written in this repository will be the results around November 2022.

## Solution

## Environment

## Setup

Connect to the docker container environment with the following command.
```
$ docker compose up -d
$ docker exec -it opt-sota-challenge /bin/bash
```

And install [RecBole](https://recbole.io/) (efficient recommendation library).
```
$ cd RecBole
$ pip install -e . --verbose
```

If you get an error when importing cudf or torch, the following commands may help. It seems that tensorboard installed with Recbole may conflict with cudf and torch.
```
$ pip install protobuf==3.20
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/rapids/lib
```

## Execution

The final submit can be roughly reproduced with the following command.
```
$ sh run.sh
```