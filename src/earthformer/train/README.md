## Train

The full argument list for *train_ims.py* is as follows: 

**`--logging-dir`**: The directory in which the logs are saved. Warning! the logs can take up a lot of disk space. 

**`--results-dir`**: The directory in which test results are saved. This parameter is effective only when "test" is set to True.

**`--gpus`** **`(default=1)`**: The number of GPUs to use (We could not test this argument since we only had access to one GPU).

**`--cfg`**: The train configuration file.

**`--seed`** **`(default=0)`**: A seed for randomization functions.

**`--ckpt`**: A Lightning checkpoint file.

**`--state-dict-file-name`**: A PyTorch state-dict file.

**`--test`**: True for running test, False for running train.