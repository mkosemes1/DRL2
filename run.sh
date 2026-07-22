docker run -it --rm \
       --runtime=nvidia \
       --gpus all \
       --ipc=host \
       -v $HOME/.secrets/wandb.key:/run/secrets/wandb.key:ro \
       -v $PWD:/drone \
       drone
