FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /drone

ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9;9.0;12.0"

COPY requirements.txt .

RUN pip install wandb && \
    pip install -r requirements.txt 

COPY . .

CMD ["/bin/bash", "-c", "if [ -f /run/secrets/wandb.key ]; then export WANDB_API_KEY=$(cat /run/secrets/wandb.key | xargs); fi && exec /bin/bash"]
