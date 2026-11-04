FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir tiktoken==0.11.0 numpy==2.2.6

ENTRYPOINT ["/usr/bin/tini","--"]
