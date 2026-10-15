FROM ghcr.io/0xsero/sglang-exl3-flashnext@sha256:86f673e630856c98c301b2731de3e1a0fe7639a3e12a7626abaea24a18e0eb5a
ENV TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 PYTHONPATH="" \
    EXL3_INT8_GEMV=0 EXL3_MOE_CPU_SWAP=0
RUN python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130 \
 && python -m pip install transformers==5.12.1 flash-attn-4==4.0.0b19 fastapi==0.141.1 uvicorn==0.53.0 numpy==2.3.5 marisa-trie matplotlib==3.11.2
COPY patches /tmp/patches
RUN git clone https://github.com/turboderp-org/exllamav3.git /opt/exllamav3-nemotron35 \
 && cd /opt/exllamav3-nemotron35 \
 && git checkout --detach 058b8630fd081f3ff46fce6246955b157c3a2e3b \
 && git apply /tmp/patches/nemotron35-text.patch \
 && git apply /tmp/patches/cpu-wide-activations.patch \
 && python -m pip install --no-build-isolation --no-deps .
ENV EXL3_MOE_CPU_WIDE=1
COPY server /kit/server
COPY tools /kit/tools
WORKDIR /kit
ENTRYPOINT ["python", "/kit/server/serve.py"]
CMD ["-m", "/models", "-cs", "131072", "-gs", "27", "-mcs", "320", "-mct", "8", "-ambs", "4", "-chunk_size", "2048", "-rcs", "1", "--host", "0.0.0.0", "--port", "30000"]
