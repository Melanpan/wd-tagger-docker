# A simple docker container around wd-tagger-v1.4 
Based on https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags

Leverages the GPU by using NVIDIA Container Toolkit.


## Usage
```
git clone
docker build -t wdtagger .
docker run --rm --gpus all -p 8005:8000 wdtagger
```

### Docker compose example (without GPU acceleration):
```yaml
services:
    tagger:
        build: .
        ports: "8005:8000"
        environment:
            - CACHE_PATH=/cache
            - USE_CUDA=0
        volumes:
           - .cache:/cache
         
```
### Docker compose example (with GPU acceleration):
```yaml
services:
   tagger:
        build: .
        ports: "8005:8000"
        environment:
            - CACHE_PATH=/cache
            - USE_CUDA=0
        volumes:
           - .cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

```

## Environment variables

| Name              | Default   | Explanation
|---                |---        |---
|MODEL              |SWIN       | The model to use, choose between MOAT, SWIN, CONV, CONV2 and VIT.
|USE_CUDA           |1          | Set to 0 to disable GPU acceleration.
|GENERAL_THRESHOLD  |0.35      | Only general tags with this value or above are returned to the client
|CHARACTER_THRESHOLD|0.85       | Only character tags with this value or above are returned to the client
|CACHE_PATH         |/tmp/model_cache|Location where to save the models, useful if you don't want to constantly redownload the models.
|MOAT_MODEL_REPO    |SmilingWolf/wd-v1-4-moat-tagger-v2|Hugging face repo for the moat model.
|SWIN_MODEL_REPO    |SmilingWolf/wd-v1-4-swinv2-tagger-v2|Hugging face repo for the swinv2 model.
|CONV_MODEL_REPO    |SmilingWolf/wd-v1-4-convnext-tagger-v2|Hugging face repo for the convnext model.
|CONV2_MODEL_REPO   |SmilingWolf/wd-v1-4-convnextv2-tagger-v2|Hugging face repo for the convnext v2 model.
|VIT_MODEL_REPO     |SmilingWolf/wd-v1-4-vit-tagger-v2|Hugging face repo for the vit model.