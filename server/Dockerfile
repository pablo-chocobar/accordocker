FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libsndfile1-dev tesseract-ocr espeak-ng python3.11 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip


ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# If set to nothing, will install the latest version
ARG PYTORCH='2.5.1'
ARG TORCH_VISION=''
ARG TORCH_AUDIO=''
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu121'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='TORCH_VISION'.*' ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='TORCH_AUDIO'.*' ||  VERSION='torchaudio'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

RUN python3 -m pip install --no-cache-dir numpy pillow Flask flask_cors

RUN python3 -m pip uninstall -y tensorflow flax

RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract
RUN python3 -m pip install -U "itsdangerous<2.1.0"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

RUN cd ..

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install Cython==0.29.36 
RUN python3 -m pip install --no-cache-dir spacy==3.0.7

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3", "temp.py"]