ARG TAG=3.8

FROM python:$TAG
ENV PYTHONUNBUFFERED 1

ARG USER
ARG USER_ID
ARG GROUP_ID
ARG WORKDIR

RUN apt-get update \
    && apt-get clean \
    && apt-get update -qqq \
    && apt-get install -y -q build-essential graphviz graphviz-dev \
    && apt-get install -y -q ffmpeg libsm6 libxext6 \
    && pip install --upgrade pip \
    && pip install Cython scipy

RUN groupadd --gid 1000 $USER
RUN useradd --create-home --uid $USER_ID --gid $GROUP_ID $USER

USER ${USER}
ENV PATH "$PATH:/home/$USER/.local/bin"

COPY ./requirements.txt requirements.txt
RUN pip install --user -r requirements.txt

RUN pip install --user torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR $WORKDIR
