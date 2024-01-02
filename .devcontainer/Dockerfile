ARG VARIANT="3.11-bullseye" 
# https://hub.docker.com/_/microsoft-devcontainers?tab=description
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT} AS dev-base
# FROM mcr.microsoft.com/devcontainers/base:jammy 

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \ 
    && apt-get install -y build-essential --no-install-recommends make \
        ca-certificates \
        git \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        python3-opencv \
        libgl1 \
        poppler-utils \
        tesseract-ocr \
        libtesseract-dev \
        ghostscript python3-tk

# Python and poetry installation
USER $USER
ARG HOME="/home/$USER"
ARG PROJECT="ml-learning"

ENV VENV_ROOT="/workspaces/${PROJECT}/.venv"
ENV PATH="${VENV_ROOT}/bin:${HOME}/.local/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org

ENV PYTHONPATH="${PYTHONPATH}:/workspaces/${PROJECT}/src"