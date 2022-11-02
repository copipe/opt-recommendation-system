FROM nvcr.io/nvidia/rapidsai/rapidsai-core:22.08-cuda11.2-base-ubuntu20.04-py3.9

ARG PROJECT_NAME=ascender
ARG USER_NAME=challenger
ARG GROUP_NAME=challengers
ARG UID=1000
ARG GID=1000
ARG USER_DIRECTORY=/home/${USER_NAME}
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}
ARG RECBOLE_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}/RecBole

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}

RUN apt update && apt install --no-install-recommends -y \
    git curl make ssh openssh-client vim tmux

RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}
USER ${USER_NAME}

WORKDIR ${USER_DIRECTORY}
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR ${APPLICATION_DIRECTORY}
