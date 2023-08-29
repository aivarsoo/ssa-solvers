# syntax=docker/dockerfile:1
ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime


# installing a new user and packages
ARG USERNAME="docker_user"
RUN apt update && apt install -y git && pip install --upgrade pip && useradd -m -u 1001 ${USERNAME}
USER ${USERNAME}
WORKDIR /home/${USERNAME}/project

# installing the required packages for docker_user
COPY --chown=${USERNAME}:${USERNAME} requirements.txt /home/${USERNAME}/project/
RUN pip install --user -r requirements.txt

# copying the remaining files and installing the main packages for docker_user (Extra layer for faster re-build in development)
COPY --chown=${USERNAME}:${USERNAME} . /home/${USERNAME}/project/
RUN pip install --user -e .

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
COPY --chown=${USERNAME}:${USERNAME} . .
