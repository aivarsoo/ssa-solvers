# syntax=docker/dockerfile:1
ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

# install app dependencies
COPY requirements.txt ./.
RUN pip install -r ./requirements.txt
WORKDIR /project/
COPY . /project/
RUN pip install -e .

# RUN useradd -d /project -u 12567 --create-home user
# USER user

# CMD [ "python", "tests/test_auto.py"]
