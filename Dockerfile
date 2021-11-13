FROM ubuntu:18.04
#MAINTAINER <your name> "<your email>"
ENV PYTHONUNBUFFERED 1

EXPOSE 5222

RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev
#We copy just the requirements.txt first to leverage Docker cache

RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
