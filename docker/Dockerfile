FROM ubuntu:latest

RUN apt update 
RUN apt upgrade
RUN apt install -y python3
RUN apt install -y python3-pip
RUN pip3 install numpy
RUN pip3 install matplotlib

CMD [ "/bin/bash" ]