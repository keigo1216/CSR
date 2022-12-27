FROM ubuntu:latest

RUN apt update 
RUN apt upgrade
RUN apt install -y python3.10
RUN apt install -y python3-pip
RUN pip3 install numpy==1.24.0
RUN pip3 install matplotlib==3.6.2
RUN pip3 install tqdm==4.64.1

CMD [ "/bin/bash" ]