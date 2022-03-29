FROM ubuntu
MAINTAINER leo.cazenille@gmail.com

RUN \
		DEBIAN_FRONTEND=noninteractive \
		apt-get update && \
		apt-get upgrade -y && \
		apt-get install -yq python3-pip git
RUN useradd -ms /bin/bash user

USER user
WORKDIR /home/user
RUN pip3 install qdpy matplotlib pyyaml scoop
RUN git clone https://gitlab.com/leo.cazenille/qdpy.git
