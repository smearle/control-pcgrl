FROM ubuntu
MAINTAINER leo.cazenille@gmail.com

ENV DEBIAN_FRONTEND noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3-yaml \
    gosu \
    rsync \
    python3-opengl \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    swig \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install gym box2d-py PyOpenGL setproctitle pybullet qdpy[all] cma

RUN git clone --branch develop https://gitlab.com/leo.cazenille/qdpy.git /home/user/qdpy
RUN pip3 uninstall -y qdpy && pip3 install --upgrade --no-cache-dir git+https://gitlab.com/leo.cazenille/qdpy.git@develop

RUN mkdir -p /home/user

# Prepare for entrypoint execution
#CMD ["bash"]
ENTRYPOINT ["/home/user/qdpy/examples/bipedal_walker/entrypoint.sh"]

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
