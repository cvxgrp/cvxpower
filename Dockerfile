FROM debian:latest

# Debian packages
RUN apt-get update && apt-get install -y \
  autoconf \
  autotools-dev \
  build-essential \
  bzip2 \
  cmake \
  curl \
  g++ \
  gfortran \
  git \
  libc-dev \
  libopenblas-dev \
  libquadmath0 \
  libtool \
  make \
  parallel \
  pkg-config \
  unzip \
  timelimit \
  wget \
  zip && apt-get clean

# Python 2
RUN apt-get install -y \
    python-dev \
    python-pip
RUN python2 -m pip install -U pip
RUN python2 -m pip install -U numpy scipy
RUN python2 -m pip install -U cvxpy

# Python 3
RUN apt-get install -y \
    python3-dev \
    python3-pip
RUN python3 -m pip install -U pip
RUN python3 -m pip install -U numpy scipy
RUN python3 -m pip install -U cvxpy

CMD ["bash"]
