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

# Python 2.7
RUN apt-get install -y \
    python-dev \
    python-pip
RUN pip2 install -U numpy scipy nose wheel
RUN pip2 install -U cvxpy

# Python 3.4
RUN apt-get install -y \
    python3-dev \
    python3-pip
RUN pip3 install -U numpy scipy nose wheel
RUN pip3 install -U cvxpy

CMD ["bash"]
