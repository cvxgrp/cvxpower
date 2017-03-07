#!/bin/bash -eu

tag=cvxgrp/dem

docker build -t $tag .
docker push $tag
