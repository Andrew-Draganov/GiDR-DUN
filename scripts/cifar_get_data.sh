#!/usr/bin/env bash
#from https://github.com/sorki/python-mnist/blob/master/bin/mnist_get_data.sh

if [ -d data/cifar ]; then
    echo "data directory already present, exiting"
    exit 1
fi

mkdir -p data/cifar
wget --directory-prefix=data/cifar https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
pushd data/cifar
tar -xvzf cifar-10-python.tar.gz
mv cifar-10-batches-py/* .
rm -r cifar-10-batches-py
rm -r cifar-10-python.tar.gz
popd
