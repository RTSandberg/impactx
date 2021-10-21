#!/usr/bin/env bash
#
# Copyright 2021 The ImpactX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

set -eu -o pipefail

sudo apt-get -qqq update
sudo apt-get install -y \
    build-essential     \
    ca-certificates     \
    cmake               \
    gnupg               \
    libopenmpi-dev      \
    ninja-build         \
    pkg-config          \
    wget
