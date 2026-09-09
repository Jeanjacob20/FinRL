# FinRL Cloud Agent base image.
#
# Python 3.10 is required on purpose: FinRL pulls in `elegantrl`, which depends on
# the unmaintained `gym[Box2D]` and pins `pygame==2.1.0`. That pygame release only
# ships wheels up to CPython 3.10 and fails to compile on 3.11+ (it uses the private
# `longintrepr.h` header that was removed in Python 3.12). Pinning 3.10 lets the
# whole dependency set install from prebuilt wheels.
FROM python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# System dependencies:
# - build-essential / swig / cmake: build native wheels (e.g. Box2D).
# - SDL2 + freetype / portmidi / jpeg: required to build/run pygame.
# - TA-Lib C library: required by the `TA-lib` Python bindings listed in requirements.txt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential swig cmake wget git curl ca-certificates \
        libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
        libfreetype6-dev libportmidi-dev libjpeg-dev \
    && wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb \
    && dpkg -i ta-lib_0.6.4_amd64.deb \
    && rm -f ta-lib_0.6.4_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

# Cursor Cloud Agents run commands as the `ubuntu` user. Create it and a virtualenv
# that ubuntu owns so `pip install` works without root.
RUN useradd --create-home --shell /bin/bash ubuntu \
    && python -m venv /opt/venv \
    && chown -R ubuntu:ubuntu /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

USER ubuntu
