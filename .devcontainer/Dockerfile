FROM ubuntu:latest

ARG USERNAME=developer
ARG GROUPNAME=$USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user; adapted from:
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd -f $USERNAME \
    && useradd -m -s /bin/bash -g $GROUPNAME $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
