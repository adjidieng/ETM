##
# ETM
#
# @file
# @version 0.1

user := $(shell whoami)
userid := $(shell id -u)
groupid := $(shell id -g)
workdir := $(shell pwd)

.PHONY: build
build:
	docker-compose build \
	--build-arg USER=$(user) \
	--build-arg USER_ID=${userid} \
	--build-arg GROUP_ID=$(groupid) \
	--build-arg WORKDIR=$(workdir)

.PHONY: clean-container
clean-container:
	docker rmi etm_etm

.PHONY: clean-python
clean-python:
	rm -rf __pycache__
# end
