# How does it work?

To avoid rebuilding an entire docker container at each build, we use base images for each cuda version (currently 8.0, 9.2 and 10.1). Those images are built locally and pushed to [dockerhub](https://hub.docker.com/u/stitchem). Then, the travis CI builds the ci image (that depends on the base images), adds the stitchEm sources and builds the applications.

# Modifying the images

## Modifying base image (base.dockerfile)

* Push your changes, create PR and merge it in master
* Ask for the rights to https://hub.docker.com/r/stitchem/stitchem-base
* Build the image locally: `docker build --tag stitchem/stitchem-base:latest --file docker/base.dockerfile .`
* Push the image: `docker push stitchem/stitchem-base:latest`
* Do the next section for x in 8, 9, 10

## Modifying base_cuda_x (base_cuda_x.dockerfile)

* Push your changes, create PR and merge it in master
* Ask for the rights to https://hub.docker.com/r/stitchem/stitchem-base-cudax
* Build the image locally: `docker build --tag stitchem/stitchem-base-cudax:latest --file docker/base_cuda_x.dockerfile .`
* Push the image: `docker push stitchem/stitchem-base-cudax:latest`

## Modifying CI image (ci.dockerfile)

* Push your changes, create PR and merge it in master
* You're all set
