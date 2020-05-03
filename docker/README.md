# How does it work?

To cache apt dependency, and avoid installing them all at each CI build, we use this docker image.

You need rights to [dockerhub](https://hub.docker.com/u/stitchem).
```
docker build --tag stitchem/stitchem:latest --file docker/stitchem.dockerfile .
docker push stitchem/stitchem:latest
```
