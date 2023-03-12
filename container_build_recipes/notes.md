# Notes on each of the build files

## alpine_vibro_20220310.def >>> FAIL

- Failed, for some reason when using the remote build, keeps getting errored out.

## alpine_vibro_20220310_1.def >>> FAIL

- Same reason as above, decided to give Ubuntu Jammy (LTS) base image a try.

## ubuntuJammy_vibro_20220310.def >>> SUCCESS

- Works, and builds, does take a while and the file is about 1.2GB
- However, installed conda in a folder named /root and requires privilege to access, thus have issue running conda when on the cluster.

## alpine_conda.def >>> SUCCESS

- Translated from the dockerfile maintained by frolvlad.
- This image address the glibc that is needed by Conda Python yet missing in Alpine Linux.
- The image is much smaller, 71MB, however, complexity of using a glibc workaround may cause issue later on.

## continuumio_miniconda3.def >>> 

- Had to drop all the package version restrictions and set Python at 3.10 instead of 3.11.0.

## continuumio_miniconda3_alpine_22_11_1.def >>> SUCCESS

- Had to drop all the package version restrictions and set Python at 3.10 instead of 3.11.0.

## continuumio_miniconda3_22_11_1 >>> SUCCESS

- Restricting to a specific version of the image 22.11.1 and drop Python version restriction to 3.10