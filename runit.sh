# docker run --rm -it --memory=4g -p 8889:8889 --mount type=bind,src="$(pwd)",dst=/home/jovyan/work ragsc bash
docker run --rm -it --memory=4g -p 8889:8889 -v "$(pwd):/home/jovyan/work" ragsc bash
