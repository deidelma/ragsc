# docker run -it -p 8889 --mount='type=bind,src=/mnt/md0/david/ragsc,dst=/home/jovyan/work' ragsc bash
docker run --rm -it --memory=4g -p 8889:8889 --mount type=bind,src="$(pwd)",dst=/home/jovyan/work ragsc bash
