FROM cawqlqyj/naoth:ultralyticsv8.1.42

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt


# TODO probably add helper script as well
# TODO there are two different usecases for docker on a local machine you want to start it and use the code on the host, in a k8s context you cant do that there you need to copy the files
COPY ./ /usr/src/ultralytics/

#WORKDIR /usr/src/datasets
WORKDIR /usr/src/ultralytics