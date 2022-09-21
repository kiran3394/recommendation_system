# start by pulling the python image
FROM python:3.8-alpine

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt
RUN
# switch working directory
WORKDIR /app

RUN apk add --no-cache gcc

RUN apk add --no-cache gcc

RUN apk update && apk add python-devel

RUN pip install --upgrade pip
# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["server.py" ]
