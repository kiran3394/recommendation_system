# start by pulling the python image
FROM python:3.6-alpine

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

RUN apk update && apk add --virtual build-deps gcc python3-dev musl-dev

RUN pip install --upgrade pip
# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

EXPOSE 5000
CMD ["python", "app.py"]
