FROM python:3.7

RUN mkdir /flask
WORKDIR /flask
ADD /flask/requirements.txt /flask/
RUN pip install -r requirements.txt
ADD /flask/ /flask/

RUN apt-get update \
        && apt-get install -y --no-install-recommends dialog \
        && apt-get update \
	&& apt-get install -y libglib2.0-0

EXPOSE 8000 2222 5000
CMD ["flask", "run", "--host", "0.0.0.0"]
