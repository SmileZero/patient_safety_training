# python2 as the base image
FROM python:2

# Install the packages
RUN apt-get update && \
    apt-get -y install bash nginx supervisor

# install uwsgi now because it takes a little while
RUN pip install uwsgi

# setup all the config files
COPY patient_safety_training.conf /etc/nginx/sites-available/default
COPY supervisor-app.conf /etc/supervisor/conf.d/

RUN mkdir -p /var/www/patient_safety_training

# Open the following ports
EXPOSE 8080

WORKDIR /var/www/patient_safety_training/

ADD . /var/www/patient_safety_training/

RUN pip install -r requirements.txt

# Define the command which runs when the container starts
CMD ["supervisord", "-n", "-c", "supervisor-app.conf"]
