FROM python:3.6-jessie

COPY docker-requirements.txt /tmp/
COPY predictor/ /opt/predictor
WORKDIR /opt/predictor

RUN pip3 install --upgrade setuptools pip && \
    pip3 install --no-cache-dir -r /tmp/docker-requirements.txt


ENTRYPOINT ["python3", "/opt/predictor/run.py"]
