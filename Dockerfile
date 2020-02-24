FROM python:3.7

COPY . /nailgun
WORKDIR /nailgun

RUN pip install -r requirements.txt
RUN python setup.py install

CMD exec gunicorn -b :5000 --access-logfile - --error-logfile - nailgun.api:app
