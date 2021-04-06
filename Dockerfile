From python:3.8.5

WORKDIR /home/

RUN git clone https://github.com/wkddnjswns3793/pragmaticEmo.git

WORKDIR /home/pragmaticEmo/

RUN pip3 install -vvv --no-cache-dir -r requirements.txt

RUN pip3 install gunicorn

RUN apt-get update && apt-get install -y python3-opencv

RUN pip install opencv-python

RUN echo "SECRET_KEY=3#+t!lv&*#c2ykt)!$b1%equ(($l6gzvzi$8r$+lpboo+%z+e*" > .env

RUN python3 manage.py migrate

EXPOSE 8000

CMD ["gunicorn", "pragmatic.wsgi", "--bind", "0.0.0.0:8000"]