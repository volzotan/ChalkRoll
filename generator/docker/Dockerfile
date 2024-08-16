FROM python:3.12
WORKDIR /usr/src/generator

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generator .

RUN useradd app
USER app

#EXPOSE 5000
#CMD python3 app.py