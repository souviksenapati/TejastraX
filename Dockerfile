FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

RUN mkdir -p /code/app/temp && chmod 777 /code/app/temp

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]