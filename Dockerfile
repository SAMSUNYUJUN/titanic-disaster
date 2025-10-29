# lightweight base
FROM python:3.11-slim

# avoid python writing .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# workdir
WORKDIR /app

# copy requirements and install first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY src /app/src

# default command: run the analysis script
CMD ["python", "src/app/main.py"]
