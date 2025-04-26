# 1) Pick an official Python base image
FROM python:3.11.11

# 2) Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Set your working directory in the container
WORKDIR /app

# 4) Copy only requirements first (leverages Docker cache)
COPY requirements.txt /app/

# 5) Install Python deps
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 6) Copy the rest of your application code
COPY . /app

# 7) (Optional) expose whichever port your app listens on
EXPOSE 5000

# 8) Final command to run your app; change to app_finetuned.py or 
#    use gunicorn/uvicorn if you prefer
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
