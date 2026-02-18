# base image 
FROM python:3.11  

# workdir
WORKDIR /app

# COPY
COPY . /app

# run 
RUN pip install -r requirements.xt

#port 
EXPOSE 8080

# commands 
CMP ['python3','demo.py']