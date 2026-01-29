FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your files (including the brain folder)
COPY . .

# create a cache folder that Hugging Face needs
RUN mkdir -p /code/.cache && chmod 777 /code/.cache

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]