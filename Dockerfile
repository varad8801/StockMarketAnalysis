FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port that the Flask app runs on
EXPOSE 5000

# Define the command to run your Flask app when the container starts
CMD python ./app.py 
