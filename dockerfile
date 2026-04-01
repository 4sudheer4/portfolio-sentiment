# Start with an official Python 3.11 image as the base
# (like starting with a fresh Linux machine that has Python pre-installed)
FROM python:3.11-slim

# Set the working directory inside the container
# (like doing: mkdir /app && cd /app)
WORKDIR /app

# Copy requirements.txt from your Mac into the container
COPY requirements.txt .

# Install all your Python packages inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Download the VADER sentiment dictionary
RUN python -m nltk.downloader vader_lexicon

# Copy the rest of your code into the container
COPY . .

# Tell Docker your app listens on port 5000
# (just documentation — doesn't actually open the port)
EXPOSE 5000

# The command to run when the container starts
# gunicorn is a production-grade server, better than Flask's built-in one
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]