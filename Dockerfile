FROM python:3.13-slim

WORKDIR /app

# Copy dependency files first
COPY requirements.txt pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir .

# Copy remaining project files
COPY . .

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords punkt_tab

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]