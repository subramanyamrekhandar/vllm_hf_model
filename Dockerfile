FROM python:3.10-slim       

# Set working directory
WORKDIR /app                
# Copy all files into container
COPY . .                    

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt  

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]  