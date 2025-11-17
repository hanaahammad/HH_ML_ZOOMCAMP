# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and the dataset into the container
# The app.py and OnlineRetail.csv must be in the same directory as the Dockerfile
COPY . .

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Define the command to run the Streamlit application
# The CMD specifies the primary command that will be executed when the container starts
CMD ["streamlit", "run", "app.py"]