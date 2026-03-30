# 1. Use a lightweight Python version
FROM python:3.9-slim

# 2. Set the folder where our app will live inside the container
WORKDIR /app

# 3. Copy the list of libraries first (helps with faster builds)
COPY requirements.txt .

# 4. Install the libraries (Flask, numpy, sklearn, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy everything else from your folder into the container
# This includes app.py, model.pkl, and the templates folder
COPY . .

# 6. Tell the container which port to use 
# Render uses 10000 by default for free services
EXPOSE 10000

# 7. Start the app
# We use gunicorn because it is much stronger for web traffic than "python app.py"
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
