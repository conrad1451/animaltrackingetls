from flask import Flask, jsonify
import requests
import json
import os 

# ### How to Use:

# 1. **Run the Code:** Copy and paste the entire code block into a Python file (e.g., `app.py`) and run it. In this environment, you can simply run the cell.
# 2. **Access the Web Interface:** Once the Flask server starts, it will typically be accessible at `http://127.0.0.1:5000/` or `http://localhost:5000/` in your browser.
# 3. **Enter Addresses:** Try entering one of the following addresses into the form to see the mock data:
#     * `123 Main St, Anytown, USA`
#     * `456 Oak Ave, Somewhere, USA`
#     * `789 Pine Ln, Floodsburg, USA`
#     * You can also enter any other address to see the "No data found" message.

# ### Explanation of the ETL Process:

# * **Extract (`/process_flood_data` route):**
#     * The `request.form.get('address')` captures the user's input.
#     * Instead of `requests.get(...)` to a real API, `MOCK_FLOOD_DATA.get(...)` simulates fetching data.
#     * Error handling is included for potential issues during extraction (e.g., address not found in mock data).
# * **Transform (`/process_flood_data` route):**
#     * The `raw_flood_data` (the dictionary from the mock API) is processed.
#     * Specific fields (`address`, `flood_factor`, etc.) are selected.
#     * A new field, `risk_category`, is *derived* based on the `flood_factor` value. This is a common transformation step where new insights are generated from raw data.
# * **Load (`/process_flood_data` route):**
#     * The `transformed_data` dictionary is passed to `render_template_string()`.
#     * The HTML template then displays this processed data to the user, effectively "loading" it into the user interface. In a production ETL, this "load" step might involve writing to a database, a data warehouse, a file, or another system.

# This example provides a clear, runnable demonstration of a simple ETL workflow within a web application context, even with a simulated external API.


app = Flask(__name__)

# It's better to store your API key as an environment variable
ai_api_key = os.environ.get("GEMINI_API_KEY")


if not ai_api_key:
    print("Error: GEMINI_API_KEY environment variable not set.")

# --- Database Connection ---
def get_db_connection():
    conn = None
    try:
        # Parse the DATABASE_URL from environment variables
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is not set.")

        url = urlparse(db_url)
        conn = psycopg2.connect(
            host=url.hostname,
            database=url.path[1:],  # Remove leading slash from path
            user=url.username,
            password=url.password,
            port=url.port if url.port else 5432, # Default PostgreSQL port if not specified
            sslmode='require' # Neon requires SSL
        )
        print("Successfully connected to PostgreSQL database!") # For initial testing, you can leave this.
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# --- API Endpoints for Students ---

# GET all students
@app.route('/process_flood_data', methods=['GET'])
def get_students(input_address):

    # EXTRACT: extracting data based on user input
    user_address = requests.form.get(input_address)
    try:
        raw_data = requests.get(params=user_address)

        # TRANSFORM: transform the data to get new insights from the raw data

        # select address and flood factor from the raw data

        # derive new field, risk cateogry, from the flood_factor value
        # response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes
        response_json = response.json()

        # LOAD: load the data to the client calling this endpoint
        return jsonify(response_json)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error during API call: {e}"})
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding JSON response."})

 