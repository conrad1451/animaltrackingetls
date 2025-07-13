# CHQ: Gemini AI generated the following file

import os
from flask import Flask, Response, request, send_file
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
# Get OpenWeatherMap API key from environment variable
# IMPORTANT: Replace 'YOUR_OPENWEATHERMAP_API_KEY' with your actual key
# or set it as an environment variable named OPENWEATHERMAP_API_KEY
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', 'YOUR_OPENWEATHERMAP_API_KEY')

# Base URL for OpenWeatherMap tile API
OPENWEATHERMAP_TILE_URL = "https://tile.openweathermap.org/map/{layer}/{z}/{x}/{y}.png"

# --- Helper Functions ---

def fetch_image_from_url(url):
    """
    Fetches an image from a given URL.
    Returns the image content if successful, None otherwise.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from {url}: {e}")
        return None

def transform_image(image_bytes, layer, z, x, y):
    """
    Applies a simple transformation (text overlay) to the image.
    In a real ETL, this could be resizing, watermarking, format conversion, etc.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        draw = ImageDraw.Draw(img)

        # Try to load a default font, or use a generic one if not found
        try:
            # You might need to adjust the font path based on your OS
            # For example, on Windows: "arial.ttf"
            # On Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font = ImageFont.truetype("arial.ttf", 16) # Adjust font size as needed
        except IOError:
            font = ImageFont.load_default() # Fallback to default font

        text_to_add = f"Layer: {layer}\nZ:{z} X:{x} Y:{y}"
        text_color = (255, 255, 255) # White color
        text_outline_color = (0, 0, 0) # Black outline

        # Get text size
        bbox = draw.textbbox((0, 0), text_to_add, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position text (e.g., top-left corner with some padding)
        x_pos = 10
        y_pos = 10

        # Draw text with outline for better visibility
        # Draw outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0: # Avoid drawing directly on the text itself for outline
                    draw.text((x_pos + dx, y_pos + dy), text_to_add, font=font, fill=text_outline_color)
        # Draw main text
        draw.text((x_pos, y_pos), text_to_add, font=font, fill=text_color)

        # Save the transformed image to a BytesIO object
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG") # Save as PNG
        output_buffer.seek(0)
        return output_buffer
    except Exception as e:
        print(f"Error transforming image: {e}")
        return None

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Simple index page to provide instructions.
    """
    return """
    <h1>OpenWeatherMap ETL Proxy</h1>
    <p>This Flask server acts as an ETL (Extract, Transform, Load) proxy for OpenWeatherMap weather tiles.</p>
    <p>To use, access the tiles via:</p>
    <code>/tile/&lt;layer&gt;/&lt;z&gt;/&lt;x&gt;/&lt;y&gt;</code>
    <p>
        Replace <code>&lt;layer&gt;</code> with the weather layer (e.g., <code>temp_new</code>, <code>pressure_new</code>, <code>wind_new</code>, <code>precipitation_new</code>, <code>clouds_new</code>).<br>
        Replace <code>&lt;z&gt;</code>, <code>&lt;x&gt;</code>, <code>&lt;y&gt;</code> with the zoom level and tile coordinates.
    </p>
    <p>
        Example: <a href="/tile/temp_new/5/10/15">/tile/temp_new/5/10/15</a>
    </p>
    <p>Make sure your <code>OPENWEATHERMAP_API_KEY</code> environment variable is set!</p>
    """

@app.route('/tile/<layer>/<int:z>/<int:x>/<int:y>')
def get_weather_tile(layer, z, x, y):
    """
    ETL endpoint for OpenWeatherMap tiles.
    1. Extracts the tile from OpenWeatherMap.
    2. Transforms the tile (adds an overlay).
    3. Loads (serves) the transformed tile.
    """
    if OPENWEATHERMAP_API_KEY == 'YOUR_OPENWEATHERMAP_API_KEY':
        return "Error: OpenWeatherMap API key not set. Please set the OPENWEATHERMAP_API_KEY environment variable.", 500

    # 1. Extract: Construct the URL for the OpenWeatherMap tile
    openweathermap_url = OPENWEATHERMAP_TILE_URL.format(layer=layer, z=z, x=x, y=y)
    openweathermap_url += f"?appid={OPENWEATHERMAP_API_KEY}"

    print(f"Fetching tile from: {openweathermap_url}")
    image_data = fetch_image_from_url(openweathermap_url)

    if not image_data:
        return Response("Failed to fetch image from OpenWeatherMap.", status=500)

    # 2. Transform: Apply transformation to the image
    transformed_image_buffer = transform_image(image_data, layer, z, x, y)

    if not transformed_image_buffer:
        return Response("Failed to transform image.", status=500)

    # 3. Load: Serve the transformed image
    return send_file(transformed_image_buffer, mimetype='image/png')

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run:
    # 1. pip install Flask requests Pillow
    # 2. Set your API key:
    #    On Linux/macOS: export OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    #    On Windows (Command Prompt): set OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    #    On Windows (PowerShell): $env:OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    # 3. python your_flask_app_name.py
    # 4. Access in browser: http://127.0.0.1:5000/
    #    Example Tile: http://127.0.0.1:5000/tile/temp_new/5/10/15
    app.run(debug=True) # debug=True enables auto-reloading and better error messages
