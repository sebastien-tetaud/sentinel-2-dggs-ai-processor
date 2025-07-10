import re
import os
from io import BytesIO
from PIL import Image

def extract_url_after_filename(url):
    """Extract the filename from the URL."""
    match = re.search(r'\?filename=(.*)', url)
    return match.group(1) if match else None

def convert_jp2_to_image(content):
    """Convert JP2 image content to PIL Image."""
    try:
        return Image.open(BytesIO(content))
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
