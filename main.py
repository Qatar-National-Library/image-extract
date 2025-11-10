import requests
import json
import base64
import time
import os
import random
from dotenv import load_dotenv # NEW: Import load_dotenv

# Load environment variables from a .env file (if it exists)
# The .env file should contain a line like: GEMINI_API_KEY="YOUR_API_KEY"
load_dotenv()

# --- Configuration ---
# Read API Key from environment variable 'GEMINI_API_KEY'.
# It will check the system environment first, and then the variables loaded from .env.
API_KEY = os.getenv("GEMINI_API_KEY", "")
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"{API_URL_BASE}{MODEL_NAME}:generateContent?key={API_KEY}"

# --- Utility Functions ---

def image_to_base64(image_path: str, mime_type: str = "image/jpeg") -> str | None:
    """
    Converts a local image file to a base64 encoded string.
    This simulates how image data needs to be passed to the Gemini API.
    
    NOTE: For this example to run, you must replace 'path/to/your/image.jpg' with a 
    valid image file path on your system.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the image: {e}")
        return None

# --- Main Function ---

def process_image_to_json(
    image_base64: str,
    mime_type: str,
    prompt: str,
    response_schema: dict,
    max_retries: int = 5
) -> dict | None:
    """
    Uses the Gemini API to analyze an image based on a prompt and return 
    the result structured according to the provided JSON schema.

    Args:
        image_base64: The base64-encoded string of the image data.
        mime_type: The MIME type of the image (e.g., 'image/jpeg', 'image/png').
        prompt: The specific question or task for the model regarding the image.
        response_schema: The JSON schema dictionary defining the required output structure.
        max_retries: Maximum number of retries for the API call (for backoff).

    Returns:
        A dictionary (the parsed JSON response) or None if the call fails.
    """
    # 1. Check for API Key
    if not API_KEY:
        print("Error: API Key is missing. Please set GEMINI_API_KEY in your environment or .env file.")
        return None

    # 2. Define the System Instruction (Model's Role)
    system_instruction = (
        "You are an expert visual data extractor and summarizer. "
        "Your task is to analyze the provided image and generate a concise, "
        "accurate JSON object that strictly adheres to the given schema and "
        "answers the user's prompt. Do not include any external commentary."
    )

    # 3. Construct the API Payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }

    # 4. Perform the API Call with Exponential Backoff
    for attempt in range(max_retries):
        try:
            print(f"Attempting API call (Attempt {attempt + 1}/{max_retries})...")
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            
            # Check for content in the response structure
            candidate = result.get('candidates', [{}])[0]
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                json_string = candidate['content']['parts'][0]['text']
                # The model returns a string that represents the JSON structure, so we parse it.
                return json.loads(json_string)
            else:
                print("API response received but contained no generated content.")
                return None

        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                # Retry on rate limit (429) or server errors (500, 503)
                wait_time = 2 ** attempt
                print(f"HTTP Error {response.status_code}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Fatal HTTP error or failed after max retries: {e}")
                print(f"API Response Text: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding the JSON response from the model: {e}")
            return None

    print("Max retries reached. Failed to get a structured response.")
    return None

# --- Example Usage ---

if __name__ == "__main__":
    # --- STEP 1: Define your JSON Schema for ID Card Analysis ---
    ID_CARD_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "Name": {"type": "STRING", "description": "The full name of the person."},
            "IDNumber": {"type": "STRING", "description": "The national ID or driver's license number."},
            "DateOfBirth": {"type": "STRING", "description": "The date of birth, standardized to YYYY-MM-DD format if possible."},
            "IDExpiryDate": {"type": "STRING", "description": "The ID card expiration date, standardized to YYYY-MM-DD format if present. Use an empty string if not found."},
            "PassportNumber": {"type": "STRING", "description": "The passport number. Use an empty string if not present."},
            "PassportExpiryDate": {"type": "STRING", "description": "The passport expiration date, standardized to YYYY-MM-DD format if present. Use an empty string if not found."},
            "Occupation": {"type": "STRING", "description": "The occupation or job title of the person, if listed on the ID. Use an empty string if not found."}
        },
        # These fields MUST be included in the output JSON
        "required": ["Name", "IDNumber", "DateOfBirth","IDExpiryDate"]
    }

    # --- STEP 2: Encode the Image ---
    # NOTE: YOU MUST CHANGE THIS PATH TO A REAL IMAGE FILE (e.g., a photo of an ID or passport)!
    MOCK_IMAGE_PATH = f"images/{random.randint(1, 4)}.jpg"
    MOCK_IMAGE_PATH = f"images/4.jpg"
    IMAGE_MIME_TYPE = "image/jpeg"

    # For demonstration, we check if the file exists before attempting conversion
    if not os.path.exists(MOCK_IMAGE_PATH):
        print("\n*** IMPORTANT ***")
        print(f"Please replace '{MOCK_IMAGE_PATH}' with the actual path to an ID card or passport image file on your computer.")
        print(f"Also, ensure you have python-dotenv installed (`pip install python-dotenv`) and a .env file is present.")
    else:
        print(f"Attempting to encode image from: {MOCK_IMAGE_PATH}")
        base64_data = image_to_base64(MOCK_IMAGE_PATH, IMAGE_MIME_TYPE)

        if base64_data:
            # --- STEP 3: Define the Prompt ---
            analysis_prompt = (
                "Analyze the identification document in this image. Extract the Name, ID Number, "
                "and Date of Birth. Also find and include the ID Expiry Date, Passport Number, "
                "Passport Expiry Date, and Occupation if they are present on the document. "
                "If a field is not present, return an empty string for that field."
            )

            # --- STEP 4: Call the Processor ---
            print("\n--- Starting Gemini API structured VLM call for ID Analysis ---")
            
            structured_result = process_image_to_json(
                image_base64=base64_data,
                mime_type=IMAGE_MIME_TYPE,
                prompt=analysis_prompt,
                response_schema=ID_CARD_SCHEMA
            )

            # --- STEP 5: Print the Result ---
            print("\n--- Final Structured ID Card Data ---")
            if structured_result:
                print(json.dumps(structured_result, indent=2))
            else:
                print("Failed to get a structured result.")