import requests
import json
import base64
import time
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if it exists)
load_dotenv()

# --- Configuration ---
# Read API Key from environment variable 'GEMINI_API_KEY'.
API_KEY = os.getenv("GEMINI_API_KEY", "")
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"{API_URL_BASE}{MODEL_NAME}:generateContent?key={API_KEY}"

# --- Utility Functions ---

def bytes_to_base64(image_bytes: bytes) -> str:
    """
    Converts raw image bytes to a base64 encoded string.
    This is used to prepare the image for the Gemini API payload.
    """
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return encoded_string

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
        return {"error": "API Key is missing"}, 401

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
                return None

        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                # Retry on rate limit (429) or server errors (500, 503)
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                print(f"Fatal HTTP error or failed after max retries: {e}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding the JSON response from the model: {e}")
            return None

    return None