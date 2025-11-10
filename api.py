import json
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import HTMLResponse
from image_processor import process_image_to_json, bytes_to_base64
import os

app = FastAPI(
    title="Gemini ID Card Data Extractor",
    description="An API to extract structured data from uploaded ID card images using the Gemini Vision Model."
)

# Define the path to the HTML template file
INDEX_HTML = "html/index.html" 

def get_html_content(file_path: str = INDEX_HTML) -> str:
    """Reads and returns the content of the local HTML file."""
    try:
        # Use 'with' statement for file handling to ensure file is closed
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Return a simple error message if the file is missing
        print(f"Error: HTML input form file not found at {file_path}")
        return "<h1>Error: HTML input form not found. Please ensure 'index.html' is in the same directory.</h1>"

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the main HTML interface for the application, reading content from index.html."""
    html_content = get_html_content()
    # If the content is the error message, use status 500
    if "Error: HTML input form not found" in html_content:
        return HTMLResponse(content=html_content, status_code=500)
    
    return HTMLResponse(content=html_content, status_code=200)


# --- JSON Schema Definition ---
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
    "required": ["Name", "IDNumber", "DateOfBirth"]
}

# --- Prompt Definition ---
ANALYSIS_PROMPT = (
    "Analyze the identification document in this image. Extract the Name, ID Number, "
    "and Date of Birth. Also find and include the ID Expiry Date, Passport Number, "
    "Passport Expiry Date, and Occupation if they are present on the document. "
    "If a field is not present, return an empty string for that field."
)


@app.post("/extract")
async def extract_id_data(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (ID card or Passport) and returns extracted data
    in a structured JSON format using the Gemini API.
    """
    
    # 1. Read file content from the upload
    try:
        # FastAPI's UploadFile reads the data into memory
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not read file content.")

    # 2. Get MIME type
    mime_type = file.content_type
    
    # Check if the content type is an image
    if not mime_type or not mime_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {mime_type}. Only image files are supported."
        )

    # 3. Convert bytes to base64
    base64_data = bytes_to_base64(contents)
    
    if not base64_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode image data for API submission."
        )

    # 4. Call the Gemini Processor
    structured_result = process_image_to_json(
        image_base64=base64_data,
        mime_type=mime_type,
        prompt=ANALYSIS_PROMPT,
        response_schema=ID_CARD_SCHEMA
    )

    if isinstance(structured_result, dict) and structured_result.get('error'):
         # Extract the specific error message if available, otherwise use a generic message
         error_message = structured_result.get('error', 'Authentication failed or API key missing.')
         # Using 401 Unauthorized here since the most common error from the utility function is API key missing.
         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=error_message)


    # 5. Handle the result
    if structured_result:
        return structured_result
    else:
        # If the API returns None (due to failure or no content)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API call failed or returned an unstructured response after multiple retries."
        )