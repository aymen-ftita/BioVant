import os
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
import uuid

# Backblaze B2 S3-compatible Credentials
# TODO: In production, store these in a .env file!
B2_KEY_ID = os.getenv("B2_KEY_ID", "003807cc139097e0000000001")
B2_APP_KEY = os.getenv("B2_APP_KEY", "K003vMFfMziTO2T9UWB5jZr3AHqSkVI")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "hypnoria-files")
B2_ENDPOINT = os.getenv("B2_ENDPOINT", "https://s3.eu-central-003.backblazeb2.com") 

def get_s3_client():
    return boto3.client(
        service_name='s3',
        endpoint_url=B2_ENDPOINT,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APP_KEY
    )

import random
import urllib.parse
import datetime

def upload_file_to_b2(file_obj, filename: str, content_type: str = None, patient_name: str = None, date_str: str = None) -> str:
    """
    Uploads a file object to Backblaze B2 and returns the public URL.
    Places the file in a structured folder named after the patient and names it starting with "patient's name_date_random number".
    """
    try:
        s3_client = get_s3_client()
        
        ext = filename.split('.')[-1] if '.' in filename else ''
        
        if patient_name:
            # Clean and sanitize patient name
            safe_patient_name = patient_name.strip()
            safe_patient_name = "".join(c for c in safe_patient_name if c.isalnum() or c in (" ", "-", "_"))
            
            if not date_str:
                date_str = datetime.date.today().isoformat()
            
            rand_num = random.randint(1000, 9999)
            # Name pattern: patient's name_date_random number
            file_key = f"{safe_patient_name}_{date_str}_{rand_num}.{ext}"
            
            # Implicit directory key
            unique_filename = f"{safe_patient_name}/{file_key}"
        else:
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
        
        ExtraArgs = {}
        if content_type:
            ExtraArgs['ContentType'] = content_type

        # Upload
        s3_client.upload_fileobj(
            file_obj,
            B2_BUCKET_NAME,
            unique_filename,
            ExtraArgs=ExtraArgs
        )
        
        # Construct the public URL
        b2_domain = B2_ENDPOINT.replace("https://", "")
        # Ensure spaces and slashes are fully safe in browsers
        encoded_filename = urllib.parse.quote(unique_filename)
        public_url = f"https://{B2_BUCKET_NAME}.{b2_domain}/{encoded_filename}"
        
        return public_url

    except Exception as e:
        print(f"Error uploading to B2: {e}")
        return None

def get_presigned_download_url(file_url: str) -> str:
    """
    Given a public/private B2 file URL, generates a secure pre-signed download/view URL
    that bypasses bucket visibility constraints and works for private buckets.
    The returned URL is valid for 24 hours.
    """
    if not file_url:
        return None
    try:
        # If it is not a Backblaze B2 domain, or is already pre-signed, return as is
        if "backblazeb2.com" not in file_url:
            return file_url
        if "AWSAccessKeyId=" in file_url or "Signature=" in file_url:
            return file_url
            
        parsed_url = urllib.parse.urlparse(file_url)
        path = urllib.parse.unquote(parsed_url.path.lstrip("/"))
        
        # Strip the bucket name from the key if it resides in the URL path segment
        if path.startswith(f"{B2_BUCKET_NAME}/"):
            key = path[len(f"{B2_BUCKET_NAME}/"):]
        else:
            key = path
            
        s3_client = get_s3_client()
        params = {
            'Bucket': B2_BUCKET_NAME,
            'Key': key
        }
        
        # Force download for raw EDF or CSV files, while keeping hypnograms (PNG) and reports (HTML) viewable inline in the browser
        if key.lower().endswith(('.edf', '.csv')):
            filename = key.split('/')[-1]
            params['ResponseContentDisposition'] = f'attachment; filename="{filename}"'

        presigned_url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params=params,
            ExpiresIn=86400  # Valid for 24 hours
        )
        return presigned_url
    except Exception as e:
        print(f"Error generating B2 presigned URL: {e}")
        return file_url

