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

def upload_file_to_b2(file_obj, filename: str, content_type: str = None) -> str:
    """
    Uploads a file object to Backblaze B2 and returns the public URL.
    """
    try:
        s3_client = get_s3_client()
        
        # Generate a unique filename to prevent overwrites
        ext = filename.split('.')[-1] if '.' in filename else ''
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
        # Format: https://<bucket_name>.s3.<region>.backblazeb2.com/<filename>
        b2_domain = B2_ENDPOINT.replace("https://", "")
        public_url = f"https://{B2_BUCKET_NAME}.{b2_domain}/{unique_filename}"
        
        return public_url

    except ClientError as e:
        print(f"Error uploading to B2: {e}")
        return None
