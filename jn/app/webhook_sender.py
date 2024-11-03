import requests
import subprocess
import os
import random
import string
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compress_video(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-crf', '28',  # Increase compression (lower quality, smaller file size)
        '-preset', 'veryslow',  # Use a slower preset for better compression
        '-acodec', 'aac',
        '-strict', 'experimental',
        '-vf', 'scale=720:-2',  # Reduce resolution to 720p
        output_path
    ]
    subprocess.run(command, check=True)

def generate_random_string(length=4):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def upload_to_0x0(file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(file_path, 'rb') as file:
                response = requests.post('https://0x0.st', files={'file': file})
            
            if response.status_code == 200:
                original_url = response.text.strip()
                # Extract the random part from the original URL
                random_part = original_url.split('/')[-1].split('.')[0]
                modified_url = f"https://0x0.st/{random_part}.mp4"
                return modified_url
            else:
                print(f"Upload attempt {attempt + 1} failed. Status code: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise Exception(f"Failed to upload file after {max_retries} attempts.")
        except requests.RequestException as e:
            print(f"Upload attempt {attempt + 1} failed due to network error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise Exception(f"Failed to upload file after {max_retries} attempts due to network errors.")

def send_webhook(webhook_url, video_url):
    """
    Sends a webhook with the video URL.

    Args:
        webhook_url (str): The URL to send the webhook to.
        video_url (str): The URL of the generated video.

    Returns:
        bool: True if the webhook was sent successfully, False otherwise.
    """
    try:
        payload = {
            "video_url": video_url
        }
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logging.info(f"Webhook sent successfully to {webhook_url}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending webhook: {str(e)}")
        return False
