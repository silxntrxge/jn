import json
import os
import uuid
import requests
import tempfile
from io import BytesIO

import numpy as np
from PIL import Image, ImageSequence
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    CompositeAudioClip,
    ImageSequenceClip,
    concatenate_videoclips
)
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import ffmpy
import io
import math  # Added for ceiling function
import logging  # Added for improved logging
import imageio
from moviepy.video.VideoClip import VideoClip
import os
import stat
import gc
import psutil
import multiprocessing
from PIL import Image, ImageDraw, ImageFont
from moviepy.config import change_settings
from bs4 import BeautifulSoup
import re
import imghdr  # Import to check image type
import mimetypes
from time import time as current_time  # Update the time import at the top
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
import subprocess
import shutil
import cv2

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the ImageMagick configuration after imports
magick_home = os.environ.get('MAGICK_HOME', '/usr')
imagemagick_binary = os.path.join(magick_home, "bin", "convert")

if os.path.exists(imagemagick_binary):
    change_settings({"IMAGEMAGICK_BINARY": imagemagick_binary})
    logging.info(f"ImageMagick binary set to: {imagemagick_binary}")
else:
    logging.warning(f"ImageMagick binary not found at {imagemagick_binary}. Using default.")

# Set debug flag
DEBUG = os.environ.get('DEBUG', '0') == '1'

def extract_image_url(url):
    """
    Extracts direct image URL from webpage if needed.
    Returns the direct URL for downloading.
    """
    try:
        # Check if it's already a direct image URL
        if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
            return url
            
        # If URL contains 'pexels.com' and looks like an image URL
        if 'pexels.com' in url.lower() and 'photos' in url.lower():
            return url  # Pexels URLs can be used directly
            
        # For other cases, try to fetch the webpage and find image
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find the main image
        img_tags = soup.find_all('img')
        for img in img_tags:
            # Look for likely main image tags
            src = img.get('src') or img.get('data-src') or img.get('srcset')
            if src:
                # If srcset, take the first URL
                if ',' in str(src):
                    src = src.split(',')[0].strip().split(' ')[0]
                return src
                
        return url  # Return original URL if no better option found
    except Exception as e:
        logging.warning(f"Error extracting image URL from {url}: {e}")
        return url

def download_file(url, suffix=''):
    """
    Downloads a file from the specified URL to a temporary file.
    Returns tuple of (file_path, detected_extension)
    """
    try:
        # Get the direct download URL
        direct_url = extract_image_url(url)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Download the file
        response = requests.get(direct_url, stream=True, headers=headers)
        response.raise_for_status()
        
        # Create temporary file without extension first
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        # Write the content
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
        temp_file.close()

        # Detect the actual file type
        detected_type = imghdr.what(temp_file.name)
        if detected_type:
            actual_extension = f'.{detected_type}'
        else:
            # Fallback to content-type header
            content_type = response.headers.get('content-type', '')
            actual_extension = mimetypes.guess_extension(content_type) or suffix

        # Rename the file with correct extension
        new_path = f"{temp_file.name}{actual_extension}"
        os.rename(temp_file.name, new_path)
        
        # Set permissions
        os.chmod(new_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
        
        return new_path, actual_extension

    except Exception as e:
        logging.error(f"Error downloading file from {url}: {e}")
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        return None, None


def parse_percentage(value, total, video_height=None):
    """
    Parses a percentage string or vmin value and converts it to an absolute value.

    Args:
        value (str or int or float): The value to parse (e.g., "50%", "0.4 vmin", 100).
        total (int): The total value to calculate the percentage against.
        video_height (int): The height of the video, used for vmin calculations.

    Returns:
        int: The absolute value.
    """
    # Handle None or empty values
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0
        
    # Handle numeric values
    if isinstance(value, (int, float)):
        return int(value)
        
    if isinstance(value, str):
        value = value.strip().lower()
        if value.endswith('%'):
            try:
                percentage = float(value.rstrip('%'))
                percentage = max(0, min(percentage, 100))  # Clamp between 0% and 100%
                return int((percentage / 100) * total)
            except ValueError:
                logging.error(f"Invalid percentage value: {value}")
                return 0
        elif 'vmin' in value:
            try:
                vmin_value = float(value.replace('vmin', '').strip())
                if video_height is None:
                    logging.error("Video height is required for vmin calculations")
                    return 0
                # For stroke width, use the actual vmin value without percentage conversion
                min_dimension = total  # total should be min(video_width, video_height)
                return max(1, int(vmin_value * (min_dimension / 100)))
            except ValueError:
                logging.error(f"Invalid vmin value: {value}")
                return 0
    return 0


def parse_size(size_str, reference_size, video_width, video_height):
    """
    Parses size strings which can be percentages or absolute values.

    Args:
        size_str (str or int or float): The size string to parse.
        reference_size (int): The reference size (width or height).
        video_width (int): The width of the video.
        video_height (int): The height of the video.

    Returns:
        int or None: The parsed size in pixels or None if invalid.
    """
    if size_str is None:
        return None
    if isinstance(size_str, (int, float)):
        return max(0, min(int(size_str), reference_size))
    if isinstance(size_str, str):
        size_str = size_str.strip().lower()
        if size_str.endswith('%'):
            return parse_percentage(size_str, reference_size)
        elif size_str.endswith('vmin'):
            try:
                vmin_value = float(size_str.rstrip('vmin').strip())
                vmin = min(video_width, video_height)
                vmin_value = max(0, min(vmin_value, 100))
                return int((vmin_value / 100) * vmin)
            except ValueError:
                print(f"Invalid vmin value: {size_str}")
        else:
            try:
                return max(0, min(int(float(size_str)), reference_size))
            except ValueError:
                print(f"Invalid size format: {size_str}")
    return None


def resize_clip(clip, target_width, target_height):
    """
    Resizes the clip to cover the target dimensions while maintaining aspect ratio.

    Args:
        clip (VideoFileClip or ImageClip or ImageSequenceClip or TextClip): The clip to resize.
        target_width (int): The target width in pixels.
        target_height (int): The target height in pixels.

    Returns:
        Clip: The resized clip.
    """
    original_ratio = clip.w / clip.h
    target_ratio = target_width / target_height

    if original_ratio > target_ratio:
        # Clip is wider than target: set height to target and scale width
        new_height = target_height
        new_width = int(new_height * original_ratio)
    else:
        # Clip is taller than target: set width to target and scale height
        new_width = target_width
        new_height = int(new_width / original_ratio)

    return clip.resize(width=new_width, height=new_height)


def position_clip(clip, x, y):
    """
    Positions the clip based on x and y coordinates.

    Args:
        clip (Clip): The clip to position.
        x (int): The x position in pixels (top-left based).
        y (int): The y position in pixels (top-left based).

    Returns:
        Clip: The positioned clip.
    """
    return clip.set_position((x, y))


def create_audio_clip(element):
    """
    Creates an audio clip from the provided element.

    Args:
        element (dict): The JSON element for the audio.

    Returns:
        AudioFileClip or None: The created audio clip or None if failed.
    """
    source = element.get('source')
    start_time = element.get('time', 0.0)
    duration = element.get('duration')

    if not source:
        print(f"Audio element {element['id']} has no source.")
        return None

    temp_audio, _ = download_file(source, suffix='.mp3')  # Now handling tuple return
    if not temp_audio:
        return None

    try:
        audio_clip = AudioFileClip(temp_audio).set_start(start_time)
        if duration:
            audio_clip = audio_clip.subclip(0, duration)
        volume = element.get('volume')
        if volume:
            try:
                volume_value = float(volume.rstrip('%')) / 100
                audio_clip = audio_clip.volumex(volume_value)
            except ValueError:
                print(f"Invalid volume value for audio element: {element['id']}, using default volume.")
        audio_clip.name = element['id']
        audio_clip.track = element.get('track', 0)
        return audio_clip
    except Exception as e:
        print(f"Error creating audio clip for element {element['id']}: {e}")
        return None
    finally:
        if temp_audio and os.path.exists(temp_audio):
            os.unlink(temp_audio)


def process_gif_with_ffmpeg(gif_path, duration, output_path):
    """
    Processes the GIF using FFmpeg to loop it until the specified duration and save as a video file.
    
    Args:
        gif_path (str): Path to the original GIF file.
        duration (float): Desired duration in seconds.
        output_path (str): Path to save the processed video file.
    
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        ff = ffmpy.FFmpeg(
            inputs={gif_path: None},
            outputs={output_path: f'-stream_loop -1 -t {duration} -c:v libx264 -pix_fmt yuv420p -y'}
        )
        ff.run()
        logging.info(f"Processed GIF with FFmpeg: {gif_path} -> {output_path}")
        return True
    except ffmpy.FFmpegError as e:
        logging.error(f"FFmpeg error while processing GIF {gif_path}: {e}")
        return False


# Add this helper function at the top level, before create_image_clip
def get_element_position_value(element, key, default_value):
    """
    Helper function to get positioning values with defaults.
    
    Args:
        element (dict): The element configuration
        key (str): The key to look for
        default_value (str): The default value if key is not found
        
    Returns:
        str: The value to use
    """
    return element.get(key, default_value)

# Modify create_image_clip function to add default values handling
def create_image_clip(element, video_width, video_height):
    """
    Creates an image clip from the provided element.
    """
    source = element.get('source')
    start_time = element.get('time', 0.0)
    duration = element.get('duration')
    repeat = element.get('repeat', False)
    speed_factor = element.get('speed', 1.0)

    if not source:
        logging.error(f"Image/GIF element {element['id']} has no source.")
        return None

    temp_image, detected_ext = download_file(source)
    if not temp_image:
        logging.error(f"Failed to download file from {source} for element {element['id']}.")
        return None

    try:
        # Check if the file is actually a GIF
        is_gif = detected_ext.lower() == '.gif'
        
        if is_gif:
            # Handle GIF
            gif = imageio.get_reader(temp_image)
            frames = []
            durations = []
            
            try:
                for frame in gif:
                    frames.append(frame)
                    durations.append(frame.meta.get('duration', 100) / 1000)  # Convert to seconds
            except Exception as e:
                logging.error(f"Error reading GIF frames: {e}")
                return None

            original_duration = sum(durations)
            frame_count = len(frames)

            if duration and repeat:
                loop_count = math.ceil(duration / original_duration)
                frames = frames * loop_count
                durations = durations * loop_count
                total_duration = loop_count * original_duration
            else:
                total_duration = original_duration

            def make_frame(t):
                t = (t * speed_factor) % total_duration
                frame_index = 0
                accumulated_time = 0
                for i, d in enumerate(durations):
                    if accumulated_time + d > t:
                        frame_index = i
                        break
                    accumulated_time += d
                return frames[frame_index % frame_count]

            clip = VideoClip(make_frame, duration=duration or total_duration)
        else:
            # Handle static image
            img = Image.open(temp_image)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGBA')
                mask = np.array(img.split()[-1]) / 255.0
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb)
                clip = ImageClip(img_array).set_mask(ImageClip(mask, ismask=True))
            else:
                img = img.convert("RGB")
                img_array = np.array(img)
                clip = ImageClip(img_array)

        # Set clip duration and start time
        clip = clip.set_duration(duration or None).set_start(start_time)

        # Define default values
        default_values = {
            'x': "50%",        # Center horizontally
            'y': "50%",        # Center Vertically
            'width': "100%",   # Full width
            'height': "100%",  # Full height
            'x_anchor': "50%", # Center anchor
            'y_anchor': "50%"  # Center anchor
        }

        # Helper function to get value or default
        def get_valid_value(key):
            value = element.get(key, '')
            if value is None or value == '':
                return default_values[key]
            return value

        # Check if ALL positioning/sizing attributes are missing
        if all(element.get(attr) is None or element.get(attr) == '' 
               for attr in ['width', 'height', 'x', 'y']):
            # If none are specified, make the image cover the entire video
            aspect_ratio = clip.w / clip.h
            video_aspect_ratio = video_width / video_height
            
            if aspect_ratio > video_aspect_ratio:
                new_height = video_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = video_width
                new_height = int(new_width / aspect_ratio)
            
            resized_clip = clip.resize(height=new_height, width=new_width)
            x_offset = (video_width - new_width) // 2
            y_offset = (video_height - new_height) // 2
            final_clip = resized_clip.set_position((x_offset, y_offset))
        else:
            # Get values with defaults
            width_value = get_valid_value('width')
            height_value = get_valid_value('height')
            x_value = get_valid_value('x')
            y_value = get_valid_value('y')
            
            # Parse dimensions
            target_width = parse_percentage(width_value, video_width)
            target_height = parse_percentage(height_value, video_height)
            
            # Calculate scaling to maintain aspect ratio while filling the target area
            aspect_ratio = clip.w / clip.h
            target_ratio = target_width / target_height
            
            if aspect_ratio > target_ratio:
                # Image is wider than target area - scale to height and crop width
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
                resized_clip = clip.resize(height=new_height)
                # Crop excess width from center
                excess_width = new_width - target_width
                x1 = excess_width // 2
                resized_clip = resized_clip.crop(x1=x1, width=target_width)
            else:
                # Image is taller than target area - scale to width and crop height
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
                resized_clip = clip.resize(width=new_width)
                # Crop excess height from center
                excess_height = new_height - target_height
                y1 = excess_height // 2
                resized_clip = resized_clip.crop(y1=y1, height=target_height)

            # Calculate position
            x_pos = parse_percentage(x_value, video_width)
            y_pos = parse_percentage(y_value, video_height)
            
            # Position relative to center
            x_pos = x_pos - (target_width / 2)
            y_pos = y_pos - (target_height / 2)
            
            final_clip = resized_clip.set_position((x_pos, y_pos))

            # If there are animations, adjust the initial position based on animation anchor points
            if element.get('animations'):
                final_clip = apply_animations(final_clip, element, duration or clip.duration, video_width, video_height)

        final_clip = final_clip.set_start(start_time)
        final_clip.name = element['id']
        final_clip.track = element.get('track', 0)

        # Get the first scale animation if it exists
        scale_anim = next((anim for anim in element.get('animations', []) 
                          if anim.get('type') == 'scale'), None)
        
        if scale_anim:
            # Get anchor points from animation
            x_anchor = float(scale_anim.get('x_anchor', '50%').rstrip('%')) / 100
            y_anchor = float(scale_anim.get('y_anchor', '50%').rstrip('%')) / 100
            
            # Calculate initial position based on anchor points
            # For x=0: left edge of screen
            # For x=0.5: center of screen
            # For x=1: right edge of screen
            x_pos = (video_width - final_clip.w) * x_anchor
            y_pos = (video_height - final_clip.h) * y_anchor
            
            logging.info(f"""
            Setting initial position:
            Video dimensions: {video_width}x{video_height}
            Element dimensions: {final_clip.w}x{final_clip.h}
            Anchor points: x={x_anchor*100}%, y={y_anchor*100}%
            Calculated position: {x_pos}, {y_pos}
            """)
            
            # Set the initial position
            final_clip = final_clip.set_position((x_pos, y_pos))
        
        # Apply animations (scaling will work from this initial position)
        if element.get('animations'):
            final_clip = apply_animations(final_clip, element, duration, video_width, video_height)
            
        return final_clip

    except Exception as e:
        logging.error(f"Error creating image/GIF clip for element {element['id']}: {e}")
        return None
    finally:
        os.unlink(temp_image)


def create_text_clip(element, video_width, video_height, total_duration):
    """
    Creates a text clip from the provided element using PIL for stroke support.
    """
    # Get basic text element properties
    text = element.get('text')
    start_time = element.get('time', 0.0)
    duration = element.get('duration')
    raw_font_size = element.get('font_size', "5%")
    font_url = element.get('font_family')

    # Log the incoming JSON data for stroke parameters
    logging.info("\n=== Text Element Stroke Parameters ===")
    logging.info("Received JSON data:")
    logging.info(f"  - Element ID: {element.get('id')}")
    logging.info(f"  - Raw stroke_color: {element.get('stroke_color')}")
    logging.info(f"  - Raw stroke_width: {element.get('stroke_width')}")

    try:
        # Parse stroke parameters
        stroke_color = element.get('stroke_color')
        stroke_width_raw = element.get('stroke_width')
        stroke_width_px = 0

        logging.info("\nProcessing stroke parameters:")
        if stroke_color and stroke_width_raw:
            if isinstance(stroke_width_raw, str) and 'vmin' in stroke_width_raw.lower():
                try:
                    vmin_value = float(stroke_width_raw.split('vmin')[0].strip())
                    min_dimension = min(video_width, video_height)
                    stroke_width_px = int(vmin_value * (min_dimension / 100))
                    stroke_width_px = max(1, stroke_width_px)
                    
                    logging.info("Converting vmin to pixels:")
                    logging.info(f"  - Original vmin value: {stroke_width_raw}")
                    logging.info(f"  - Numeric vmin value: {vmin_value}")
                    logging.info(f"  - Video dimensions: {video_width}x{video_height}")
                    logging.info(f"  - Min dimension: {min_dimension}")
                    logging.info(f"  - Calculation: {vmin_value} * ({min_dimension}/100)")
                    logging.info(f"  - Final pixel width: {stroke_width_px}px")
                except ValueError as e:
                    logging.error(f"Failed to parse vmin value: {e}")
                    stroke_width_px = 0
            else:
                try:
                    stroke_width_px = int(float(stroke_width_raw))
                    logging.info(f"Using direct pixel value: {stroke_width_px}px")
                except (ValueError, TypeError) as e:
                    logging.error(f"Invalid stroke width value: {e}")
                    stroke_width_px = 0

        # Modify font size calculation to reduce vmin values by 25%
        if isinstance(raw_font_size, str) and 'vmin' in raw_font_size.lower():
            try:
                vmin_value = float(raw_font_size.split('vmin')[0].strip())
                # Reduce vmin value by 25%
                vmin_value = vmin_value * 1
                font_size = parse_percentage(f"{vmin_value}vmin", min(video_width, video_height), video_height)
            except ValueError:
                font_size = parse_percentage("5%", min(video_width, video_height), video_height)
        else:
            font_size = parse_percentage(raw_font_size, min(video_width, video_height), video_height)

        # Create a transparent image for the text
        img = Image.new('RGBA', (video_width, video_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # Load the font
        try:
            if font_url and font_url.startswith('http'):
                font_path_tuple = download_file(font_url, suffix='.ttf')
                font_path = font_path_tuple[0] if font_path_tuple[0] else "Arial"
            else:
                font_path = element.get('font_family', "Arial")

            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logging.error(f"Font could not be loaded. Using default font.")
            font = ImageFont.load_default()

        # Get fill color
        fill_color = element.get('fill_color', 'white')

        # Calculate text size
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width = draw.textlength(text, font=font)
            text_height = font_size

        # Calculate position from the middle of the text
        x_percentage = element.get('x', "0%")
        y_percentage = element.get('y', "0%")
        x_pos = parse_percentage(x_percentage, video_width)
        y_pos = parse_percentage(y_percentage, video_height)

        # Adjust position to center the text
        x_pos = x_pos - (text_width / 2)
        y_pos = y_pos - (text_height / 2)

        # Draw text with stroke
        try:
            draw.text(
                (x_pos, y_pos),
                text,
                font=font,
                fill=fill_color,
                stroke_width=stroke_width_px,
                stroke_fill=stroke_color
            )
            logging.info("Successfully applied stroke using built-in parameters")
        except TypeError:
            logging.info("Falling back to manual stroke method")
            for adj in range(-stroke_width_px, stroke_width_px+1):
                for adj_y in range(-stroke_width_px, stroke_width_px+1):
                    if adj != 0 or adj_y != 0:
                        draw.text((x_pos + adj, y_pos + adj_y), text, font=font, fill=stroke_color)
            draw.text((x_pos, y_pos), text, font=font, fill=fill_color)

        # Create the clip
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        text_image = np.array(Image.open(buffer))
        
        if duration is None:
            duration = total_duration - start_time

        text_clip = ImageClip(text_image, transparent=True).set_duration(duration).set_start(start_time)
        text_clip.name = element['id']
        text_clip.track = element.get('track', 0)

        return text_clip

    except Exception as e:
        logging.error(f"Error creating text clip: {str(e)}")
        return None
    finally:
        # Clean up downloaded font file if it exists
        if 'font_path_tuple' in locals() and font_url and font_url.startswith('http'):
            if font_path_tuple[0] and os.path.exists(font_path_tuple[0]):
                os.unlink(font_path_tuple[0])


def create_clip(element, video_width, video_height, video_spec):
    """
    Creates a clip based on the element type.
    """
    element_type = element.get('type')
    
    # Get total duration from video spec
    total_duration = video_spec.get('duration')
    if total_duration is None:
        logging.error("No duration specified in video specification")
        return None
    
    # Set default values for missing parameters
    if 'width' not in element:
        element['width'] = '100%'
    if 'height' not in element:
        element['height'] = '100%'
    if 'x' not in element:
        element['x'] = '50%'
    if 'y' not in element:
        element['y'] = '50%'
    
    # Initialize animations as empty list if null
    if element.get('animations') is None:
        element['animations'] = []
        
    # Handle different element types
    if element_type == 'video':
        # If element has loop=True but no duration, set duration to total_duration
        if element.get('loop', False) and not element.get('duration'):
            element['duration'] = total_duration
            # Store the total duration in the element for use in create_video_clip
            element['total_duration'] = total_duration
        return create_video_clip(element, video_width, video_height)
    elif element_type == 'text':
        return create_text_clip(element, video_width, video_height, total_duration)
    elif element_type == 'image':
        # Use the new OpenCV-based function for image clips
        return create_image_clip_with_opencv(element, video_width, video_height)
    elif element_type == 'audio':
        return create_audio_clip(element)
    else:
        logging.error(f"Unknown element type: {element_type}")
        return None


def resize_maintaining_ratio(target_w, target_h, clip):
    """
    Helper function to calculate dimensions that maintain aspect ratio while filling target area.
    Scales to cover target area without warping, may result in cropping.
    """
    clip_ratio = clip.w / clip.h
    target_ratio = target_w / target_h

    if clip_ratio > target_ratio:
        # Image is wider - scale to height
        new_height = target_h
        new_width = int(new_height * clip_ratio)
    else:
        # Image is taller - scale to width
        new_width = target_w
        new_height = int(new_width / clip_ratio)

    return new_width, new_height

# First, let's update the easing functions to be more precise
def get_easing_function(easing_type):
    """
    Returns an easing function based on the specified type.
    Implements standard easing functions for smooth animations.
    """
    def linear(t):
        return t

    def sinusoid_in(t):
        return 1 - math.cos((t * math.pi) / 2)

    def sinusoid_out(t):
        return math.sin((t * math.pi) / 2)

    def sinusoid_in_out(t):
        return -(math.cos(math.pi * t) - 1) / 2

    def quadratic_in(t):
        return t * t

    def quadratic_out(t):
        return 1 - (1 - t) * (1 - t)

    def quadratic_in_out(t):
        if t < 0.5:
            return 2 * t * t
        return 1 - (-2 * t + 2)**2 / 2

    def cubic_in(t):
        return t * t * t

    def cubic_out(t):
        return 1 - (1 - t)**3

    def cubic_in_out(t):
        if t < 0.5:
            return 4 * t * t * t
        return 1 - (-2 * t + 2)**3 / 2

    def quartic_in(t):
        return t * t * t * t

    def quartic_out(t):
        return 1 - (1 - t)**4

    def quartic_in_out(t):
        if t < 0.5:
            return 8 * t * t * t * t
        return 1 - (-2 * t + 2)**4 / 2

    def quintic_in(t):
        return t * t * t * t * t

    def quintic_out(t):
        return 1 - (1 - t)**5

    def quintic_in_out(t):
        if t < 0.5:
            return 16 * t * t * t * t * t
        return 1 - (-2 * t + 2)**5 / 2

    def exponential_in(t):
        return 0 if t == 0 else math.pow(2, 10 * t - 10)

    easing_functions = {
        'linear': linear,
        'sinusoid-in': sinusoid_in,
        'sinusoid-out': sinusoid_out,
        'sinusoid-in-out': sinusoid_in_out,
        'quadratic-in': quadratic_in,
        'quadratic-out': quadratic_out,
        'quadratic-in-out': quadratic_in_out,
        'cubic-in': cubic_in,
        'cubic-out': cubic_out,
        'cubic-in-out': cubic_in_out,
        'quartic-in': quartic_in,
        'quartic-out': quartic_out,
        'quartic-in-out': quartic_in_out,
        'quintic-in': quintic_in,
        'quintic-out': quintic_out,
        'quintic-in-out': quintic_in_out,
        'exponential-in': exponential_in
    }

    # Add math import at the top of the file if not already present
    return easing_functions.get(easing_type, linear)

# Then modify the apply_animations function to use the easing
def apply_animations(clip, element, duration, video_width, video_height):
    animations = element.get('animations', [])
    if not animations:
        return clip

    for anim in animations:
        try:
            if anim.get('type') != 'scale':
                continue

            element_start = element.get('time', 0)
            anim_start = element_start + anim.get('time', 0)
            anim_duration = anim.get('duration', duration)

            start_scale = 0.05  
            if anim.get('start_scale'):
                start_scale = max(0.05, float(anim.get('start_scale', '5%').rstrip('%')) / 100)
            end_scale = max(0.05, float(anim.get('end_scale', '100%').rstrip('%')) / 100)
            
            # Get easing function
            easing_type = anim.get('easing', 'linear')
            easing_function = get_easing_function(easing_type)
            
            x_anchor = float(anim.get('x_anchor', '50%').strip('%')) / 100
            y_anchor = float(anim.get('y_anchor', '50%').strip('%')) / 100

            element_w = parse_percentage(element.get('width', '100%'), video_width)
            element_h = parse_percentage(element.get('height', '100%'), video_height)
            element_x = parse_percentage(element.get('x', '50%'), video_width)
            element_y = parse_percentage(element.get('y', '50%'), video_height)

            def get_progress(t):
                if t < anim_start:
                    return 0
                elif t > anim_start + anim_duration:
                    return 1
                # Apply easing to the progress
                linear_progress = (t - anim_start) / anim_duration
                return easing_function(linear_progress)

            def make_frame_resize(t):
                progress = get_progress(t)
                scale = start_scale + (end_scale - start_scale) * progress
                scaled_w = element_w * scale
                scaled_h = element_h * scale
                return resize_maintaining_ratio(scaled_w, scaled_h, clip)

            def make_frame_position(t):
                current_w, current_h = make_frame_resize(t)
                progress = get_progress(t)
                
                width_diff = current_w - element_w
                height_diff = current_h - element_h
                
                x_offset = width_diff * x_anchor
                y_offset = height_diff * y_anchor
                
                pos_x = element_x - (element_w / 2) - x_offset
                pos_y = element_y - (element_h / 2) - y_offset
                
                return (pos_x, pos_y)

            clip = clip.resize(make_frame_resize)
            clip = clip.set_position(make_frame_position)

        except Exception as e:
            logging.error(f"Error in animation for {element['id']}: {e}")
            logging.error(f"Animation data: {anim}")
            return clip

    return clip

def upload_video(file_path):
    """
    Uploads video to 0x0.st and returns the URL.
    """
    try:
        logging.info(f"Starting video upload from path: {file_path}")
        logging.info(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        with open(file_path, 'rb') as file:
            logging.info("Uploading to 0x0.st...")
            response = requests.post('https://0x0.st', files={'file': file})
            
            logging.info(f"Upload response status: {response.status_code}")
            if response.status_code == 200:
                url = response.text.strip()
                logging.info(f"Upload successful. URL: {url}")
                return url
            else:
                logging.error(f"Upload failed. Response: {response.text}")
                return None
    except Exception as e:
        logging.error(f"Upload error: {str(e)}", exc_info=True)
        return None

def generate_video(json_data):
    try:
        video_spec = json_data
        video_duration = float(video_spec.get('duration') or 15.0)
        video_fps = float(video_spec.get('fps') or 30)
        video_width = int(video_spec.get('width') or 720)
        video_height = int(video_spec.get('height') or 1280)

        logging.info(f"\nVideo Specifications:")
        logging.info(f"Duration: {video_duration}s")
        logging.info(f"FPS: {video_fps}")
        logging.info(f"Resolution: {video_width}x{video_height}")

        # Create temporary directory for assets
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join('/tmp', f"output_video_{uuid.uuid4().hex}.mp4")
        logging.info(f"Created temp directory: {temp_dir}")

        try:
            # Sort elements by track and time
            elements = sorted(
                video_spec['elements'], 
                key=lambda x: (x.get('track', 0), x.get('time', 0))
            )

            # Prepare FFmpeg inputs and filters
            inputs = []
            filter_complex = []

            # Create background
            filter_complex.append(
                f'color=c=black:s={video_width}x{video_height}:'
                f'd={video_duration}:r={video_fps}[bg]'
            )
            last_output = 'bg'

            # Process each element
            for i, element in enumerate(elements):
                element_type = element.get('type')
                start_time = float(element.get('time', 0.0))
                element_duration = element.get('duration')
                if element_duration is None:
                    element_duration = video_duration - start_time
                else:
                    element_duration = float(element_duration)

                if element_type in ['image', 'video']:
                    clip = create_clip(element, video_width, video_height, video_spec)
                    if not clip:
                        continue

                    inputs.append(clip)
                    input_idx = len(inputs) - 1
                    next_output = f'[temp{i}]' if i < len(elements) - 1 else '[outv]'
                    
                    # Add setpts filter to ensure proper timing
                    filter_complex.append(
                        f'[{input_idx}:v]setpts=PTS-STARTPTS[v{i}];'
                        f'[{last_output}][v{i}]overlay='
                        f'shortest=0:enable=between(t\\,{start_time}\\,{start_time + element_duration})'
                        f'{next_output}'
                    )
                    last_output = next_output.strip('[]')

                elif element_type == 'text':
                    next_output = f'[temp{i}]' if i < len(elements) - 1 else '[outv]'
                    text = element.get('text', '')
                    escaped_text = text.replace("'", "\\'").replace(':', '\\:')
                    font_size = parse_percentage(element.get('font_size', '5%'), 
                                              min(video_width, video_height), 
                                              video_height)
                    y_pos = parse_percentage(element.get('y', '50%'), video_height)
                    
                    drawtext_filter = (
                        f'[{last_output}]drawtext=text=\'{escaped_text}\':'
                        f'fontsize={font_size}:fontcolor={element.get("fill_color", "white")}:'
                        f'x=(w-text_w)/2:y={y_pos}'
                    )
                    
                    if element.get('stroke_color') and element.get('stroke_width'):
                        stroke_width = parse_percentage(element.get('stroke_width', '0'), 
                                                     min(video_width, video_height), 
                                                     video_height)
                        drawtext_filter += f':bordercolor={element["stroke_color"]}:borderw={stroke_width}'
                    
                    drawtext_filter += f':enable=between(t\\,{start_time}\\,{start_time + element_duration}){next_output}'
                    
                    filter_complex.append(drawtext_filter)
                    last_output = next_output.strip('[]')

            # Build FFmpeg command with loop options
            ffmpeg_cmd = ['ffmpeg', '-y']
            
            # Add input files with stream loop
            for input_file in inputs:
                ffmpeg_cmd.extend(['-stream_loop', '-1', '-i', input_file])

            # Add filter complex
            filter_complex_str = ';'.join(filter_complex)
            ffmpeg_cmd.extend([
                '-filter_complex', filter_complex_str,
                '-map', '[outv]',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-profile:v', 'baseline',
                '-level', '3.0',
                '-t', str(video_duration)
            ])
            ffmpeg_cmd.append(output_path)

            logging.info("Executing FFmpeg command:")
            logging.info(' '.join(ffmpeg_cmd))

            # Execute FFmpeg
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Monitor FFmpeg output
            for line in process.stderr:
                if 'error' in line.lower():
                    logging.error(f"FFmpeg error: {line.strip()}")
                elif 'warning' in line.lower():
                    logging.warning(f"FFmpeg warning: {line.strip()}")
                else:
                    logging.debug(f"FFmpeg: {line.strip()}")

            if process.wait() == 0:
                # Verify the output video
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logging.info(f"Video generated successfully. Size: {file_size/1024/1024:.2f}MB")
                    
                    # Verify video can be opened
                    cap = cv2.VideoCapture(output_path)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            logging.info("Video file verified - can be read correctly")
                        else:
                            logging.error("Video file verification failed - cannot read frames")
                        cap.release()
                    else:
                        logging.error("Video file verification failed - cannot open file")
                    
                    return upload_video(output_path)
                else:
                    logging.error("Output video file not found")
                    return None
            else:
                logging.error("FFmpeg process failed")
                return None

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
            if os.path.exists(output_path):
                os.unlink(output_path)

    except Exception as e:
        logging.error(f"Error in video generation: {str(e)}", exc_info=True)
        return None


def process_video_element(element, total_duration):
    """
    Process a video element, handling looping and duration.
    """
    try:
        source = element.get('source')
        input_path = None
        temp_video_tuple = None

        if source.startswith('http'):
            temp_video_tuple = download_file(source, suffix='.mp4')
            if not temp_video_tuple[0]:
                logging.error(f"Failed to download video from {source}")
                return None
            input_path = temp_video_tuple[0]
        else:
            input_path = source

        element_duration = element.get('duration')
        if element_duration is None:
            element_duration = total_duration

        if element.get('loop', False):
            try:
                # First, create a video clip to get its duration
                original_clip = VideoFileClip(input_path)
                original_duration = original_clip.duration
                original_clip.close()

                if element_duration:
                    # Create a temporary file for concatenated video
                    with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as concat_list:
                        # Calculate how many times to repeat the video
                        num_loops = math.ceil(element_duration / original_duration)
                        
                        # Write the list of files to concatenate
                        for _ in range(num_loops):
                            concat_list.write(f"file '{input_path}'\n")
                    
                    # Create output path for concatenated video
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as output_file:
                        output_path = output_file.name

                    # Use ffmpeg concat demuxer
                    ff = ffmpy.FFmpeg(
                        inputs={concat_list.name: '-f concat -safe 0'},
                        outputs={
                            output_path: [
                                '-c:v', 'libx264',
                                '-preset', 'ultrafast',
                                '-t', str(element_duration),
                                '-y'
                            ]
                        }
                    )
                    ff.run()

                    # Clean up concat list file
                    os.unlink(concat_list.name)

                    # Load the final video
                    video_clip = VideoFileClip(output_path)
                    
                    # Clean up the output file after loading
                    os.unlink(output_path)
                    
                    return video_clip
                else:
                    # No duration specified, use concatenation with total_duration
                    video_clip = VideoFileClip(input_path)
                    num_loops = int(np.ceil(total_duration / video_clip.duration))
                    clips = [video_clip] * num_loops
                    final_clip = concatenate_videoclips(clips)
                    final_clip = final_clip.subclip(0, total_duration)
                    return final_clip
                    
            except Exception as e:
                logging.error(f"Error processing looped video: {str(e)}")
                return VideoFileClip(input_path)
        else:
            # If not looping, just load and trim if needed
            video_clip = VideoFileClip(input_path)
            if element_duration:
                video_clip = video_clip.subclip(0, min(element_duration, video_clip.duration))
            return video_clip

    except Exception as e:
        logging.error(f"Error in process_video_element: {str(e)}")
        return None
    finally:
        if temp_video_tuple and temp_video_tuple[0] and os.path.exists(temp_video_tuple[0]):
            os.unlink(temp_video_tuple[0])

# Modify create_video_clip to use process_video_element
def create_video_clip(element, video_width, video_height):
    source = element.get('source')
    if not source:
        logging.error(f"Video element {element['id']} has no source.")
        return None

    temp_video, detected_ext = download_file(source)
    if not temp_video:
        logging.error(f"Failed to download video from {source}")
        return None

    try:
        # Get video dimensions
        cap = cv2.VideoCapture(temp_video)
        if not cap.isOpened():
            logging.error("Failed to open video file")
            return None

        # Get video properties
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logging.info(f"Original video: {orig_width}x{orig_height} @ {fps}fps")

        # Calculate target dimensions
        target_height = parse_percentage(element.get('height', '100%'), video_height)
        # If width not specified, calculate to maintain aspect ratio
        if 'width' in element:
            target_width = parse_percentage(element.get('width'), video_width)
        else:
            target_width = int(target_height * (orig_width / orig_height))

        # Calculate scaling factors to maintain aspect ratio
        width_scale = target_width / orig_width
        height_scale = target_height / orig_height
        
        # Use the larger scale to ensure coverage (fit to cover)
        scale = max(width_scale, height_scale)
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        logging.info(f"Scaling to cover: {new_width}x{new_height}")

        # Calculate crop offsets to center the video
        x_crop = (new_width - target_width) // 2
        y_crop = (new_height - target_height) // 2

        # Create output video
        output_path = f"{tempfile.mktemp()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame maintaining aspect ratio
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Crop to target dimensions
            cropped_frame = resized_frame[y_crop:y_crop+target_height, x_crop:x_crop+target_width]
            
            # Write the frame
            out.write(cropped_frame)

        cap.release()
        out.release()

        logging.info(f"Video clip created at: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error creating video clip: {e}")
        return None
    finally:
        if os.path.exists(temp_video):
            os.unlink(temp_video)

def monitor_resources():
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.Process().memory_info()
    return {
        'cpu_usage': cpu_percent,
        'memory_usage': memory_info.rss / 1024 / 1024  # MB
    }

def monitor_rendering_progress(current_frame, total_frames):
    """Monitor and log rendering progress with resource usage"""
    if current_frame % 10 == 0:  # Log every 10 frames
        progress = (current_frame / total_frames) * 100
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logging.info(f"Rendering progress: {progress:.1f}% | Frame: {current_frame}/{total_frames} | "
                    f"CPU: {cpu_percent}% | Memory: {memory:.1f}MB")
        
        # Check if resources are strained
        if cpu_percent > 90 or memory > (psutil.virtual_memory().total * 0.9):
            logging.warning("Resource usage high during rendering, triggering GC")
            gc.collect()

def format_time(seconds):
    """Format seconds into human readable time"""
    return str(timedelta(seconds=int(seconds)))

class RenderingProgress:
    def __init__(self, total_frames):
        self.total_frames = total_frames
        self.start_time = current_time()
        self.last_frame = 0
        self.last_update = self.start_time
        self.fps_samples = []
        
    def __call__(self, t):
        current = current_time()
        frame = int(t * self.total_frames)
        elapsed = current - self.last_update
        
        # Update every 0.5 seconds
        if elapsed > 0.5:
            frames_processed = frame - self.last_frame
            current_fps = frames_processed / elapsed if elapsed > 0 else 0
            self.fps_samples.append(current_fps)
            
            # Keep only last 5 samples for average
            if len(self.fps_samples) > 5:
                self.fps_samples.pop(0)
            
            avg_fps = sum(self.fps_samples) / len(self.fps_samples)
            progress = (frame / self.total_frames) * 100
            
            # Calculate ETA
            frames_remaining = self.total_frames - frame
            eta_seconds = frames_remaining / avg_fps if avg_fps > 0 else 0
            
            # Get resource usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logging.info(
                f"Rendering: {progress:.1f}% | "
                f"Frame: {frame}/{self.total_frames} | "
                f"Speed: {avg_fps:.1f} fps | "
                f"ETA: {format_time(eta_seconds)} | "
                f"CPU: {cpu_percent}% | "
                f"Memory: {memory:.1f}MB"
            )
            
            self.last_frame = frame
            self.last_update = current
            
            # Trigger GC if memory usage is high
            if memory > psutil.virtual_memory().total * 0.8:
                gc.collect()

def create_image_clip_with_opencv(element, video_width, video_height):
    """
    Creates an image clip using OpenCV for animations.
    """
    source = element.get('source')
    start_time = float(element.get('time', 0.0))
    duration = float(element.get('duration', 5.0))
    
    logging.info(f"\nProcessing Image/GIF Element:")
    for key, value in element.items():
        logging.info(f"{key}: {value}")

    if not source:
        logging.error(f"Element {element['id']} has no source.")
        return None

    temp_file, detected_ext = download_file(source)
    if not temp_file:
        logging.error(f"Failed to download file from {source}")
        return None

    try:
        # Check multiple ways if file is a GIF
        is_gif = (
            detected_ext.lower() == '.gif' or 
            imghdr.what(temp_file) == 'gif' or
            'gif' in source.lower() or
            Image.open(temp_file).format == 'GIF'
        )
        
        logging.info(f"File type detection: Extension={detected_ext}, Is GIF={is_gif}")

        if is_gif:
            logging.info("Processing as GIF animation")
            try:
                # Open GIF with PIL
                gif = Image.open(temp_file)
                frames = []
                frame_count = 0
                
                while True:
                    try:
                        gif.seek(frame_count)
                        # Convert frame to RGB and then to numpy array
                        frame = gif.convert('RGB')
                        frame_array = np.array(frame)
                        # Convert from RGB to BGR for OpenCV
                        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                        frames.append(frame_array)
                        frame_count += 1
                    except EOFError:
                        break

                logging.info(f"Successfully extracted {frame_count} frames from GIF")
                
                if not frames:
                    logging.error("No frames extracted from GIF")
                    return None
                    
                img = frames[0]  # Use first frame for initial sizing
            except Exception as e:
                logging.error(f"Error processing GIF: {e}")
                return None
        else:
            # Regular image processing
            img = cv2.imread(temp_file, cv2.IMREAD_UNCHANGED)
            if img is None:
                logging.error(f"Failed to read image from {temp_file}")
                return None
            frames = [img]  # Single frame for static images

        # Get original image dimensions
        img_height, img_width = img.shape[:2]
        logging.info(f"Original image dimensions: {img_width}x{img_height}")

        # Calculate dimensions based on percentages
        width_value = element.get('width', '100%')
        height_value = element.get('height', '100%')
        x_value = element.get('x', '50%')
        y_value = element.get('y', '50%')

        # Calculate target dimensions
        target_width = parse_percentage(width_value, video_width)
        target_height = parse_percentage(height_value, video_height)
        x_pos = parse_percentage(x_value, video_width)
        y_pos = parse_percentage(y_value, video_height)

        logging.info(f"Target dimensions: {target_width}x{target_height}")
        logging.info(f"Position: ({x_pos}, {y_pos})")

        # Calculate scaling factors
        width_scale = target_width / img_width
        height_scale = target_height / img_height
        scale = max(width_scale, height_scale)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        logging.info(f"Scaling to cover: {new_width}x{new_height}")

        # Resize and crop all frames
        processed_frames = []
        for frame in frames:
            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Calculate crop offsets
            x_crop = (new_width - target_width) // 2
            y_crop = (new_height - target_height) // 2
            
            # Crop frame
            cropped_frame = resized_frame[y_crop:y_crop+target_height, x_crop:x_crop+target_width]
            processed_frames.append(cropped_frame)

        # Get animation parameters
        animations = element.get('animations', [])
        scale_animation = next((anim for anim in animations if anim.get('type') == 'scale'), None)
        
        if scale_animation:
            anim_start = float(scale_animation.get('time', 0))
            anim_duration = float(scale_animation.get('duration', duration))
            end_scale = float(scale_animation.get('end_scale', '100').rstrip('%')) / 100
            start_scale = max(0.01, float(scale_animation.get('start_scale', '100').rstrip('%')) / 100)
            easing = scale_animation.get('easing', 'linear')
            
            # Get anchor points from animation (as percentages)
            x_anchor_pct = float(scale_animation.get('x_anchor', '50%').rstrip('%')) / 100
            y_anchor_pct = float(scale_animation.get('y_anchor', '50%').rstrip('%')) / 100
            
            logging.info(f"Animation anchors (relative to element): x={x_anchor_pct*100}%, y={y_anchor_pct*100}%")

        # Calculate total frames needed for the full duration
        fps = 30
        total_frames = int(duration * fps)
        frame_count = len(processed_frames)
        
        # Create output video
        output_path = f"{tempfile.mktemp()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

        # Generate frames
        for frame_num in range(total_frames):
            frame_time = frame_num / fps
            
            # Create base frame
            frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            
            # Calculate current scale based on animation
            current_scale = 1.0
            if scale_animation and frame_time >= anim_start:
                progress = min(1.0, (frame_time - anim_start) / anim_duration)
                
                # Apply easing
                if easing == 'quadratic-out':
                    progress = 1 - (1 - progress) ** 2
                elif easing == 'quadratic-in':
                    progress = progress ** 2
                
                current_scale = start_scale + (end_scale - start_scale) * progress
                current_scale = max(0.01, current_scale)
            
            # Select base frame
            if is_gif:
                # If looping, use modulo to cycle through GIF frames
                frame_idx = frame_num % frame_count
                frame_img = processed_frames[frame_idx]
            else:
                # Use first frame for static image
                frame_img = processed_frames[0]
            
            # Calculate scaled dimensions
            scaled_width = int(target_width * current_scale)
            scaled_height = int(target_height * current_scale)
            
            if scaled_width > 0 and scaled_height > 0:
                # Scale the frame
                scaled_img = cv2.resize(frame_img, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
                
                if scale_animation:
                    # Get the element's original position (this is the center point)
                    element_center_x = x_pos
                    element_center_y = y_pos
                    
                    # Calculate the anchor point offset from the center
                    anchor_offset_x = (x_anchor_pct - 0.5) * target_width
                    anchor_offset_y = (y_anchor_pct - 0.5) * target_height
                    
                    # Calculate the fixed anchor point in absolute coordinates
                    anchor_x = element_center_x + anchor_offset_x
                    anchor_y = element_center_y + anchor_offset_y
                    
                    # Calculate the scaled offset from anchor point
                    x_offset = int(anchor_x - (scaled_width * x_anchor_pct))
                    y_offset = int(anchor_y - (scaled_height * y_anchor_pct))
                else:
                    # Use center anchoring if no animation
                    x_offset = int(x_pos - scaled_width // 2)
                    y_offset = int(y_pos - scaled_height // 2)

                # Ensure coordinates are within bounds
                x_start = max(0, x_offset)
                y_start = max(0, y_offset)
                x_end = min(video_width, x_offset + scaled_width)
                y_end = min(video_height, y_offset + scaled_height)

                # Calculate ROI
                roi_x_start = max(0, -x_offset)
                roi_y_start = max(0, -y_offset)
                roi_x_end = min(roi_x_start + (x_end - x_start), scaled_width)
                roi_y_end = min(roi_y_start + (y_end - y_start), scaled_height)

                if frame_num % fps == 0:  # Log every second
                    logging.debug(f"Frame {frame_num}/{total_frames}: Scale={current_scale:.2f}, "
                                f"Size={scaled_width}x{scaled_height}")

                try:
                    # Place image in frame
                    frame[y_start:y_end, x_start:x_end] = scaled_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                except Exception as e:
                    logging.error(f"Error placing frame {frame_num}: {e}")
                    continue

            out.write(frame)

        out.release()
        logging.info(f"Video clip created at: {output_path}")
        return output_path

    except Exception as e:
        logging.error(f"Error creating image/GIF clip: {e}")
        return None
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

# Replace the existing create_image_clip function call with create_image_clip_with_opencv







