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

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    Args:
        url (str): The URL to download the file from.
        suffix (str): The suffix for the temporary file.

    Returns:
        str: The path to the downloaded temporary file.
    """
    try:
        # Get the direct download URL
        direct_url = extract_image_url(url)
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download the file
        response = requests.get(direct_url, stream=True, headers=headers)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Write the content
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
        temp_file.close()
        
        # Set permissions
        os.chmod(temp_file.name, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
        
        return temp_file.name
    except Exception as e:
        logging.error(f"Error downloading file from {url}: {e}")
        return None


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

    temp_audio = download_file(source, suffix='.mp3')  # Assuming mp3, adjust if necessary
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

    temp_image = download_file(source)
    if not temp_image:
        logging.error(f"Failed to download file from {source} for element {element['id']}.")
        return None

    try:
        if source.lower().endswith('.gif'):
            # Handle GIF (existing GIF handling code remains the same)
            gif = imageio.get_reader(temp_image)
            frames = []
            durations = []
            for frame in gif:
                frames.append(frame)
                durations.append(frame.meta.get('duration', 100) / 1000)  # Convert to seconds

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
            # Handle static image (including transparent PNGs)
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
            # Get values with defaults - now properly checking for empty values
            width_value = get_valid_value('width')
            height_value = get_valid_value('height')
            x_value = get_valid_value('x')
            y_value = get_valid_value('y')
            
            # Log the values being used
            logging.info(f"Using values - x: {x_value}, y: {y_value}, width: {width_value}, height: {height_value}")
            
            # Parse dimensions with defaults
            target_width = parse_percentage(width_value, video_width)
            target_height = parse_percentage(height_value, video_height)
            
            # Resize the clip to cover the target dimensions
            aspect_ratio = clip.w / clip.h
            target_ratio = target_width / target_height
            
            if aspect_ratio > target_ratio:
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
            
            resized_clip = clip.resize(height=new_height, width=new_width)
            
            # Crop to fit the target dimensions
            x_center = resized_clip.w / 2
            y_center = resized_clip.h / 2
            final_clip = resized_clip.crop(
                x1=x_center - target_width / 2,
                y1=y_center - target_height / 2,
                width=target_width,
                height=target_height
            )
            
            # Position the clip using the default-aware values
            final_x = parse_percentage(x_value, video_width - final_clip.w)
            final_y = parse_percentage(y_value, video_height - final_clip.h)
            
            # **Start of Edit**
            # Ensure that y + height does not exceed video height
            if (final_y + target_height) > video_height:
                final_y = video_height - target_height
                logging.info(f"Adjusted y position to {final_y} to fit height within video frame.")
            # **End of Edit**
    
            final_clip = final_clip.set_position((final_x, final_y))

        # Apply animations if present
        if element.get('animations'):
            final_clip = apply_animations(final_clip, element, duration or clip.duration)

        final_clip = final_clip.set_start(start_time)
        final_clip.name = element['id']
        final_clip.track = element.get('track', 0)

        logging.info(f"Created {'GIF' if source.lower().endswith('.gif') else 'image'} clip for element {element['id']} positioned at {final_clip.pos} with size {final_clip.w}x{final_clip.h}.")
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

        logging.info("\nFinal stroke parameters to be applied:")
        logging.info(f"  - Stroke color: {stroke_color}")
        logging.info(f"  - Stroke width: {stroke_width_px}px")
        logging.info("=====================================")

        # Modify font size calculation to reduce vmin values by 25%
        if isinstance(raw_font_size, str) and 'vmin' in raw_font_size.lower():
            try:
                vmin_value = float(raw_font_size.split('vmin')[0].strip())
                # Reduce vmin value by 25%
                vmin_value = vmin_value * 0.75
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
                font_path = download_file(font_url, suffix='.ttf')
                if not font_path:
                    font_path = "Arial"
            else:
                font_path = element.get('font_family', "Arial")

            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logging.error(f"Font at {font_path} could not be loaded. Using default font.")
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

        # Calculate position
        x_percentage = element.get('x', "0%")
        y_percentage = element.get('y', "0%")
        x_pos = parse_percentage(x_percentage, video_width - text_width)
        y_pos = parse_percentage(y_percentage, video_height - text_height)

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
        if 'font_path' in locals() and font_url and font_url.startswith('http') and os.path.exists(font_path):
            os.unlink(font_path)


def create_clip(element, video_width, video_height, video_spec):
    """
    Creates a clip based on the element type.

    Args:
        element (dict): The JSON element.
        video_width (int): The width of the video.
        video_height (int): The height of the video.
        video_spec (dict): The overall video specifications.

    Returns:
        Clip or None: The created clip or None if failed.
    """
    element_type = element.get('type')
    
    # Set default values for missing parameters
    if 'width' not in element:
        element['width'] = '100%'  # Default width to 100%
    if 'x' not in element:
        element['x'] = '50%'  # Default x to center
        
    # Handle video positioning
    if element_type == 'video':
        # Adjust video position up by 13%
        current_y = parse_percentage(element.get('y', '0%'), video_height)
        adjusted_y = max(0, current_y - (video_height * 0.13))
        element['y'] = f"{(adjusted_y / video_height) * 100}%"
        
        # Create video clip
        return create_video_clip(element, video_width, video_height)
    
    # Handle text positioning
    elif element_type == 'text':
        if element.get('text') == 'Your text here':
            # Center the text
            element['x'] = '50%'
            element['x_anchor'] = '50%'  # This centers the text around its x position
        elif element.get('text') == 'RageCvlt':
            # Move RageCvlt slightly right
            current_x = parse_percentage(element.get('x', '37.7451%'), video_width)
            element['x'] = f"{((current_x + 20) / video_width) * 100}%"
        return create_text_clip(element, video_width, video_height, video_spec.get('duration', 15.0))
    
    # Handle image sizing and positioning
    elif element_type == 'image':
        # Ensure image width is correct (101.5848% if specified, 100% if not)
        if element.get('width') in [None, '', '100%']:  # Check actual value instead of presence
            element['width'] = '101.5848%'
        
        # Ensure proper vertical positioning
        if element.get('y') == '50%' or element.get('height') == '100%':  # Check if using default values
            # Set defaults for missing values
            element['y'] = '64.4274%'
            element['height'] = '71.1453%'
        return create_image_clip(element, video_width, video_height)
    
    elif element_type == 'audio':
        return create_audio_clip(element)
    else:
        logging.error(f"Unknown element type: {element_type}")
        return None


def apply_animations(clip, element, duration):
    """
    Applies animations to a clip based on the element's animation settings.
    """
    animations = element.get('animations', [])
    if not animations:
        return clip

    for anim in animations:
        anim_type = anim.get('type')
        anim_time = anim.get('time', 0)
        anim_duration = anim.get('duration', duration)
        
        if anim_type == 'scale':
            start_scale = parse_percentage(anim.get('start_scale', '100%'), 100) / 100
            end_scale = parse_percentage(anim.get('end_scale', '100%'), 100) / 100
            
            def scale_func(t):
                if t < anim_time:
                    return start_scale
                elif t > anim_time + anim_duration:
                    return end_scale
                else:
                    progress = (t - anim_time) / anim_duration
                    return start_scale + (end_scale - start_scale) * progress
            
            clip = clip.resize(lambda t: scale_func(t))
            
            # Apply fade if specified
            if anim.get('fade', False):
                def fade_func(t):
                    if t < anim_time:
                        return 0
                    elif t > anim_time + anim_duration:
                        return 1
                    else:
                        return (t - anim_time) / anim_duration
                
                clip = clip.set_opacity(lambda t: fade_func(t))

    return clip


def generate_video(json_data):
    """
    Generates a video based on the provided JSON configuration.
    """
    try:
        video_spec = json_data
        
        # Log the complete JSON data for text elements
        logging.info("=== Processing Video Generation Request ===")
        for element in video_spec['elements']:
            if element.get('type') == 'text':
                logging.info(f"\nText Element Processing:")
                logging.info(json.dumps(element, indent=2))

        video_clips = []
        audio_clips = []

        logging.info("Starting video generation process...")
        logging.info(f"Video specification: {json.dumps(video_spec, indent=2)}")

        # Set default values if not provided
        video_duration = video_spec.get('duration', 15.0)
        video_fps = video_spec.get('fps', 30)
        video_width = video_spec.get('width', 720)
        video_height = video_spec.get('height', 1280)

        for index, element in enumerate(video_spec['elements']):
            logging.info(f"Processing element {index + 1}/{len(video_spec['elements'])}: {json.dumps(element, indent=2)}")
            clip = create_clip(element, video_width, video_height, video_spec)
            if clip:
                if isinstance(clip, AudioFileClip):
                    audio_clips.append(clip)
                    print(f"Added audio clip: {element['id']} on track {element.get('track', 0)}")
                else:
                    video_clips.append(clip)
                    print(f"Added video/image/GIF/text clip: {element['id']} on track {element.get('track', 0)}")
            else:
                print(f"Failed to create clip for element: {element['id']}")

            # Force garbage collection after each element
            gc.collect()
            
            # Log memory usage
            process = psutil.Process(os.getpid())
            logging.info(f"Memory usage after processing element {index + 1}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        print(f"Total video/image/GIF/text clips created: {len(video_clips)}")
        print(f"Total audio clips created: {len(audio_clips)}")

        if video_clips or audio_clips:
            # Sort video clips based on track number and start time
            video_clips.sort(key=lambda c: (getattr(c, 'track', 0), getattr(c, 'start', 0)))
            print("Sorted video/image/GIF/text clips based on track number and start time")

            try:
                # Create the final composite video
                final_video = CompositeVideoClip(video_clips, size=(video_width, video_height), bg_color=None).set_duration(video_duration)
                print("Created CompositeVideoClip with all video/image/GIF/text clips")

                # Combine audio clips
                if audio_clips:
                    composite_audio = CompositeAudioClip(audio_clips)
                    final_video = final_video.set_audio(composite_audio)
                    print("Added CompositeAudioClip to the final video")

                # Generate a unique filename for the output video
                unique_filename = f"output_video_{uuid.uuid4().hex}.mp4"
                desktop_path = os.path.expanduser("~/Desktop")
                output_path = os.path.join(desktop_path, unique_filename)

                print(f"Attempting to write video file to: {output_path}")

                num_cores = multiprocessing.cpu_count()
                ffmpeg_params = [
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-tune", "fastdecode,zerolatency",
                    "-movflags", "+faststart",
                    "-bf", "0",
                    "-flags:v", "+global_header",
                    "-vf", "format=yuv420p",
                    "-maxrate", "4M",
                    "-bufsize", "4M",
                    "-threads", str(num_cores)
                ]

                final_video.write_videofile(
                    output_path,
                    fps=video_fps,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    ffmpeg_params=ffmpeg_params
                )

                if os.path.exists(output_path):
                    print(f"Video exported successfully to: {output_path}")
                    return output_path
                else:
                    print(f"Error: Video file was not created at {output_path}")
                    return None
            except Exception as e:
                print(f"Error creating or writing the final video: {e}")
                return None
        else:
            print("Error: No valid clips were created.")
            return None

    except MemoryError:
        logging.error("Out of memory error occurred. Try reducing video quality or duration.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during video generation: {str(e)}", exc_info=True)
        return None


def create_video_clip(element, video_width, video_height):
    """
    Creates a video clip from the provided element.
    """
    source = element.get('source')
    start_time = element.get('time', 0.0)
    duration = element.get('duration')
    loop = element.get('loop', False)

    if not source:
        logging.error(f"Video element {element['id']} has no source.")
        return None

    try:
        # Download video if it's a URL
        if source.startswith('http'):
            temp_video = download_file(source, suffix='.mp4')
            if not temp_video:
                logging.error(f"Failed to download video from {source}")
                return None
            video_clip = VideoFileClip(temp_video)
        else:
            video_clip = VideoFileClip(source)

        # Set clip duration and start time
        if duration:
            video_clip = video_clip.subclip(0, duration)
        video_clip = video_clip.set_start(start_time)

        # Handle looping
        if loop:
            video_clip = video_clip.loop(duration=duration)

        # Get target dimensions from element
        target_width = parse_percentage(element.get('width', '100%'), video_width)
        target_height = parse_percentage(element.get('height', '100%'), video_height)

        # Calculate scaling to maintain aspect ratio
        original_aspect = video_clip.w / video_clip.h
        target_aspect = target_width / target_height

        if original_aspect > target_aspect:
            # Video is wider than target: scale to height and crop width
            new_height = target_height
            new_width = int(new_height * original_aspect)
            resized_clip = video_clip.resize(height=new_height)
            
            # Center crop to target width
            excess_width = new_width - target_width
            x1 = excess_width // 2
            resized_clip = resized_clip.crop(x1=x1, width=target_width)
        else:
            # Video is taller than target: scale to width and crop height
            new_width = target_width
            new_height = int(new_width / original_aspect)
            resized_clip = video_clip.resize(width=new_width)
            
            # Center crop to target height
            excess_height = new_height - target_height
            y1 = excess_height // 2
            resized_clip = resized_clip.crop(y1=y1, height=target_height)

        # Position the clip
        x_pos = parse_percentage(element.get('x', '50%'), video_width - target_width)
        y_pos = parse_percentage(element.get('y', '50%'), video_height - target_height)
        final_clip = resized_clip.set_position((x_pos, y_pos))

        final_clip.name = element['id']
        final_clip.track = element.get('track', 0)

        logging.info(f"Created video clip for element {element['id']} positioned at ({x_pos}, {y_pos}) with size {target_width}x{target_height}")
        return final_clip

    except Exception as e:
        logging.error(f"Error creating video clip for element {element['id']}: {e}")
        return None
    finally:
        # Clean up temporary file if it exists
        if 'temp_video' in locals() and temp_video and os.path.exists(temp_video):
            os.unlink(temp_video)








