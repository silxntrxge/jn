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
        value (str or int or float): The value to parse (e.g., "50%", "7 vmin", 100).
        total (int): The total value to calculate the percentage against.
        video_height (int): The height of the video, used for vmin calculations.

    Returns:
        int: The absolute value.
    """
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
        elif 'vmin' in value:
            try:
                vmin_value = float(value.replace('vmin', '').strip())
                if video_height is None:
                    logging.error("Video height is required for vmin calculations")
                    return 0
                # Divide vmin value by 2 for font sizes
                return int((vmin_value / 200) * video_height)  # Changed from 100 to 200
            except ValueError:
                logging.error(f"Invalid vmin value: {value}")
    logging.error(f"Invalid value for parsing: {value}")
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


def create_image_clip(element, video_width, video_height):
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

        # Check if width, height, x, and y are specified
        if all(element.get(attr) is None for attr in ['width', 'height', 'x', 'y']):
            # If none are specified, make the image cover the entire video
            aspect_ratio = clip.w / clip.h
            video_aspect_ratio = video_width / video_height
            
            if aspect_ratio > video_aspect_ratio:
                # Image is wider, fit to height
                new_height = video_height
                new_width = int(new_height * aspect_ratio)
            else:
                # Image is taller, fit to width
                new_width = video_width
                new_height = int(new_width / aspect_ratio)
            
            resized_clip = clip.resize(height=new_height, width=new_width)
            x_offset = (video_width - new_width) // 2
            y_offset = (video_height - new_height) // 2
            final_clip = resized_clip.set_position((x_offset, y_offset))
        else:
            # Use the existing logic for resizing and positioning when dimensions are specified
            target_width = parse_percentage(element.get('width', '100%'), video_width)
            target_height = parse_percentage(element.get('height', '100%'), video_height)
            
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
            
            # Position the clip
            x_percentage = element.get('x', "0%")
            y_percentage = element.get('y', "0%")
            final_x = parse_percentage(x_percentage, video_width - final_clip.w)
            final_y = parse_percentage(y_percentage, video_height - final_clip.h)
            final_clip = final_clip.set_position((final_x, final_y))

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
    Creates a text clip from the provided element.
    """
    text = element.get('text')
    start_time = element.get('time', 0.0)
    duration = element.get('duration')

    if not text:
        logging.error(f"Text element {element['id']} has no text content.")
        return None

    font_size = parse_percentage(element.get('font_size', "5%"), min(video_width, video_height), video_height)
    font_url = element.get('font_family')

    if font_url and font_url.startswith('http'):
        font_path = download_file(font_url, suffix='.ttf')
        if not font_path:
            font_path = "Arial"  # Fallback
    else:
        font_path = "Arial"  # Default font

    try:
        # If duration is not specified, use the remaining video duration
        if duration is None:
            duration = total_duration - start_time

        # Create the text clip
        text_clip = TextClip(
            txt=text,
            fontsize=font_size,
            font=font_path,
            color=element.get('fill_color', 'white'),
            method='label',
            transparent=True
        ).set_duration(duration)

        # Parse position percentages
        x_percentage = element.get('x', "0%")
        y_percentage = element.get('y', "0%")
        
        # Calculate positions - now using direct percentage without centering adjustment
        x_pos = parse_percentage(x_percentage, video_width)
        final_y = parse_percentage(y_percentage, video_height)

        logging.info(f"Text positioning - x: {x_pos}, y: {final_y}, text width: {text_clip.w}")

        final_clip = text_clip.set_position((x_pos, final_y)).set_start(start_time)
        final_clip.name = element['id']
        final_clip.track = element.get('track', 0)
        return final_clip

    except Exception as e:
        logging.error(f"Error creating text clip for element {element['id']}: {e}")
        return None
    finally:
        if font_url and font_url.startswith('http') and os.path.exists(font_path):
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
    if element_type == 'audio':
        return create_audio_clip(element)
    elif element_type in ['image', 'video']:
        return create_image_clip(element, video_width, video_height)
    elif element_type == 'text':
        return create_text_clip(element, video_width, video_height, video_spec.get('duration', 15.0))
    else:
        print(f"Unknown element type: {element_type}")
        return None


def apply_animations(clip, animations, element_duration):
    """
    Applies animations to a clip.
    """
    if not animations:
        return clip

    for anim in animations:
        if anim['type'] == 'scale':
            start_time = anim.get('time', 0)
            duration = anim.get('duration', element_duration)
            end_time = start_time + duration
            start_scale = parse_percentage(anim.get('start_scale', '100%'), 100) / 100
            end_scale = parse_percentage(anim.get('end_scale', '100%'), 100) / 100
            easing = anim.get('easing', 'linear')

            def scale_func(t):
                if t < start_time:
                    return start_scale
                elif t > end_time:
                    return end_scale
                else:
                    progress = (t - start_time) / duration
                    
                    # Apply easing function
                    if easing == 'linear':
                        pass
                    elif easing == 'quadratic-out':
                        progress = 1 - (1 - progress) ** 2
                    elif easing == 'quadratic-in':
                        progress = progress ** 2
                    elif easing == 'quadratic-in-out':
                        progress = 2 * progress ** 2 if progress < 0.5 else 1 - (-2 * progress + 2) ** 2 / 2
                    
                    return start_scale + (end_scale - start_scale) * progress

            clip = clip.resize(lambda t: scale_func(t))

            if anim.get('fade', False):
                def fade_func(t):
                    if t < start_time:
                        return 1
                    elif t > end_time:
                        return 0
                    else:
                        return 1 - (t - start_time) / duration
                
                clip = clip.set_opacity(lambda t: fade_func(t))

    return clip


def generate_video(json_data):
    """
    Generates a video based on the provided JSON configuration.

    Args:
        json_data (dict): The JSON configuration.

    Returns:
        str or None: The path to the generated video or None if failed.
    """
    try:
        video_spec = json_data
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







