from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from video_generator import generate_video
from webhook_sender import send_webhook
import os
import logging
import resource
import gc
import psutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import math

app = FastAPI()

class Animation(BaseModel):
    """Model for animation properties"""
    time: float = 0
    duration: float
    type: str  # e.g., 'scale'
    easing: str = 'linear'  # e.g., 'quadratic-out', 'linear', etc.
    start_scale: Optional[str] = '0%'  # e.g., '0%', '100%'
    end_scale: Optional[str] = '100%'  # e.g., '100%', '110%'
    x_anchor: Optional[str] = '50%'  # Default top-left
    y_anchor: Optional[str] = '50%'  # Default top-left
    axis: Optional[str] = 'both'  # 'x', 'y', or 'both'
    fade: bool = False

class SubElement(BaseModel):
    id: str
    type: str
    name: Optional[str] = None
    track: Optional[int] = None
    time: Optional[float] = None
    duration: Optional[float] = None
    source: Optional[str] = None
    # Add loop parameter for video elements
    loop: Optional[bool] = False
    # Add default values for x and width
    x: Optional[str] = "50%"  # Default center position
    y: Optional[str] = "50%"
    width: Optional[str] = "100%"  # Default full width
    height: Optional[str] = "100%"
    x_anchor: Optional[str] = None
    y_anchor: Optional[str] = None
    fill_color: Optional[str] = None
    stroke_color: Optional[str] = None
    stroke_width: Optional[str] = None
    text: Optional[str] = None
    font_family: Optional[str] = None
    font_size: Optional[str] = None
    # Add animations field
    animations: Optional[List[Animation]] = None

class Element(SubElement):
    elements: Optional[List[SubElement]] = None

class VideoRequest(BaseModel):
    output_format: str
    width: int
    height: int
    duration: float
    snapshot_time: Optional[float] = None  # Make this optional
    elements: List[Union[SubElement, Element]]  # Allow both SubElement and Element
    fps: Optional[int] = 30  # Added fps field to set frames per second

# Set memory limits
def set_memory_limits(soft_limit, hard_limit):
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))

# Periodic garbage collection
gc.collect()

def get_system_resources():
    """Get available system resources"""
    cpu_count = multiprocessing.cpu_count()
    memory = psutil.virtual_memory()
    return {
        'cpu_cores': cpu_count,
        'total_memory': memory.total,
        'available_memory': memory.available,
        'memory_percent': memory.percent
    }

def calculate_resource_limits():
    """Calculate safe resource limits"""
    resources = get_system_resources()
    
    # Leave 20% of resources for system
    safe_memory = int(resources['available_memory'] * 0.8)
    safe_cpu_cores = max(1, math.floor(resources['cpu_cores'] * 0.8))
    
    return {
        'memory_limit': safe_memory,
        'cpu_cores': safe_cpu_cores,
        'worker_threads': safe_cpu_cores
    }

@app.post("/generate_video")
async def create_video(request: VideoRequest, background_tasks: BackgroundTasks, x_webhook_url: str = Header(...)):
    # Check available resources
    resources = get_system_resources()
    if resources['memory_percent'] > 90:
        raise HTTPException(status_code=503, detail="System resources currently exhausted")
    
    # Calculate safe limits
    limits = calculate_resource_limits()
    
    # Add resource limits to the request
    request_dict = request.dict()
    request_dict['resource_limits'] = limits
    
    # Generate video in the background
    background_tasks.add_task(process_video_request, request_dict, x_webhook_url)
    return {"message": "Video generation started"}

async def process_video_request(json_data: dict, webhook_url: str):
    logging.info(f"Starting video generation with webhook URL: {webhook_url}")
    # Generate the video
    video_url = generate_video(json_data)
    
    if video_url:
        try:
            logging.info(f"Video generated successfully, URL: {video_url}")
            # Send the video via webhook
            success = send_webhook(webhook_url, video_url)
            if success:
                logging.info(f"Webhook sent successfully to {webhook_url}")
            else:
                logging.error("Webhook sending failed")
        except Exception as e:
            logging.error(f"Error sending webhook: {str(e)}", exc_info=True)
    else:
        logging.error("Video generation failed; webhook not sent.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
