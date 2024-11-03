from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from video_generator import generate_video
from webhook_sender import send_webhook
import os
import logging

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

@app.post("/generate_video")
async def create_video(request: VideoRequest, background_tasks: BackgroundTasks, x_webhook_url: str = Header(...)):
    # Generate video in the background
    background_tasks.add_task(process_video_request, request.dict(), x_webhook_url)
    return {"message": "Video generation started"}

async def process_video_request(json_data: dict, webhook_url: str):
    # Generate the video
    video_path = generate_video(json_data)
    
    if video_path:
        try:
            # Send the video via webhook
            send_webhook(webhook_url, video_path)
        except Exception as e:
            print(f"Error sending webhook: {str(e)}")
    else:
        print("Video generation failed; webhook not sent.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
