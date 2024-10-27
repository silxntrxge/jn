from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from video_generator import generate_video
from webhook_sender import send_webhook
import os

app = FastAPI()

class Animation(BaseModel):
    easing: str
    type: str
    fade: bool
    scope: str
    end_scale: str
    start_scale: str

class SubElement(BaseModel):
    id: str
    type: str
    name: Optional[str] = None
    track: Optional[int] = None
    time: Optional[float] = None
    duration: Optional[float] = None
    source: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    width: Optional[str] = None
    height: Optional[str] = None
    x_anchor: Optional[str] = None
    y_anchor: Optional[str] = None
    fill_color: Optional[str] = None
    text: Optional[str] = None
    font_family: Optional[str] = None
    font_size: Optional[str] = None

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
