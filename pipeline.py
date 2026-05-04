from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse
from datetime import datetime, timedelta

import crud
from database import SessionLocal, init_db
from services import export_clip_for_later_use, extract_video_clip

app = FastAPI()

init_db()

# It lets clients request footage from archived camera videos
@app.get("/footage")
def get_footage(
    camera_id: int = Query(..., ge=1, le=64, description="Camera number 1-64"),
    days_ago: int = Query(0, ge=0, le=90, description="How many days back, 0 = today"),
    timestamp: str = Query(None, description="HH:MM:SS-HH:MM:SS, optional"),
):
    # 1. Resolve the date
    target_date = datetime.today() - timedelta(days=days_ago)
    date_str = target_date.strftime("%Y-%m-%d")

    db = SessionLocal()
    try:
        camera = crud.get_camera_by_number(db, camera_id)
        if camera is None:
            raise HTTPException(
                status_code=404,
                detail=f"No camera found with number {camera_id}"
            )

        archive = crud.get_archive_for_date(db, camera.id, target_date.date())
        if archive is None:
            raise HTTPException(
                status_code=404,
                detail=f"No footage found for camera {camera_id} on {date_str}"
            )

        # 3. No timestamp -> redirect to the Cloudinary-hosted full video
        if not timestamp:
            return RedirectResponse(url=archive.video_path)

        # 4. Parse timestamp
        try:
            start_time, end_time = timestamp.split("-")
            parsed_start = datetime.strptime(start_time, "%H:%M:%S").time()
            parsed_end = datetime.strptime(end_time, "%H:%M:%S").time()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="timestamp must be in format HH:MM:SS-HH:MM:SS"
            )

        # 5. Validate start is before end
        if parsed_start >= parsed_end:
            raise HTTPException(
                status_code=400,
                detail="start time must be before end time"
            )

        # 6. Cut with FFmpeg, upload the generated clip to Cloudinary, and redirect
        try:
            clip_path = extract_video_clip(
                video_path=archive.video_path,
                clip_start_time=parsed_start,
                clip_end_time=parsed_end,
                clip_prefix=f"camera_{camera_id}_{date_str}_{timestamp.replace(':', '')}",
                source_start_time=archive.recording_start_time,
            )
            clip_url = export_clip_for_later_use(
                clip_path,
                export_name=f"camera_{camera_id}_{date_str}_{timestamp.replace(':', '')}.mp4",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc)
            ) from exc

        return RedirectResponse(url=clip_url)
    finally:
        db.close()
