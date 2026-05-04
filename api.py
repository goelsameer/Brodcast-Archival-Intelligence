from datetime import date, time
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import crud
from database import get_db, init_db
from models import Camera
from schemas import (
    CaptionJobResponse,
    CaptionSearchRequest,
    ArchiveQueryParams,
    CameraArchiveCreate,
    CameraArchiveFetchResponse,
    CameraArchiveResponse,
    CameraArchiveUploadResponse,
    CameraCreate,
    CameraResponse,
)
from services import (
    filter_caption_segments,
    generate_bilingual_captions,
    parse_timestamp_range,
    save_uploaded_video,
    tag_sentiment
)

init_db()

app = FastAPI(
    title="BAI MVP Camera API",
    version="1.0.0",
    description="API for registering cameras and storing/fetching archived camera video metadata.",
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)) -> Camera:
    existing_camera = crud.get_camera_by_number(db, payload.camera_number)
    if existing_camera:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Camera number {payload.camera_number} already exists.",
        )

    try:
        return crud.create_camera(db, payload)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Camera could not be created because of a uniqueness conflict.",
        ) from exc


@app.get("/cameras", response_model=list[CameraResponse])
def list_cameras(db: Session = Depends(get_db)):
    return crud.list_cameras(db)


@app.post("/camera-archives", response_model=CameraArchiveResponse, status_code=status.HTTP_201_CREATED)
def create_camera_archive(payload: CameraArchiveCreate, db: Session = Depends(get_db)):
    camera = crud.get_camera_by_id(db, payload.camera_id)
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {payload.camera_id} was not found.",
        )

    return crud.create_or_update_archive(db, payload)


@app.post("/camera-archives/upload", response_model=CameraArchiveUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_camera_archive(
    camera_id: int = Form(...),
    recorded_date: date = Form(...),
    recording_start_time: Optional[time] = Form(default=None),
    recording_end_time: Optional[time] = Form(default=None),
    video_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    camera = crud.get_camera_by_id(db, camera_id)
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {camera_id} was not found.",
        )

    if (recording_start_time is None) != (recording_end_time is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="recording_start_time and recording_end_time must be provided together.",
        )

    if (
        recording_start_time is not None
        and recording_end_time is not None
        and recording_start_time >= recording_end_time
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="recording_end_time must be later than recording_start_time.",
        )

    try:
        stored_path, filename = save_uploaded_video(camera.camera_number, recorded_date, video_file)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    archive = crud.create_or_update_archive_for_date(
        db=db,
        camera_id=camera_id,
        recorded_date=recorded_date,
        video_path=stored_path,
        recording_start_time=recording_start_time,
        recording_end_time=recording_end_time,
    )

    return CameraArchiveUploadResponse(
        id=archive.id,
        camera_id=archive.camera_id,
        camera_number=camera.camera_number,
        recorded_date=archive.recorded_date,
        video_path=archive.video_path,
        recording_start_time=archive.recording_start_time,
        recording_end_time=archive.recording_end_time,
        filename=filename,
    )


@app.get("/camera-archives", response_model=CameraArchiveFetchResponse)
def fetch_camera_archive(
    params: ArchiveQueryParams = Depends(),
    db: Session = Depends(get_db),
):
    camera = crud.get_camera_by_id(db, params.camera_id)
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {params.camera_id} was not found.",
        )

    archive = crud.get_archive_for_day(db, params.camera_id, params.previous_days_to_current_day)
    if archive is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No archive was found for camera id {params.camera_id} "
                f"and {params.previous_days_to_current_day} day(s) back."
            ),
        )

    if params.start_time and params.end_time:
        if params.start_time < archive.recording_start_time or params.end_time > archive.recording_end_time:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Requested time range is outside the stored recording window. "
                    f"Available range is {archive.recording_start_time} to {archive.recording_end_time}."
                ),
            )

    return CameraArchiveFetchResponse(
        camera_id=archive.camera_id,
        camera_number=camera.camera_number,
        day_number=params.previous_days_to_current_day,
        recorded_date=archive.recorded_date,
        requested_start_time=params.start_time,
        requested_end_time=params.end_time,
        available_start_time=archive.recording_start_time,
        available_end_time=archive.recording_end_time,
        video_path=archive.video_path,
    )

@app.post("/captions/search", response_model=CaptionJobResponse, status_code=status.HTTP_201_CREATED)
def generate_and_search_captions(payload: CaptionSearchRequest, db: Session = Depends(get_db)):
    camera = crud.get_camera_by_id(db, payload.camera_id)
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera with id {payload.camera_id} was not found.",
        )

    archive = crud.get_archive_for_date(db, payload.camera_id, payload.recorded_date)
    if archive is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No archive was found for camera id {payload.camera_id} "
                f"on {payload.recorded_date.isoformat()}."
            ),
        )

    requested_start_time, requested_end_time = parse_timestamp_range(payload.timestamp)
    if requested_start_time and requested_end_time:
        if (
            requested_start_time < archive.recording_start_time
            or requested_end_time > archive.recording_end_time
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Requested timestamp is outside the stored recording window. "
                    f"Available range is {archive.recording_start_time} to {archive.recording_end_time}."
                ),
            )

    try:
        caption_context = generate_bilingual_captions(
            archive.video_path,
            payload.timestamp,
            base_start_time=archive.recording_start_time,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    filtered_segments = filter_caption_segments(caption_context.segments, payload.search_sentence)
    caption_job = crud.create_caption_job(
        db=db,
        camera_id=camera.id,
        archive_id=archive.id,
        recorded_date=archive.recorded_date,
        requested_start_time=caption_context.requested_start_time,
        requested_end_time=caption_context.requested_end_time,
        search_sentence=payload.search_sentence,
        matched_results_only=bool(payload.search_sentence),
        segments=filtered_segments,
    )

    return CaptionJobResponse(
        id=caption_job.id,
        camera_id=caption_job.camera_id,
        archive_id=caption_job.archive_id,
        recorded_date=caption_job.recorded_date,
        requested_start_time=caption_job.requested_start_time,
        requested_end_time=caption_job.requested_end_time,
        search_sentence=caption_job.search_sentence,
        matched_results_only=caption_job.matched_results_only,
        segments=caption_job.segments,
    )
