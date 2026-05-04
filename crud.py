from __future__ import annotations

from datetime import date, timedelta, time
from typing import Callable

from sqlalchemy.orm import Session, joinedload

from models import Camera, CameraArchive
from models import CaptionJob, CaptionSegment
from models import ClipRecord, Face, FaceDetection, WatchlistItem
from schemas import CameraArchiveCreate, CameraCreate
from services import (
    SEMANTIC_CAPTION_MATCH_THRESHOLD,
    build_searchable_caption_text,
    cosine_similarity,
    generate_text_embedding,
    generate_text_embeddings,
)


def create_camera(db: Session, payload: CameraCreate) -> Camera:
    camera = Camera(**payload.model_dump())
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


def list_cameras(db: Session) -> list[Camera]:
    return db.query(Camera).order_by(Camera.camera_number.asc()).all()


def get_camera_by_number(db: Session, camera_number: int) -> Camera | None:
    return db.query(Camera).filter(Camera.camera_number == camera_number).first()


def get_camera_by_id(db: Session, camera_id: int) -> Camera | None:
    return db.query(Camera).filter(Camera.id == camera_id).first()


def create_or_update_archive(db: Session, payload: CameraArchiveCreate) -> CameraArchive:
    recorded_date = date.today() - timedelta(days=payload.day_number)
    return create_or_update_archive_for_date(
        db=db,
        camera_id=payload.camera_id,
        recorded_date=recorded_date,
        video_path=payload.video_path,
        recording_start_time=payload.recording_start_time or time(0, 0, 0),
        recording_end_time=payload.recording_end_time or time(23, 59, 59),
    )


def create_or_update_archive_for_date(
    db: Session,
    camera_id: int,
    recorded_date: date,
    video_path: str,
    recording_start_time: time | None = None,
    recording_end_time: time | None = None,
) -> CameraArchive:
    archive_start_time = recording_start_time or time(0, 0, 0)
    archive_end_time = recording_end_time or time(23, 59, 59)

    archive = (
        db.query(CameraArchive)
        .filter(
            CameraArchive.camera_id == camera_id,
            CameraArchive.recorded_date == recorded_date,
        )
        .first()
    )

    archive_data = {
        "video_path": video_path,
        "recording_start_time": archive_start_time,
        "recording_end_time": archive_end_time,
    }

    if archive is None:
        archive = CameraArchive(
            camera_id=camera_id,
            recorded_date=recorded_date,
            **archive_data,
        )
        db.add(archive)
    else:
        for field, value in archive_data.items():
            setattr(archive, field, value)

    db.commit()
    db.refresh(archive)
    return archive


def get_archive_for_day(db: Session, camera_id: int, previous_days_to_current_day: int) -> CameraArchive | None:
    recorded_date = date.today() - timedelta(days=previous_days_to_current_day)
    return (
        db.query(CameraArchive)
        .filter(
            CameraArchive.camera_id == camera_id,
            CameraArchive.recorded_date == recorded_date,
        )
        .first()
    )


def get_archive_for_date(db: Session, camera_id: int, recorded_date: date) -> CameraArchive | None:
    return (
        db.query(CameraArchive)
        .filter(
            CameraArchive.camera_id == camera_id,
            CameraArchive.recorded_date == recorded_date,
        )
        .first()
    )


def create_caption_job(
    db: Session,
    camera_id: int,
    archive_id: int,
    recorded_date: date,
    requested_start_time: time | None,
    requested_end_time: time | None,
    search_sentence: str | None,
    matched_results_only: bool,
    segments: list[dict],
    progress_callback: Callable[[int, str], None] | None = None,
) -> CaptionJob:
    def report(percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, percent)), message)

    report(92, "Creating caption job")
    caption_job = CaptionJob(
        camera_id=camera_id,
        archive_id=archive_id,
        recorded_date=recorded_date,
        requested_start_time=requested_start_time,
        requested_end_time=requested_end_time,
        search_sentence=search_sentence,
        matched_results_only=matched_results_only,
    )
    db.add(caption_job)
    db.flush()

    report(94, "Preparing searchable captions")
    searchable_texts = [
        build_searchable_caption_text(segment["hindi_caption"], segment["english_caption"])
        for segment in segments
    ]

    report(95, "Generating search embeddings")
    embeddings = generate_text_embeddings(searchable_texts)

    total_segments = max(1, len(segments))
    for index, (segment, searchable_text, embedding) in enumerate(
        zip(segments, searchable_texts, embeddings)
    ):
        db.add(
            CaptionSegment(
                caption_job_id=caption_job.id,
                segment_index=index,
                start_time=segment.get("start_time"),
                end_time=segment.get("end_time"),
                hindi_caption=segment["hindi_caption"],
                english_caption=segment["english_caption"],
                searchable_text=searchable_text,
                embedding=embedding,
            )
        )
        if index % 5 == 0 or index == total_segments - 1:
            report(96 + int(((index + 1) / total_segments) * 2), "Adding caption segments")

    report(99, "Committing caption index")
    db.commit()
    db.refresh(caption_job)
    return caption_job


def get_latest_caption_job_for_archive(db: Session, archive_id: int) -> CaptionJob | None:
    return (
        db.query(CaptionJob)
        .options(joinedload(CaptionJob.segments))
        .filter(CaptionJob.archive_id == archive_id)
        .order_by(CaptionJob.created_at.desc(), CaptionJob.id.desc())
        .first()
    )


def list_latest_caption_jobs(
    db: Session,
    camera_id: int | None = None,
    recorded_date: date | None = None,
) -> list[CaptionJob]:
    query = (
        db.query(CaptionJob)
        .options(
            joinedload(CaptionJob.segments),
            joinedload(CaptionJob.archive),
            joinedload(CaptionJob.camera),
        )
    )

    if camera_id is not None:
        query = query.filter(CaptionJob.camera_id == camera_id)
    if recorded_date is not None:
        query = query.filter(CaptionJob.recorded_date == recorded_date)

    jobs = query.order_by(CaptionJob.created_at.desc(), CaptionJob.id.desc()).all()
    latest_by_archive = {}
    for job in jobs:
        if job.archive_id not in latest_by_archive:
            latest_by_archive[job.archive_id] = job
    return list(latest_by_archive.values())


def search_caption_segments(
    db: Session,
    query_text: str,
    camera_id: int | None = None,
    recorded_date: date | None = None,
    range_start_time: time | None = None,
    range_end_time: time | None = None,
    confidence_threshold: float = SEMANTIC_CAPTION_MATCH_THRESHOLD,
    limit: int = 10,
) -> list[CaptionSegment]:
    normalized_query = query_text.strip()
    if not normalized_query:
        return []

    query = (
        db.query(CaptionSegment)
        .join(CaptionSegment.caption_job)
        .join(CaptionJob.archive)
        .join(CaptionJob.camera)
        .options(
            joinedload(CaptionSegment.caption_job).joinedload(CaptionJob.archive),
            joinedload(CaptionSegment.caption_job).joinedload(CaptionJob.camera),
        )
    )

    if camera_id is not None:
        query = query.filter(CaptionJob.camera_id == camera_id)

    if recorded_date is not None:
        query = query.filter(CaptionJob.recorded_date == recorded_date)

    if range_start_time is not None:
        query = query.filter(CaptionSegment.start_time.isnot(None), CaptionSegment.start_time >= range_start_time)

    if range_end_time is not None:
        query = query.filter(CaptionSegment.end_time.isnot(None), CaptionSegment.end_time <= range_end_time)

    candidates = query.order_by(CaptionJob.recorded_date.desc(), CaptionSegment.segment_index.asc()).all()
    query_embedding = generate_text_embedding(normalized_query)
    changed_segments = False

    ranked_candidates = []
    for segment in candidates:
        searchable_text = segment.searchable_text or build_searchable_caption_text(
            segment.hindi_caption,
            segment.english_caption,
        )
        if not segment.searchable_text:
            segment.searchable_text = searchable_text
            changed_segments = True

        segment_embedding = segment.embedding
        if not segment_embedding:
            segment_embedding = generate_text_embedding(searchable_text)
            segment.embedding = segment_embedding
            changed_segments = True

        confidence = cosine_similarity(segment_embedding, query_embedding)
        if confidence < confidence_threshold:
            continue

        setattr(segment, "match_confidence", confidence)
        ranked_candidates.append((confidence, segment))

    if changed_segments:
        db.commit()

    ranked_candidates.sort(key=lambda item: item[0], reverse=True)
    return [segment for _, segment in ranked_candidates[:limit]]


def create_face(db: Session, image_path: str, embedding: list[float]) -> Face:
    face = Face(image_path=image_path, embedding=embedding)
    db.add(face)
    db.commit()
    db.refresh(face)
    return face


def list_faces(db: Session) -> list[Face]:
    return db.query(Face).order_by(Face.uploaded_at.desc()).all()


def create_face_detection(
    db: Session,
    camera_id: int,
    archive_id: int,
    recorded_date: date,
    start_time: time,
    end_time: time,
    face_embedding: list[float],
    image_path: str,
    confidence: float,
) -> FaceDetection:
    detection = FaceDetection(
        camera_id=camera_id,
        archive_id=archive_id,
        recorded_date=recorded_date,
        start_time=start_time,
        end_time=end_time,
        face_embedding=face_embedding,
        image_path=image_path,
        confidence=confidence,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def search_face_matches(
    db: Session,
    query_embedding: list[float],
    camera_id: int | None = None,
    recorded_date: date | None = None,
    threshold: float = 0.6,
    limit: int = 10,
) -> list[FaceDetection]:
    from services import search_face_matches as service_search
    return service_search(db, query_embedding, camera_id, recorded_date, threshold, limit)


def list_watchlist_items(db: Session, active_only: bool = False) -> list[WatchlistItem]:
    query = db.query(WatchlistItem)
    if active_only:
        query = query.filter(WatchlistItem.is_active.is_(True))
    return query.order_by(WatchlistItem.created_at.desc(), WatchlistItem.id.desc()).all()


def create_or_update_watchlist_item(
    db: Session,
    *,
    term: str,
    item_type: str = "keyword",
    language: str | None = None,
    reference_image_path: str | None = None,
    face_embedding: list[float] | None = None,
    is_active: bool = True,
) -> WatchlistItem:
    normalized_term = term.strip()
    if not normalized_term:
        raise ValueError("Watchlist term cannot be empty.")

    watchlist_item = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.term == normalized_term)
        .first()
    )
    if watchlist_item is None:
        watchlist_item = WatchlistItem(
            term=normalized_term,
            item_type=item_type,
            language=language,
            reference_image_path=reference_image_path,
            face_embedding=face_embedding,
            is_active=is_active,
        )
        db.add(watchlist_item)
    else:
        watchlist_item.item_type = item_type
        watchlist_item.language = language
        if reference_image_path is not None:
            watchlist_item.reference_image_path = reference_image_path
        if face_embedding is not None:
            watchlist_item.face_embedding = face_embedding
        watchlist_item.is_active = is_active

    db.commit()
    db.refresh(watchlist_item)
    return watchlist_item


def list_watchlist_face_items(db: Session, active_only: bool = True) -> list[WatchlistItem]:
    query = db.query(WatchlistItem).filter(
        WatchlistItem.item_type == "person",
        WatchlistItem.face_embedding.isnot(None),
    )
    if active_only:
        query = query.filter(WatchlistItem.is_active.is_(True))
    return query.order_by(WatchlistItem.created_at.desc(), WatchlistItem.id.desc()).all()


def list_archives(
    db: Session,
    camera_id: int | None = None,
    recorded_date: date | None = None,
) -> list[CameraArchive]:
    query = db.query(CameraArchive).options(joinedload(CameraArchive.camera))
    if camera_id is not None:
        query = query.filter(CameraArchive.camera_id == camera_id)
    if recorded_date is not None:
        query = query.filter(CameraArchive.recorded_date == recorded_date)
    return query.order_by(CameraArchive.recorded_date.desc(), CameraArchive.id.desc()).all()


def update_watchlist_item_status(db: Session, watchlist_item_id: int, is_active: bool) -> WatchlistItem | None:
    watchlist_item = db.query(WatchlistItem).filter(WatchlistItem.id == watchlist_item_id).first()
    if watchlist_item is None:
        return None
    watchlist_item.is_active = is_active
    db.commit()
    db.refresh(watchlist_item)
    return watchlist_item


def get_clip_record_by_source(
    db: Session,
    *,
    source_type: str,
    source_reference: str,
) -> ClipRecord | None:
    return (
        db.query(ClipRecord)
        .filter(
            ClipRecord.source_type == source_type,
            ClipRecord.source_reference == source_reference,
        )
        .first()
    )


def get_clip_record_by_path(db: Session, clip_path: str) -> ClipRecord | None:
    return db.query(ClipRecord).filter(ClipRecord.clip_path == clip_path).first()


def get_clip_record_by_selection(
    db: Session,
    *,
    source_type: str,
    camera_id: int | None,
    archive_id: int | None,
    recorded_date: date | None,
    start_time: time | None,
    end_time: time | None,
) -> ClipRecord | None:
    return (
        db.query(ClipRecord)
        .filter(
            ClipRecord.source_type == source_type,
            ClipRecord.camera_id == camera_id,
            ClipRecord.archive_id == archive_id,
            ClipRecord.recorded_date == recorded_date,
            ClipRecord.start_time == start_time,
            ClipRecord.end_time == end_time,
        )
        .first()
    )


def list_clip_records(db: Session) -> list[ClipRecord]:
    return db.query(ClipRecord).order_by(ClipRecord.created_at.desc(), ClipRecord.id.desc()).all()


def create_or_get_clip_record(
    db: Session,
    *,
    clip_path: str,
    source_type: str,
    source_reference: str | None,
    camera_id: int | None,
    archive_id: int | None,
    recorded_date: date | None,
    start_time: time | None,
    end_time: time | None,
    watchlist_item_id: int | None = None,
    matched_text: str | None = None,
    sentiment: str | None = None,
    context_text: str | None = None,
) -> ClipRecord:
    clip_record = get_clip_record_by_path(db, clip_path)
    if clip_record is None:
        clip_record = ClipRecord(
            clip_path=clip_path,
            source_type=source_type,
            source_reference=source_reference,
            camera_id=camera_id,
            archive_id=archive_id,
            recorded_date=recorded_date,
            start_time=start_time,
            end_time=end_time,
            watchlist_item_id=watchlist_item_id,
            matched_text=matched_text,
            sentiment=sentiment,
            context_text=context_text,
        )
        db.add(clip_record)
        db.commit()
        db.refresh(clip_record)
        return clip_record

    changed = False
    clip_fields = {
        "source_type": source_type,
        "source_reference": source_reference,
        "camera_id": camera_id,
        "archive_id": archive_id,
        "recorded_date": recorded_date,
        "start_time": start_time,
        "end_time": end_time,
        "watchlist_item_id": watchlist_item_id,
        "matched_text": matched_text,
        "sentiment": sentiment,
        "context_text": context_text,
    }
    for field, value in clip_fields.items():
        if getattr(clip_record, field) is None and value is not None:
            setattr(clip_record, field, value)
            changed = True

    if changed:
        db.commit()
        db.refresh(clip_record)

    return clip_record


def update_clip_record_status(
    db: Session,
    clip_record_id: int,
    *,
    is_reviewed: bool | None = None,
    is_important: bool | None = None,
    export_path: str | None = None,
    exported_at=None,
) -> ClipRecord | None:
    clip_record = db.query(ClipRecord).filter(ClipRecord.id == clip_record_id).first()
    if clip_record is None:
        return None

    if is_reviewed is not None:
        clip_record.is_reviewed = is_reviewed
    if is_important is not None:
        clip_record.is_important = is_important
    if export_path is not None:
        clip_record.export_path = export_path
        clip_record.exported_at = exported_at

    db.commit()
    db.refresh(clip_record)
    return clip_record
