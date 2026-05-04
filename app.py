from __future__ import annotations

from datetime import datetime
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace
import os

import streamlit as st

import crud
from database import SessionLocal, init_db
from schemas import CameraCreate, CaptionSearchRequest
from services import (
    SEMANTIC_CAPTION_MATCH_THRESHOLD,
    build_searchable_caption_text,
    cosine_similarity,
    detect_matching_faces_in_video,
    export_clip_for_later_use,
    extract_video_clip,
    generate_bilingual_captions,
    generate_face_embedding,
    generate_text_embedding,
    get_stored_file_name,
    is_stored_file_available,
    read_stored_file_bytes,
    parse_timestamp_range,
    save_clip_from_archive,
    save_watchlist_reference_image,
    save_uploaded_video,
    tag_sentiment,
)

init_db()

st.set_page_config(page_title="BAI MVP Camera Manager", page_icon="camera", layout="wide")


def _show_pending_toast() -> None:
    pending_toast = st.session_state.pop("pending_toast", None)
    if pending_toast:
        st.toast(pending_toast)


def _namespace_from_dict(payload: dict | None):
    return SimpleNamespace(**(payload or {}))


def _format_bytes(bytes_value: int) -> str:
    if bytes_value == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} TB"


def render_endpoint_reference(method: str, path: str, description: str) -> None:
    st.caption(f"Endpoint: `{method} {path}`")
    st.write(description)


def _build_export_filename(
    *,
    source_type: str,
    camera_number: int | None,
    recorded_date: date | None,
    start_time: time | None,
    end_time: time | None,
) -> str:
    date_label = recorded_date.isoformat() if recorded_date is not None else datetime.utcnow().date().isoformat()
    start_label = start_time.strftime("%H%M%S") if start_time is not None else "unknown_start"
    end_label = end_time.strftime("%H%M%S") if end_time is not None else "unknown_end"
    camera_label = f"camera_{camera_number}" if camera_number is not None else "camera_unknown"
    return f"{source_type}_{camera_label}_{date_label}_{start_label}_{end_label}.mp4"


def _render_save_clip_action(
    db,
    *,
    source_type: str,
    source_reference: str,
    camera,
    archive,
    start_time: time | None,
    end_time: time | None,
    action_key: str,
) -> None:
    clip_record = crud.get_clip_record_by_selection(
        db,
        source_type=source_type,
        camera_id=getattr(camera, "id", None),
        archive_id=getattr(archive, "id", None),
        recorded_date=getattr(archive, "recorded_date", None),
        start_time=start_time,
        end_time=end_time,
    )
    if clip_record is not None:
        st.caption("Clip already saved. You can manage it in the Saved Clips tab.")
        return

    if st.button("Save Clip", key=f"save_{action_key}"):
        try:
            saved_clip_path = save_clip_from_archive(
                video_path=archive.video_path,
                clip_start_time=start_time,
                clip_end_time=end_time,
                clip_prefix=f"saved_{source_type}_{getattr(camera, 'camera_number', 'unknown')}_{getattr(archive, 'recorded_date', 'unknown')}",
                source_start_time=getattr(archive, "recording_start_time", None),
                saved_name=_build_export_filename(
                    source_type=f"saved_{source_type}",
                    camera_number=getattr(camera, "camera_number", None),
                    recorded_date=getattr(archive, "recorded_date", None),
                    start_time=start_time,
                    end_time=end_time,
                ),
            )
        except Exception as exc:
            st.error(f"Could not save clip: {exc}")
            return

        crud.create_or_get_clip_record(
            db,
            clip_path=saved_clip_path,
            source_type=source_type,
            source_reference=source_reference,
            camera_id=getattr(camera, "id", None),
            archive_id=getattr(archive, "id", None),
            recorded_date=getattr(archive, "recorded_date", None),
            start_time=start_time,
            end_time=end_time,
        )
        st.session_state["pending_toast"] = "Clip saved"
        st.rerun()


def _render_saved_clip_actions(db, clip_record, action_key: str) -> None:
    status_parts = []
    if clip_record.is_reviewed:
        status_parts.append("Reviewed")
    if clip_record.is_important:
        status_parts.append("Important")
    if clip_record.export_path:
        status_parts.append(f"Exported: `{clip_record.export_path}`")
    if clip_record.matched_text:
        status_parts.append(f"Matched: `{clip_record.matched_text}`")
    if clip_record.sentiment:
        status_parts.append(f"Sentiment: {clip_record.sentiment}")

    st.caption(" | ".join(status_parts) if status_parts else "Status: saved")

    review_col, important_col, export_col, download_col = st.columns(4)

    review_label = "Mark Unreviewed" if clip_record.is_reviewed else "Mark Reviewed"
    if review_col.button(review_label, key=f"review_{action_key}"):
        crud.update_clip_record_status(
            db,
            clip_record.id,
            is_reviewed=not clip_record.is_reviewed,
        )
        st.rerun()

    important_label = "Unflag Important" if clip_record.is_important else "Flag Important"
    if important_col.button(important_label, key=f"important_{action_key}"):
        crud.update_clip_record_status(
            db,
            clip_record.id,
            is_important=not clip_record.is_important,
        )
        st.rerun()

    if export_col.button("Export Clip", key=f"export_{action_key}"):
        try:
            export_path = export_clip_for_later_use(
                clip_record.clip_path,
                export_name=_build_export_filename(
                    source_type=clip_record.source_type,
                    camera_number=getattr(clip_record.camera, "camera_number", None),
                    recorded_date=clip_record.recorded_date,
                    start_time=clip_record.start_time,
                    end_time=clip_record.end_time,
                ),
            )
            crud.update_clip_record_status(
                db,
                clip_record.id,
                export_path=export_path,
                exported_at=datetime.utcnow(),
            )
            st.success(f"Clip exported to Cloudinary: `{export_path}`")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not export clip: {exc}")

    try:
        download_col.download_button(
            "Download Clip",
            data=read_stored_file_bytes(clip_record.clip_path),
            file_name=get_stored_file_name(clip_record.clip_path),
            mime="video/mp4",
            key=f"download_{action_key}",
        )
    except Exception as exc:
        download_col.caption(f"Download unavailable: {exc}")


def _render_caption_results(db, results: list[dict], sentiment_label: str | None = None) -> None:
    if not results:
        return

    st.success(f"Found {len(results)} matching caption segment(s).")

    for result in results:
        camera = _namespace_from_dict(result["camera"])
        archive = _namespace_from_dict(result["archive"])
        confidence = result.get("confidence")
        confidence_label = f" | Confidence: {confidence * 100:.0f}%" if confidence is not None else ""

        st.markdown(
            f"**Camera {camera.camera_number}** | `{result['recorded_date'].isoformat()}` | "
            f"`{result['start_time']}` to `{result['end_time']}`"
            f"{confidence_label}"
        )
        st.write(f"Hindi : {result['hindi_caption']}")
        st.write(f"English : {result['english_caption']}")

        if result["start_time"] is None or result["end_time"] is None:
            st.caption("No segment timestamps were stored for this caption, so a clip preview is not available.")
            if sentiment_label:
                st.text(f"Query Sentiment: {sentiment_label}")
            st.divider()
            continue

        clip_path = result.get("clip_path")
        if clip_path and is_stored_file_available(clip_path):
            st.video(clip_path)
        else:
            st.error("Clip preview is no longer available for this result.")

        _render_save_clip_action(
            db,
            source_type="caption",
            source_reference=result["source_reference"],
            camera=camera,
            archive=archive,
            start_time=result["start_time"],
            end_time=result["end_time"],
            action_key=result["action_key"],
        )
        if sentiment_label:
            st.text(f"Query Sentiment: {sentiment_label}")
        st.divider()


def _render_face_results(db, results: list[dict], errors: list[str] | None = None) -> None:
    if not results:
        return

    st.success(f"Found {len(results)} matching face detection(s).")
    if errors:
        st.warning(f"Skipped {len(errors)} archive(s) because detection failed.")
        for error in errors:
            st.caption(error)

    for result in results:
        camera = _namespace_from_dict(result["camera"])
        archive = _namespace_from_dict(result["archive"])

        st.markdown(
            f"**Camera {camera.camera_number}** | `{archive.recorded_date.isoformat()}` | "
            f"`{result['start_time']}` to `{result['end_time']}` | Similarity: {result['confidence']:.2f}"
        )
        if result.get("image_path"):
            st.image(result["image_path"], caption="Detected Face", width=100)

        clip_path = result.get("clip_path")
        if clip_path and is_stored_file_available(clip_path):
            st.video(clip_path)
        else:
            st.error("Clip preview is no longer available for this result.")

        _render_save_clip_action(
            db,
            source_type="face",
            source_reference=result["source_reference"],
            camera=camera,
            archive=archive,
            start_time=result["start_time"],
            end_time=result["end_time"],
            action_key=result["action_key"],
        )
        st.divider()


def _camera_to_result_dict(camera) -> dict:
    return {
        "id": getattr(camera, "id", None),
        "camera_number": getattr(camera, "camera_number", None),
        "name": getattr(camera, "name", None),
    }


def _archive_to_result_dict(archive) -> dict:
    return {
        "id": getattr(archive, "id", None),
        "video_path": getattr(archive, "video_path", None),
        "recorded_date": getattr(archive, "recorded_date", None),
        "recording_start_time": getattr(archive, "recording_start_time", None),
        "recording_end_time": getattr(archive, "recording_end_time", None),
    }


def _make_progress_reporter(progress_bar, progress_status):
    last_percent = {"value": -1}

    def report(percent: int, message: str) -> None:
        safe_percent = max(0, min(100, int(percent)))
        if safe_percent < last_percent["value"]:
            safe_percent = last_percent["value"]
        last_percent["value"] = safe_percent
        progress_bar.progress(safe_percent)
        progress_status.caption(f"{safe_percent}% - {message}")

    return report


def _auto_save_watchlist_clips_for_caption_job(db, caption_job, progress_callback=None) -> tuple[list, list[str]]:
    watchlist_items = crud.list_watchlist_items(db, active_only=True)
    if not watchlist_items:
        return [], []

    phrase_embeddings = {
        watchlist_item.id: generate_text_embedding(watchlist_item.term)
        for watchlist_item in watchlist_items
    }
    saved_records = []
    errors = []
    segments = list(getattr(caption_job, "segments", []) or [])
    total_checks = max(1, len(segments) * len(watchlist_items))
    checks_done = 0

    for segment in segments:
        context_text = build_searchable_caption_text(segment.hindi_caption, segment.english_caption)
        if segment.start_time is None or segment.end_time is None:
            continue

        caption_embedding = segment.embedding
        if not caption_embedding:
            caption_embedding = generate_text_embedding(context_text)
            segment.embedding = caption_embedding
            db.commit()

        for watchlist_item in watchlist_items:
            checks_done += 1
            if progress_callback is not None:
                progress_callback(
                    99,
                    "Scanning watchlist matches",
                )

            confidence = cosine_similarity(caption_embedding, phrase_embeddings.get(watchlist_item.id))
            if confidence < SEMANTIC_CAPTION_MATCH_THRESHOLD:
                continue

            source_reference = f"watchlist:{watchlist_item.id}:caption_segment:{segment.id}"
            existing_record = crud.get_clip_record_by_source(
                db,
                source_type="watchlist",
                source_reference=source_reference,
            )
            if existing_record is not None:
                saved_records.append(existing_record)
                continue

            archive = caption_job.archive
            camera = caption_job.camera
            sentiment = tag_sentiment(context_text)
            try:
                saved_clip_path = save_clip_from_archive(
                    video_path=archive.video_path,
                    clip_start_time=segment.start_time,
                    clip_end_time=segment.end_time,
                    clip_prefix=(
                        f"watchlist_{watchlist_item.id}_camera_{camera.camera_number}_"
                        f"{caption_job.recorded_date.isoformat()}_{segment.id}"
                    ),
                    source_start_time=archive.recording_start_time,
                    saved_name=_build_export_filename(
                        source_type=f"watchlist_{watchlist_item.id}",
                        camera_number=getattr(camera, "camera_number", None),
                        recorded_date=caption_job.recorded_date,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                    ),
                )
                clip_record = crud.create_or_get_clip_record(
                    db,
                    clip_path=saved_clip_path,
                    source_type="watchlist",
                    source_reference=source_reference,
                    camera_id=caption_job.camera_id,
                    archive_id=caption_job.archive_id,
                    recorded_date=caption_job.recorded_date,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    watchlist_item_id=watchlist_item.id,
                    matched_text=watchlist_item.term,
                    sentiment=sentiment,
                    context_text=f"{context_text}\nSemantic match confidence: {confidence:.2f}",
                )
                saved_records.append(clip_record)
            except Exception as exc:
                errors.append(f"{watchlist_item.term} at {segment.start_time}-{segment.end_time}: {exc}")

    return saved_records, errors


def _auto_save_watchlist_face_clips_for_archives(
    db,
    archives,
    progress_callback=None,
    threshold: float = 0.6,
) -> tuple[list, list[str]]:
    watchlist_items = crud.list_watchlist_face_items(db, active_only=True)
    if not watchlist_items or not archives:
        return [], []

    saved_records = []
    errors = []
    total_checks = max(1, len(watchlist_items) * len(archives))
    checks_done = 0

    for watchlist_item in watchlist_items:
        for archive in archives:
            checks_done += 1
            if progress_callback is not None:
                progress_callback(
                    90 + int((checks_done / total_checks) * 9),
                    f"Scanning face watchlist: {watchlist_item.term}",
                )

            try:
                matches = detect_matching_faces_in_video(
                    video_path=archive.video_path,
                    query_embedding=watchlist_item.face_embedding,
                    archive=archive,
                    threshold=threshold,
                    sample_interval_seconds=2,
                )
            except Exception as exc:
                camera_number = (
                    archive.camera.camera_number
                    if getattr(archive, "camera", None) is not None
                    else archive.camera_id
                )
                errors.append(f"{watchlist_item.term} on camera {camera_number}: {exc}")
                continue

            for match in matches:
                start_time = match.get("start_time") or match.get("timestamp")
                end_time = match.get("end_time") or match.get("timestamp")
                source_reference = f"watchlist_face:{watchlist_item.id}:archive:{archive.id}:{start_time}"
                existing_record = crud.get_clip_record_by_source(
                    db,
                    source_type="watchlist_face",
                    source_reference=source_reference,
                )
                if existing_record is not None:
                    saved_records.append(existing_record)
                    continue

                camera = getattr(archive, "camera", None)
                try:
                    saved_clip_path = save_clip_from_archive(
                        video_path=archive.video_path,
                        clip_start_time=start_time,
                        clip_end_time=end_time,
                        clip_prefix=(
                            f"watchlist_face_{watchlist_item.id}_camera_"
                            f"{getattr(camera, 'camera_number', archive.camera_id)}_{archive.recorded_date.isoformat()}"
                        ),
                        source_start_time=archive.recording_start_time,
                        saved_name=_build_export_filename(
                            source_type=f"watchlist_face_{watchlist_item.id}",
                            camera_number=getattr(camera, "camera_number", None),
                            recorded_date=archive.recorded_date,
                            start_time=start_time,
                            end_time=end_time,
                        ),
                    )
                    clip_record = crud.create_or_get_clip_record(
                        db,
                        clip_path=saved_clip_path,
                        source_type="watchlist_face",
                        source_reference=source_reference,
                        camera_id=archive.camera_id,
                        archive_id=archive.id,
                        recorded_date=archive.recorded_date,
                        start_time=start_time,
                        end_time=end_time,
                        watchlist_item_id=watchlist_item.id,
                        matched_text=watchlist_item.term,
                        sentiment=None,
                        context_text=f"Face match similarity: {match['confidence']:.2f}",
                    )
                    saved_records.append(clip_record)
                except Exception as exc:
                    errors.append(f"{watchlist_item.term} at {start_time}-{end_time}: {exc}")

    return saved_records, errors


def render_camera_form() -> None:
    st.subheader("Register Camera")
    render_endpoint_reference(
        "POST",
        "/cameras",
        "Create a camera entry so uploaded recordings can be attached to it later.",
    )

    with st.form("camera_form", clear_on_submit=True):
        camera_number = st.number_input("Camera Number", min_value=1, max_value=64, value=1, step=1)
        name = st.text_input("Camera Name", placeholder="Front Gate")
        location = st.text_input("Location", placeholder="Main entrance")
        stream_url = st.text_input("Stream URL", placeholder="rtsp://camera-stream")
        submitted = st.form_submit_button("Add Camera")

    if not submitted:
        return

    if not name.strip():
        st.error("Camera name is required.")
        return

    db = SessionLocal()
    try:
        existing_camera = crud.get_camera_by_number(db, int(camera_number))
        if existing_camera:
            st.error(f"Camera number {int(camera_number)} already exists.")
            return

        camera = crud.create_camera(
            db,
            CameraCreate(
                camera_number=int(camera_number),
                name=name.strip(),
                location=location.strip() or None,
                stream_url=stream_url.strip() or None,
            ),
        )
        st.success(f"Camera created with database id {camera.id}.")
    finally:
        db.close()


def render_archive_form() -> None:
    st.subheader("Upload Camera Video")
    render_endpoint_reference(
        "POST",
        "/camera-archives/upload",
        "Pick a camera, choose the day, upload a video file, and store the archive record in the database.",
    )

    db = SessionLocal()
    try:
        cameras = crud.list_cameras(db)
        camera_options = {f"{camera.camera_number} - {camera.name}": camera for camera in cameras}

        if not camera_options:
            st.info("No cameras exist yet. Add a camera first.")
            return

        with st.form("archive_form", clear_on_submit=True):
            selected_label = st.selectbox("Camera", options=list(camera_options.keys()))
            recorded_date = st.date_input("Recorded Date", value=date.today())
            recording_window = st.checkbox("Specify recording start and end time", value=False)

            start_time_value = None
            end_time_value = None
            if recording_window:
                start_time_value = st.time_input("Recording Start Time", value=time(0, 0, 0))
                end_time_value = st.time_input("Recording End Time", value=time(23, 59, 59))

            uploaded_file = st.file_uploader(
                "Upload Video File",
                type=["mp4", "mov", "avi", "mkv"],
            )
            submitted = st.form_submit_button("Upload and Save")

        if not submitted:
            return

        if uploaded_file is None:
            st.error("Please upload a video file.")
            return

        if recording_window and start_time_value >= end_time_value:
            st.error("Recording end time must be later than the start time.")
            return

        selected_camera = camera_options[selected_label]
        try:
            saved_path, original_filename = save_uploaded_video(
                selected_camera.camera_number,
                recorded_date,
                uploaded_file,
            )
        except Exception as exc:
            st.error(f"Could not upload video to Cloudinary: {exc}")
            return

        archive = crud.create_or_update_archive_for_date(
            db=db,
            camera_id=selected_camera.id,
            recorded_date=recorded_date,
            video_path=saved_path,
            recording_start_time=start_time_value if recording_window else None,
            recording_end_time=end_time_value if recording_window else None,
        )

        st.success(
            f"Saved archive for camera {selected_camera.camera_number} on {archive.recorded_date.isoformat()}."
        )
        st.write(f"Uploaded file: `{original_filename}`")
        st.write(f"Cloudinary URL: `{archive.video_path}`")
    finally:
        db.close()


def _generate_caption_index_for_archive(
    db,
    archive,
    camera_id: int,
    timestamp: str | None,
    progress_callback=None,
):
    caption_context = generate_bilingual_captions(
        archive.video_path,
        timestamp,
        base_start_time=archive.recording_start_time,
        progress_callback=progress_callback,
    )
    return crud.create_caption_job(
        db=db,
        camera_id=camera_id,
        archive_id=archive.id,
        recorded_date=archive.recorded_date,
        requested_start_time=caption_context.requested_start_time,
        requested_end_time=caption_context.requested_end_time,
        search_sentence=None,
        matched_results_only=False,
        segments=caption_context.segments,
        progress_callback=progress_callback,
    )


def render_watchlist_processing() -> None:
    st.subheader("Automatic Watchlist")
    st.caption("Manage Hindi/English names, keywords, or phrases and auto-save matching caption clips.")

    db = SessionLocal()
    try:
        with st.form("watchlist_item_form", clear_on_submit=True):
            term = st.text_input("Watchlist Term", placeholder="Person A or sensitive phrase")
            item_type = st.selectbox("Type", options=["keyword", "person", "sensitive_phrase", "incident"])
            language = st.selectbox("Language", options=["auto", "hi", "en"])
            uploaded_reference = None
            if item_type == "person":
                uploaded_reference = st.file_uploader(
                    "Reference Photo",
                    type=["jpg", "jpeg", "png"],
                    help="Used for face matching when automatic watchlist scans run.",
                )
            submitted = st.form_submit_button("Add Or Update Watchlist Item")

        if submitted:
            reference_image_path = None
            face_embedding = None
            try:
                if not term.strip():
                    raise ValueError("Watchlist term cannot be empty.")
                if item_type == "person" and uploaded_reference is not None:
                    image_bytes = uploaded_reference.read()
                    reference_image_path = save_watchlist_reference_image(
                        uploaded_reference.name,
                        image_bytes,
                        term,
                    )
                    try:
                        face_embedding = generate_face_embedding(reference_image_path)
                    except Exception:
                        Path(reference_image_path).unlink(missing_ok=True)
                        raise

                crud.create_or_update_watchlist_item(
                    db,
                    term=term,
                    item_type=item_type,
                    language=None if language == "auto" else language,
                    reference_image_path=reference_image_path,
                    face_embedding=face_embedding,
                    is_active=True,
                )
                st.session_state["pending_toast"] = "Watchlist item saved"
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save watchlist item: {exc}")

        watchlist_items = crud.list_watchlist_items(db)
        if not watchlist_items:
            st.info("No watchlist items yet.")
            return

        st.dataframe(
            [
                {
                    "id": item.id,
                    "term": item.term,
                    "type": item.item_type,
                    "language": item.language or "auto",
                    "photo": "yes" if item.reference_image_path else "",
                    "active": item.is_active,
                }
                for item in watchlist_items
            ],
            use_container_width=True,
        )

        st.markdown("**Toggle Items**")
        for item in watchlist_items:
            label = "Deactivate" if item.is_active else "Activate"
            col_a, col_b = st.columns([3, 1])
            with col_a:
                if item.reference_image_path and Path(item.reference_image_path).exists():
                    st.image(item.reference_image_path, caption=f"{item.term} reference", width=80)
            with col_b:
                if st.button(f"{label}: {item.term}", key=f"toggle_watchlist_{item.id}"):
                    crud.update_watchlist_item_status(db, item.id, not item.is_active)
                    st.rerun()

        st.markdown("**Run Automatic Scan**")
        cameras = crud.list_cameras(db)
        camera_options = {"All cameras": None}
        camera_options.update({f"{camera.camera_number} - {camera.name}": camera for camera in cameras})

        with st.form("watchlist_scan_form"):
            selected_label = st.selectbox("Camera", options=list(camera_options.keys()), key="watchlist_scan_camera")
            use_date_filter = st.checkbox("Scan one date only", value=True)
            recorded_date = None
            if use_date_filter:
                recorded_date = st.date_input("Archive Date", value=date.today(), key="watchlist_scan_date")
            scan_submitted = st.form_submit_button("Scan Caption Indexes And Save Clips")

        if not scan_submitted:
            return

        selected_camera = camera_options[selected_label]
        caption_jobs = crud.list_latest_caption_jobs(
            db,
            camera_id=selected_camera.id if selected_camera is not None else None,
            recorded_date=recorded_date,
        )
        face_watchlist_items = crud.list_watchlist_face_items(db, active_only=True)
        archives = crud.list_archives(
            db,
            camera_id=selected_camera.id if selected_camera is not None else None,
            recorded_date=recorded_date,
        )
        if not caption_jobs and not (face_watchlist_items and archives):
            st.info("No caption indexes or face-watchlist archives found for that selection.")
            return

        progress_bar = st.progress(0)
        progress_status = st.empty()
        report_progress = _make_progress_reporter(progress_bar, progress_status)

        saved_records = []
        errors = []
        total_jobs = max(1, len(caption_jobs))
        for index, caption_job in enumerate(caption_jobs, start=1):
            report_progress(int((index - 1) / total_jobs * 90), "Scanning caption index")
            job_saved_records, job_errors = _auto_save_watchlist_clips_for_caption_job(
                db,
                caption_job,
                progress_callback=report_progress,
            )
            saved_records.extend(job_saved_records)
            errors.extend(job_errors)

        face_saved_records, face_errors = _auto_save_watchlist_face_clips_for_archives(
            db,
            archives,
            progress_callback=report_progress,
        )
        saved_records.extend(face_saved_records)
        errors.extend(face_errors)

        report_progress(100, "Watchlist scan complete")
        st.success(f"Watchlist scan finished. {len(saved_records)} matching clip record(s) found or saved.")
        if errors:
            st.warning(f"{len(errors)} matching clip(s) could not be saved.")
            for error in errors:
                st.caption(error)
    finally:
        db.close()


def render_caption_generation() -> None:
    st.subheader("Generate Caption Index")
    render_endpoint_reference(
        "POST",
        "/captions/search",
        "Generate and store searchable captions for a selected camera video or clip.",
    )

    db = SessionLocal()
    try:
        cameras = crud.list_cameras(db)
        camera_options = {f"{camera.camera_number} - {camera.name}": camera for camera in cameras}

        if not camera_options:
            st.info("No cameras exist yet. Add a camera and upload a video first.")
            return

        with st.form("caption_generation_form"):
            selected_label = st.selectbox("Camera For Caption Generation", options=list(camera_options.keys()))
            recorded_date = st.date_input("Archive Date", value=date.today(), key="caption_generation_date")
            use_timestamp = st.checkbox("Generate only for a timestamp range", value=False)

            timestamp = None
            if use_timestamp:
                start_time_value = st.time_input("Timestamp Start", value=time(0, 0, 0), key="caption_generation_start")
                end_time_value = st.time_input("Timestamp End", value=time(0, 5, 0), key="caption_generation_end")
                timestamp = f"{start_time_value.strftime('%H:%M:%S')}-{end_time_value.strftime('%H:%M:%S')}"
            submitted = st.form_submit_button("Generate Captions For Archive")

        if not submitted:
            return

        if use_timestamp and start_time_value >= end_time_value:
            st.error("Timestamp end must be later than timestamp start.")
            return

        selected_camera = camera_options[selected_label]

        try:
            payload = CaptionSearchRequest(
                camera_id=selected_camera.id,
                recorded_date=recorded_date,
                timestamp=timestamp,
                search_sentence=None,
            )
        except Exception as exc:
            st.error(str(exc))
            return

        archive = crud.get_archive_for_date(db, payload.camera_id, payload.recorded_date)
        if archive is None:
            st.error(
                f"No archive found for camera {selected_camera.camera_number} on {payload.recorded_date.isoformat()}."
            )
            return

        requested_start_time, requested_end_time = parse_timestamp_range(payload.timestamp)
        if requested_start_time and requested_end_time:
            if (
                requested_start_time < archive.recording_start_time
                or requested_end_time > archive.recording_end_time
            ):
                st.error(
                    "Requested timestamp is outside the stored recording window: "
                    f"{archive.recording_start_time} to {archive.recording_end_time}."
                )
                return

        st.caption("Generating caption index")
        progress_bar = st.progress(0)
        progress_status = st.empty()
        report_progress = _make_progress_reporter(progress_bar, progress_status)
        try:
            caption_job = _generate_caption_index_for_archive(
                db=db,
                archive=archive,
                camera_id=selected_camera.id,
                timestamp=payload.timestamp,
                progress_callback=report_progress,
            )
            watchlist_clip_records, watchlist_errors = _auto_save_watchlist_clips_for_caption_job(
                db,
                caption_job,
                progress_callback=report_progress,
            )
            report_progress(100, "Caption index complete")
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except RuntimeError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Caption generation failed: {exc}")
            return

        st.success(
            f"Generated {len(caption_job.segments)} caption segment(s) for camera {selected_camera.camera_number}."
        )
        if watchlist_clip_records:
            st.success(f"Auto-saved {len(watchlist_clip_records)} watchlist clip(s).")
        if watchlist_errors:
            st.warning(f"{len(watchlist_errors)} watchlist clip(s) could not be saved.")
            for error in watchlist_errors:
                st.caption(error)
        st.write(f"Video source: `{archive.video_path}`")
        if payload.timestamp:
            st.write(f"Timestamp range: `{payload.timestamp}`")
        if caption_job.segments:
            st.dataframe(
                [
                    {
                        "segment_index": segment.segment_index,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "original_or_hindi_caption": segment.hindi_caption,
                        "translated_or_english_caption": segment.english_caption,
                    }
                    for segment in caption_job.segments
                ],
                use_container_width=True,
            )
    finally:
        db.close()


def render_caption_search() -> None:
    st.subheader("Search Stored Captions")
    render_endpoint_reference(
        "POST",
        "/captions/search",
        "Search stored caption text by keyword, tag, or phrase and return the matching clip.",
    )

    db = SessionLocal()
    try:
        cameras = crud.list_cameras(db)
        camera_options = {"All cameras": None}
        camera_options.update({f"{camera.camera_number} - {camera.name}": camera for camera in cameras})

        with st.form("caption_search_form"):
            selected_label = st.selectbox("Camera Filter", options=list(camera_options.keys()))
            use_date_filter = st.checkbox("Filter by date", value=False)
            recorded_date = None
            if use_date_filter:
                recorded_date = st.date_input("Day Filter", value=date.today(), key="caption_search_date")

            use_timestamp_filter = st.checkbox("Filter by timestamp range", value=False)
            timestamp = None
            if use_timestamp_filter:
                start_time_value = st.time_input("Search Start Time", value=time(0, 0, 0), key="caption_search_start")
                end_time_value = st.time_input("Search End Time", value=time(0, 5, 0), key="caption_search_end")
                timestamp = f"{start_time_value.strftime('%H:%M:%S')}-{end_time_value.strftime('%H:%M:%S')}"

            search_sentence = st.text_input(
                "Search Tags, Keywords, or Phrases",
                placeholder="shouting near gate",
            )
            submitted = st.form_submit_button("Search Matching Clips")

        if submitted:
            st.session_state.pop("caption_search_results", None)
            st.session_state.pop("caption_search_sentiment", None)

            if not search_sentence.strip():
                st.error("Enter a keyword, tag, or phrase to search.")
                return

            if use_timestamp_filter and start_time_value >= end_time_value:
                st.error("Timestamp end must be later than timestamp start.")
                return

            selected_camera = camera_options[selected_label]
            camera_id = selected_camera.id if selected_camera is not None else None

            range_start_time, range_end_time = parse_timestamp_range(timestamp)
            try:
                caption_sentiment_pos_neg_neutral = tag_sentiment(search_sentence)
            except Exception as exc:
                st.error(f"Sentiment tagging failed: {exc}")
                return

            if camera_id is not None and recorded_date is not None:
                archive = crud.get_archive_for_date(db, camera_id, recorded_date)
                if archive is not None and crud.get_latest_caption_job_for_archive(db, archive.id) is None:
                    st.caption("No stored captions were found for this archive. Generating them now.")
                    progress_bar = st.progress(0)
                    progress_status = st.empty()
                    report_progress = _make_progress_reporter(progress_bar, progress_status)
                    try:
                        generated_job = _generate_caption_index_for_archive(
                            db=db,
                            archive=archive,
                            camera_id=camera_id,
                            timestamp=timestamp,
                            progress_callback=report_progress,
                        )
                        _auto_save_watchlist_clips_for_caption_job(
                            db,
                            generated_job,
                            progress_callback=report_progress,
                        )
                        report_progress(100, "Caption index complete")
                    except FileNotFoundError as exc:
                        st.error(str(exc))
                        return
                    except RuntimeError as exc:
                        st.error(str(exc))
                        return
                    except Exception as exc:
                        st.error(f"Caption generation failed: {exc}")
                        return

            matches = crud.search_caption_segments(
                db=db,
                query_text=search_sentence,
                camera_id=camera_id,
                recorded_date=recorded_date,
                range_start_time=range_start_time,
                range_end_time=range_end_time,
                limit=10,
            )

            if not matches:
                st.info("No matching captions were found in the stored caption index.")
                nearest_matches = crud.search_caption_segments(
                    db=db,
                    query_text=search_sentence,
                    camera_id=camera_id,
                    recorded_date=recorded_date,
                    range_start_time=range_start_time,
                    range_end_time=range_end_time,
                    confidence_threshold=0.0,
                    limit=5,
                )
                if nearest_matches:
                    best_confidence = getattr(nearest_matches[0], "match_confidence", 0.0)
                    st.caption(
                        f"Best semantic score was {best_confidence * 100:.0f}%, "
                        f"below the required {SEMANTIC_CAPTION_MATCH_THRESHOLD * 100:.0f}%."
                    )
                    st.dataframe(
                        [
                            {
                                "confidence": f"{getattr(candidate, 'match_confidence', 0.0) * 100:.0f}%",
                                "start_time": candidate.start_time,
                                "end_time": candidate.end_time,
                                "hindi_caption": candidate.hindi_caption,
                                "english_caption": candidate.english_caption,
                            }
                            for candidate in nearest_matches
                        ],
                        use_container_width=True,
                    )
                if camera_id is None or recorded_date is None:
                    st.caption("Tip: generate captions for archives first, or narrow the search with camera/date.")
                return

            results = []
            for match in matches:
                caption_job = match.caption_job
                archive = caption_job.archive
                camera = caption_job.camera
                clip_path = None

                if match.start_time is not None and match.end_time is not None:
                    try:
                        clip_path = extract_video_clip(
                            video_path=archive.video_path,
                            clip_start_time=match.start_time,
                            clip_end_time=match.end_time,
                            clip_prefix=f"camera_{camera.camera_number}_{caption_job.recorded_date.isoformat()}_{match.id}",
                            source_start_time=archive.recording_start_time,
                        )
                        crud.create_or_get_clip_record(
                            db,
                            clip_path=clip_path,
                            source_type="caption",
                            source_reference=str(match.id),
                            camera_id=getattr(camera, "id", None),
                            archive_id=getattr(archive, "id", None),
                            recorded_date=caption_job.recorded_date,
                            start_time=match.start_time,
                            end_time=match.end_time,
                            matched_text=search_sentence.strip(),
                            sentiment=caption_sentiment_pos_neg_neutral,
                            context_text=(
                                f"{build_searchable_caption_text(match.hindi_caption, match.english_caption)}\n"
                                f"Semantic match confidence: {getattr(match, 'match_confidence', 0.0):.2f}"
                            ),
                        )
                    except Exception as exc:
                        st.error(f"Could not extract clip for this match: {exc}")

                results.append(
                    {
                        "camera": _camera_to_result_dict(camera),
                        "archive": _archive_to_result_dict(archive),
                        "recorded_date": caption_job.recorded_date,
                        "start_time": match.start_time,
                        "end_time": match.end_time,
                        "hindi_caption": match.hindi_caption,
                        "english_caption": match.english_caption,
                        "confidence": getattr(match, "match_confidence", 0.0),
                        "clip_path": clip_path,
                        "source_reference": str(match.id),
                        "action_key": f"caption_{match.id}",
                    }
                )

            st.session_state["caption_search_results"] = results
            st.session_state["caption_search_sentiment"] = caption_sentiment_pos_neg_neutral

        _render_caption_results(
            db,
            st.session_state.get("caption_search_results", []),
            st.session_state.get("caption_search_sentiment"),
        )

    finally:
        db.close()


def render_camera_table() -> None:
    st.subheader("Current Cameras")

    db = SessionLocal()
    try:
        cameras = crud.list_cameras(db)
    finally:
        db.close()

    if not cameras:
        st.caption("No cameras added yet.")
        return

    st.dataframe(
        [
            {
                "db_id": camera.id,
                "camera_number": camera.camera_number,
                "name": camera.name,
                "location": camera.location,
                "stream_url": camera.stream_url,
            }
            for camera in cameras
        ],
        use_container_width=True,
    )


def render_face_search() -> None:
    st.subheader("Search Faces")
    st.caption("Upload a face image to search for matches in stored video detections.")

    db = SessionLocal()
    try:
        cameras = crud.list_cameras(db)
        camera_options = {"All cameras": None}
        camera_options.update({f"{camera.camera_number} - {camera.name}": camera for camera in cameras})

        with st.form("face_search_form"):
            selected_label = st.selectbox("Camera Filter", options=list(camera_options.keys()))
            use_date_filter = st.checkbox("Filter by date", value=True)
            recorded_date = date.today()
            if use_date_filter:
                recorded_date = st.date_input("Day Filter", value=date.today(), key="face_search_date")

            uploaded_face = st.file_uploader(
                "Upload Face Image",
                type=["jpg", "jpeg", "png"],
            )
            threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
            submitted = st.form_submit_button("Search Matching Faces")

        if submitted:
            st.session_state.pop("face_search_results", None)
            st.session_state.pop("face_search_errors", None)

            if uploaded_face is None:
                st.error("Please upload a face image.")
                return

            selected_camera = camera_options[selected_label]

            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_face.read())
                face_path = tmp_file.name

            try:
                embedding = generate_face_embedding(face_path)
            except Exception as exc:
                st.error(f"Face processing failed: {exc}")
                return
            finally:
                Path(face_path).unlink(missing_ok=True)

            if selected_camera is None:
                target_archives = []
                for camera in cameras:
                    archive = crud.get_archive_for_date(db, camera.id, recorded_date)
                    if archive is not None:
                        target_archives.append(archive)
            else:
                archive = crud.get_archive_for_date(db, selected_camera.id, recorded_date)
                target_archives = [archive] if archive is not None else []

            if not target_archives:
                if selected_camera is None:
                    st.error(f"No archives found on {recorded_date.isoformat()} for any camera.")
                else:
                    st.error(
                        f"No archive found for camera {selected_camera.camera_number} on {recorded_date.isoformat()}."
                    )
                return

            matches = []
            errors = []
            with st.spinner(f"Searching {len(target_archives)} archive(s) for matching faces..."):
                for target_archive in target_archives:
                    try:
                        archive_matches = detect_matching_faces_in_video(
                            video_path=target_archive.video_path,
                            query_embedding=embedding,
                            archive=target_archive,
                            threshold=threshold,
                            sample_interval_seconds=2,
                        )
                        matches.extend(archive_matches)
                    except Exception as exc:
                        camera_number = (
                            target_archive.camera.camera_number
                            if getattr(target_archive, "camera", None) is not None
                            else target_archive.camera_id
                        )
                        errors.append(f"Camera {camera_number}: {exc}")

            if not matches:
                if errors:
                    st.error("Face detection failed for all matching archives.")
                    for error in errors:
                        st.caption(error)
                    return
                st.info("No matching faces were found in the selected camera archives.")
                return

            matches.sort(
                key=lambda match: (
                    getattr(match.get("archive"), "recorded_date", recorded_date),
                    getattr(match.get("camera"), "camera_number", 0),
                    match.get("start_time") or match.get("timestamp"),
                )
            )

            results = []
            for match in matches:
                match_archive = match.get("archive")
                camera = match.get("camera") or getattr(match_archive, "camera", None) or selected_camera
                start_time = match.get("start_time") or match.get("timestamp")
                end_time = match.get("end_time") or match.get("timestamp")
                clip_path = None

                try:
                    clip_path = extract_video_clip(
                        video_path=match_archive.video_path,
                        clip_start_time=start_time,
                        clip_end_time=end_time,
                        clip_prefix=f"face_camera_{camera.camera_number}_{match_archive.recorded_date.isoformat()}_{int(match['confidence']*100)}",
                        source_start_time=match_archive.recording_start_time,
                    )
                    action_suffix = Path(get_stored_file_name(clip_path)).stem
                except Exception as exc:
                    st.error(f"Could not extract clip for this match: {exc}")
                    action_suffix = (
                        f"face_camera_{camera.camera_number}_{match_archive.recorded_date.isoformat()}_"
                        f"{start_time}_{end_time}_{int(match['confidence']*100)}"
                    )

                results.append(
                    {
                        "camera": _camera_to_result_dict(camera),
                        "archive": _archive_to_result_dict(match_archive),
                        "start_time": start_time,
                        "end_time": end_time,
                        "confidence": match["confidence"],
                        "image_path": match.get("image_path"),
                        "clip_path": clip_path,
                        "source_reference": action_suffix,
                        "action_key": f"face_{action_suffix}",
                    }
                )

            st.session_state["face_search_results"] = results
            st.session_state["face_search_errors"] = errors

        _render_face_results(
            db,
            st.session_state.get("face_search_results", []),
            st.session_state.get("face_search_errors", []),
        )

    finally:
        db.close()


def render_saved_clips() -> None:
    st.subheader("Saved Clips")
    st.caption("Manage saved clips and mark them reviewed, important, exported, or download them later.")

    db = SessionLocal()
    try:
        saved_clips = crud.list_clip_records(db)

        if not saved_clips:
            st.info("No clips have been saved yet.")
            return

        # Calculate total storage
        total_bytes = 0
        for clip_record in saved_clips:
            if not is_stored_file_available(clip_record.clip_path):
                continue
            if not clip_record.clip_path.startswith(("http://", "https://")):
                total_bytes += os.path.getsize(clip_record.clip_path)

        total_storage = _format_bytes(total_bytes)
        total_capacity_tb = 20
        total_capacity_bytes = total_capacity_tb * 1024**4
        fraction = min(1.0, total_bytes / total_capacity_bytes) if total_capacity_bytes > 0 else 0

        st.subheader("Storage Usage")
        st.write(f"Local saved clips storage: {total_storage} out of {total_capacity_tb} TB")
        st.progress(fraction)
        st.caption("Cloudinary-hosted clips are stored remotely and are not counted in local disk usage.")

        for clip_record in saved_clips:
            camera = clip_record.camera
            archive = clip_record.archive

            camera_label = (
                f"Camera {camera.camera_number}"
                if camera is not None
                else f"Camera ID {clip_record.camera_id}"
            )
            date_label = (
                clip_record.recorded_date.isoformat()
                if clip_record.recorded_date is not None
                else "Unknown date"
            )
            st.markdown(
                f"**{camera_label}** | `{date_label}` | "
                f"`{clip_record.start_time}` to `{clip_record.end_time}` | Source: `{clip_record.source_type}`"
            )
            if clip_record.context_text:
                st.write(clip_record.context_text)

            if is_stored_file_available(clip_record.clip_path):
                st.video(clip_record.clip_path)
            else:
                st.error(f"Saved clip file is missing: {clip_record.clip_path}")

            _render_saved_clip_actions(
                db,
                clip_record,
                action_key=f"saved_{clip_record.id}",
            )

            if archive is not None and archive.video_path:
                st.caption(f"Archive source: `{archive.video_path}`")

            st.divider()
    finally:
        db.close()


st.title("BAI MVP Camera Frontend")
st.caption("Streamlit frontend for separate camera registration, video upload, and caption search workflows.")
_show_pending_toast()

register_tab, upload_tab, caption_tab, watchlist_tab, face_tab, saved_tab = st.tabs(
    ["Register Camera", "Upload Camera Video", "Search Captions", "Watchlist", "Face Search", "Saved Clips"]
)

with register_tab:
    render_camera_form()

with upload_tab:
    render_archive_form()

with caption_tab:
    render_caption_generation()
    st.divider()
    render_caption_search()

with watchlist_tab:
    render_watchlist_processing()

with face_tab:
    render_face_search()

with saved_tab:
    render_saved_clips()

st.divider()
render_camera_table()
