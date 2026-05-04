from __future__ import annotations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from io import BytesIO
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable
from urllib.parse import urlparse
from urllib.request import urlopen

from dotenv import load_dotenv
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

from faster_whisper import WhisperModel
import numpy as np

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover - dependency error path
    GoogleTranslator = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - dependency error path
    SentenceTransformer = None
    
try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover - optional dependency path
    FaceAnalysis = None

try:
    import cloudinary
    import cloudinary.uploader
except ImportError:  # pragma: no cover - optional dependency path
    cloudinary = None

CLIPS_DIR = Path("tmp_clips")
CLOUDINARY_CACHE_DIR = CLIPS_DIR / "cloudinary_cache"
WATCHLIST_REFERENCE_DIR = Path("watchlist_reference_images")
SEMANTIC_CAPTION_MATCH_THRESHOLD = 0.4

CLIPS_DIR.mkdir(exist_ok=True)
CLOUDINARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
WATCHLIST_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


@dataclass
class CaptionContext:
    requested_start_time: time | None
    requested_end_time: time | None
    segments: list[dict]


def is_remote_resource(path_or_url: str | None) -> bool:
    if not path_or_url:
        return False
    parsed = urlparse(path_or_url)
    return parsed.scheme in {"http", "https"}


def _configure_cloudinary() -> None:
    if cloudinary is None:
        raise RuntimeError(
            "Cloudinary uploads require the 'cloudinary' package. Run: pip install -r requirements.txt"
        )

    cloud_name = (os.getenv("CLOUDINARY_CLOUD_NAME") or "").strip()
    api_key = (os.getenv("CLOUDINARY_API_KEY") or "").strip()
    api_secret = (os.getenv("CLOUDINARY_API_SECRET") or "").strip()
    cloudinary_url = (os.getenv("CLOUDINARY_URL") or "").strip()

    if cloud_name and api_key and api_secret:
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
        )
        return

    if cloudinary_url:
        try:
            cloudinary.config()._load_from_url(cloudinary_url)
        except Exception as exc:
            raise RuntimeError("Cloudinary URL is invalid. Expected cloudinary://api_key:api_secret@cloud_name") from exc
        cloudinary.config(secure=True)
        return

    raise RuntimeError(
        "Cloudinary is not configured. Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, "
        "and CLOUDINARY_API_SECRET, or set CLOUDINARY_URL."
    )


def _safe_public_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_") or "video"


def _cloudinary_upload(
    file_source,
    *,
    folder: str,
    public_id: str,
    resource_type: str = "video",
) -> str:
    _configure_cloudinary()
    result = cloudinary.uploader.upload(
        file_source,
        folder=folder,
        public_id=public_id,
        resource_type=resource_type,
        overwrite=True,
        use_filename=False,
        unique_filename=False,
    )
    secure_url = result.get("secure_url") or result.get("url")
    if not secure_url:
        raise RuntimeError("Cloudinary upload succeeded but no URL was returned.")
    return secure_url


def _local_path_for_processing(video_path: str) -> str:
    if not is_remote_resource(video_path):
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Stored video file was not found at {video_path}")
        return video_path

    parsed = urlparse(video_path)
    suffix = Path(parsed.path).suffix or ".mp4"
    cache_name = hashlib.sha256(video_path.encode("utf-8")).hexdigest() + suffix
    cache_path = CLOUDINARY_CACHE_DIR / cache_name
    if not cache_path.exists():
        with urlopen(video_path) as response, cache_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
    return str(cache_path.resolve())


def is_stored_file_available(path_or_url: str | None) -> bool:
    if not path_or_url:
        return False
    if is_remote_resource(path_or_url):
        return True
    return Path(path_or_url).exists()


def get_stored_file_name(path_or_url: str) -> str:
    parsed = urlparse(path_or_url)
    name = Path(parsed.path if is_remote_resource(path_or_url) else path_or_url).name
    return name or "clip.mp4"


def read_stored_file_bytes(path_or_url: str) -> bytes:
    if is_remote_resource(path_or_url):
        with urlopen(path_or_url) as response:
            return response.read()
    return Path(path_or_url).read_bytes()


def save_uploaded_video(camera_number: int, recorded_date: date, uploaded_file) -> tuple[str, str]:
    timestamp_label = datetime.utcnow().strftime("%H%M%S")
    upload_name = getattr(uploaded_file, "filename", None) or getattr(uploaded_file, "name", None)
    safe_name = Path(upload_name or "uploaded_video.mp4").name

    if hasattr(uploaded_file, "file"):
        payload = uploaded_file.file.read()
    elif hasattr(uploaded_file, "getbuffer"):
        payload = uploaded_file.getbuffer()
    elif hasattr(uploaded_file, "read"):
        payload = uploaded_file.read()
    else:
        raise RuntimeError("Unsupported uploaded file type.")

    folder = f"bai-mvp/uploads/camera_{camera_number:02d}/{recorded_date.isoformat()}"
    public_id = _safe_public_id(f"{timestamp_label}_{Path(safe_name).stem}")
    payload_file = BytesIO(payload)
    payload_file.name = safe_name
    cloudinary_url = _cloudinary_upload(
        payload_file,
        folder=folder,
        public_id=public_id,
        resource_type="video",
    )
    return cloudinary_url, safe_name


def parse_timestamp_range(timestamp: str | None) -> tuple[time | None, time | None]:
    if not timestamp:
        return None, None

    raw_start, raw_end = timestamp.split("-", maxsplit=1)
    start_time = time.fromisoformat(raw_start)
    end_time = time.fromisoformat(raw_end)
    return start_time, end_time


def clip_video_if_needed(
    video_path: str,
    timestamp: str | None,
    source_start_time: time | None = None,
) -> tuple[str, time | None, time | None]:
    source_video_path = _local_path_for_processing(video_path)

    start_time, end_time = parse_timestamp_range(timestamp)
    if start_time is None or end_time is None:
        return source_video_path, None, None

    start_seconds = _time_to_seconds(start_time)
    end_seconds = _time_to_seconds(end_time)
    if end_seconds <= start_seconds:
        raise ValueError("Clip end time must be later than clip start time.")

    source_offset_seconds = _time_to_seconds(source_start_time) if source_start_time is not None else 0
    relative_start = start_seconds - source_offset_seconds
    if relative_start < 0:
        raise ValueError(
            "Requested clip start time occurs before the stored video source start time."
        )

    duration_seconds = end_seconds - start_seconds
    ffmpeg_start = _format_ffmpeg_offset(relative_start)
    ffmpeg_duration = _format_ffmpeg_offset(duration_seconds)

    clip_name = (
        f"{Path(source_video_path).stem}_{start_time.strftime('%H%M%S')}_{end_time.strftime('%H%M%S')}.mp4"
    )
    clip_path = CLIPS_DIR / clip_name

    if not clip_path.exists():
        command = [
            "ffmpeg",
            "-y",
            "-i",
            source_video_path,
            "-ss",
            ffmpeg_start,
            "-t",
            ffmpeg_duration,
            "-c",
            "copy",
            str(clip_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg failed while clipping the video")

    return str(clip_path.resolve()), start_time, end_time


def _time_to_seconds(value: time) -> int:
    return value.hour * 3600 + value.minute * 60 + value.second


def _format_ffmpeg_offset(total_seconds: int) -> str:
    safe_seconds = max(0, total_seconds)
    hours = safe_seconds // 3600
    minutes = (safe_seconds % 3600) // 60
    seconds = safe_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _load_whisper_model() -> WhisperModel:
    return WhisperModel("small", device="cpu", compute_type="int8")


def _load_embedding_model() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError(
            "Semantic caption search requires the 'sentence-transformers' package. "
            "Run: pip install -r requirements.txt"
        )
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _load_face_model() -> FaceAnalysis:
    if FaceAnalysis is None:
        raise RuntimeError(
            "Face recognition requires the 'insightface' package. "
            "Run: pip install insightface"
        )
    app = FaceAnalysis(name="buffalo_s")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def build_searchable_caption_text(hindi_caption: str, english_caption: str) -> str:
    return f"{hindi_caption.strip()} {english_caption.strip()}".strip()


def watchlist_term_matches_text(term: str, text: str) -> bool:
    normalized_term = term.strip().casefold()
    normalized_text = text.strip().casefold()
    return bool(normalized_term and normalized_term in normalized_text)


def semantic_caption_match_confidence(phrase: str, caption_text: str) -> float:
    phrase_embedding = generate_text_embedding(phrase)
    caption_embedding = generate_text_embedding(caption_text)
    return cosine_similarity(caption_embedding, phrase_embedding)


def filter_caption_segments(
    segments: list[dict],
    search_sentence: str | None,
    threshold: float = SEMANTIC_CAPTION_MATCH_THRESHOLD,
) -> list[dict]:
    phrase = (search_sentence or "").strip()
    if not phrase:
        return segments

    query_embedding = generate_text_embedding(phrase)
    searchable_texts = [
        build_searchable_caption_text(segment["hindi_caption"], segment["english_caption"])
        for segment in segments
    ]
    caption_embeddings = generate_text_embeddings(searchable_texts)

    matched_segments = []
    for segment, caption_embedding in zip(segments, caption_embeddings):
        confidence = cosine_similarity(caption_embedding, query_embedding)
        if confidence >= threshold:
            matched_segment = dict(segment)
            matched_segment["match_confidence"] = confidence
            matched_segments.append(matched_segment)

    return matched_segments


def generate_text_embedding(text: str) -> list[float]:
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    model = _load_embedding_model()
    embedding = model.encode(cleaned_text, normalize_embeddings=True)
    return embedding.tolist()


def generate_text_embeddings(texts: list[str]) -> list[list[float]]:
    cleaned_texts = [text.strip() for text in texts]
    non_empty_texts = [text for text in cleaned_texts if text]
    if not non_empty_texts:
        return [[] for _ in cleaned_texts]

    model = _load_embedding_model()
    encoded_embeddings = model.encode(non_empty_texts, normalize_embeddings=True)

    embeddings_by_text = iter(encoded_embeddings.tolist())
    return [next(embeddings_by_text) if text else [] for text in cleaned_texts]


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return -1.0

    left_vector = np.array(left, dtype=float)
    right_vector = np.array(right, dtype=float)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if math.isclose(float(denominator), 0.0):
        return -1.0
    return float(np.dot(left_vector, right_vector) / denominator)


def _translate_text(text: str, source_language: str, target_language: str) -> str:
    cleaned_text = text.strip()
    if not cleaned_text:
        return cleaned_text

    normalized_source = (source_language or "auto").lower()
    normalized_target = (target_language or "").lower()
    if normalized_source == normalized_target:
        return cleaned_text

    if GoogleTranslator is None:
        raise RuntimeError(
            "Hindi translation requires the 'deep-translator' package. "
            "Run: pip install -r requirements.txt"
        )

    source_attempts = [normalized_source]
    if normalized_source != "auto":
        source_attempts.append("auto")

    for source_attempt in source_attempts:
        try:
            translator = GoogleTranslator(source=source_attempt, target=normalized_target)
            translated_text = translator.translate(cleaned_text)
        except Exception:
            continue

        if translated_text and translated_text.strip():
            return translated_text

    return cleaned_text


def _seconds_to_time(seconds: float | int | None) -> time:
    if seconds is None:
        return time(0, 0, 0)
    try:
        whole_seconds = max(0, int(seconds))
        # Cap at 23:59:59 to stay within a single day
        whole_seconds = min(whole_seconds, 86399)
        return (datetime.min + timedelta(seconds=whole_seconds)).time()
    except (ValueError, TypeError, OverflowError):
        return time(0, 0, 0)


def extract_video_clip(
    video_path: str,
    clip_start_time: time,
    clip_end_time: time,
    clip_prefix: str,
    source_start_time: time | None = None,
) -> str:
    source_video_path = _local_path_for_processing(video_path)

    buffer_seconds = 5
    start_seconds = max(0, _time_to_seconds(clip_start_time) - buffer_seconds)
    end_seconds = _time_to_seconds(clip_end_time) + buffer_seconds

    if end_seconds <= start_seconds:
        raise ValueError("Buffered clip end time must be later than buffered clip start time.")

    source_offset_seconds = _time_to_seconds(source_start_time) if source_start_time is not None else 0
    relative_start = max(0, start_seconds - source_offset_seconds)

    duration_seconds = end_seconds - start_seconds
    ffmpeg_start = _format_ffmpeg_offset(relative_start)
    ffmpeg_duration = _format_ffmpeg_offset(duration_seconds)

    buffered_start_time = _seconds_to_time(start_seconds)
    buffered_end_time = _seconds_to_time(end_seconds)
    clip_name = (
        f"{clip_prefix}_{buffered_start_time.strftime('%H%M%S')}_{buffered_end_time.strftime('%H%M%S')}.mp4"
    )
    clip_path = CLIPS_DIR / clip_name

    if not clip_path.exists():
        command = [
            "ffmpeg",
            "-y",
            "-i",
            source_video_path,
            "-ss",
            ffmpeg_start,
            "-t",
            ffmpeg_duration,
            "-c",
            "copy",
            str(clip_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "FFmpeg failed while extracting the clip")

    return str(clip_path.resolve())


def export_clip_for_later_use(clip_path: str, export_name: str | None = None) -> str:
    if not is_stored_file_available(clip_path):
        raise FileNotFoundError(f"Clip file was not found at {clip_path}")

    safe_name = Path(export_name).name if export_name else get_stored_file_name(clip_path)
    public_id = _safe_public_id(Path(safe_name).stem)
    return _cloudinary_upload(
        clip_path,
        folder="bai-mvp/exported_clips",
        public_id=public_id,
        resource_type="video",
    )


def save_clip_from_archive(
    video_path: str,
    clip_start_time: time,
    clip_end_time: time,
    clip_prefix: str,
    source_start_time: time | None = None,
    saved_name: str | None = None,
) -> str:
    extracted_clip_path = extract_video_clip(
        video_path=video_path,
        clip_start_time=clip_start_time,
        clip_end_time=clip_end_time,
        clip_prefix=clip_prefix,
        source_start_time=source_start_time,
    )

    source_path = Path(extracted_clip_path)
    destination_name = Path(saved_name).name if saved_name else source_path.name
    public_id = _safe_public_id(Path(destination_name).stem)
    return _cloudinary_upload(
        str(source_path),
        folder="bai-mvp/saved_clips",
        public_id=public_id,
        resource_type="video",
    )


def generate_bilingual_captions(
    video_path: str,
    timestamp: str | None,
    base_start_time: time | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> CaptionContext:
    def report(percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(100, percent)), message)

    report(2, "Preparing video")
    target_video_path, requested_start_time, requested_end_time = clip_video_if_needed(
        video_path,
        timestamp,
        source_start_time=base_start_time,
    )
    report(8, "Loading caption model")
    model = _load_whisper_model()

    report(15, "Transcribing audio")
    # Always transcribe first to get timing information
    original_segments_iter, original_info = model.transcribe(
        target_video_path,
        task="transcribe",
        vad_filter=True,
        beam_size=5,
    )
    original_segments = []
    duration = float(getattr(original_info, "duration", 0) or 0)
    for index, segment in enumerate(original_segments_iter, start=1):
        original_segments.append(segment)
        segment_end = float(getattr(segment, "end", 0) or 0)
        if duration > 0:
            transcribe_percent = 15 + int(min(segment_end / duration, 1.0) * 50)
        else:
            transcribe_percent = min(65, 15 + index * 5)
        report(transcribe_percent, f"Transcribing audio ({transcribe_percent}%)")
    report(65, "Transcription complete")

    detected_language = original_info.language if original_info else None
    normalized_language = (detected_language or "").lower()

    # Extract original language text with timing
    original_texts = []
    segment_timings = []  # Store timing from original transcription
    
    for segment in original_segments:
        original_texts.append(segment.text.strip())
        segment_timings.append({
            "start": getattr(segment, "start", 0),
            "end": getattr(segment, "end", 0),
        })

    # Determine English text
    if normalized_language == "en":
        english_texts = original_texts
    else:
        # Translate from detected language to English
        english_texts = []
        total_texts = max(1, len(original_texts))
        for index, text in enumerate(original_texts, start=1):
            english_texts.append(_translate_text(text, normalized_language, "en"))
            report(65 + int(index / total_texts * 15), "Translating captions to English")

    # Determine Hindi text
    if normalized_language == "hi":
        hindi_texts = original_texts
    else:
        # Translate from English to Hindi
        hindi_texts = []
        total_texts = max(1, len(english_texts))
        for index, text in enumerate(english_texts, start=1):
            hindi_texts.append(_translate_text(text, "en", "hi"))
            report(80 + int(index / total_texts * 12), "Translating captions to Hindi")

    # Build final segments using timing from original transcription
    report(94, "Building caption segments")
    generated_segments: list[dict] = []
    offset_time = requested_start_time or base_start_time
    offset_seconds = _time_to_seconds(offset_time) if offset_time is not None else 0

    for index in range(len(original_texts)):
        timing = segment_timings[index]
        
        # Extract and validate timing from Whisper
        try:
            start_seconds = float(timing["start"]) if timing["start"] is not None else 0
            end_seconds = float(timing["end"]) if timing["end"] is not None else 0
        except (ValueError, TypeError):
            start_seconds = 0
            end_seconds = 0
        
        start_time_value = _seconds_to_time(start_seconds + offset_seconds)
        end_time_value = _seconds_to_time(end_seconds + offset_seconds)

        generated_segments.append(
            {
                "start_time": start_time_value,
                "end_time": end_time_value,
                "hindi_caption": hindi_texts[index],
                "english_caption": english_texts[index],
            }
        )

    report(98, "Caption segments ready")
    return CaptionContext(
        requested_start_time=requested_start_time,
        requested_end_time=requested_end_time,
        segments=generated_segments,
    )


def generate_face_embedding(image_path: str) -> list[float]:
    if FaceAnalysis is None:
        raise RuntimeError("Face recognition requires insightface. Run: pip install insightface")

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Face image not found at {image_path}")

    if cv2 is not None:
        img = cv2.imread(image_path)
        if img is None and Image is not None:
            img = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    elif Image is not None:
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        raise RuntimeError("OpenCV or PIL is required to load face images.")

    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    app = _load_face_model()
    faces = app.get(img)
    if faces is None or len(faces) == 0:
        raise ValueError("No face detected in the image")

    # Use the first face
    embedding = faces[0].embedding.tolist()
    return embedding


def save_watchlist_reference_image(file_name: str, image_bytes: bytes, term: str) -> str:
    extension = Path(file_name or "reference.jpg").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png"}:
        extension = ".jpg"

    safe_term = "".join(char.lower() if char.isalnum() else "_" for char in term).strip("_") or "person"
    digest = hashlib.sha1(image_bytes).hexdigest()[:12]
    image_path = WATCHLIST_REFERENCE_DIR / f"{safe_term}_{digest}{extension}"
    image_path.write_bytes(image_bytes)
    return str(image_path.resolve())


def _save_face_crop_image(frame, bbox, timestamp_seconds: float) -> str:
    x1, y1, x2, y2 = [int(max(0, coord)) for coord in bbox[:4]]
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = frame[y1:y2, x1:x2]

    face_dir = CLIPS_DIR / "face_matches"
    face_dir.mkdir(parents=True, exist_ok=True)
    timestamp_label = _seconds_to_time(int(timestamp_seconds)).strftime("%H%M%S")
    face_path = face_dir / f"face_match_{timestamp_label}_{x1}_{y1}.jpg"

    if crop.size == 0:
        face_path = face_dir / f"face_match_{timestamp_label}_unknown.jpg"
        crop = frame

    cv2.imwrite(str(face_path), crop)
    return str(face_path.resolve())


def detect_matching_faces_in_video(
    video_path: str,
    query_embedding: list[float],
    archive=None,
    threshold: float = 0.6,
    sample_interval_seconds: int = 2,
) -> list[dict]:
    """
    Detects matching faces in a video using InsightFace (buffalo_s model).

    Args:
        video_path (str): Path to the video file.
        query_embedding (list[float]): The face embedding to search for.
        archive: Metadata object containing recording start time.
        threshold (float): Similarity threshold for a match.
        sample_interval_seconds (int): Interval in seconds to sample frames.

    Returns:
        list[dict]: A list of matches with timestamps and confidence scores.
    """
    if FaceAnalysis is None:
        raise RuntimeError("Face recognition requires insightface. Run: pip install insightface")

    if cv2 is None:
        raise RuntimeError("Video face detection requires OpenCV. Run: pip install opencv-python")

    source_video_path = _local_path_for_processing(video_path)

    # Load the InsightFace model (buffalo_s) for CPU
    app = _load_face_model()

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_seconds = frame_count / fps if frame_count > 0 else 0.0
    current_seconds = 0.0
    matches: list[dict] = []
    offset_seconds = _time_to_seconds(archive.recording_start_time) if archive and archive.recording_start_time else 0

    while current_seconds <= duration_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_seconds * 1000)
        success, frame = cap.read()
        if not success or frame is None:
            break

        # Detect faces in the current frame
        faces = app.get(frame) if frame is not None else []
        if not isinstance(faces, list):
            faces = []

        for face in faces:
            face_embedding = face.embedding.tolist()
            similarity = cosine_similarity(face_embedding, query_embedding)
            if similarity >= threshold:
                timestamp_seconds = offset_seconds + current_seconds
                timestamp = _seconds_to_time(int(timestamp_seconds))
                image_path = _save_face_crop_image(frame, face.bbox, timestamp_seconds)
                matches.append({
                    "camera": getattr(archive, "camera", None) if archive is not None else None,
                    "archive": archive,
                    "timestamp": timestamp,
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "image_path": image_path,
                    "face_embedding": face_embedding,
                    "confidence": float(similarity),
                })
                break  # Stop processing other faces in this frame

        current_seconds += sample_interval_seconds

    cap.release()
    return matches


def search_face_matches(
    db,
    query_embedding: list[float],
    camera_id: int | None = None,
    recorded_date: date | None = None,
    threshold: float = 0.6,
    limit: int = 10,
):
    from models import FaceDetection  # avoid circular import

    detections = db.query(FaceDetection).join(FaceDetection.camera).join(FaceDetection.archive)

    if camera_id is not None:
        detections = detections.filter(FaceDetection.camera_id == camera_id)

    if recorded_date is not None:
        detections = detections.filter(FaceDetection.recorded_date == recorded_date)

    candidates = detections.order_by(FaceDetection.recorded_date.desc(), FaceDetection.start_time.desc()).all()

    matches = []
    for detection in candidates:
        similarity = cosine_similarity(detection.face_embedding, query_embedding)
        if similarity >= threshold:
            matches.append((similarity, detection))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [detection for _, detection in matches[:limit]]


def tag_sentiment(text:str)->str:
    analyzer = SentimentIntensityAnalyzer()
    score=analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'
