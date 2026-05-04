from __future__ import annotations

from datetime import date, datetime, time
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, Time, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base
from vector_types import EmbeddingVector


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_number: Mapped[int] = mapped_column(Integer, unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    stream_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    archives: Mapped[list["CameraArchive"]] = relationship(
        back_populates="camera",
        cascade="all, delete-orphan",
    )
    caption_jobs: Mapped[list["CaptionJob"]] = relationship(
        back_populates="camera",
        cascade="all, delete-orphan",
    )
    face_detections: Mapped[list["FaceDetection"]] = relationship(
        back_populates="camera",
        cascade="all, delete-orphan",
    )


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    __table_args__ = (
        UniqueConstraint("term", name="uq_watchlist_item_term"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    term: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    item_type: Mapped[str] = mapped_column(String(50), default="keyword", nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    reference_image_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    face_embedding: Mapped[Optional[list[float]]] = mapped_column(EmbeddingVector(512), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class CameraArchive(Base):
    __tablename__ = "camera_archives"
    __table_args__ = (
        UniqueConstraint("camera_id", "recorded_date", name="uq_camera_archive_recorded_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), nullable=False, index=True)
    recorded_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    video_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    recording_start_time: Mapped[time] = mapped_column(Time, default=time(0, 0, 0), nullable=False)
    recording_end_time: Mapped[time] = mapped_column(Time, default=time(23, 59, 59), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    camera: Mapped["Camera"] = relationship(back_populates="archives")
    caption_jobs: Mapped[list["CaptionJob"]] = relationship(
        back_populates="archive",
        cascade="all, delete-orphan",
    )
    face_detections: Mapped[list["FaceDetection"]] = relationship(
        back_populates="archive",
        cascade="all, delete-orphan",
    )


class CaptionJob(Base):
    __tablename__ = "caption_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), nullable=False, index=True)
    archive_id: Mapped[int] = mapped_column(ForeignKey("camera_archives.id"), nullable=False, index=True)
    recorded_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    requested_start_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    requested_end_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    search_sentence: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    matched_results_only: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    camera: Mapped["Camera"] = relationship(back_populates="caption_jobs")
    archive: Mapped["CameraArchive"] = relationship(back_populates="caption_jobs")
    segments: Mapped[list["CaptionSegment"]] = relationship(
        back_populates="caption_job",
        cascade="all, delete-orphan",
    )


class CaptionSegment(Base):
    __tablename__ = "caption_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    caption_job_id: Mapped[int] = mapped_column(ForeignKey("caption_jobs.id"), nullable=False, index=True)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    end_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    hindi_caption: Mapped[str] = mapped_column(Text, nullable=False)
    english_caption: Mapped[str] = mapped_column(Text, nullable=False)
    searchable_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[list[float]]] = mapped_column(EmbeddingVector(384), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    caption_job: Mapped["CaptionJob"] = relationship(back_populates="segments")


class Face(Base):
    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingVector(512), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class FaceDetection(Base):
    __tablename__ = "face_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), nullable=False, index=True)
    archive_id: Mapped[int] = mapped_column(ForeignKey("camera_archives.id"), nullable=False, index=True)
    recorded_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    face_embedding: Mapped[list[float]] = mapped_column(EmbeddingVector(512), nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    camera: Mapped["Camera"] = relationship()
    archive: Mapped["CameraArchive"] = relationship()


class ClipRecord(Base):
    __tablename__ = "clip_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    clip_path: Mapped[str] = mapped_column(String(1024), unique=True, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_reference: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    camera_id: Mapped[Optional[int]] = mapped_column(ForeignKey("cameras.id"), nullable=True, index=True)
    archive_id: Mapped[Optional[int]] = mapped_column(ForeignKey("camera_archives.id"), nullable=True, index=True)
    recorded_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    start_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    end_time: Mapped[Optional[time]] = mapped_column(Time, nullable=True)
    watchlist_item_id: Mapped[Optional[int]] = mapped_column(ForeignKey("watchlist_items.id"), nullable=True, index=True)
    matched_text: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    context_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_important: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    export_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    exported_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    camera: Mapped[Optional["Camera"]] = relationship()
    archive: Mapped[Optional["CameraArchive"]] = relationship()
    watchlist_item: Mapped[Optional["WatchlistItem"]] = relationship()
