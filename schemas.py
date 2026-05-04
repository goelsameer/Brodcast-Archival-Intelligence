from __future__ import annotations

from datetime import date, time

from pydantic import BaseModel, Field, field_validator, model_validator


class CameraCreate(BaseModel):
    camera_number: int = Field(..., ge=1, le=64)
    name: str = Field(..., min_length=1, max_length=255)
    location: str | None = Field(default=None, max_length=255)
    stream_url: str | None = Field(default=None, max_length=1024)


class CameraResponse(BaseModel):
    id: int
    camera_number: int
    name: str
    location: str | None
    stream_url: str | None

    model_config = {"from_attributes": True}


class CameraArchiveCreate(BaseModel):
    camera_id: int
    day_number: int = Field(..., ge=1, le=90)
    video_path: str = Field(..., min_length=1, max_length=1024)
    recording_start_time: time | None = None
    recording_end_time: time | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> "CameraArchiveCreate":
        start = self.recording_start_time
        end = self.recording_end_time

        if (start is None) != (end is None):
            raise ValueError("recording_start_time and recording_end_time must be provided together")

        if start is not None and end is not None and start >= end:
            raise ValueError("recording_end_time must be later than recording_start_time")

        return self


class CameraArchiveResponse(BaseModel):
    id: int
    camera_id: int
    recorded_date: date
    video_path: str
    recording_start_time: time
    recording_end_time: time

    model_config = {"from_attributes": True}


class CameraArchiveUploadResponse(CameraArchiveResponse):
    camera_number: int
    filename: str


class CameraArchiveFetchResponse(BaseModel):
    camera_id: int
    camera_number: int
    day_number: int
    recorded_date: date
    requested_start_time: time | None
    requested_end_time: time | None
    available_start_time: time
    available_end_time: time
    video_path: str


class ArchiveQueryParams(BaseModel):
    camera_id: int = Field(..., ge=1)
    previous_days_to_current_day: int = Field(..., ge=1, le=90)
    timestamp: str | None = None
    start_time: time | None = None
    end_time: time | None = None

    @model_validator(mode="after")
    def validate_time_inputs(self) -> "ArchiveQueryParams":
        if self.timestamp and (self.start_time or self.end_time):
            raise ValueError("Use either timestamp or start_time/end_time, not both")

        if self.timestamp:
            try:
                raw_start, raw_end = self.timestamp.split("-", maxsplit=1)
                self.start_time = time.fromisoformat(raw_start)
                self.end_time = time.fromisoformat(raw_end)
            except ValueError as exc:
                raise ValueError("timestamp must be in the format HH:MM:SS-HH:MM:SS") from exc

        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("start_time and end_time must be provided together")

        if self.start_time is not None and self.end_time is not None and self.start_time >= self.end_time:
            raise ValueError("end_time must be later than start_time")

        return self

    @field_validator("timestamp")
    @classmethod
    def empty_timestamp_to_none(cls, value: str | None) -> str | None:
        if value == "":
            return None
        return value


class CaptionSearchRequest(BaseModel):
    camera_id: int = Field(..., ge=1)
    recorded_date: date
    timestamp: str | None = None
    search_sentence: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_timestamp(self) -> "CaptionSearchRequest":
        if not self.timestamp:
            return self

        try:
            raw_start, raw_end = self.timestamp.split("-", maxsplit=1)
            start_time = time.fromisoformat(raw_start)
            end_time = time.fromisoformat(raw_end)
        except ValueError as exc:
            raise ValueError("timestamp must be in the format HH:MM:SS-HH:MM:SS") from exc

        if start_time >= end_time:
            raise ValueError("timestamp end must be later than start")

        return self


class CaptionSegmentResponse(BaseModel):
    id: int
    segment_index: int
    start_time: time | None
    end_time: time | None
    hindi_caption: str
    english_caption: str

    model_config = {"from_attributes": True}


class CaptionJobResponse(BaseModel):
    id: int
    camera_id: int
    archive_id: int
    recorded_date: date
    requested_start_time: time | None
    requested_end_time: time | None
    search_sentence: str | None
    matched_results_only: bool
    segments: list[CaptionSegmentResponse]


class TimestampRange(BaseModel):
    start_time: time
    end_time: time
