"""Data structures for the JumpTo worker service."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobData:
    """Data for a transcription job as returned by the backend."""

    job_id: str
    video_id: str
    youtube_video_id: str
    youtube_url: str
    status: str


@dataclass(frozen=True)
class TranscriptWordData:
    """A single word with timing information for storage."""

    word_index: int
    word: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class TranscriptSubmission:
    """Payload sent to the backend to store a transcript."""

    title: str
    duration_seconds: int
    language: str
    transcript_text: str
    words: list[TranscriptWordData] = field(default_factory=list)
