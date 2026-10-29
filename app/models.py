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
class DownloadData:
    """Data for an offloaded video download as returned by the backend."""

    download_id: str
    video_id: str
    youtube_video_id: str
    youtube_url: str
    status: str
    start_seconds: int | None = None
    end_seconds: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class NoteBlock:
    """One rendered block of a lecture-notes document."""

    kind: str
    text: str = ""
    latex: str = ""
    evidence: str = ""
    note: str = ""


@dataclass(frozen=True)
class NotesLineData:
    """One timestamped transcript line sent to the notes worker."""

    seconds: float
    text: str


@dataclass(frozen=True)
class NotesData:
    """Data for one lecture-notes request as returned by the backend."""

    note_id: str
    video_id: str
    youtube_video_id: str
    youtube_url: str
    title: str
    status: str
    duration_seconds: int | None = None
    language: str | None = None
    is_rtl: bool = False
    transcript_lines: list[NotesLineData] = field(default_factory=list)


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
    provider: str = ""
