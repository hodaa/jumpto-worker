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
class SummaryLineData:
    """One timestamped transcript line sent to the summary worker."""

    seconds: float
    text: str


@dataclass(frozen=True)
class SummaryData:
    """Data for one original-footage summary request from the backend."""

    summary_id: str
    video_id: str
    youtube_video_id: str
    youtube_url: str
    title: str
    status: str
    duration_seconds: int | None = None
    language: str | None = None
    target_seconds: int | None = None
    transcript_lines: list[SummaryLineData] = field(default_factory=list)


@dataclass(frozen=True)
class SummarySegment:
    """One contiguous transcript window with an optional AI score."""

    index: int
    start_seconds: float
    end_seconds: float
    text: str
    score: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """How long the segment runs."""
        return max(self.end_seconds - self.start_seconds, 0.0)


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
