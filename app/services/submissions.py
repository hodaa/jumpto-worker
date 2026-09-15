"""Transcript submission payload building."""

from app.models import TranscriptSubmission, TranscriptWordData
from app.providers import TranscriptData
from app.utils.text import normalize_word


def build_result_submission(result, provider: str) -> TranscriptSubmission:
    """Build a submission payload directly from a provider strategy result."""
    return build_submission(
        title=result.title or "Untitled video",
        duration_seconds=result.duration_seconds,
        transcript=result.transcript,
        provider=provider,
    )


def build_submission(
    *,
    title: str,
    duration_seconds: int,
    transcript: TranscriptData,
    provider: str = "",
) -> TranscriptSubmission:
    """Build a transcript submission payload from media and transcript data."""
    words: list[TranscriptWordData] = []
    for word in transcript.words:
        normalized = normalize_word(word.word)
        if normalized:
            words.append(
                TranscriptWordData(
                    word_index=len(words),
                    word=normalized,
                    start_time=word.start_time,
                    end_time=word.end_time,
                )
            )
    return TranscriptSubmission(
        title=title,
        duration_seconds=duration_seconds,
        language=transcript.language,
        transcript_text=transcript.text,
        words=words,
        provider=provider,
    )
