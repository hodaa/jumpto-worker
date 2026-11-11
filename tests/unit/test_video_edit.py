"""Unit tests for ffmpeg cutting and concatenation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.exceptions import ExternalServiceError
from app.integrations import video_edit
from app.integrations.video_edit import _quote, concat_clips, cut_clip, render_summary


def _fake_run(monkeypatch, *, returncode: int = 0, write_output: bool = True) -> list[list[str]]:
    """Capture ffmpeg commands and optionally create their output file."""
    commands: list[list[str]] = []

    def run(command, *, capture_output, text, check):
        commands.append(command)
        if returncode == 0 and write_output:
            Path(command[-1]).write_bytes(b"video")
        return SimpleNamespace(returncode=returncode, stderr="boom" if returncode else "")

    monkeypatch.setattr(video_edit.subprocess, "run", run)
    return commands


class TestCutClip:
    """Cuts are accurate re-encodes, not keyframe-only stream copies."""

    def test_seeks_before_input_and_re_encodes(self, tmp_path, monkeypatch) -> None:
        commands = _fake_run(monkeypatch)
        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")

        cut_clip(source, tmp_path / "clip.mp4", 10.0, 20.5)

        command = commands[0]
        assert command[0] == "ffmpeg"
        assert command.index("-ss") < command.index("-i")
        assert command[command.index("-ss") + 1] == "10.000"
        assert command[command.index("-t") + 1] == "10.500"
        assert "libx264" in command
        assert "aac" in command
        assert "+faststart" in command

    def test_maps_optional_audio(self, tmp_path, monkeypatch) -> None:
        commands = _fake_run(monkeypatch)
        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")

        cut_clip(source, tmp_path / "clip.mp4", 0, 1)

        assert "0:a:0?" in commands[0]

    def test_raises_when_ffmpeg_fails(self, tmp_path, monkeypatch) -> None:
        _fake_run(monkeypatch, returncode=1)
        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")

        with pytest.raises(ExternalServiceError) as caught:
            cut_clip(source, tmp_path / "clip.mp4", 0, 1)

        assert "Could not cut" in caught.value.message

    def test_raises_when_no_output_is_produced(self, tmp_path, monkeypatch) -> None:
        _fake_run(monkeypatch, write_output=False)
        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")

        with pytest.raises(ExternalServiceError) as caught:
            cut_clip(source, tmp_path / "clip.mp4", 0, 1)

        assert "produced no file" in caught.value.message


class TestConcatClips:
    """Concatenation stream-copies identically encoded clips."""

    def test_writes_a_concat_list_and_copies_streams(self, tmp_path, monkeypatch) -> None:
        commands = _fake_run(monkeypatch)
        clips = [tmp_path / "clip-0.mp4", tmp_path / "clip-1.mp4"]
        for clip in clips:
            clip.write_bytes(b"clip")
        output = tmp_path / "summary.mp4"

        concat_clips(clips, output)

        list_path = tmp_path / "clips.txt"
        assert list_path.read_text() == (f"file '{clips[0]}'\nfile '{clips[1]}'\n")
        command = commands[0]
        assert command[command.index("-f") + 1] == "concat"
        assert command[command.index("-i") + 1] == str(list_path)
        assert command[command.index("-c") + 1] == "copy"

    def test_rejects_an_empty_clip_list(self, tmp_path) -> None:
        with pytest.raises(ExternalServiceError):
            concat_clips([], tmp_path / "summary.mp4")

    def test_escapes_single_quotes_in_paths(self) -> None:
        assert _quote(Path("/tmp/it's.mp4")) == "/tmp/it'\\''s.mp4"


class TestRenderSummary:
    """Rendering cuts every selected range then concatenates them."""

    def test_cuts_each_clip_then_concatenates(self, tmp_path, monkeypatch) -> None:
        commands = _fake_run(monkeypatch)
        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")

        output = render_summary(source, [(0, 5), (10, 15)], tmp_path)

        assert output == tmp_path / "summary.mp4"
        assert output.read_bytes() == b"video"
        assert len(commands) == 3
        assert "+faststart" in commands[-1]
        assert (tmp_path / "clips.txt").is_file()

    def test_rejects_an_empty_selection(self, tmp_path) -> None:
        with pytest.raises(ExternalServiceError):
            render_summary(tmp_path / "source.mp4", [], tmp_path)
