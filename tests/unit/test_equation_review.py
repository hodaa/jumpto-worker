"""Unit tests for the equation review pass."""

from app.integrations.equation_review import (
    EQUATION_REVIEW_PROMPT,
    RECONSTRUCTED_MARKER,
    EquationReview,
    apply_reviews,
    equation_review_prompt,
    reviews_from_payload,
)
from app.models import NoteBlock


class TestEquationReviewPrompt:
    """The prompt lists every equation with its transcript evidence."""

    def test_numbers_equations_and_includes_evidence(self) -> None:
        prompt = equation_review_prompt(
            "the variance is p times one minus p",
            [
                NoteBlock(kind="equation", latex="p(1-p)", evidence="variance is p times"),
                NoteBlock(kind="equation", latex="E[X]"),
            ],
        )

        assert "Transcript:\nthe variance is p times one minus p" in prompt
        assert "[0] p(1-p)\nEvidence: variance is p times" in prompt
        assert "[1] E[X]" in prompt

    def test_requires_domain_and_convention_checks(self) -> None:
        assert "PMF, PDF, CDF, survival function, and MGF" in EQUATION_REVIEW_PROMPT
        assert "E[X] = 1/p" in EQUATION_REVIEW_PROMPT
        assert "E[X] = (1-p)/p" in EQUATION_REVIEW_PROMPT
        assert 'use "uncertain"' in EQUATION_REVIEW_PROMPT


class TestReviewsFromPayload:
    """Malformed verdicts are ignored instead of failing the chunk."""

    def test_parses_valid_entries(self) -> None:
        reviews = reviews_from_payload(
            {
                "reviews": [
                    {
                        "index": 0,
                        "verdict": "Corrected",
                        "corrected_latex": " p(1-p) ",
                        "reason": "sign",
                    }
                ]
            },
            1,
        )

        assert reviews == [
            EquationReview(index=0, verdict="corrected", corrected_latex="p(1-p)", reason="sign")
        ]

    def test_ignores_invalid_entries(self) -> None:
        reviews = reviews_from_payload(
            {
                "reviews": [
                    {"index": "x", "verdict": "supported"},
                    {"index": 5, "verdict": "supported"},
                    {"index": 0, "verdict": "maybe"},
                    "not a dict",
                    {"index": 1, "verdict": "supported"},
                ]
            },
            2,
        )

        assert reviews == [EquationReview(index=1, verdict="supported")]


class TestApplyReviews:
    """Review never removes an equation; it only applies corrections."""

    def test_applies_a_correction(self) -> None:
        blocks = [NoteBlock(kind="equation", latex="p^x", evidence="boom")]

        result = apply_reviews(
            blocks,
            [EquationReview(index=0, verdict="corrected", corrected_latex="p^{x}")],
        )

        assert result[0].latex == "p^{x}"
        assert result[0].evidence == "boom"

    def test_keeps_unsupported_and_uncertain_equations(self) -> None:
        blocks = [
            NoteBlock(kind="paragraph", text="Intro"),
            NoteBlock(kind="equation", latex="a"),
            NoteBlock(kind="equation", latex="b"),
        ]

        result = apply_reviews(
            blocks,
            [
                EquationReview(index=0, verdict="unsupported"),
                EquationReview(index=1, verdict="uncertain"),
            ],
        )

        assert [(block.kind, block.text, block.latex) for block in result] == [
            ("paragraph", "Intro", ""),
            ("equation", "", "a"),
            ("paragraph", RECONSTRUCTED_MARKER, ""),
            ("equation", "", "b"),
            ("paragraph", RECONSTRUCTED_MARKER, ""),
        ]

    def test_keeps_unreviewed_equations(self) -> None:
        blocks = [NoteBlock(kind="equation", latex="a")]

        assert apply_reviews(blocks, []) == blocks

    def test_normalizes_a_correction(self) -> None:
        blocks = [NoteBlock(kind="equation", latex="p")]

        result = apply_reviews(
            blocks,
            [EquationReview(index=0, verdict="corrected", corrected_latex=r"\frac12\le p")],
        )

        assert result[0].latex == r"\frac{1}{2}\leq p"

    def test_does_not_mark_supported_or_corrected(self) -> None:
        blocks = [NoteBlock(kind="equation", latex="a"), NoteBlock(kind="equation", latex="b")]

        result = apply_reviews(
            blocks,
            [
                EquationReview(index=0, verdict="supported"),
                EquationReview(index=1, verdict="corrected", corrected_latex="b^2"),
            ],
        )

        assert [(block.kind, block.latex) for block in result] == [
            ("equation", "a"),
            ("equation", "b^2"),
        ]

    def test_does_not_duplicate_an_existing_marker(self) -> None:
        blocks = [
            NoteBlock(kind="equation", latex="a"),
            NoteBlock(kind="paragraph", text=RECONSTRUCTED_MARKER),
        ]

        result = apply_reviews(blocks, [EquationReview(index=0, verdict="uncertain")])

        assert [(block.kind, block.text) for block in result] == [
            ("equation", ""),
            ("paragraph", RECONSTRUCTED_MARKER),
        ]

    def test_does_not_duplicate_a_marker_before_the_equation(self) -> None:
        blocks = [
            NoteBlock(kind="paragraph", text=RECONSTRUCTED_MARKER),
            NoteBlock(kind="equation", latex="a"),
        ]

        result = apply_reviews(blocks, [EquationReview(index=0, verdict="uncertain")])

        assert [(block.kind, block.text) for block in result] == [
            ("paragraph", RECONSTRUCTED_MARKER),
            ("equation", ""),
        ]

    def test_restores_control_escaped_corrected_latex(self) -> None:
        reviews = reviews_from_payload(
            {
                "reviews": [
                    {
                        "index": 0,
                        "verdict": "corrected",
                        "corrected_latex": "\x0crac{1}{12}",
                        "reason": "dropped backslash",
                    }
                ]
            },
            1,
        )

        assert reviews[0].corrected_latex == r"\frac{1}{12}"

    def test_correction_preserves_description_and_note(self) -> None:
        original = NoteBlock(
            kind="equation",
            text="وصف المعادلة",
            latex="rac{1}{2}",
            evidence="نص",
            note="ملاحظة",
        )

        result = apply_reviews(
            [original],
            [EquationReview(index=0, verdict="corrected", corrected_latex=r"\frac12")],
        )

        assert result[0].text == "وصف المعادلة"
        assert result[0].note == "ملاحظة"
        assert result[0].latex == r"\frac{1}{2}"
