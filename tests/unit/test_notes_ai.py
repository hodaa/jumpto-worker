"""Unit tests for shared notes-AI prompt and block normalization."""

from app.integrations.notes_ai import blocks_from_payload, notes_prompt, parse_json


class TestNotesPrompt:
    """The prompt asks for structured notes with optional equations."""

    def test_requests_structured_notes_and_optional_equations(self) -> None:
        prompt = notes_prompt("محاضرة", "ar", "نص المحاضرة", 0, 60)

        assert "structured study notes" in prompt
        assert '"heading"' in prompt
        assert '"paragraph"' in prompt
        assert '"bullet"' in prompt
        assert "Equations are optional" in prompt
        assert "If the lecture has no formulas, return no equation blocks" in prompt
        assert "Return at least one block" in prompt

    def test_asks_for_language_matched_description_and_note(self) -> None:
        prompt = notes_prompt("محاضرة", "ar", "نص المحاضرة", 0, 60)

        assert "in the lecture's language" in prompt
        assert 'write "text" as a one- or two-sentence description' in prompt
        assert '"note" as an optional short clarification' in prompt

    def test_lists_mathtext_rules(self) -> None:
        prompt = notes_prompt("Linear Algebra", "ar", "نص المحاضرة", 0, 60)

        assert "Always brace \\frac" in prompt
        assert "Use \\leq/\\geq not \\le/\\ge" in prompt
        assert "Never use \\begin, \\end, cases" in prompt
        assert "valid LaTeX for matplotlib" in prompt
        assert "Never use \\big, \\Big, \\bigg" in prompt
        assert "Repair obvious ASR errors only when unambiguous." in prompt
        assert "Include closest transcript phrase in" in prompt


class TestBlocksFromPayload:
    """Model blocks are normalized before they reach the PDF renderer."""

    def test_strips_a_trailing_period_from_a_heading(self) -> None:
        blocks = blocks_from_payload({"blocks": [{"type": "heading", "text": "Overview."}]})

        assert blocks[0].text == "Overview"

    def test_strips_a_trailing_arabic_comma_from_a_heading(self) -> None:
        blocks = blocks_from_payload({"blocks": [{"type": "heading", "text": "نظرة عامة،"}]})

        assert blocks[0].text == "نظرة عامة"

    def test_strips_a_markdown_bullet_marker(self) -> None:
        blocks = blocks_from_payload({"blocks": [{"type": "bullet", "text": "• first point"}]})

        assert blocks[0].text == "first point"

    def test_drops_an_orphan_equation_label(self) -> None:
        blocks = blocks_from_payload(
            {
                "blocks": [
                    {"type": "paragraph", "text": "(1)"},
                    {"type": "paragraph", "text": "See equation (1)"},
                ]
            }
        )

        assert [block.text for block in blocks] == ["See equation (1)"]

    def test_keeps_sentence_punctuation_in_a_paragraph(self) -> None:
        blocks = blocks_from_payload(
            {"blocks": [{"type": "paragraph", "text": "This is a sentence."}]}
        )

        assert blocks[0].text == "This is a sentence."

    def test_keeps_equation_evidence_for_the_review_pass(self) -> None:
        blocks = blocks_from_payload(
            {
                "blocks": [
                    {
                        "type": "equation",
                        "latex": "p(1-p)",
                        "evidence": "the variance is p times one minus p",
                    }
                ]
            }
        )

        assert blocks[0].evidence == "the variance is p times one minus p"

    def test_normalizes_leq_and_geq_aliases(self) -> None:
        blocks = blocks_from_payload(
            {"blocks": [{"type": "equation", "text": "", "latex": r"a\le x\ge 0"}]}
        )

        assert blocks[0].latex == r"a\leq x\geq 0"

    def test_leaves_left_and_right_untouched(self) -> None:
        blocks = blocks_from_payload(
            {"blocks": [{"type": "equation", "text": "", "latex": r"\left(x\right)"}]}
        )

        assert blocks[0].latex == r"\left(x\right)"

    def test_braces_frac_shorthand(self) -> None:
        blocks = blocks_from_payload(
            {"blocks": [{"type": "equation", "text": "", "latex": r"\frac12+\frac1\lambda"}]}
        )

        assert blocks[0].latex == r"\frac{1}{2}+\frac{1}{\lambda}"


class TestLatexEscapeRepair:
    """Single-backslash LaTeX survives JSON string decoding."""

    def test_restores_control_escaped_commands(self) -> None:
        blocks = blocks_from_payload(
            {"blocks": [{"type": "equation", "latex": "\x0crac{1}{12}+\x09imes 3"}]}
        )

        assert blocks[0].latex == r"\frac{1}{12}+\times 3"

    def test_parses_bare_latex_in_json(self) -> None:
        payload = parse_json('{"blocks":[{"type":"equation","latex":"a \\leq b"}]}', "test")

        assert payload["blocks"][0]["latex"] == r"a \leq b"

    def test_parses_control_and_bare_latex_together(self) -> None:
        payload = parse_json(
            '{"blocks":[{"type":"equation","latex":"\\frac{a}{b} \\leq c"}]}', "test"
        )

        blocks = blocks_from_payload(payload)

        assert blocks[0].latex == r"\frac{a}{b} \leq c"

    def test_keeps_valid_unicode_escapes(self) -> None:
        payload = parse_json('{"blocks":[{"latex":"\\u0645"}]}', "test")

        assert payload["blocks"][0]["latex"] == "م"

    def test_flattens_mathtext_unsupported_environment(self) -> None:
        blocks = blocks_from_payload(
            {
                "blocks": [
                    {
                        "type": "equation",
                        "latex": r"\begin{cases} a \\ b \end{cases}",
                    }
                ]
            }
        )

        assert blocks[0].latex == r"a ,\quad  b"

    def test_strips_size_and_style_commands(self) -> None:
        blocks = blocks_from_payload(
            {
                "blocks": [
                    {
                        "type": "equation",
                        "latex": r"\displaystyle \int_0^x \bigg|_0^x",
                    }
                ]
            }
        )

        assert blocks[0].latex == r"\int_0^x |_0^x"

    def test_drops_arabic_text_labels(self) -> None:
        blocks = blocks_from_payload(
            {"blocks": [{"type": "equation", "latex": r"x = 1 \text{مجموع}"}]}
        )

        assert blocks[0].latex == "x = 1"

    def test_keeps_equation_description_and_note(self) -> None:
        blocks = blocks_from_payload(
            {
                "blocks": [
                    {
                        "type": "equation",
                        "text": "احتمال النجاح في تجربة واحدة.",
                        "note": "بشرط أن تكون p بين صفر وواحد.",
                        "latex": "P(X=1)=p",
                        "evidence": "احتمال النجاح",
                    }
                ]
            }
        )

        assert blocks[0].text == "احتمال النجاح في تجربة واحدة."
        assert blocks[0].note == "بشرط أن تكون p بين صفر وواحد."

    def test_drops_empty_fraction_arguments(self) -> None:
        blocks = blocks_from_payload(
            {"blocks": [{"type": "equation", "latex": r"\frac{1}{b} \frac{1}{}"}]}
        )

        assert "\\frac{1}{}" not in blocks[0].latex
        assert r"\frac{1}{b}" in blocks[0].latex

    def test_drops_stray_arabic_inside_equations(self) -> None:
        blocks = blocks_from_payload(
            {"blocks": [{"type": "equation", "latex": "x = 1 \\text{احتمال} مجموع"}]}
        )

        assert blocks[0].latex == "x = 1"
