from scirag.chunk.section_parser import split_sections


def names(md: str) -> list[str]:
    return [s.name for s in split_sections(md)]


def test_numbered_and_plain_headings():
    md = (
        "Title of Paper\n\nAbstract\n\nWe study things.\n\n1 Introduction\n\nIntro text.\n\n"
        "2.1 Training Details\n\nWe train.\n\n3 Conclusion\n\nDone."
    )
    assert names(md) == ["Abstract", "Introduction", "Training Details", "Conclusion"]


def test_heading_levels_from_numbering():
    md = "1 Method\n\nx\n\n1.2 Retrieval Setup\n\ny"
    secs = split_sections(md)
    assert [s.level for s in secs] == [1, 2]


def test_markdown_headings():
    md = "# Introduction\n\nhello\n\n## Related Work\n\nworld"
    assert names(md) == ["Introduction", "Related Work"]


def test_references_tail_dropped_but_appendix_kept():
    md = (
        "1 Introduction\n\ntext\n\nReferences\n\n[1] Foo. Experiments in retrieval. 2020.\n\n"
        "2 Results Analysis\n\nnot a heading, inside references\n\nAppendix A\n\nextra material"
    )
    assert names(md) == ["Introduction", "References", "Appendix A"]


def test_figure_captions_are_not_headings():
    md = "1 Method\n\nbody\n\nFigure 2: Retrieval results\n\nmore body"
    assert names(md) == ["Method"]


def test_no_headings_gives_single_body_section():
    secs = split_sections("just some text without any headings")
    assert len(secs) == 1 and secs[0].name == "Body"


def test_section_bodies_cover_text():
    md = "Abstract\n\nAAA\n\n1 Introduction\n\nBBB"
    secs = split_sections(md)
    assert secs[0].body == "AAA" and secs[1].body == "BBB"
