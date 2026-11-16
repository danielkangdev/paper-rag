from scirag.chunk.chunker import build_chunks, chunk_id_for, count_tokens


def _md(n_sentences: int) -> str:
    sents = " ".join(f"Sentence number {i} talks about retrieval augmented generation." for i in range(n_sentences))
    return (
        "<!-- page 1 -->\n\nAbstract\n\nShort abstract here.\n\n1 Introduction\n\n"
        f"{sents}\n\n<!-- page 2 -->\n\nMore introduction text on page two.\n\n"
        "2 Method\n\nMethod text.\n\nReferences\n\n[1] Some reference."
    )


def test_chunks_never_cross_sections():
    secs, chunks = build_chunks("p", _md(200), title="T", target_tokens=100, overlap_tokens=20)
    by_section = {s.name for s in secs}
    assert {"Abstract", "Introduction", "Method", "References"} <= by_section
    for c in chunks:
        assert c.section in {"Abstract", "Introduction", "Method"}
    assert not any("Method text" in c.text for c in chunks if c.section == "Introduction")


def test_token_budget_and_overlap():
    _, chunks = build_chunks("p", _md(200), title="T", target_tokens=100, overlap_tokens=20)
    intro = [c for c in chunks if c.section == "Introduction"]
    assert len(intro) > 3
    for c in intro:
        assert count_tokens(c.text) <= 100 + 20
    # consecutive chunks share their boundary sentence(s)
    for prev, nxt in zip(intro, intro[1:]):
        last_sentence = prev.text.split(". ")[-1]
        assert last_sentence[:30] in nxt.text


def test_pages_tracked_and_references_not_chunked():
    secs, chunks = build_chunks("p", _md(5), title="T", target_tokens=500, overlap_tokens=50)
    assert all(c.page >= 1 for c in chunks)
    assert any(c.page == 2 for c in chunks)
    assert not any("Some reference" in c.text for c in chunks)
    assert "<!--" not in " ".join(c.text for c in chunks)
    refs = next(s for s in secs if s.name == "References")
    assert "Some reference" in refs.text


def test_ids_are_deterministic_and_indices_sequential():
    _, a = build_chunks("p", _md(50), title="T", target_tokens=80, overlap_tokens=10)
    _, b = build_chunks("p", _md(50), title="T", target_tokens=80, overlap_tokens=10)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.chunk_index for c in a] == list(range(len(a)))
    assert a[0].chunk_id == chunk_id_for("p", a[0].section_idx, 0)
    assert len({c.chunk_id for c in a}) == len(a)
