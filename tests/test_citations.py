from scirag.chunk.models import Hit
from scirag.rag import citations


def _ev(n: int) -> list[Hit]:
    return [Hit(chunk_id=f"c{i}", paper_id="arxiv:1", title="Self-RAG: Learning to Retrieve",
                section="Method", page=i + 1, text="t") for i in range(n)]


def test_invalid_tags_removed_and_valid_kept():
    chk = citations.validate("Claim one holds here [E1]. Claim two holds too [E7].", n_evidence=3)
    assert chk.valid == [1]
    assert chk.invalid == [7]
    assert "[E7]" not in chk.text and "[E1]" in chk.text


def test_numeric_and_author_year_styles_stripped():
    chk = citations.validate("RAG is useful [3] (Lewis et al., 2020) [E2].", n_evidence=2)
    assert chk.stripped_styles == 2
    assert "[3]" not in chk.text and "Lewis" not in chk.text
    assert chk.text.endswith("[E2].")


def test_uncited_claim_sentences_reported():
    chk = citations.validate(
        "The model uses reflection tokens [E1]. It also improves accuracy a lot. "
        "The papers do not contain more details.", n_evidence=1)
    assert chk.uncited_sentences == ["It also improves accuracy a lot."]


def test_refusal_sentences():
    assert citations.is_refusal_sentence("The indexed papers do not contain the answer.")
    assert citations.is_refusal_sentence("The evidence does not mention the price of an H100 GPU.")
    assert not citations.is_refusal_sentence("BEIR does not include training data for all tasks [E1].")
    assert not citations.is_refusal_sentence("Self-RAG does not require retrieval for every query.")


def test_render_labels():
    out = citations.render("Fact [E1][E1]. Other fact [E2].", _ev(2))
    assert out == "Fact [Self-RAG, §Method, p.1]. Other fact [Self-RAG, §Method, p.2]."


def test_cited_indices_order():
    assert citations.cited_indices("[E3] x [E1] y [E3]") == [3, 1]


def test_split_sentences():
    assert citations.split_sentences("One fact [E1]. Two facts here! Three?") == [
        "One fact [E1].", "Two facts here!", "Three?"]
