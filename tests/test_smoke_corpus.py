"""The fixture corpus indexes cleanly through the project's own code paths."""

from tests.conftest import db_rows
from tests.fixtures.corpus import VIDEOS


def test_corpus_rows_exist(corpus):
    db = corpus["db_path"]
    assert db_rows(db, "SELECT COUNT(*) FROM videos")[0][0] == len(VIDEOS)
    assert db_rows(db, "SELECT COUNT(*) FROM transcripts WHERE shorty IS NOT NULL")[0][0] == len(VIDEOS)
    assert db_rows(db, "SELECT COUNT(*) FROM segment_index")[0][0] == 3 * len(VIDEOS)
    assert db_rows(db, "SELECT COUNT(*) FROM video_signatures")[0][0] == len(VIDEOS)


def test_chroma_has_every_doc_type(corpus):
    col = corpus["rag"].collection
    for t in ("chunk", "shorty", "synthetic_question"):
        got = col.get(where={"type": t})
        assert {m["video_id"] for m in got["metadatas"]} == {v["video_id"] for v in VIDEOS}, t
