# K2 static fixture

Small, deterministic fixture used by the unit / integration tests so CI does not
download `bge-m3` and does not need a production Qdrant.

* `points.jsonl` — 5 mdd-style points with ready-made dense (and sparse) vectors
  and full K2 payloads.
* `query_vectors.jsonl` — 10 questions with ready-made dense + sparse query
  vectors; each question knows which chunk it should find.

`query_vectors.jsonl` plays the role of the embedding model in tests: the fake
embedder returns the stored vector for a question text.  The fixture is generated
deterministically by `tests/fixtures/k2/generate.py`.
