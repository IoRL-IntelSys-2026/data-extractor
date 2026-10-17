"""Deterministically generate the K2 static fixture (points + query vectors).

Run: ``python tests/fixtures/k2/generate.py``.  Vectors are random but seeded, so
re-generation is reproducible.  The *payload* fields follow K2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIM = 1024


def _rand_vector(rng: np.random.Generator) -> list[float]:
    vec = rng.standard_normal(DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return [float(x) for x in vec]


def _sparse(rng: np.random.Generator) -> dict:
    idx = sorted(rng.choice(range(1000), size=8, replace=False).tolist())
    vals = [float(abs(rng.standard_normal())) for _ in idx]
    return {"indices": [int(i) for i in idx], "values": vals}


DOCS = [
    {"doc_id": "mdd-dmv-1", "title": "Renewing a driver license", "section": "dmv", "lang": "en",
     "url": "https://example.org/dmv/renew", "text": "You may renew a suspended license online after paying the reinstatement fee."},
    {"doc_id": "mdd-dmv-2", "title": "License suspension", "section": "dmv", "lang": "en",
     "url": "https://example.org/dmv/suspend", "text": "A license is suspended when you accumulate too many points."},
    {"doc_id": "mdd-housing-1", "title": "Dormitories", "section": "housing", "lang": "en",
     "url": "https://example.org/housing/dorms", "text": "Each dormitory room has a private bathroom and shower."},
    {"doc_id": "rudn-obshaga-1", "title": "Общежития РУДН", "section": "admissions", "lang": "ru",
     "url": "https://www.rudn.ru/housing", "text": "В общежитии душ находится в блоке на этаже."},
    {"doc_id": "rudn-fee-1", "title": "Стоимость обучения", "section": "admissions", "lang": "ru",
     "url": "https://www.rudn.ru/fee", "text": "Стоимость обучения для иностранных студентов указана на сайте."},
]


def main() -> None:
    rng = np.random.default_rng(1234)

    points = []
    for i, doc in enumerate(DOCS):
        points.append(
            {
                "id": i + 1,
                "dense": _rand_vector(rng),
                "sparse": _sparse(rng),
                "payload": {
                    "chunk_id": f"{doc['doc_id']}#0",
                    "doc_id": doc["doc_id"],
                    "url": doc["url"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "embed_text": doc["text"],
                    "lang": doc["lang"],
                    "section": doc["section"],
                    "source": "mdd" if doc["doc_id"].startswith("mdd") else "rudn",
                    "content_type": "page",
                    "chunk_index": 0,
                    "n_chunks": 1,
                    "chunker": "v1",
                    "embedding_model": "BAAI/bge-m3",
                },
            }
        )

    with (HERE / "points.jsonl").open("w", encoding="utf-8") as fh:
        for p in points:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Query vectors: each query is engineered to point at a specific chunk by
    # re-using that chunk's dense vector (so the target is unambiguously nearest).
    queries = [
        ("How do I renew a suspended license?", 0, "en"),
        ("license suspension points", 1, "en"),
        ("does the dorm room have a bathroom", 2, "en"),
        ("shower in dorm", 2, "en"),
        ("driver license renewal", 0, "en"),
        ("suspension of license", 1, "en"),
        ("а в общаге там душ в блоке или на этаже?", 3, "ru"),
        ("душ общежитие", 3, "ru"),
        ("сколько стоит обучение для иностранцев", 4, "ru"),
        ("стоимость обучения", 4, "ru"),
    ]
    with (HERE / "query_vectors.jsonl").open("w", encoding="utf-8") as fh:
        for text, target_idx, lang in queries:
            vec = points[target_idx]["dense"]
            # tiny perturbation so the target is nearest but not identical
            vec = [v + float(rng.standard_normal()) * 0.01 for v in vec]
            fh.write(
                json.dumps(
                    {
                        "query": text,
                        "lang": lang,
                        "dense": vec,
                        "sparse": points[target_idx]["sparse"],
                        "target_chunk_id": points[target_idx]["payload"]["chunk_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote {len(points)} points and {len(queries)} query vectors to {HERE}")


if __name__ == "__main__":
    main()
