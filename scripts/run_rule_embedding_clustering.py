from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import openai
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenAI package not found. Please run 'pip install openai'.") from exc


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _build_client() -> Any:
    if load_dotenv:
        load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _normalize(vector: Iterable[float]) -> List[float]:
    values = [float(item) for item in vector]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0:
        return values
    return [item / norm for item in values]


def _cosine(left: List[float], right: List[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _connected_components(vectors: List[List[float]], *, threshold: float) -> List[List[int]]:
    parent = list(range(len(vectors)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if _cosine(vectors[i], vectors[j]) >= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(vectors)):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda item: (-len(item), item[0]))


def _cluster_topic_rules(
    rules: List[Dict[str, Any]],
    embeddings: Dict[str, List[float]],
    *,
    threshold: float,
    min_cluster_size: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    indexed = [rule for rule in rules if rule.get("rule_id") in embeddings]
    vectors = [_normalize(embeddings[str(rule["rule_id"])]) for rule in indexed]
    components = _connected_components(vectors, threshold=threshold) if vectors else []

    clusters: List[Dict[str, Any]] = []
    residual_rule_ids: List[str] = []
    for index, component in enumerate(components, start=1):
        rule_ids = [str(indexed[i]["rule_id"]) for i in component]
        if len(rule_ids) < min_cluster_size:
            residual_rule_ids.extend(rule_ids)
            continue
        exemplar_rules = [indexed[i] for i in component[:5]]
        clusters.append(
            {
                "cluster_id": f"embedding_cluster_{index:02d}",
                "rule_ids": rule_ids,
                "size": len(rule_ids),
                "representative_rules": [
                    {
                        "rule_id": str(rule.get("rule_id") or ""),
                        "title": _text(rule.get("title") or ""),
                        "summary": _text(rule.get("summary") or ""),
                        "trigger": _text(rule.get("trigger") or ""),
                    }
                    for rule in exemplar_rules
                ],
            }
        )
    return clusters, residual_rule_ids


def _embed_rules(
    *,
    client: Any,
    model: str,
    rules: List[Dict[str, Any]],
    batch_size: int,
    existing: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    embeddings = dict(existing)
    pending = [rule for rule in rules if str(rule.get("rule_id") or "") not in embeddings]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [_text(rule.get("embedding_text") or "") for rule in batch]
        result = client.embeddings.create(model=model, input=texts)
        for rule, item in zip(batch, result.data):
            embeddings[str(rule["rule_id"])] = [float(value) for value in item.embedding]
        print(f"[embedding] {min(start + batch_size, len(pending))}/{len(pending)} new embeddings")
    return embeddings


def run_embedding_clustering(
    *,
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    embedding_model: str,
    similarity_threshold: float,
    min_cluster_size: int,
    batch_size: int,
) -> Dict[str, Any]:
    payload = _load_json(input_path)
    rules = [item for item in payload.get("rules", []) if isinstance(item, dict)]
    cache = _load_json(cache_path) if cache_path.exists() else {"embeddings": {}}
    existing = cache.get("embeddings") if isinstance(cache, dict) else {}
    if not isinstance(existing, dict):
        existing = {}

    client = _build_client()
    embeddings = _embed_rules(
        client=client,
        model=embedding_model,
        rules=rules,
        batch_size=batch_size,
        existing={str(k): v for k, v in existing.items() if isinstance(v, list)},
    )
    _write_json(cache_path, {"embedding_model": embedding_model, "embeddings": embeddings})

    by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        by_topic[str(rule.get("topic_key") or "")].append(rule)

    topics = []
    for topic_key, topic_rules in sorted(by_topic.items()):
        clusters, residual_rule_ids = _cluster_topic_rules(
            topic_rules,
            embeddings,
            threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )
        domain, _, topic = topic_key.partition("::")
        topics.append(
            {
                "domain": domain,
                "topic": topic,
                "topic_key": topic_key,
                "rule_count": len(topic_rules),
                "cluster_count": len(clusters),
                "clusters": clusters,
                "residual_rule_ids": residual_rule_ids,
            }
        )

    result = {
        "metadata": {
            "generator": "topic_local_rule_embedding_clustering_v1",
            "embedding_model": embedding_model,
            "similarity_threshold": similarity_threshold,
            "min_cluster_size": min_cluster_size,
            "rule_count": len(rules),
            "topic_count": len(topics),
        },
        "topics": topics,
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed cleaned rules and cluster them within each topic.")
    parser.add_argument("--input", default="results/unified_rules_3000/rule_embedding_input.json")
    parser.add_argument("--output", default="results/unified_rules_3000/rule_embedding_clusters.json")
    parser.add_argument("--cache", default="results/unified_rules_3000/rule_embedding_cache.json")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true", help="Kept for command stability; cache is always reused if present.")
    args = parser.parse_args()

    result = run_embedding_clustering(
        input_path=Path(args.input),
        output_path=Path(args.output),
        cache_path=Path(args.cache),
        embedding_model=args.embedding_model,
        similarity_threshold=float(args.similarity_threshold),
        min_cluster_size=int(args.min_cluster_size),
        batch_size=int(args.batch_size),
    )
    print(json.dumps({"metadata": result["metadata"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
