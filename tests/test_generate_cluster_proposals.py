from __future__ import annotations

import unittest

from scripts.generate_cluster_proposals import _assert_english_only_proposal, _extract_json_object


class GenerateClusterProposalTests(unittest.TestCase):
    def test_extract_json_object_accepts_plain_json(self) -> None:
        data = _extract_json_object('{"topic_summary":"x","should_add_clusters":true}')
        self.assertEqual(data["topic_summary"], "x")
        self.assertTrue(data["should_add_clusters"])

    def test_extract_json_object_accepts_fenced_json(self) -> None:
        data = _extract_json_object(
            """Here is the proposal:

```json
{"topic_summary":"mechanics summary","should_add_clusters":false}
```"""
        )
        self.assertEqual(data["topic_summary"], "mechanics summary")
        self.assertFalse(data["should_add_clusters"])

    def test_extract_json_object_accepts_loose_wrapped_json(self) -> None:
        data = _extract_json_object(
            'I think this is the right output: {"topic_summary":"wrapped","should_add_clusters":true}'
        )
        self.assertEqual(data["topic_summary"], "wrapped")
        self.assertTrue(data["should_add_clusters"])

    def test_english_only_proposal_rejects_cjk_content(self) -> None:
        with self.assertRaises(RuntimeError):
            _assert_english_only_proposal(
                {
                    "topic_summary": "mechanics summary",
                    "rationale": "ok",
                    "clusters": [
                        {
                            "cluster_id": "bad_cluster",
                            "name": "中文名称",
                            "summary": "summary",
                            "description": "description",
                            "includes": [],
                            "excludes": [],
                            "entry_cues": [],
                            "related_clusters": [],
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
