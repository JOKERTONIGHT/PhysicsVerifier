from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from core.rule_catalog_retrieval import norm_text
from core.unified_semantic_matcher import UnifiedSemanticMatcher


CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "catalogs"
    / "rules_unified_3000_runtime_backfilled.json"
)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "typical_samples.json"


CASES: list[dict[str, Any]] = [
    {
        "id": "154795",
        "question": "Calculate the magnetic flux Phi through the loop.",
        "prediction": "Use the straight-wire field and integrate it over the loop area.",
        "domain": "Electromagnetism",
        "topic_id": "electromagnetism.biot_savart_law_and_ampere_s_law",
        "topic": "Biot-Savart Law and Ampere's Law",
        "cluster_id": "symmetry_loop_and_field_direction",
        "rule_ids": ["norm_7ad1b18ce5022390"],
        "rule_expectation": "none_required",
        "background_analysis": {
            "task_focus": "Compute the magnetic flux through a loop.",
            "objects": ["magnetic-field source", "loop"],
            "processes": ["static magnetic-field flux integration"],
            "conditions": [],
            "target_quantity": "magnetic flux Phi",
            "symbols_and_units": ["Phi"],
            "missing_information": [
                "The source-loop geometry, dimensions, and referenced diagram are absent."
            ],
            "inactive_context": [],
        },
    },
    {
        "id": "62392",
        "question": (
            "A conducting sphere with total charge Q and radius R is cut into two identical "
            "hemispheres. Find the minimum force required to keep them together."
        ),
        "prediction": "Treat each hemisphere as a point charge Q/2 at its center.",
        "domain": "Electromagnetism",
        "topic_id": "electromagnetism.electrostatics_in_conductors_and_insulators",
        "topic": "Electrostatics in Conductors and Insulators",
        "cluster_id": "boundary_images_forces_conductors_dielectrics",
        "rule_ids": ["norm_08cd0ebc3859c90c", "norm_196d7e6e165108a2"],
        "rule_expectation": "any_candidate",
        "background_analysis": {
            "task_focus": "Find the force holding two charged conducting hemispheres together.",
            "objects": ["charged conducting sphere", "two hemispheres"],
            "processes": ["electrostatic pressure on a conductor surface"],
            "conditions": ["electrostatic equilibrium", "identical hemispheres"],
            "target_quantity": "minimum holding force",
            "symbols_and_units": ["Q", "R"],
            "missing_information": [],
            "inactive_context": [],
        },
    },
    {
        "id": "142854",
        "question": (
            "Connect instead an ideal ammeter between A and B. Determine the current in terms "
            "of any or all of R and I_s."
        ),
        "prediction": "The ideal ammeter does not affect the circuit, so symmetry gives I_s/3.",
        "domain": "Electromagnetism",
        "topic_id": "electromagnetism.dc_circuits_and_kirchhoff_s_laws",
        "topic": "DC Circuits and Kirchhoff's Laws",
        "cluster_id": "general_reasoning",
        "rule_ids": ["norm_787b1832fa50e8d8"],
        "rule_expectation": "required",
        "background_analysis": {
            "task_focus": "Find the current through an ideal ammeter connected between A and B.",
            "objects": ["ideal ammeter", "DC resistor network", "current source"],
            "processes": ["short-circuit equivalent and Kirchhoff analysis"],
            "conditions": ["ideal ammeter"],
            "target_quantity": "ammeter current",
            "symbols_and_units": ["R", "I_s"],
            "missing_information": [
                "The circuit diagram and the original definitions and topology of A and B are absent."
            ],
            "inactive_context": [],
        },
    },
    {
        "id": "170364",
        "question": (
            "A heater wire cools from 800 C to 100 C after switch-off. Its heat capacity is "
            "C=10 J/K and the attached plot gives heat exchange rate P as a function of temperature."
        ),
        "prediction": "Use an average heat exchange rate to obtain a cooling time of 70 s.",
        "domain": "Thermodynamics & Statistical Physics",
        "topic_id": (
            "thermodynamics_statistical_physics.heat_transfer_conduction_convection_radiation"
        ),
        "topic": "Heat Transfer (Conduction, Convection, Radiation)",
        "cluster_id": "heating_cooling_and_capacity_model",
        "rule_ids": ["norm_0ff28d9260e4368b", "norm_884ae6d1d84de68e"],
        "rule_expectation": "any_candidate",
        "background_analysis": {
            "task_focus": "Compute cooling time with temperature-dependent heat-loss power.",
            "objects": ["heater wire", "surroundings"],
            "processes": ["nonsteady cooling with P(T)"],
            "conditions": ["heater switched off", "heat capacity C=10 J/K"],
            "target_quantity": "cooling time from 800 C to 100 C",
            "symbols_and_units": ["P(T)", "C=10 J/K", "T"],
            "missing_information": ["The attached P(T) plot and its numerical data are absent."],
            "inactive_context": [],
        },
    },
    {
        "id": "167622",
        "question": (
            "For a nearly circular ISS orbit undergoing a small altitude loss under drag, find "
            "the descent per revolution and the total fall time from altitude h."
        ),
        "prediction": "Equate drag work to the change of circular-orbit mechanical energy.",
        "domain": "Mechanics",
        "topic_id": "mechanics.gravitation_and_kepler_s_laws",
        "topic": "Gravitation and Kepler's Laws",
        "cluster_id": "orbital_decay_and_orbit_accounting",
        "rule_ids": ["norm_b9ce23b5c4014ece"],
        "rule_expectation": "required",
        "background_analysis": {
            "task_focus": "Relate drag work to slow circular-orbit decay.",
            "objects": ["ISS", "Earth", "atmospheric drag"],
            "processes": ["orbital mechanical-energy loss", "slow altitude decay"],
            "conditions": ["nearly circular orbit", "altitude loss per orbit is small"],
            "target_quantity": "descent per revolution and total fall time",
            "symbols_and_units": ["M_S", "F_drag", "h", "R_E", "G"],
            "missing_information": [],
            "inactive_context": [
                "ISS dimensions, historical altitude data, Ampere force, and atomic oxygen are not "
                "needed for Part B.5."
            ],
        },
    },
    {
        "id": "194394",
        "question": (
            "Near a black-hole horizon, write the circumference L at radial coordinate r_h+epsilon "
            "and express it in terms of P, epsilon, and F'(r_h)."
        ),
        "prediction": "Use L=2*pi*(r_h+epsilon)*sqrt(F'(r_h)*epsilon).",
        "domain": "Modern Physics",
        "topic_id": "modern_physics.cosmology_and_general_relativity_basics",
        "topic": "Cosmology and General Relativity (Basics)",
        "cluster_id": "horizon_and_cosmic_evolution",
        "rule_ids": ["norm_828e5c4ce683fa5b"],
        "rule_expectation": "candidate_only",
        "background_analysis": {
            "task_focus": "Check the near-horizon metric scaling of a proper length.",
            "objects": ["black-hole horizon", "near-horizon circle"],
            "processes": ["first-order expansion of a metric function at a zero"],
            "conditions": ["epsilon is infinitesimal", "F(r_h)=0"],
            "target_quantity": "near-horizon circumference L",
            "symbols_and_units": ["L", "P", "epsilon", "F'(r_h)"],
            "missing_information": [
                "P, F, and r_h are not defined consistently with the supplied f, g, and r_S metric."
            ],
            "inactive_context": ["The general black-hole introduction is not needed for the local expansion."],
        },
    },
]


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _RoutingFakeCompletions:
    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.requests: list[dict[str, object]] = []
        self.seen_target_rule_ids: set[str] = set()

    def create(self, **request: object) -> _FakeResponse:
        self.requests.append(request)
        payload = _request_payload(request)
        if "candidate_domains" in payload:
            candidate = next(
                item
                for item in payload["candidate_domains"]
                if item.get("domain") == self.case["domain"]
            )
            return _FakeResponse(
                json.dumps(
                    {
                        "background_analysis": self.case["background_analysis"],
                        "domains": [
                            {
                                "domain_id": candidate["domain_id"],
                                "relevant": True,
                                "score": 0.99,
                                "reason": "routing regression domain",
                            }
                        ],
                    }
                )
            )
        if "candidate_topics" in payload:
            candidate = next(
                item
                for item in payload["candidate_topics"]
                if item.get("topic_id") == self.case["topic_id"]
            )
            return _FakeResponse(
                json.dumps(
                    {
                        "topics": [
                            {
                                "topic_id": candidate["topic_id"],
                                "relevant": True,
                                "score": 0.98,
                                "reason": "routing regression topic",
                            }
                        ]
                    }
                )
            )
        if "candidate_clusters" in payload:
            candidate = next(
                item
                for item in payload["candidate_clusters"]
                if item.get("cluster_id") == self.case["cluster_id"]
            )
            return _FakeResponse(
                json.dumps(
                    {
                        "clusters": [
                            {
                                "cluster_id": candidate["cluster_id"],
                                "relevant": True,
                                "score": 0.97,
                                "reason": "routing regression cluster",
                            }
                        ]
                    }
                )
            )
        if "candidate_rules" in payload:
            candidate_ids = {item["rule_id"] for item in payload["candidate_rules"]}
            target_ids = [
                rule_id for rule_id in self.case["rule_ids"] if rule_id in candidate_ids
            ]
            self.seen_target_rule_ids.update(target_ids)
            if self.case["rule_expectation"] == "none_required" or not target_ids:
                return _FakeResponse('{"rules":[]}')
            return _FakeResponse(
                json.dumps(
                    {
                        "rules": [
                            {
                                "rule_id": target_ids[0],
                                "applicable": True,
                                "score": 0.96,
                                "reason": "routing regression candidate",
                            }
                        ]
                    }
                )
            )
        raise AssertionError(f"Unknown semantic request payload keys: {sorted(payload)}")


class _FakeClient:
    def __init__(self, case: dict[str, Any]) -> None:
        self.chat = type("FakeChat", (), {"completions": _RoutingFakeCompletions(case)})()

    @property
    def requests(self) -> list[dict[str, object]]:
        return self.chat.completions.requests

    @property
    def seen_target_rule_ids(self) -> set[str]:
        return self.chat.completions.seen_target_rule_ids


class _SequenceFakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> _FakeResponse:
        if not self.responses:
            raise AssertionError("No fake semantic response remains.")
        self.requests.append(request)
        return _FakeResponse(self.responses.pop(0))


class _SequenceFakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = type("FakeChat", (), {"completions": _SequenceFakeCompletions(responses)})()

    @property
    def requests(self) -> list[dict[str, object]]:
        return self.chat.completions.requests


def _find_path(catalog: dict[str, Any], case: dict[str, Any]) -> tuple[dict, dict, dict, list[dict]]:
    domain = next(item for item in catalog["domains"] if item.get("name") == case["domain"])
    topic = next(item for item in domain["topics"] if item.get("id") == case["topic_id"])
    cluster = next(
        item for item in topic["scenario_clusters"] if item.get("id") == case["cluster_id"]
    )
    rule_index = {item.get("rule_id"): item for item in topic["rules"]}
    rules = [rule_index[rule_id] for rule_id in case["rule_ids"]]
    return domain, topic, cluster, rules


def _request_payload(request: dict[str, object]) -> dict[str, Any]:
    messages = request["messages"]
    if not isinstance(messages, list):
        raise AssertionError("messages must be a list")
    user_message = next(message for message in messages if message["role"] == "user")
    return json.loads(user_message["content"])


class TypicalSemanticNavigationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CATALOG_PATH.open(encoding="utf-8") as handle:
            cls.catalog = json.load(handle)
        with DATA_PATH.open(encoding="utf-8") as handle:
            cls.samples = {str(item["id"]): item for item in json.load(handle)}

    def test_runtime_catalog_contains_all_expected_paths(self) -> None:
        for case in CASES:
            with self.subTest(sample_id=case["id"]):
                _domain, topic, cluster, rules = _find_path(self.catalog, case)
                self.assertEqual(topic["name"], case["topic"])
                self.assertTrue(set(case["rule_ids"]).issubset(set(cluster.get("rule_ids", []))))
                self.assertEqual([rule["rule_id"] for rule in rules], case["rule_ids"])

    def test_full_catalog_routing_keeps_six_real_samples_on_expected_paths(self) -> None:
        for case in CASES:
            with self.subTest(sample_id=case["id"]):
                client = _FakeClient(case)
                matcher = UnifiedSemanticMatcher(
                    model="fake-model",
                    client=client,
                    max_selected_rules=6,
                )
                sample = self.samples[case["id"]]

                result = matcher.select_tree_semantically(
                    sample,
                    self.catalog,
                )

                self.assertEqual(result["background_analysis"], case["background_analysis"])
                self.assertEqual(result["selected_domains"], [case["domain"]])
                self.assertEqual(
                    [item["topic_id"] for item in result["selected_topics"]],
                    [case["topic_id"]],
                )
                self.assertEqual(
                    [item["cluster_id"] for item in result["selected_clusters"]],
                    [case["cluster_id"]],
                )
                selected_rule_ids = {item["rule_id"] for item in result["selected_rules"]}
                if case["rule_expectation"] == "none_required":
                    self.assertEqual(selected_rule_ids, set())
                else:
                    self.assertTrue(selected_rule_ids.intersection(case["rule_ids"]))
                self.assertTrue(client.seen_target_rule_ids)
                self.assertGreaterEqual(len(client.requests), 4)

                payloads = [_request_payload(request) for request in client.requests]
                for payload in payloads:
                    self.assertIn(norm_text(sample["question"]), payload["problem_background"])
                for payload in payloads[:3]:
                    self.assertNotIn("student_solution", payload)
                    self.assertNotIn(norm_text(sample["prediction"]), json.dumps(payload))
                for payload in payloads[1:]:
                    self.assertEqual(payload["background_analysis"], case["background_analysis"])
                rule_payloads = [payload for payload in payloads if "candidate_rules" in payload]
                self.assertTrue(rule_payloads)
                self.assertTrue(
                    all(payload["student_solution"] == norm_text(sample["prediction"]) for payload in rule_payloads)
                )

                trace = result["navigation_trace"]["stages"]
                self.assertIn(case["topic_id"], {item["topic_id"] for item in trace["topic"]["candidates"]})
                self.assertIn(case["cluster_id"], {item["cluster_id"] for item in trace["cluster"]["candidates"]})
                self.assertTrue(
                    set(case["rule_ids"]).intersection(
                        item["rule_id"] for item in trace["rule"]["candidates"]
                    )
                )

                missing_information = result["background_analysis"]["missing_information"]
                self.assertEqual(bool(missing_information), case["id"] in {
                    "154795",
                    "142854",
                    "170364",
                    "194394",
                })
                if case["id"] == "167622":
                    self.assertIn("Ampere", sample["question"])
                    self.assertTrue(
                        any("Ampere" in item for item in result["background_analysis"]["inactive_context"])
                    )

    def test_two_domain_topic_prompt_stays_within_server_context_budget(self) -> None:
        client = _SequenceFakeClient(
            [
                json.dumps(
                    {
                        "background_analysis": {
                            "task_focus": "orbital decay under drag",
                            "inactive_context": ["electromagnetic mechanisms are not active here"],
                        },
                        "domains": [
                            {
                                "domain_id": "mechanics",
                                "relevant": True,
                                "score": 0.99,
                                "reason": "active orbital dynamics",
                            },
                            {
                                "domain_id": "electromagnetism",
                                "relevant": True,
                                "score": 0.5,
                                "reason": "mentioned in the long stem",
                            },
                        ],
                    }
                ),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)
        matcher.select_tree_semantically(
            {
                "question": "Orbital decay under constant drag. " * 350,
                "context": "A long competition stem with auxiliary mechanisms. " * 100,
                "prediction": "must not enter topic navigation",
            },
            self.catalog,
        )

        topic_payload = next(
            message["content"]
            for message in client.requests[1]["messages"]
            if message["role"] == "user"
        )
        self.assertLess(len(topic_payload), 120_000)


if __name__ == "__main__":
    unittest.main()
