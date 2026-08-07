from __future__ import annotations

import unittest

from scripts.run_llm_checker_baseline import extract_json, structured_output_kwargs


class RunLlmCheckerBaselineTest(unittest.TestCase):
    def test_parse_failure_is_not_silently_treated_as_empty_diagnostics(self) -> None:
        parsed = extract_json('{"diagnostics":[{"message":"bad \\q"}]}')

        self.assertIn("_parse_error", parsed)

    def test_forced_tool_call_has_required_diagnostics_schema(self) -> None:
        kwargs = structured_output_kwargs("forced_tool_call")

        self.assertEqual("submit_diagnostics", kwargs["tool_choice"]["function"]["name"])
        parameters = kwargs["tools"][0]["function"]["parameters"]
        self.assertIn("diagnostics", parameters["required"])


if __name__ == "__main__":
    unittest.main()
