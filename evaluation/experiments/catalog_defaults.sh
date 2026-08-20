#!/usr/bin/env bash
# Canonical defaults for main PhysicsVerifier experiments (900-scale rule library).
#
# Source from other bash scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/catalog_defaults.sh"

CATALOG_DEFAULTS_ROOT="${CATALOG_DEFAULTS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEGACY_CATALOG_ROOT="${LEGACY_CATALOG_ROOT:-/slow_share/jinjianhan/workspace/catalogs}"

export MAIN_UNIFIED_CATALOG="${MAIN_UNIFIED_CATALOG:-$LEGACY_CATALOG_ROOT/rules_unified_0900.json}"
export MAIN_EXPERIENCE_MANIFEST="${MAIN_EXPERIENCE_MANIFEST:-$CATALOG_DEFAULTS_ROOT/results/experience_symbolic_program_manifest_0900.json}"
export MAIN_EXPERIENCE_MODULE="${MAIN_EXPERIENCE_MODULE:-evaluation.generated.generated_experience_checks_0900}"

export MAIN_ERROR_DATASET="${MAIN_ERROR_DATASET:-$CATALOG_DEFAULTS_ROOT/data/derived/expansion3000_scale_seed20260508/error_eval_dataset_100.json}"
export MAIN_QUESTION_DATASET="${MAIN_QUESTION_DATASET:-$CATALOG_DEFAULTS_ROOT/data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain/question_eval_dataset_50_50.json}"
export MAIN_PRECISION_DATASET="${MAIN_PRECISION_DATASET:-$CATALOG_DEFAULTS_ROOT/data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain/question_right_only_50.json}"

export MAIN_MODEL_30B="${MAIN_MODEL_30B:-qwen3-30b-a3b-instruct-2507}"
export MAIN_MODEL_4B="${MAIN_MODEL_4B:-qwen3-4b-instruct-2507}"
export MAIN_MODEL_235B="${MAIN_MODEL_235B:-qwen3-235b-a22b-instruct-2507}"

export MAIN_SYMBOLIC_MODEL="${MAIN_SYMBOLIC_MODEL:-gemini-3-flash-preview-thinking}"
export SYMBOLIC_MIN_CHECKS="${SYMBOLIC_MIN_CHECKS:-1150}"
export SYMBOLIC_MIN_LLM_OK="${SYMBOLIC_MIN_LLM_OK:-1100}"
