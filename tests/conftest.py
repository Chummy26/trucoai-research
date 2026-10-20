"""Explicit optional-artifact boundaries for the public research package."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_TESTS = {
    "test_network_can_model_all_actions_when_requested",
    "test_refined_hidden_state_invariance_and_legal_actions",
    "test_acceleration_preserves_seeded_portfolio_decisions",
    "test_portfolio_file_loads_exact_policies_and_completes_legal_match",
    "test_learned_search_registration_and_legal_play",
    "test_adaptive_expert_preserves_champion_against_nonfolding_style",
    "test_adaptive_can_load_general_network_and_resets_between_matches",
    "test_style_agent_registration_legality_and_reset",
    "test_asymmetric_search_uses_the_opponent_model_only_for_the_other_seat",
    "test_iteration_factory_loads_explicit_models_and_finishes_a_legal_match",
    "test_parallel_panel_matches_seeded_serial_games_and_preserves_outputs",
}

def pytest_collection_modifyitems(items):
    if not (ROOT / "runs/mccfr_rs_o1_s7/compact/meta.json").exists():
        for item in items:
            if item.originalname in ARTIFACT_TESTS:
                item.add_marker(pytest.mark.skip(reason="Historical blueprint/configuration omitted from public release; see docs/REPRODUCIBILITY.md"))

    config_tests = {"test_refined_hidden_state_invariance_and_legal_actions",
                    "test_acceleration_preserves_seeded_portfolio_decisions"}
    if not (ROOT / "runs/e9_frozen_20261020/portfolio.json").exists():
        for item in items:
            if item.originalname in config_tests:
                item.add_marker(pytest.mark.skip(reason="Historical E9 portfolio configuration omitted"))
