"""Tests for calibration bundles and registry."""

import pytest

from liquiditylife.calibrations.registry import list_calibrations, load_calibration


class TestCalibrationRegistry:
    def test_list_calibrations(self) -> None:
        names = list_calibrations()
        assert "adams_baseline" in names
        assert "adams_frictionless" in names
        assert "adams_moderate_friction" in names
        assert "adams_high_friction" in names
        assert "toy_demo_small_grid" in names
        assert len(names) == 5

    def test_load_adams_baseline(self) -> None:
        cal = load_calibration("adams_baseline")
        assert cal.preferences.gamma == 5.0
        assert cal.preferences.psi == 0.5
        assert cal.preferences.beta == 0.85
        assert cal.adjustment_cost.phi_c == 10.0
        assert cal.asset_returns.rf == 0.02

    def test_load_frictionless(self) -> None:
        cal = load_calibration("adams_frictionless")
        assert cal.adjustment_cost.phi_c == 0.0

    def test_load_moderate_friction(self) -> None:
        cal = load_calibration("adams_moderate_friction")
        assert cal.adjustment_cost.phi_c == 5.0

    def test_load_high_friction(self) -> None:
        cal = load_calibration("adams_high_friction")
        assert cal.adjustment_cost.phi_c == 10.0

    def test_load_toy_demo(self) -> None:
        cal = load_calibration("toy_demo_small_grid")
        assert cal.lifecycle.age_max == 50
        assert cal.lifecycle.age_retire == 40

    def test_load_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown calibration"):
            load_calibration("nonexistent")

    def test_load_with_override(self) -> None:
        cal = load_calibration("adams_baseline", phi_c=3.0)
        assert cal.adjustment_cost.phi_c == 3.0

    def test_fingerprint_deterministic(self) -> None:
        cal1 = load_calibration("adams_baseline")
        cal2 = load_calibration("adams_baseline")
        assert cal1.fingerprint() == cal2.fingerprint()

    def test_fingerprint_differs(self) -> None:
        cal1 = load_calibration("adams_frictionless")
        cal2 = load_calibration("adams_high_friction")
        assert cal1.fingerprint() != cal2.fingerprint()

    def test_calibration_serialization_round_trip(self) -> None:
        cal = load_calibration("adams_baseline")
        json_str = cal.model_dump_json()
        from liquiditylife.calibrations.bundles import CalibrationBundle

        cal2 = CalibrationBundle.model_validate_json(json_str)
        assert cal == cal2

    def test_source_field(self) -> None:
        cal = load_calibration("adams_baseline")
        assert cal.source == "public_approximation"
