"""Tests for Lifecycle domain model."""

import pytest

from liquiditylife.core.lifecycle import Lifecycle


class TestLifecycleConstruction:
    def test_defaults(self) -> None:
        lc = Lifecycle()
        assert lc.age_start == 25
        assert lc.age_retire == 60
        assert lc.age_max == 99

    def test_custom(self) -> None:
        lc = Lifecycle(age_start=30, age_retire=65, age_max=90)
        assert lc.age_start == 30

    def test_invalid_order(self) -> None:
        with pytest.raises(ValueError, match="age_start < age_retire < age_max"):
            Lifecycle(age_start=60, age_retire=25, age_max=99)

    def test_equal_ages_invalid(self) -> None:
        with pytest.raises(ValueError):
            Lifecycle(age_start=25, age_retire=25, age_max=99)


class TestLifecycleProperties:
    def test_n_working_periods(self) -> None:
        lc = Lifecycle()
        assert lc.n_working_periods == 35

    def test_n_retirement_periods(self) -> None:
        lc = Lifecycle()
        assert lc.n_retirement_periods == 40

    def test_n_total_periods(self) -> None:
        lc = Lifecycle()
        assert lc.n_total_periods == 75

    def test_ages_range(self) -> None:
        lc = Lifecycle(age_start=25, age_retire=60, age_max=99)
        ages = list(lc.ages)
        assert ages[0] == 25
        assert ages[-1] == 99
        assert len(ages) == 75

    def test_is_retired(self) -> None:
        lc = Lifecycle()
        assert not lc.is_retired(25)
        assert not lc.is_retired(59)
        assert lc.is_retired(60)
        assert lc.is_retired(99)
