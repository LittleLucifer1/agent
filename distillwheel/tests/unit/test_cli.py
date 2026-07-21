import argparse

import pytest

from distillwheel.cli import _positive_float


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_positive_float_rejects_non_finite_or_non_positive_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite number"):
        _positive_float(value)


def test_positive_float_accepts_finite_positive_value():
    assert _positive_float("1.5") == 1.5
