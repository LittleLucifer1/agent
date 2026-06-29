import pytest

from distillwheel.core.adapter import BackendAdapter
from distillwheel.core.envspec import EnvSpec
from distillwheel.core.errors import RegistryError, RoutingError
from distillwheel.core.ir.recipe import IOConfig, OptimConfig, Recipe, TrainConfig
from distillwheel.core.registry import (
    get_adapter,
    list_adapters,
    register_adapter,
    unregister_adapter,
)
from distillwheel.core.router import resolve


class _StubAdapter(BackendAdapter):
    name = "_stub"
    supported_stages = ("sft",)
    env_spec = EnvSpec(venv_path="/tmp/nope", python_executable="/tmp/nope/bin/python")  # type: ignore[arg-type]

    def prepare_data(self, stream, recipe, workdir): ...
    def prepare_config(self, recipe, data_path, workdir): ...
    def build_launcher(self, config_path, recipe, workdir): ...
    def checkpoint_normalizer(self): ...
    def log_parser(self): ...


def test_register_and_lookup():
    unregister_adapter("_stub")
    register_adapter(_StubAdapter)
    assert "_stub" in list_adapters()
    inst = get_adapter("_stub")
    assert isinstance(inst, _StubAdapter)


def test_double_register_rejected():
    unregister_adapter("_stub")
    register_adapter(_StubAdapter)
    with pytest.raises(RegistryError):
        # different identity, same name
        class _StubAdapter2(_StubAdapter):
            pass
        _StubAdapter2.name = "_stub"
        register_adapter(_StubAdapter2)


def test_route_via_backend_hint():
    unregister_adapter("_stub")
    register_adapter(_StubAdapter)
    r = Recipe(
        stage="sft", base_model="m",
        train=TrainConfig(), optim=OptimConfig(), io=IOConfig(output_dir="o"),
        backend_hint="_stub",
    )
    assert resolve(r).name == "_stub"


def test_route_no_match_raises():
    unregister_adapter("_stub")
    r = Recipe(
        stage="rloo", base_model="m",
        train=TrainConfig(), optim=OptimConfig(), io=IOConfig(output_dir="o"),
    )
    # no verl/swift registered in this test isolation
    with pytest.raises(RoutingError):
        resolve(r)
