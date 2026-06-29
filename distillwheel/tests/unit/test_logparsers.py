from distillwheel.backends.swift.logparser import SwiftLogParser
from distillwheel.backends.verl.logparser import VerlLogParser


def test_swift_parses_hf_trainer_dict():
    p = SwiftLogParser()
    p.stage = "sft"
    m = p.parse_line("{'loss': 1.234, 'learning_rate': 5e-05, 'epoch': 0.01, 'step': 10}")
    assert m is not None
    assert m.step == 10
    assert m.loss == 1.234
    assert m.learning_rate == 5e-5
    assert m.stage == "sft"


def test_swift_ignores_garbage_lines():
    p = SwiftLogParser()
    assert p.parse_line("training started...") is None
    assert p.parse_line("") is None


def test_verl_parses_kv_line():
    p = VerlLogParser()
    p.stage = "grpo"
    m = p.parse_line(
        "step=42 actor/loss=0.21 actor/kl=0.013 reward/mean=0.81 reward/std=0.12 "
        "actor/pg_loss=0.18 actor/entropy=2.3 critic/loss=0.045"
    )
    assert m is not None
    assert m.step == 42
    assert m.loss == 0.21
    assert m.kl == 0.013
    assert m.reward_mean == 0.81
    assert m.policy_loss == 0.18
    assert m.entropy == 2.3
    assert m.value_loss == 0.045


def test_verl_no_step_returns_none():
    p = VerlLogParser()
    assert p.parse_line("actor/loss=0.1 reward/mean=0.5") is None
