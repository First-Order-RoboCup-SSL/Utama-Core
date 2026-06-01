from __future__ import annotations

import py_trees
import pytest
from py_trees.blackboard import Blackboard

from utama_core.strategy.common import AbstractBehaviour
from utama_core.strategy.common.blackboard_contract import (
    BlackboardKeySpec,
    declared_contract,
    register_blackboard_contract,
    validate_blackboard_contracts,
)


class ContractOnlyBehaviour(AbstractBehaviour):
    BLACKBOARD_CONTRACT = (
        BlackboardKeySpec(
            name="robot_id",
            key_attr="robot_id_key",
            access="READ",
            type_name="int",
        ),
        BlackboardKeySpec(
            name="target_orientation",
            key_attr="target_orientation_key",
            access=py_trees.common.Access.WRITE,
            type_name="float",
        ),
        BlackboardKeySpec(
            name="optional_target",
            key_attr="optional_target_key",
            access="READ",
            required=False,
        ),
    )

    def __init__(self):
        super().__init__(name="ContractOnly")
        self.robot_id_key = "robot_id"
        self.target_orientation_key = "target_orientation"
        self.optional_target_key = None

    def setup_(self):
        pass

    def update(self):
        return py_trees.common.Status.SUCCESS


def test_runtime_key_resolves_direct_key_key_attr_optional_and_class_level_docs():
    node = ContractOnlyBehaviour()

    assert BlackboardKeySpec(name="direct", key="actual", access="READ").runtime_key(node) == "actual"
    assert BlackboardKeySpec(name="robot_id", key_attr="robot_id_key", access="READ").runtime_key(node) == "robot_id"
    assert (
        BlackboardKeySpec(name="optional", key_attr="optional_target_key", access="READ", required=False).runtime_key(
            node
        )
        is None
    )
    assert BlackboardKeySpec(name="robot_id", key_attr="robot_id_key", access="READ").runtime_key() == "robot_id"


def test_declared_contract_is_doc_friendly_without_instance():
    contracts = declared_contract(ContractOnlyBehaviour)

    assert contracts[0]["key"] == "robot_id"
    assert contracts[0]["access"] == "READ"
    assert contracts[1]["access"] == "WRITE"


def test_setup_auto_registers_contract_keys():
    Blackboard.clear()
    node = ContractOnlyBehaviour()

    node.setup(is_opp_strat=False)

    assert "/My/robot_id" in node.blackboard.read
    assert "/My/target_orientation" in node.blackboard.write
    assert "/My/optional_target" not in node.blackboard.read
    Blackboard.clear()


def test_register_blackboard_contract_rejects_missing_required_attr():
    Blackboard.clear()
    node = ContractOnlyBehaviour()
    node.BLACKBOARD_CONTRACT = (BlackboardKeySpec(name="missing", key_attr="missing_key", access="READ"),)
    node.blackboard = py_trees.blackboard.Client(name="Test", namespace="/My")

    with pytest.raises(ValueError, match="missing attribute 'missing_key'"):
        register_blackboard_contract(node)
    Blackboard.clear()


def test_register_blackboard_contract_rejects_invalid_access():
    Blackboard.clear()
    node = ContractOnlyBehaviour()
    node.BLACKBOARD_CONTRACT = (BlackboardKeySpec(name="robot_id", key_attr="robot_id_key", access="MUTATE"),)
    node.blackboard = py_trees.blackboard.Client(name="Test", namespace="/My")

    with pytest.raises(ValueError, match="invalid access 'MUTATE'"):
        register_blackboard_contract(node)
    Blackboard.clear()


def test_validate_blackboard_contracts_reports_missing_read_writer():
    node = ContractOnlyBehaviour()

    validation = validate_blackboard_contracts(node)

    assert validation["status"] == "invalid"
    assert {error["code"] for error in validation["errors"]} == {"missing_writer"}
    assert {error["key"] for error in validation["errors"]} == {"robot_id"}


def test_validate_blackboard_contracts_accepts_seeded_inputs():
    node = ContractOnlyBehaviour()

    validation = validate_blackboard_contracts(node, seeded_keys={"robot_id"})

    assert validation["status"] == "valid"
    assert validation["errors"] == []
