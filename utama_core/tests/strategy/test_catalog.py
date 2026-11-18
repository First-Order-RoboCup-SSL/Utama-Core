"""`catalog.current_name`: old strategy names (pre-2026-11-12, and the older `build_*_kernel_strategy`)
load today's strategy, while `plain_name` keeps the saved name that replay files are found by."""

from utama_core.strategy import catalog


def test_every_old_name_maps_to_a_strategy_in_the_catalog():
    names = set(catalog.names())
    assert set(catalog.RENAMED.values()) <= names
    assert not set(catalog.RENAMED) & names  # an old name is never also a current one


def test_old_names_in_both_old_forms_resolve_to_the_new_name():
    assert catalog.current_name("split_shape") == "pendulum"
    assert catalog.current_name("build_split_shape_kernel_strategy") == "pendulum"
    assert catalog.current_name("pendulum") == "pendulum"


def test_plain_name_keeps_the_saved_name_replay_files_are_named_by():
    assert catalog.plain_name("build_split_shape_kernel_strategy") == "split_shape"
