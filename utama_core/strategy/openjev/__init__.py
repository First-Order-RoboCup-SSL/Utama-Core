"""OpenJev decision engines: an LLM decision model (served by FOR-Engine) as the kernel Partitioner
(`partitioner.py`, the split engine) or choosing every outfield robot's action (`flat.py`, the flat
engine).

The model-driven engines are deliberately not in `kernel_strategy.py`, so tournament auto-discovery
(`tools/evaluation/match.py`'s `_CONFIG_NAMES`) never picks them up without a running server. This
package imports nothing, so one engine's code never brings in the other's (the split engine's
fallback runs `tiki_taka`'s picker): import from the modules.
"""
