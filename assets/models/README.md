# Bundled models

`striker_rl2_w150.npz` is the trained 1v1 striker used by
[`examples/real_striker.py`](../../examples/real_striker.py). Inference uses NumPy and runs on
the CPU. Training needs Torch; running this checkpoint does not.

The network has 28 inputs, two 128-unit tanh layers and 16 actions: nine aimed shots, an
immediate kick and six dribble directions. It was trained for 150 PPO iterations, 450,000
episodes, on the standard field in rsim against the default and flat goalkeepers.

[`striker_rl2_w150.json`](striker_rl2_w150.json) preserves the original training metadata,
architecture, action definitions, evaluation results and weight checksum. Those results are
from the original standard-field simulation. Deployment adapts the coordinate frame and
field dimensions; it does not retrain the model or establish its real-hardware performance.

Weight SHA-256:

```text
d6e20d4b5bab24424c5d3a391bff38c1700aee7d921788d15e55e978c06a34f0
```

For the laptop connected to the transmission board, follow
[`docs/real_robot_setup.md`](../../docs/real_robot_setup.md#testing-the-rl-striker).
