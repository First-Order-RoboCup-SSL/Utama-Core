# Your first tactic

Part 2 of [Your first strategy](first_strategy.md): do that one first. Here `rookie` gets a
second tactic, a partitioner that shares the robots between them, a memory, and tests.
Allow about an hour.

## 1. What rookie is missing

Watch it again:

```
pixi run main rookie wolfpack
```

Every robot runs `ChaseAndKick`, so when wolfpack breaks through, nobody is at the back. A real
team splits its robots: some attack, some defend. In code that is two tactics and a
**partitioner**, a function that decides which robots each tactic gets, every tick.

## 2. Reuse before you write

Look in `utama_core/tactics/` first: most of what you need exists. Defending is
`ShadowAndMarkTactic` (`tactics/shadow_and_mark.py`): its first two robots stand on the line
between the ball and our goal, and any others mark an opponent each. You will reuse it as it is.

## 3. The partitioner

A partitioner gets four things and returns which robots each tactic gets:

- `game`: the field state, as in a tactic.
- `free`: the robots to share out.
- `previous`: last tick's split (`None` on the first tick).
- `available`: the tactics that may take robots now. A tactic can be busy or not apply, so
  never give robots to one that is missing.

Rookie's rule: the two robots nearest our goal defend, the rest attack.

## 4. A memory: stop the chaser switching

In part 1 the chaser is whichever robot is nearest the ball, recomputed every tick. When two
robots are about as near, the job flips between them and both stop and start. A tactic can
remember things between ticks in its **mem**: `tick` gets last tick's mem and returns the new
one. Rookie remembers its chaser and only hands the job over when another robot is clearly
(0.3 m) nearer.

The decision lives in a small function of its own, `pick_chaser`, so a test can check it
without a simulator (step 7).

## 5. The code

Replace everything in `utama_core/strategy/rookie.py` with:

```python
# utama_core/strategy/rookie.py
"""`rookie`: two robots guard our goal; the others chase the ball and kick it forward."""

from dataclasses import dataclass
from typing import Optional

from utama_core.engine.context import TickContext
from utama_core.engine.strategy import Strategy
from utama_core.engine.tactic import BaseTactic, TacticTag
from utama_core.entities.data.vector import Vector2D
from utama_core.skills.go_to_ball import go_to_ball
from utama_core.skills.go_to_point import go_to_point
from utama_core.skills.kick_upfield import kick_upfield
from utama_core.tactics.shadow_and_mark import ShadowAndMarkTactic

DEFENDERS = 2  # robots that guard our goal
LANES = [-1.5, 0.0, 1.5]  # where the helpers stand across the pitch (y, metres)
SWITCH_MARGIN = 0.3  # metres: another robot must be this much nearer the ball to take over


@dataclass
class ChaseMem:
    chaser: Optional[int] = None  # the robot chasing the ball, kept from tick to tick


def pick_chaser(game, robot_ids, previous):
    """The robot nearest the ball, but last tick's chaser keeps the job unless another robot
    is SWITCH_MARGIN nearer."""
    ball = game.ball.p
    robots = game.friendly_robots

    def distance(i):
        return robots[i].p.distance_to(ball)

    nearest = min(robot_ids, key=distance)
    if previous in robot_ids and distance(nearest) > distance(previous) - SWITCH_MARGIN:
        return previous
    return nearest


class ChaseAndKick(BaseTactic[ChaseMem]):
    tag = TacticTag.ATTACK

    def initial_mem(self):
        return ChaseMem()

    def tick(self, game, ctx, robot_ids, mem):
        if not robot_ids or game.ball is None:
            return {}, mem
        ball = game.ball.p
        robots = game.friendly_robots
        chaser = pick_chaser(game, robot_ids, mem.chaser)

        commands = {}
        if robots[chaser].has_ball:
            commands[chaser] = kick_upfield(game, ctx.motion_controller, chaser)
        else:
            commands[chaser] = go_to_ball(game, ctx.motion_controller, chaser)

        # The helpers stand 1 m behind the ball, spread across the pitch.
        back = 1 if game.field.my_goal_line[0][0] > 0 else -1
        x = max(-3.3, min(3.3, ball.x + back * 1.0))
        helpers = [i for i in robot_ids if i != chaser]
        for i, y in zip(helpers, LANES):
            commands[i] = go_to_point(game, ctx.motion_controller, i, Vector2D(x, y))
        return commands, ChaseMem(chaser)


def split(game, free, previous, available):
    """The DEFENDERS robots nearest our goal defend; the rest attack."""
    if "defend" not in available:
        return {"attack": free}
    if "attack" not in available:
        return {"defend": free}
    goal = Vector2D(game.field.my_goal_line[0][0], 0.0)
    by_goal = sorted(free, key=lambda i: game.friendly_robots[i].p.distance_to(goal))
    return {"defend": frozenset(by_goal[:DEFENDERS]), "attack": frozenset(by_goal[DEFENDERS:])}


def build(outfield_robot_ids, motion_controller):
    return Strategy(
        tactics={"attack": ChaseAndKick(), "defend": ShadowAndMarkTactic()},
        partitioner=split,
        outfield_robot_ids=outfield_robot_ids,
        ctx=TickContext(motion_controller=motion_controller),
    )
```

What changed from part 1:

- `tactics=` has two entries. Their names (`"attack"`, `"defend"`) are what the partitioner hands
  robots to.
- `partitioner=split` replaces `single_tactic_picker`, which gave every robot to one tactic.
- `ChaseAndKick` keeps a `ChaseMem` and returns a new one each tick.

## 6. Watch it play

```
pixi run main rookie wolfpack
```

Do two robots stay back? Does one robot keep chasing, or does the job still flip? Try
`SWITCH_MARGIN = 0.0` and watch the difference.

Watch the defenders too. When two robots are about as far from our goal, `split` swaps them
between attack and defence, back and forth (in our test match, three or four times a minute).
That is the chaser's problem again, one level up. `split` gets `previous`, last tick's split: make
it keep last tick's defenders unless another robot is clearly nearer our goal, and add a test for
it like the chaser's.

## 7. Test the decisions

Watching shows a few seconds; a test checks a decision every time anyone changes the code. Test
the pure functions, `split` and `pick_chaser`, on game states built by hand. Create
`utama_core/tests/strategy/test_rookie.py`:

```python
# utama_core/tests/strategy/test_rookie.py
"""`rookie`'s split and chaser, on game states built by hand (no simulator)."""

from utama_core.strategy.rookie import pick_chaser, split
from utama_core.tests.fixtures.game_builder import build_game, make_ball, make_robot

BOTH = frozenset({"attack", "defend"})


def game_with(friendly_xy, ball_xy=(0.0, 0.0)):
    """Our robots at the given (x, y); our goal is on the right (x = +4.5)."""
    robots = {i: make_robot(i, is_friendly=True, x=x, y=y) for i, (x, y) in friendly_xy.items()}
    return build_game(robots, {}, make_ball(*ball_xy), my_team_is_right=True)


def test_the_two_robots_nearest_our_goal_defend():
    game = game_with({1: (4.0, 0.0), 2: (-2.0, 0.0), 3: (3.0, 1.0), 4: (0.0, 0.0), 5: (-1.0, 0.0)})
    partition = split(game, frozenset({1, 2, 3, 4, 5}), None, BOTH)
    assert partition == {"defend": {1, 3}, "attack": {2, 4, 5}}


def test_every_robot_attacks_when_defending_is_not_available():
    game = game_with({1: (4.0, 0.0), 2: (-2.0, 0.0)})
    assert split(game, frozenset({1, 2}), None, frozenset({"attack"})) == {"attack": {1, 2}}


def test_the_chaser_keeps_the_job_until_another_robot_is_clearly_nearer():
    # Robot 2 is 0.2 m nearer the ball than robot 1: less than the margin, robot 1 keeps it.
    game = game_with({1: (1.0, 0.0), 2: (-0.8, 0.0)})
    assert pick_chaser(game, (1, 2), previous=1) == 1
    # Robot 2 is 0.5 m nearer: it takes over.
    game = game_with({1: (1.0, 0.0), 2: (-0.5, 0.0)})
    assert pick_chaser(game, (1, 2), previous=1) == 2


def test_with_no_chaser_yet_the_nearest_robot_chases():
    game = game_with({1: (1.0, 0.0), 2: (-0.8, 0.0)})
    assert pick_chaser(game, (1, 2), previous=None) == 2
```

```
pixi run pytest utama_core/tests/strategy/test_rookie.py
```

The chaser test sits right at the margin: 0.2 m keeps the job, 0.5 m hands it over. A test at
the edge of a rule catches the change that breaks it. Then run the checks from part 1 (step 8)
and the evaluation (step 9): `pixi run evaluate rookie`.

## 8. When a tactic is worth sharing

`ChaseAndKick` lives in `rookie.py`, so only rookie can use it. When a second strategy wants it,
move the class (with `ChaseMem` and `pick_chaser`) into a new file,
`utama_core/tactics/chase_and_kick.py`, and import it in `rookie.py`.

One rule: on a strategy branch you may **add** files to `tactics/`, never change an existing
one. The other strategies, the opponents in every evaluation, run those tactics; changing one
would change the opponents your strategy is scored against. To improve a tactic, copy it into a
new file and change the copy. CI checks this (`tools/dev/check_strategy_branch.py`).

## Where to go next

- [Debug a moment](debug_a_moment.md): find out why a robot did something wrong in a match, and
  fix it with a test.
- [strategy.md](../docs/strategy.md), "Writing a Tactic" and "Writing and evaluating a
  strategy": lessons from past tactic bugs, and the shared pieces in `strategy/pickers.py`.
