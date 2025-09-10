# Table of Contents
  * [Link to The Coach AI Tree](#the-coach-tree)
  * [Link to Our Implementation Principles](#our-implementation-principles)
  * [Link to Base Strategy](#base-strategy)

# Understanding Roles, Tactics, and Plays in Multi-Agent AI

Clarifying the distinction between roles, tactics, and plays is crucial for building a well-structured multi-agent AI system. While often used interchangeably, in our robotics context, they represent a clear hierarchy of command and action.

### Table of Differences

| Feature    | Role                    | Tactic                      | Play                               |
| :--------- | :---------------------- | :-------------------------- | :--------------------------------- |
| **Scope** | Individual Robot        | Entire Team                 | 2-3 Robots                         |
| **Purpose**| Defines a robot's job   | Defines team's strategy     | A specific, rehearsed action       |
| **Duration**| Dynamic, can change instantly | Situational, lasts for a phase | Short-term, finite sequence        |
| **Example**| `Attacker`, `Goalie`    | `ATTACKING`, `DEFENDING`    | "Dynamic Pass", "Give-and-Go"      |

---

### How They Work Together: The Hierarchy

The relationship is hierarchical:

1.  The Coach AI observes the game and selects a **Tactic** (e.g., `ATTACKING`).
2.  Based on this tactic, the Coach assigns **Roles** to each player (e.g., Robot 5 becomes `Attacker`, Robot 2 becomes `Winger`).
3.  To execute the tactic, the Coach may decide to initiate a specific **Play**.
4.  The robots, knowing their **Roles** and seeing the **Play** being called, execute the necessary low-level actions. These are either an action from a play or the role's default action.

# Py-Trees

### **Legend for Behaviour Tree in ASCII Art**

| Symbol | Name                | Description                                                                                                                                              |
| :----- | :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `(?)`  | **Selector** | Runs children in order, succeeding as soon as one succeeds. It acts like an **OR** gate or a fallback plan.                                                |
| `(S)`  | **Sequence** | Runs children in order. It succeeds only if **all** children succeed. It acts like an **AND** gate or a checklist.                                          |
| `[ ]`  | **Action Node** | A "verb" or command that performs a task and can change the world state (e.g., `[AssignRoles]`).                                                          |
| `{ }`  | **Condition Node** | A "question" or check that observes the world state without changing it (e.g., `{IsBallOurs?}`).                                                          |
| `-->`  | **Write Operation** | Indicates that the node **writes or broadcasts** information to the shared Team Blackboard. It is an **output** from the node's logic.                  |
| `<--`  | **Read Operation** | Indicates that the node **reads or subscribes** to information from the shared Team Blackboard to make its decision. It is an **input** to the node's logic.|

### **Components of the Tree**

There are three primary categories of nodes: **Composite Nodes** (which direct traffic), **Leaf Nodes** (which do the work), and **Decorator Nodes** (which modify the work).

### 1. Composite Nodes (The "Directors")

Composite nodes have one or more children and control the order in which those children are executed. They are the primary tools for creating logical flow in your tree.

#### `(S)` Sequence
* **What it is:** A `Sequence` node executes its children in order, from left to right.
* **Analogy:** A **checklist or a recipe**. You must complete each step to succeed.
* **Execution Logic:**
    * It runs the first child. If it returns `SUCCESS`, it immediately runs the second child, and so on.
    * If any child returns `FAILURE`, the `Sequence` node immediately stops and returns `FAILURE`.
    * If a child returns `RUNNING`, the `Sequence` node also returns `RUNNING` and will resume from that same child on the next tick.
    * It only returns `SUCCESS` if **all** of its children succeed.
* **Use Case:** Aiming and then shooting. You must first successfully `[AimAtGoal]` before you can `[ExecuteKick]`. One without the other is a failure.

#### `(?)` Selector
* **What it is:** A `Selector` (also called a "Fallback") executes its children in order, from left to right, until one of them succeeds.
* **Analogy:** A **series of backup plans**. Try Plan A; if it doesn't work, try Plan B; if that fails, try Plan C.
* **Execution Logic:**
    * It runs the first child. If it returns `SUCCESS`, the `Selector` immediately stops and returns `SUCCESS`.
    * If the first child returns `FAILURE`, it immediately tries the second child.
    * If a child returns `RUNNING`, the `Selector` immediately stops and returns `RUNNING`, as it has found a valid, ongoing plan.
    * It only returns `FAILURE` if **all** of its children fail.
* **Use Case:** Deciding what to do with the ball. Try `(S) ShootAtGoal`; if that's not possible (e.g., path is blocked), try `(S) PassToTeammate`; if that's not possible, fall back to `[DribbleForward]`.

#### `(P)` Parallel
* **What it is:** A `Parallel` node executes all of its children simultaneously on every tick.
* **Analogy:** A **multitasker**. You are walking and chewing gum at the same time.
* **Execution Logic:** This is the most complex composite because it needs a "success policy" to determine when the whole node is considered complete. The two most common policies are:
    * **Success on One:** The `Parallel` node returns `SUCCESS` as soon as **one** of its children returns `SUCCESS`. This is great for a "race" condition.
    * **Success on All:** The `Parallel` node only returns `SUCCESS` after **all** of its children have returned `SUCCESS`.
* **Use Case:** Our "Dynamic Pass" tactic is the perfect example. One child `[FindsBestCandidate]` while the other child `[AimsAndPasses]`. They run in parallel to ensure the aiming is always using the most up-to-date information.

---

### 2. Leaf Nodes (The "Workers")

Leaf nodes have no children. They are the endpoints of the tree where the actual work gets done.

#### `[ ]` Action
* **What it is:** An `Action` is the "verb" of the tree. It performs a task that changes the state of the world or the robot.
* **Analogy:** The **instructions** in a recipe, like "preheat the oven" or "stir the mixture."
* **Execution Logic:** An action can take time to complete. Therefore, it has three possible return statuses:
    * `SUCCESS`: The action has finished successfully (e.g., `[ExecuteKick]` is complete).
    * `FAILURE`: The action could not be completed (e.g., the kicker is broken).
    * `RUNNING`: The action is still in progress (e.g., `[GoToBall]` is still moving the robot).
* **Use Case:** `[MoveToPosition]`, `[KickBall]`, `[AssignRoles]`.

#### `{ }` Condition
* **What it is:** A `Condition` is the "question" of the tree. It checks the state of the world without changing anything.
* **Analogy:** Checking if you **have an ingredient** before starting a recipe, like "do I have eggs?"
* **Execution Logic:** A condition check is typically instantaneous. It almost always returns either `SUCCESS` or `FAILURE` on the same tick it's executed.
* **Use Case:** `{DoIHaveTheBall?}`, `{IsPathToGoalClear?}`, `{IsMyRole_Attacker?}`.

---

### 3. Decorator Nodes (The "Modifiers")

Decorator nodes have exactly one child. Their purpose is to wrap their child and modify its behaviour or the result it returns. They are like logical operators.

* **Inverter:** This decorator inverts the result of its child. `SUCCESS` becomes `FAILURE`, and `FAILURE` becomes `SUCCESS`. `RUNNING` is usually passed through unchanged.
    * **Use Case:** You want to do something only if you *don't* have the ball. You would use: `Inverter({DoIHaveTheBall?})`.

* **Succeeder:** This decorator always returns `SUCCESS`, regardless of what its child returns.
    * **Use Case:** You want to try an action, but you don't care if it fails; you want the tree to continue anyway. For example, `Succeeder([LogStatusToConsole])`. The logging might fail, but it shouldn't stop the robot's main logic.

* **Failer:** The opposite of a Succeeder. It always returns `FAILURE`, no matter what its child returns.

* **Repeater:** This decorator will re-execute its child a specific number of times, or forever.
    * **Use Case:** `Repeater([KickBall], num_repeats=3)` to kick three times in a row.

* **RetryUntilSuccessful:** This is a common type of repeater that will keep trying its child until it returns `SUCCESS`.
    * **Use Case:** Trying to connect to a server. The `[ConnectToServer]` action might fail, so you wrap it in a `RetryUntilSuccessful` decorator to keep trying.
 
## The Coach Tree 
This tree runs on a sideline computer and dictates the entire team's strategy.
<pre>
  (?) CoachRoot
  |
  |-(S) HandleOurKickOff
  |  |--{ IsGameState_OurKickOff? }
  |  `--[ SetUpKickOffFormation ]
  |
  |-(S) HandleTheirKickOff
  |  | // ... and other preset game states (referee handling) like penalties, corners, etc.
  |
  `-(S) HandleInPlay
      |
      |--[ AnalyzeFieldState ]
      |    `--< (Reads all sensor data)
      |
      |--(?) DetermineStrategicMode
      |   |
      |   |-(S) WeHavePossession
      |   |  |--{ DoWeHaveTheBall? }
      |   |  `--[ SetStrategyMode_Attacking ] --> (Writes Tactic to Blackboard)
      |   |
      |   |-(S) TheyHavePossession
      |   |  |--{ DoTheyHaveTheBall? }
      |   |  `--[ SetStrategyMode_Defending ] --> (Writes Tactic to Blackboard)
      |   |
      |   `--(S) BallIsContested
      |      `--[ SetStrategyMode_Contesting ] --> (Writes Tactic to Blackboard)
      |
      |--[ AssignDynamicRoles ]
      |    |  Description: Assigns a role to each robot based on the Tactic and world state.
      |    `--> (Writes `robot_roles` dictionary to Blackboard)
      |
      `--(?) ChooseTeamPlay
          |
          |-(S) OrchestratePassingPlay
          |  |--{ IsAttackingTactic? }
          |  |--{ IsPassTheBestOption? }
          |  `--[ CreateOrManagePassIntent ]
          |       `--> (Writes `team_pass_intent` object to Blackboard)
          |
          `-(S) OrchestrateDefensiveFormation
             |--{ IsDefendingTactic? }
             `--[ UpdateFormationCoordinates ]
                  `--> (Writes `formation_positions` to Blackboard)
</pre>

### **Part 1: How the Tactic is Chosen (The `DetermineStrategicMode` Node)**

In our tree, this looks like a simple selector based on ball possession. In a real-world implementation, this node would be more sophisticated. The decision to switch from `ATTACKING` to `DEFENDING` isn't just about who has the ball, but about the overall "game pressure" and strategic opportunity.

Here are the key factors the `DetermineStrategicMode` node would analyse on every tick to make its decision:

1.  **Ball Possession (The Primary Factor):**
    * **We have it:** Strong signal for `ATTACKING`.
    * **They have it:** Strong signal for `DEFENDING`.
    * **Nobody has it (loose ball):** Strong signal for `CONTESTING_BALL`.

2.  **Field Location of the Ball (The Context):**
    * **If we have the ball in our defensive third:** The tactic might be a more cautious "BUILD-UP ATTACK" instead of an all-out attack.
    * **If we have the ball in their final third:** The tactic becomes a very aggressive "HIGH-PRESSURE ATTACK".
    * **If they have the ball in our final third:** The tactic must be a desperate "SCRAMBLE DEFENSE", not just a standard defensive shape.

3.  **Team Momentum and Velocity:**
    * Are most of our robots moving towards the opponent's goal? This reinforces an `ATTACKING` tactic.
    * Are most of our robots moving back towards our own goal? This reinforces a `DEFENDING` tactic, even if we technically still have the ball but are under pressure.

4.  **Numerical Advantage Around the Ball:**
    * The Coach can draw a virtual circle around the ball and count the number of friendly vs. enemy robots.
    * If we have a 3v1 advantage around the ball, it will strongly favour an `ATTACKING` tactic.
    * If we are in a 2v4 disadvantage, we may choose a more conservative "HOLD_POSSESSION" or even a "CLEARANCE" tactic, even if we have the ball.

The output of all this analysis is a single, clear string written to the blackboard: the `team_strategy_mode` (e.g., `"ATTACKING"`).

---

### **Part 2: How the Tactic Directly Affects Role Assignment (The `AssignDynamicRoles` Node)**

This is where the magic happens. The `AssignDynamicRoles` node runs *immediately after* the tactic has been decided. Inside this node is a block of logic (like a `switch` statement or `if/elif/else` chain) that executes a completely different set of rules based on the `team_strategy_mode` it reads from the blackboard.

Let's look at two concrete examples.

#### **Scenario A: Tactic is set to `"ATTACKING"`**

The `AssignDynamicRoles` node executes its "Attacking" logic block:
* **`Goalie`**: Assigned to Robot 0 (static assignment).
* **`Attacker`**: Assigned to the friendly robot who currently has the ball. If no robot has it, it's assigned to the one closest to the ball with the clearest path.
* **`WingerLeft` / `WingerRight`**: Assigned to the two robots who are furthest upfield and widest apart, creating passing options. The omni-wheels allow them to take very aggressive forward-facing angles.
* **`MidfieldSupport`**: Assigned to the robot in the best central position to support the attacker, often trailing slightly to be a "drop pass" option.
* **`SecondaryDefender`**: Assigned to the deepest outfield robot (the one closest to our own goal), providing cover in case of a sudden turnover.

**Result:** The team spreads out, pushes forward, and creates a structure designed to score.

#### **Scenario B: Tactic is set to `"DEFENDING"`**

The ball is turned over. On the very next tick, the Coach sets the tactic to `"DEFENDING"`. Now, the `AssignDynamicRoles` node executes its completely different "Defending" logic block:
* **`Goalie`**: Assigned to Robot 0.
* **`PrimaryDefender`**: Assigned to the friendly robot who is closest to the *opponent robot with the ball*. Its job is to apply immediate pressure.
* **`SupportDefender (x2)`**: Assigned to the two robots best positioned to cover the most dangerous passing lanes available to the opponent attacker. Their job is to intercept, not to pressure the ball carrier.
* **`MidfieldPress (x2)`**: Assigned to the two robots highest up the field. Their job is not to win the ball directly, but to slow down the opponent's advance and prevent easy passes through the midfield.

**Result:** The team instantly becomes compact, focuses on closing down space, and prioritises interception and pressure over maintaining a wide, offensive shape.

# Our Implementation Principles

### General
We are separating the design into a Coach behaviour tree and a Player Action Map. For simplicity of design, at every moment, we shall only have 1 main play (play as defined above) occuring at once. The rest of the robots will default to a fallback behaviour based on the roles (ie, a midfielder or striker will default to moving to open space if not involved in the play)

After the play has been decided, each robot will be assigned to a specific action that is mapped to an **Action Node** on the Player Action Map. 

**Action Nodes** are `functions` that:
1. Involves only one robot
2. Stateless (only able to access global Game State)
3. Take up as much responsibility as possible from the top-level Coach tree (to prevent overcrowding of the Coach tree)

**Exhaustive List of Action Nodes**
Pass Ball (from passer)
Receive Ball
Go to Ball
Go to Point (without ball always face the ball)
Go to Point (with ball)
Turn On Spot
Clear Ball
Shoot Ball
Stop
Go To Open Space
ManMark
Block (block mouth of possessor until it faces too far from goal, then block line of sight to goal)
Defend_Goal

### Design Flow
Each Coach Behaviour Tree we design must follow the following flow:
1. Game Analysis (based on game state, decide if we should attack or defend or what tactic to employ)
2. Assign Roles
3. Run Play and Choose Involved Robots (based on roles and game state)
4. Evaluate Default Behaviour for the rest of the robots (can just be done in one node)

See below for the base strategy that we will begin with:

# Base Strategy
```
GAME ANALYSIS AND ROLE ASSIGNMENT
is_enemy_closer -> attack or defence
assign roles to attack and defence (preset)

ATTACK
ball_in_flight (passing) ->
    Y - receive_ball
    N - do_we_have_ball? -> (set robot_id closest to ball)
        Y - Is my goal chance high?
            Y - Shoot
            N - Pass (kick_ball_at_angle to receiver, get receiver to turn_to_receive parallel)
                (set receiver_id in global blackboard for next tick to know who receives ball)
        N - Who is closest?: go to ball
        
Default -> go to open space, defenders defend

DEFEND
[play] Who is closest to posessor?: Block
Default -> Man Mark (and make sure there's no overlap)
```
