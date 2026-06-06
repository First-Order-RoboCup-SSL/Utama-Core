from dataclasses import dataclass

# ============================================================
# Dummy mode
# ============================================================

USE_DUMMY_CONTROLLER = False  # dummy controller for checking ui with fake serial data.

"""
Controls (keyboard or click):
  W / S      — forward / reverse
  A / D      — strafe left / right
  Q / E      — rotate CCW / CW
  Space      — dribble
  B          — toggle kick
  C          — toggle chip
  Esc        — quit
"""

import queue
import threading
import time
import tkinter as tk

if not USE_DUMMY_CONTROLLER:

    from utama_core.team_controller.src.controllers.real.real_robot_controller import (
        RealRobotController,
        RobotCommand,
        empty_command,
    )

else:

    @dataclass
    class RobotCommand:
        local_forward_vel: float = 0.0
        local_left_vel: float = 0.0
        angular_vel: float = 0.0
        kick: bool = False
        chip: bool = False
        dribble: bool = False

    def empty_command():
        return RobotCommand()

    class DummyResponse:
        def __init__(self, robot_id, has_ball=False):
            self.id = robot_id
            self.has_ball = has_ball

    class RealRobotController:
        def __init__(self, is_team_yellow=True, n_friendly=2):

            self.n_friendly = n_friendly

            print("\n[DUMMY CONTROLLER ENABLED]")
            print("No serial/network/hardware required.\n")

        def get_robots_responses(self):

            # Fake UI feedback
            return [
                DummyResponse(0, has_ball=True),
                DummyResponse(1, has_ball=False),
            ]

        def add_robot_commands(self, cmd, robot_id):
            pass

        def send_robot_commands(self):
            pass


# --- Config ---
ROBOT_ID = 0
N_FRIENDLY = 4
IS_YELLOW = True
LOOP_HZ = 60
MAX_VEL = 0.1
FAST_VEL = 0.5
MAX_ANG_VEL = 1
CONNECTION_TIMEOUT = 0.5  # seconds without a response before a robot is marked disconnected

BG = "#1a1a1a"
SURFACE = "#2a2a2a"
BORDER = "#3a3a3a"
TEXT = "#eeeeee"
MUTED = "#888888"
ACTIVE = "#4a90d9"
KICK_C = "#d94a4a"
ON_C = "#4ad97a"
BALL_C = "#d9a84a"
DUMMY_C = "#d9a84a"  # Orange for dummy warning


class TeleopGUI:
    def __init__(self, root: tk.Tk, controller: RealRobotController):
        self.root = root
        self.controller = controller

        # Thread safety mechanics
        self._lock = threading.Lock()
        self._ui_queue = queue.Queue()
        self.held: set[str] = set()
        self._release_timers: dict[str, str] = {}

        self.dribble = False
        self.chip = False
        self._shift_held = False
        self.active_robot_id = ROBOT_ID

        # Per-robot connection tracking
        self._last_seen: dict[int, float] = {}
        self._last_has_ball: dict[int, bool] = {}

        # Impulse flags
        self._kick_impulse = False
        self._chip_impulse = False

        self._running = True

        root.title("Robot Teleop")
        root.configure(bg=BG)
        root.resizable(False, False)

        self._build_ui()
        self._bind_keys()
        self._poll_queue()

        self._loop_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._loop_thread.start()

        root.protocol("WM_DELETE_WINDOW", self._quit)

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        pad = dict(padx=12, pady=8)

        # --- Dummy Mode Indicator ---
        if USE_DUMMY_CONTROLLER:
            dummy_banner = tk.Label(
                self.root,
                text="[ DUMMY MODE ACTIVE ]",
                bg=BG,
                fg=DUMMY_C,
                font=("monospace", 10, "bold"),
                pady=4,
            )
            dummy_banner.pack(fill="x")

        # --- Status bar ---
        status_frame = tk.Frame(self.root, bg=BG)
        status_frame.pack(fill="x", **pad)

        self._metrics = {}
        for label, key in [("Forward", "fwd"), ("Left", "left"), ("Angular", "ang")]:
            col = tk.Frame(
                status_frame,
                bg=SURFACE,
                padx=14,
                pady=8,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            col.pack(side="left", expand=True, fill="x", padx=4)
            tk.Label(col, text=label, bg=SURFACE, fg=MUTED, font=("monospace", 10)).pack()
            var = tk.StringVar(value="0.00")
            tk.Label(col, textvariable=var, bg=SURFACE, fg=TEXT, font=("monospace", 18)).pack()
            self._metrics[key] = var

        # --- WASD keypad ---
        keypad = tk.Frame(self.root, bg=BG)
        keypad.pack(**pad)

        layout = [
            [None, "Q", "W", "E", None],
            [None, "A", "S", "D", None],
        ]

        self._key_buttons = {}

        for row_i, row in enumerate(layout):
            for col_i, key in enumerate(row):

                if key is None:
                    tk.Frame(
                        keypad,
                        width=58,
                        height=58,
                        bg=BG,
                    ).grid(
                        row=row_i + 1,
                        column=col_i,
                        padx=3,
                        pady=3,
                    )
                    continue

                btn = tk.Label(
                    keypad,
                    text=key,
                    width=3,
                    bg=SURFACE,
                    fg=TEXT,
                    font=("monospace", 16, "bold"),
                    highlightbackground=BORDER,
                    highlightthickness=1,
                    padx=10,
                    pady=10,
                )

                btn.grid(
                    row=row_i + 1,
                    column=col_i,
                    padx=3,
                    pady=3,
                )

                self._key_buttons[key.lower()] = btn

        # Labels ABOVE Q and E
        tk.Label(
            keypad,
            text="CCW",
            bg=BG,
            fg=MUTED,
            font=("monospace", 10),
        ).grid(row=0, column=1)

        tk.Label(
            keypad,
            text="CW",
            bg=BG,
            fg=MUTED,
            font=("monospace", 10),
        ).grid(row=0, column=3)

        # --- Space row (Toggle Dribble) ---
        space_frame = tk.Frame(self.root, bg=BG)
        space_frame.pack(**pad)

        self._dribble_btn = tk.Label(
            space_frame,
            text="SPACE  dribble: OFF",
            bg=SURFACE,
            fg=MUTED,
            font=("monospace", 13),
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=40,
            pady=10,
        )
        self._dribble_btn.pack()
        self._dribble_btn.bind("<Button-1>", lambda e: self._toggle_dribble())

        self._dribbler_timer = tk.StringVar(value="dribbler limit: --")
        self._dribbler_timer_lbl = tk.Label(
            space_frame,
            textvariable=self._dribbler_timer,
            bg=BG,
            fg=MUTED,
            font=("monospace", 10),
            pady=4,
        )
        self._dribbler_timer_lbl.pack()

        # --- B and C row (Kick and Chip) ---
        action_frame = tk.Frame(self.root, bg=BG)
        action_frame.pack(**pad)

        self._kick_btn = tk.Label(
            action_frame,
            text="B  kick",
            bg=SURFACE,
            fg=TEXT,
            font=("monospace", 12),
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=28,
            pady=8,
        )
        self._kick_btn.pack(side="left", padx=6)

        self._chip_btn = tk.Label(
            action_frame,
            text="C  chip",
            bg=SURFACE,
            fg=TEXT,
            font=("monospace", 12),
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=28,
            pady=8,
        )
        self._chip_btn.pack(side="left", padx=6)

        # --- Robot feedback ---
        fb_outer = tk.Frame(self.root, bg=BG)
        fb_outer.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(fb_outer, text="Robot Feedback", bg=BG, fg=MUTED, font=("monospace", 10)).pack(anchor="w")

        fb_frame = tk.Frame(fb_outer, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        fb_frame.pack(fill="x")

        self._fb_ball_vars: dict[int, tk.StringVar] = {}
        self._fb_ball_lbls: dict[int, tk.Label] = {}
        self._fb_status_vars: dict[int, tk.StringVar] = {}
        self._fb_rows: dict[int, tk.Frame] = {}
        self._fb_id_lbls: dict[int, tk.Label] = {}
        for i in range(N_FRIENDLY):
            row = tk.Frame(fb_frame, bg=SURFACE)
            row.pack(fill="x", padx=8, pady=4)

            id_lbl = tk.Label(
                row,
                text=f"Robot {i}",
                bg=SURFACE,
                fg=TEXT,
                font=("monospace", 11, "bold"),
                width=8,
                anchor="w",
            )
            id_lbl.pack(side="left")

            ball_var = tk.StringVar(value="ball: --")
            ball_lbl = tk.Label(
                row,
                textvariable=ball_var,
                bg=SURFACE,
                fg=MUTED,
                font=("monospace", 11),
                width=12,
                anchor="w",
            )
            ball_lbl.pack(side="left", padx=(8, 0))

            status_var = tk.StringVar(value="no data")
            status_lbl = tk.Label(
                row,
                textvariable=status_var,
                bg=SURFACE,
                fg=MUTED,
                font=("monospace", 10),
                anchor="e",
            )
            status_lbl.pack(side="right")

            # Click anywhere on the row to make this robot active.
            for widget in (row, id_lbl, ball_lbl, status_lbl):
                widget.bind("<Button-1>", lambda e, rid=i: self._set_active_robot(rid))

            self._fb_ball_vars[i] = ball_var
            self._fb_ball_lbls[i] = ball_lbl
            self._fb_status_vars[i] = status_var
            self._fb_rows[i] = row
            self._fb_id_lbls[i] = id_lbl

        self._refresh_active_robot()

        # --- Command readout ---
        readout_frame = tk.Frame(self.root, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        readout_frame.pack(fill="x", padx=12, pady=(0, 12))
        self._readout = tk.StringVar(value="RobotCommand(...)")
        tk.Label(
            readout_frame,
            textvariable=self._readout,
            bg=SURFACE,
            fg=MUTED,
            font=("monospace", 10),
            justify="left",
            anchor="w",
            padx=10,
            pady=8,
        ).pack(fill="x")

    def _bind_keys(self):

        movement = ["w", "a", "s", "d", "q", "e"]

        for key in movement:
            for k in (key.lower(), key.upper()):

                self.root.bind(
                    f"<KeyPress-{k}>",
                    lambda e, key=key: self._press(key),
                )

                self.root.bind(
                    f"<KeyRelease-{k}>",
                    lambda e, key=key: self._release(key),
                )

        # Dribbler toggle (Space)
        self.root.bind("<KeyRelease-space>", lambda e: self._toggle_dribble())

        # Chip (Impulse C)
        self.root.bind("<KeyPress-c>", lambda e: self._press("c"))
        self.root.bind("<KeyRelease-c>", lambda e: self._release("c"))
        self.root.bind("<KeyPress-C>", lambda e: self._press("c"))
        self.root.bind("<KeyRelease-C>", lambda e: self._release("c"))

        # Kick (Impulse B)
        self.root.bind("<KeyPress-b>", lambda e: self._press("b"))
        self.root.bind("<KeyRelease-b>", lambda e: self._release("b"))
        self.root.bind("<KeyPress-B>", lambda e: self._press("b"))
        self.root.bind("<KeyRelease-B>", lambda e: self._release("b"))

        # Shift boost
        self.root.bind("<KeyPress-Shift_L>", lambda e: self._set_shift(True))
        self.root.bind("<KeyRelease-Shift_L>", lambda e: self._set_shift(False))
        self.root.bind("<KeyPress-Shift_R>", lambda e: self._set_shift(True))
        self.root.bind("<KeyRelease-Shift_R>", lambda e: self._set_shift(False))

        # Number keys select the active robot
        for i in range(min(N_FRIENDLY, 10)):
            self.root.bind(f"<KeyPress-{i}>", lambda e, rid=i: self._set_active_robot(rid))

        self.root.bind("<Escape>", lambda e: self._quit())

    def _set_shift(self, value: bool):
        with self._lock:
            self._shift_held = value

    def _set_active_robot(self, robot_id: int):
        if robot_id < 0 or robot_id >= N_FRIENDLY:
            return
        with self._lock:
            self.active_robot_id = robot_id
        self._refresh_active_robot()

    def _refresh_active_robot(self):
        with self._lock:
            active = self.active_robot_id
        for rid, id_lbl in self._fb_id_lbls.items():
            if rid == active:
                id_lbl.configure(fg=ACTIVE)
                self._fb_rows[rid].configure(bg=BORDER)
            else:
                id_lbl.configure(fg=TEXT)
                self._fb_rows[rid].configure(bg=SURFACE)

    def _press(self, key: str):
        with self._lock:
            # Check for auto-repeat
            if key in self._release_timers:
                self.root.after_cancel(self._release_timers.pop(key))
                return

            # Fresh press
            if key not in self.held:
                if key == "b":
                    self._kick_impulse = True
                if key == "c":
                    self._chip_impulse = True
                self.held.add(key)

        self._refresh_key_visuals()

    def _release(self, key: str):
        # Debounce release to filter OS auto-repeat events
        with self._lock:
            if key in self._release_timers:
                self.root.after_cancel(self._release_timers.pop(key))

            timer_id = self.root.after(30, lambda k=key: self._execute_release(k))
            self._release_timers[key] = timer_id

    def _execute_release(self, key: str):
        with self._lock:
            self.held.discard(key)
            self._release_timers.pop(key, None)
        self._refresh_key_visuals()

    def _refresh_key_visuals(self):
        with self._lock:
            current_held = set(self.held)

        for k, btn in self._key_buttons.items():
            if k in current_held:
                btn.configure(bg=ACTIVE, fg="white", highlightbackground=ACTIVE)
            else:
                btn.configure(bg=SURFACE, fg=TEXT, highlightbackground=BORDER)

    def _toggle_dribble(self):
        self.dribble = not self.dribble
        if self.dribble:
            self._dribble_btn.configure(text="SPACE  dribble: ON", fg=ON_C, highlightbackground=ON_C)
        else:
            self._dribble_btn.configure(text="SPACE  dribble: OFF", fg=MUTED, highlightbackground=BORDER)

    # --------------------------------------------------------- Control loop --

    def _poll_queue(self):
        """Processes UI updates safely on the Tkinter main thread."""
        try:
            while True:
                msg_type, data = self._ui_queue.get_nowait()
                if msg_type == "readout":
                    self._update_readout(data)
                elif msg_type == "feedback":
                    self._update_feedback(data)
                elif msg_type == "dribbler_status":
                    self._update_dribbler_status(*data)
        except queue.Empty:
            pass

        if self._running:
            self.root.after(16, self._poll_queue)

    def _control_loop(self):
        dt = 1.0 / LOOP_HZ
        while self._running:
            t0 = time.perf_counter()

            with self._lock:
                keys = set(self.held)
                shift = self._shift_held
                active_id = self.active_robot_id
                # Impulse capture
                kick = self._kick_impulse  # don't worry about kick cooldown, handled in controller.
                chip = self._chip_impulse  # don't worry about kick and chip at same time, handled in controller.
                self._kick_impulse = False
                self._chip_impulse = False

            vel = FAST_VEL if shift else MAX_VEL
            fwd = vel if "w" in keys else (-vel if "s" in keys else 0.0)
            left = vel if "a" in keys else (-vel if "d" in keys else 0.0)
            ang = MAX_ANG_VEL if "q" in keys else (-MAX_ANG_VEL if "e" in keys else 0.0)

            cmd = RobotCommand(
                local_forward_vel=fwd,
                local_left_vel=left,
                angular_vel=ang,
                kick=kick,
                chip=chip,
                dribble=self.dribble,
            )

            responses = self.controller.get_robots_responses() or []

            for rid in range(N_FRIENDLY):
                self.controller.add_robot_commands(cmd if rid == active_id else empty_command(), rid)
            self.controller.send_robot_commands()
            dribbler_status = self._get_dribbler_status(active_id)

            now = time.perf_counter()
            for resp in responses:
                self._last_seen[resp.id] = now
                self._last_has_ball[resp.id] = resp.has_ball

            feedback = [
                (
                    i,
                    self._last_has_ball.get(i, False),
                    i in self._last_seen and (now - self._last_seen[i]) < CONNECTION_TIMEOUT,
                )
                for i in range(N_FRIENDLY)
            ]

            # Safely pass updates to the main thread via the queue
            self._ui_queue.put(("readout", cmd))
            self._ui_queue.put(("feedback", feedback))
            self._ui_queue.put(("dribbler_status", (active_id, dribbler_status)))

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _update_feedback(self, feedback):
        for robot_id, has_ball, connected in feedback:
            if robot_id not in self._fb_ball_vars:
                continue
            if not connected:
                self._fb_ball_vars[robot_id].set("ball: --")
                self._fb_ball_lbls[robot_id].configure(fg=MUTED)
                self._fb_status_vars[robot_id].set("no data")
                continue
            if has_ball:
                self._fb_ball_vars[robot_id].set("ball: YES")
                self._fb_ball_lbls[robot_id].configure(fg=BALL_C)
            else:
                self._fb_ball_vars[robot_id].set("ball: no")
                self._fb_ball_lbls[robot_id].configure(fg=MUTED)
            self._fb_status_vars[robot_id].set("connected")

    def _get_dribbler_status(self, robot_id: int):
        get_status = getattr(self.controller, "get_dribbler_limiter_status", None)
        if get_status is None:
            return None
        return get_status(robot_id)

    def _update_dribbler_status(self, robot_id: int, status):
        if status is None:
            self._dribbler_timer.set("dribbler limit: --")
            self._dribbler_timer_lbl.configure(fg=MUTED)
            return

        if status.throttled:
            self._dribbler_timer.set(f"R{robot_id} cooldown: {status.seconds_until_resume:04.1f}s")
            self._dribbler_timer_lbl.configure(fg=KICK_C)
            return

        self._dribbler_timer.set(f"R{robot_id} limit: {status.seconds_until_limit:05.1f}s left")
        if status.seconds_until_limit <= 10.0:
            self._dribbler_timer_lbl.configure(fg=BALL_C)
        elif self.dribble:
            self._dribbler_timer_lbl.configure(fg=ON_C)
        else:
            self._dribbler_timer_lbl.configure(fg=MUTED)

    def _update_readout(self, cmd: RobotCommand):
        self._metrics["fwd"].set(f"{cmd.local_forward_vel:+.2f}")
        self._metrics["left"].set(f"{cmd.local_left_vel:+.2f}")
        self._metrics["ang"].set(f"{cmd.angular_vel:+.2f}")
        self._readout.set(
            f"fwd={cmd.local_forward_vel:+.2f}  left={cmd.local_left_vel:+.2f}"
            f"  ang={cmd.angular_vel:+.2f}  kick={int(cmd.kick)}"
            f"  drib={int(cmd.dribble)}  chip={int(cmd.chip)}"
        )

        # UI Flash for impulse feedback
        if cmd.kick:
            self._kick_btn.configure(bg=KICK_C, fg="white", highlightbackground=KICK_C)
        else:
            self._kick_btn.configure(bg=SURFACE, fg=TEXT, highlightbackground=BORDER)

        if cmd.chip:
            self._chip_btn.configure(bg=KICK_C, fg="white", highlightbackground=KICK_C)
        else:
            self._chip_btn.configure(bg=SURFACE, fg=TEXT, highlightbackground=BORDER)

    def _quit(self):
        self._running = False

        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

        print("\nSending stop commands...")
        for _ in range(10):
            for rid in range(N_FRIENDLY):
                self.controller.add_robot_commands(empty_command(), rid)
            self.controller.send_robot_commands()

        self.root.destroy()


def main():
    controller = RealRobotController(is_team_yellow=IS_YELLOW, n_friendly=N_FRIENDLY)
    root = tk.Tk()
    TeleopGUI(root, controller)
    root.mainloop()


if __name__ == "__main__":
    main()
