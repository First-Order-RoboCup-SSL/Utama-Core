"""
Controls (keyboard or click):
  W / S      — forward / reverse
  A / D      — strafe left / right
  Q / E      — rotate CCW / CW
  Space      — kick
  B          — toggle dribbler
  C          — toggle chip
  Esc        — quit
"""

import queue
import threading
import time
import tkinter as tk

from utama_core.team_controller.src.controllers.real.real_robot_controller import (
    RealRobotController,
    RobotCommand,
    empty_command,
)

# --- Config ---
ROBOT_ID = 1
N_FRIENDLY = 2
IS_YELLOW = True
LOOP_HZ = 60
MAX_VEL = -0.1
MAX_ANG_VEL = 1

BG = "#1a1a1a"
SURFACE = "#2a2a2a"
BORDER = "#3a3a3a"
TEXT = "#eeeeee"
MUTED = "#888888"
ACTIVE = "#4a90d9"
KICK_C = "#d94a4a"
ON_C = "#4ad97a"
BALL_C = "#d9a84a"


class TeleopGUI:
    def __init__(self, root: tk.Tk, controller: RealRobotController):
        self.root = root
        self.controller = controller

        # Thread safety mechanics
        self._lock = threading.Lock()
        self._ui_queue = queue.Queue()
        self.held: set[str] = set()

        self.dribble = False
        self.chip = False
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

        # --- Kick button ---
        kick_frame = tk.Frame(self.root, bg=BG)
        kick_frame.pack(**pad)
        self._kick_btn = tk.Label(
            kick_frame,
            text="SPACE  -  kick",
            bg=SURFACE,
            fg=TEXT,
            font=("monospace", 13),
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=40,
            pady=10,
        )
        self._kick_btn.pack()

        # --- Toggle buttons ---
        toggle_frame = tk.Frame(self.root, bg=BG)
        toggle_frame.pack(**pad)

        self._dribble_btn = tk.Label(
            toggle_frame,
            text="B  dribble: OFF",
            bg=SURFACE,
            fg=MUTED,
            font=("monospace", 12),
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=16,
            pady=8,
        )
        self._dribble_btn.pack(side="left", padx=6)
        self._dribble_btn.bind("<Button-1>", lambda e: self._toggle_dribble())

        self._chip_btn = tk.Label(
            toggle_frame,
            text="C  chip: OFF",
            bg=SURFACE,
            fg=MUTED,
            font=("monospace", 12),
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=16,
            pady=8,
        )
        self._chip_btn.pack(side="left", padx=6)
        self._chip_btn.bind("<Button-1>", lambda e: self._toggle_chip())

        # --- Robot feedback ---
        fb_outer = tk.Frame(self.root, bg=BG)
        fb_outer.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(fb_outer, text="Robot Feedback", bg=BG, fg=MUTED, font=("monospace", 10)).pack(anchor="w")

        fb_frame = tk.Frame(fb_outer, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        fb_frame.pack(fill="x")

        self._fb_ball_vars: dict[int, tk.StringVar] = {}
        self._fb_ball_lbls: dict[int, tk.Label] = {}
        self._fb_status_vars: dict[int, tk.StringVar] = {}
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
            tk.Label(
                row,
                textvariable=status_var,
                bg=SURFACE,
                fg=MUTED,
                font=("monospace", 10),
                anchor="e",
            ).pack(side="right")

            self._fb_ball_vars[i] = ball_var
            self._fb_ball_lbls[i] = ball_lbl
            self._fb_status_vars[i] = status_var

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
                self.root.bind(f"<KeyPress-{k}>", lambda e, key=key: self._press(key))
                self.root.bind(f"<KeyRelease-{k}>", lambda e, key=key: self._release(key))

        special_keys = {
            "space": (self._press, self._release),
        }

        for key, (press_fn, release_fn) in special_keys.items():
            self.root.bind(f"<KeyPress-{key}>", lambda e, k=key, fn=press_fn: fn(k))
            self.root.bind(f"<KeyRelease-{key}>", lambda e, k=key, fn=release_fn: fn(k))

        toggles = {
            "b": self._toggle_dribble,
            "c": self._toggle_chip,
        }

        for key, fn in toggles.items():
            for k in (key.lower(), key.upper()):
                self.root.bind(f"<KeyRelease-{k}>", lambda e, fn=fn: fn())

        self.root.bind("<Escape>", lambda e: self._quit())

    def _press(self, key: str):
        with self._lock:
            self.held.add(key)
        self._refresh_key_visuals()

    def _release(self, key: str):
        with self._lock:
            self.held.discard(key)
        self._refresh_key_visuals()

    def _refresh_key_visuals(self):
        with self._lock:
            current_held = set(self.held)

        for k, btn in self._key_buttons.items():
            if k in current_held:
                btn.configure(bg=ACTIVE, fg="white", highlightbackground=ACTIVE)
            else:
                btn.configure(bg=SURFACE, fg=TEXT, highlightbackground=BORDER)

        if "space" in current_held:
            self._kick_btn.configure(bg=KICK_C, fg="white", highlightbackground=KICK_C)
        else:
            self._kick_btn.configure(bg=SURFACE, fg=TEXT, highlightbackground=BORDER)

    def _toggle_dribble(self):
        self.dribble = not self.dribble
        if self.dribble:
            self._dribble_btn.configure(text="B  dribble: ON", fg=ON_C, highlightbackground=ON_C)
        else:
            self._dribble_btn.configure(text="B  dribble: OFF", fg=MUTED, highlightbackground=BORDER)

    def _toggle_chip(self):
        self.chip = not self.chip
        if self.chip:
            self._chip_btn.configure(text="C  chip: ON", fg=ON_C, highlightbackground=ON_C)
        else:
            self._chip_btn.configure(text="C  chip: OFF", fg=MUTED, highlightbackground=BORDER)

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

            fwd = MAX_VEL if "w" in keys else (-MAX_VEL if "s" in keys else 0.0)
            left = MAX_VEL if "a" in keys else (-MAX_VEL if "d" in keys else 0.0)
            ang = MAX_ANG_VEL if "q" in keys else (-MAX_ANG_VEL if "e" in keys else 0.0)
            kick = "space" in keys

            cmd = RobotCommand(
                local_forward_vel=fwd,
                local_left_vel=left,
                angular_vel=ang,
                kick=kick,
                chip=self.chip,
                dribble=self.dribble,
            )

            responses = self.controller.get_robots_responses() or []

            self.controller.add_robot_commands(cmd, ROBOT_ID)
            self.controller.send_robot_commands()
            print("Sending commands", cmd)

            # Safely pass updates to the main thread via the queue
            self._ui_queue.put(("readout", cmd))
            if responses:
                self._ui_queue.put(("feedback", responses))

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _update_feedback(self, responses):
        for resp in responses:
            if resp.id not in self._fb_ball_vars:
                continue
            if resp.has_ball:
                self._fb_ball_vars[resp.id].set("ball: YES")
                self._fb_ball_lbls[resp.id].configure(fg=BALL_C)
            else:
                self._fb_ball_vars[resp.id].set("ball: no")
                self._fb_ball_lbls[resp.id].configure(fg=MUTED)
            self._fb_status_vars[resp.id].set("connected")

    def _update_readout(self, cmd: RobotCommand):
        self._metrics["fwd"].set(f"{cmd.local_forward_vel:+.2f}")
        self._metrics["left"].set(f"{cmd.local_left_vel:+.2f}")
        self._metrics["ang"].set(f"{cmd.angular_vel:+.2f}")
        self._readout.set(
            f"fwd={cmd.local_forward_vel:+.2f}  left={cmd.local_left_vel:+.2f}"
            f"  ang={cmd.angular_vel:+.2f}  kick={int(cmd.kick)}"
            f"  drib={int(cmd.dribble)}  chip={int(cmd.chip)}"
        )

    def _quit(self):
        self._running = False

        # Ensure the control loop thread fully terminates before destroying Tk root
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

        print("\nSending stop commands...")
        for _ in range(10):
            self.controller.add_robot_commands(empty_command(), ROBOT_ID)
            self.controller.send_robot_commands()

        self.root.destroy()


def main():
    controller = RealRobotController(is_team_yellow=IS_YELLOW, n_friendly=N_FRIENDLY)
    root = tk.Tk()
    TeleopGUI(root, controller)
    root.mainloop()


if __name__ == "__main__":
    main()
