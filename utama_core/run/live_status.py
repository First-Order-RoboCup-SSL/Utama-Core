"""Live terminal status panel for `StrategyRunner`, mixed in as `LiveStatusMixin`: one rich `Live`
line with FPS, referee command, stage, score and time left, refreshed every `FPS_PRINT_INTERVAL` seconds."""

import time

from rich.text import Text

from utama_core.config.settings import FPS_PRINT_INTERVAL
from utama_core.custom_referee import CustomReferee
from utama_core.run.referee_source import OfficialReferee


class LiveStatusMixin:
    """`StrategyRunner` methods for the live status panel; they read runner state set in `__init__`."""

    def _stop_fps_live(self):
        if self._fps_live:
            self._fps_live.stop()
            self._fps_live = None

    def _update_live_status(self, frame_start: float) -> None:
        """Count this frame and, once per `FPS_PRINT_INTERVAL`, refresh the live status line.

        Called at the end of `_run_step` only when `show_live_status` is set.
        """
        frame_end = time.perf_counter()
        frame_dt = frame_end - frame_start

        self.elapsed_time += frame_dt
        self.num_frames_elapsed += 1

        if self.elapsed_time >= FPS_PRINT_INTERVAL:
            fps = self.num_frames_elapsed / self.elapsed_time

            ref = self.referee_refiner
            stage_secs = ref.stage_time_left
            stage_min = int(stage_secs // 60)
            stage_sec = int(stage_secs % 60)
            display = Text()
            display.append(f"FPS: {fps:.1f}", style="bold cyan")
            display.append("  |  ")
            display.append(ref.last_command.name, style="bold yellow")
            if ref.last_next_command:
                display.append("  ->  ")
                display.append(ref.last_next_command.name, style="yellow")
            display.append("  |  ")
            display.append(ref.stage.name.replace("_", " ").title())
            display.append("  |  Blue ")
            display.append(str(ref.blue_team.score), style="bold blue")
            display.append(" - ")
            display.append(str(ref.yellow_team.score), style="bold yellow")
            display.append(" Yellow")
            display.append(f"  |  {stage_min}:{stage_sec:02d} left")
            display.append("  |  Ref: ")
            if isinstance(self.referee, CustomReferee):
                display.append("custom", style="bold magenta")
                display.append(" (")
                display.append(self.referee.profile_name, style="magenta")
                display.append(")")
            elif isinstance(self.referee, OfficialReferee):
                display.append("official", style="bold magenta")
            else:
                display.append("none", style="bold magenta")

            if ref.last_status_message:
                display.append(f"  |  {ref.last_status_message}", style="dim")

            self._fps_live.update(display)
            self._fps_live.refresh()

            self.elapsed_time = 0.0
            self.num_frames_elapsed = 0
