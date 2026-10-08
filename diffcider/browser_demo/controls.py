"""Per-viewer controls for inspecting or automatically executing browser actions."""

import asyncio
import threading

from .overlays import annotate


class RunControl:
    """Keep manual inspection, cancellation and display preferences in one run."""

    def __init__(self, *, automatic=True, boxes=True, slow=False):
        self.cancelled = threading.Event()
        self.automatic = automatic
        self.boxes, self.slow = boxes, slow
        self.phase = "opening"
        self.pending = None
        self.finished = False
        self.frame = None
        self.changed = asyncio.Event()
        self.loop = asyncio.get_running_loop()

    def cancel(self):
        """Cancel from a UI callback or Gradio's session-cleanup thread."""
        self.cancelled.set()
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.changed.set)

    def command(self, command):
        """Resume, pause, or permit exactly one decision or execution."""
        if self.finished or self.cancelled.is_set():
            return
        if command == "auto":
            self.automatic, self.pending = True, None
        elif command == "pause":
            self.automatic, self.pending = False, None
        elif command in {"choose", "execute"} and command == self.phase:
            self.automatic, self.pending = False, command
        self.changed.set()

    async def wait(self, phase, *, timeout=180):
        """Wait for permission at a decision/action boundary without holding the model."""
        self.phase = phase
        while not self.cancelled.is_set():
            if self.automatic or self.pending == phase:
                self.pending = None
                self.phase = "predicting" if phase == "choose" else "acting"
                return True
            self.changed.clear()
            try:
                await asyncio.wait_for(self.changed.wait(), timeout=timeout)
            except TimeoutError:
                self.cancel()
                return False
        return False

    def screenshot(self):
        """Render the latest observation with the current overlay preference."""
        if self.frame is None:
            return None
        image, snapshot, selected = self.frame
        return annotate(image, snapshot, selected) if self.boxes else image
