"""
DiRecorder - Screen & Audio Studio (fixed version)

Install:
    pip install opencv-python numpy mss pyautogui soundcard soundfile moviepy pywin32 pillow
    pip install -U soundcard        # make sure you have 0.4.4+ (fixes numpy 2.x errors)

Windows only (WASAPI loopback for system audio + real cursor capture).
"""
import ctypes

# --- DPI awareness MUST be set before Tk() is created ---
# Makes pixel coordinates (cursor, screen capture) match the real physical pixels.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import os
import time
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import cv2
import numpy as np
import mss
import pyautogui
import soundcard as sc
import soundfile as sf

# moviepy 1.x vs 2.x compatibility
try:
    from moviepy.editor import VideoFileClip, AudioFileClip  # moviepy 1.x
    MOVIEPY_V2 = False
except ImportError:
    from moviepy import VideoFileClip, AudioFileClip  # moviepy 2.x
    MOVIEPY_V2 = True

try:
    import win32gui
    import win32ui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

SAMPLE_RATE = 48000
FPS = 20.0
CURSOR_CANVAS = 128  # max cursor size we can render (px)


# ----------------------------------------------------------------------------
# Real cursor capture (renders the actual Windows cursor, including I-beam,
# hand, resize arrows, busy spinner, custom cursor themes, etc.)
# ----------------------------------------------------------------------------
class CursorCapturer:
    def __init__(self):
        self.cache = {}  # hcursor -> (premultiplied_rgb, alpha, hotspot_offset)

    def _render_on(self, hcursor, bg_value):
        size = CURSOR_CANVAS
        hdc_screen = win32gui.GetDC(0)
        dc = win32ui.CreateDCFromHandle(hdc_screen)
        mem = dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(dc, size, size)
        old = mem.SelectObject(bmp)
        try:
            mem.FillSolidRect((0, 0, size, size), bg_value)
            win32gui.DrawIconEx(mem.GetSafeHdc(), 0, 0, hcursor, 0, 0, 0, None, win32con.DI_NORMAL)
            bits = bmp.GetBitmapBits(True)
            arr = np.frombuffer(bits, dtype=np.uint8).reshape(size, size, 4)[:, :, :3].copy()
        finally:
            mem.SelectObject(old)
            win32gui.DeleteObject(bmp.GetHandle())
            mem.DeleteDC()
            dc.DeleteDC()
            win32gui.ReleaseDC(0, hdc_screen)
        return arr  # BGR

    def _build_sprite(self, hcursor):
        # Hotspot
        info = win32gui.GetIconInfo(hcursor)
        hot_x, hot_y = info[1], info[2]
        for bm in (info[3], info[4]):
            if bm:
                try:
                    win32gui.DeleteObject(bm)
                except Exception:
                    pass

        # Render on black and on white; the difference gives per-pixel alpha.
        on_black = self._render_on(hcursor, 0x000000).astype(np.float32)
        on_white = self._render_on(hcursor, 0xFFFFFF).astype(np.float32)
        alpha = 1.0 - (on_white - on_black).mean(axis=2) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)

        ys, xs = np.where(alpha > 0.01)
        if len(xs) == 0:
            return None
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1

        premult = on_black[y0:y1, x0:x1]            # = alpha * color
        alpha = alpha[y0:y1, x0:x1, None]
        offset = (int(x0) - hot_x, int(y0) - hot_y)  # top-left relative to pointer
        return premult, alpha, offset

    def draw(self, frame, monitor_left=0, monitor_top=0):
        if not HAS_WIN32:
            return
        try:
            flags, hcursor, (cx, cy) = win32gui.GetCursorInfo()
        except Exception:
            return
        if not (flags & 1) or not hcursor:  # CURSOR_SHOWING
            return

        if hcursor not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            try:
                self.cache[hcursor] = self._build_sprite(hcursor)
            except Exception:
                self.cache[hcursor] = None
        sprite = self.cache[hcursor]
        if sprite is None:
            return

        premult, alpha, (ox, oy) = sprite
        h, w = alpha.shape[:2]
        x = cx - monitor_left + ox
        y = cy - monitor_top + oy

        fh, fw = frame.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, fw), min(y + h, fh)
        if x1 >= x2 or y1 >= y2:
            return
        sx, sy = x1 - x, y1 - y
        sw, sh = x2 - x1, y2 - y1

        roi = frame[y1:y2, x1:x2].astype(np.float32)
        a = alpha[sy:sy + sh, sx:sx + sw]
        p = premult[sy:sy + sh, sx:sx + sw]
        frame[y1:y2, x1:x2] = np.clip(roi * (1.0 - a) + p, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_stereo(data):
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2]


def fit_length(data, n):
    """Pad with silence or trim so the track is exactly n samples long."""
    if len(data) >= n:
        return data[:n]
    pad = np.zeros((n - len(data), data.shape[1]), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
class DiRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DiRecorder")
        self.root.geometry("520x620")
        self.root.resizable(False, False)

        try:
            if os.path.exists("icon.png"):
                icon_img = tk.PhotoImage(file="icon.png")
                self.root.iconphoto(True, icon_img)
        except Exception:
            pass

        self.is_recording = False
        self.is_paused = False
        self.start_time = 0
        self.paused_duration = 0
        self.pause_start_time = 0
        self.cursor = CursorCapturer()

        # --- UI ---
        tk.Label(root, text="DiRecorder - Screen & Audio Studio", font=("Arial", 16, "bold")).pack(pady=10)

        file_frame = tk.LabelFrame(root, text=" Video Recording ", font=("Arial", 9, "bold"), padx=10, pady=8)
        file_frame.pack(fill="x", padx=20, pady=5)

        vid_path_frame = tk.Frame(file_frame)
        vid_path_frame.pack(fill="x", pady=2)
        tk.Label(vid_path_frame, text="Save As:", font=("Arial", 9), width=8).pack(side=tk.LEFT)
        self.filename_entry = tk.Entry(vid_path_frame, width=28, font=("Arial", 9))
        self.filename_entry.insert(0, "screen_record.mp4")
        self.filename_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(vid_path_frame, text="Browse", command=self.browse_video_file, width=8).pack(side=tk.LEFT)

        options_frame = tk.Frame(file_frame)
        options_frame.pack(fill="x", pady=5)

        self.sys_audio_var = tk.BooleanVar(value=True)
        self.mic_audio_var = tk.BooleanVar(value=True)
        self.facecam_var = tk.BooleanVar(value=True)

        tk.Checkbutton(options_frame, text="System Audio", variable=self.sys_audio_var, font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(options_frame, text="Microphone", variable=self.mic_audio_var, font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(options_frame, text="Facecam", variable=self.facecam_var, font=("Arial", 9)).pack(side=tk.LEFT, padx=5)

        pos_frame = tk.Frame(file_frame)
        pos_frame.pack(fill="x", pady=5)
        tk.Label(pos_frame, text="Facecam Corner:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.position_var = tk.StringVar(value="Bottom-Right")
        self.pos_dropdown = ttk.Combobox(pos_frame, textvariable=self.position_var,
                                         values=["Bottom-Right", "Bottom-Left", "Top-Right", "Top-Left"],
                                         state="readonly", width=15)
        self.pos_dropdown.pack(side=tk.LEFT, padx=5)

        self.timer_label = tk.Label(root, text="00:00:00", font=("Courier", 20, "bold"), fg="#222")
        self.timer_label.pack(pady=3)

        self.status_label = tk.Label(root, text="Status: Idle", font=("Arial", 9, "italic"), fg="gray")
        self.status_label.pack(pady=2)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.start_btn = tk.Button(btn_frame, text="Start Recording", bg="#4CAF50", fg="white",
                                   font=("Arial", 10, "bold"), width=13, command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = tk.Button(btn_frame, text="Pause", bg="#FF9800", fg="white",
                                   font=("Arial", 10, "bold"), width=10, command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="Stop", bg="#F44336", fg="white",
                                  font=("Arial", 10, "bold"), width=10, command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # --- Screenshot ---
        ss_frame = tk.LabelFrame(root, text=" Instant Screenshot Studio ", font=("Arial", 9, "bold"), padx=10, pady=8)
        ss_frame.pack(fill="x", padx=20, pady=10)

        ss_sub_frame = tk.Frame(ss_frame)
        ss_sub_frame.pack(fill="x", pady=2)
        tk.Label(ss_sub_frame, text="Delay:", font=("Arial", 9), width=6).pack(side=tk.LEFT)
        self.delay_var = tk.StringVar(value="3 Seconds")
        self.delay_dropdown = ttk.Combobox(ss_sub_frame, textvariable=self.delay_var,
                                           values=["3 Seconds", "5 Seconds", "10 Seconds"],
                                           state="readonly", width=12)
        self.delay_dropdown.pack(side=tk.LEFT, padx=5)

        self.ss_btn = tk.Button(ss_sub_frame, text="Take Screenshot", bg="#2196F3", fg="white",
                                font=("Arial", 9, "bold"), command=self.start_screenshot_countdown)
        self.ss_btn.pack(side=tk.RIGHT, padx=5)

    # ------------------------------------------------------------------ files
    def browse_video_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                 filetypes=[("MP4 files", "*.mp4")],
                                                 initialfile="screen_record.mp4")
        if file_path:
            self.filename_entry.delete(0, tk.END)
            self.filename_entry.insert(0, file_path)

    # ------------------------------------------------------------- screenshot
    def start_screenshot_countdown(self):
        delay_seconds = int(self.delay_var.get().split()[0])
        self.ss_btn.config(state=tk.DISABLED)

        def countdown(remaining):
            if remaining > 0:
                self.status_label.config(text=f"Taking screenshot in {remaining}s...", fg="blue")
                self.root.after(1000, countdown, remaining - 1)
            else:
                self.take_screenshot()

        countdown(delay_seconds)

    def take_screenshot(self):
        try:
            # Hide our own window, grab the screen, then ask where to save.
            self.root.iconify()
            self.root.update()
            time.sleep(0.4)
            img = pyautogui.screenshot()
            self.root.deiconify()
            self.root.update()

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=f"screenshot_{timestamp}.png")

            if file_path:
                img.save(file_path)
                self.status_label.config(text="Status: Screenshot Saved!", fg="green")
                messagebox.showinfo("Success", f"Screenshot saved successfully as:\n{file_path}")
            else:
                self.status_label.config(text="Status: Idle", fg="gray")
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("Error", f"Failed to take screenshot: {e}")
        finally:
            self.ss_btn.config(state=tk.NORMAL)

    # -------------------------------------------------------------- recording
    def start_recording(self):
        output_file = self.filename_entry.get().strip()
        if not output_file:
            messagebox.showerror("Error", "Please specify a valid output filename.")
            return

        self.is_recording = True
        self.is_paused = False
        self.paused_duration = 0
        self.start_time = time.time()

        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause", bg="#FF9800")
        self.stop_btn.config(state=tk.NORMAL)
        self.filename_entry.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Recording...", fg="green")

        self.update_timer()
        threading.Thread(target=self.record_loop, args=(output_file,), daemon=True).start()

    def toggle_pause(self):
        if not self.is_paused:
            self.is_paused = True
            self.pause_start_time = time.time()
            self.pause_btn.config(text="Resume", bg="#00BCD4")
            self.status_label.config(text="Status: Paused", fg="orange")
        else:
            self.paused_duration += time.time() - self.pause_start_time
            self.is_paused = False
            self.pause_btn.config(text="Pause", bg="#FF9800")
            self.status_label.config(text="Status: Recording...", fg="green")

    def update_timer(self):
        if self.is_recording:
            if not self.is_paused:
                elapsed = int(time.time() - self.start_time - self.paused_duration)
                hrs, mins, secs = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
                self.timer_label.config(text=f"{hrs:02d}:{mins:02d}:{secs:02d}")
            self.root.after(500, self.update_timer)

    def stop_recording(self):
        if self.is_paused:  # close out the pause so timing is right
            self.paused_duration += time.time() - self.pause_start_time
            self.is_paused = False
        self.is_recording = False
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Saving... (encoding video)", fg="blue")

    # ------------------------------------------------------------ audio threads
    def audio_worker(self, kind, store, ready_event, errors):
        """Records either the microphone or system loopback in its own thread."""
        try:
            if kind == "mic":
                device = sc.default_microphone()
                channels = 1
            else:
                speaker = sc.default_speaker()
                device = sc.get_microphone(id=str(speaker.name), include_loopback=True)
                channels = 2

            with device.recorder(samplerate=SAMPLE_RATE, channels=channels) as rec:
                ready_event.set()
                while self.is_recording:
                    data = rec.record(numframes=SAMPLE_RATE // 10)  # 100 ms blocks
                    if not self.is_paused:
                        store.append(data.astype("float32"))
        except Exception as e:
            errors.append(f"{kind} audio: {e}")
            ready_event.set()

    def silence_worker(self):
        """Loopback delivers no data while nothing is playing, which would
        shorten the system-audio track. Playing silence keeps it flowing."""
        try:
            speaker = sc.default_speaker()
            block = np.zeros((SAMPLE_RATE // 10, 2), dtype="float32")
            with speaker.player(samplerate=SAMPLE_RATE, channels=2) as player:
                while self.is_recording:
                    player.play(block)
        except Exception:
            pass

    # ----------------------------------------------------------- main capture
    def record_loop(self, output_filename):
        tmp_dir = tempfile.mkdtemp(prefix="direcorder_")
        temp_video_file = os.path.join(tmp_dir, "video.mp4")
        temp_audio_file = os.path.join(tmp_dir, "audio.wav")

        mic_frames, sys_frames, errors = [], [], []
        threads, ready_events = [], []

        if self.mic_audio_var.get():
            ev = threading.Event()
            ready_events.append(ev)
            threads.append(threading.Thread(target=self.audio_worker, args=("mic", mic_frames, ev, errors), daemon=True))
        if self.sys_audio_var.get():
            ev = threading.Event()
            ready_events.append(ev)
            threads.append(threading.Thread(target=self.audio_worker, args=("sys", sys_frames, ev, errors), daemon=True))
            threads.append(threading.Thread(target=self.silence_worker, daemon=True))

        for t in threads:
            t.start()
        for ev in ready_events:
            ev.wait(timeout=3)

        frames_written = 0
        video_error = None

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                width = monitor["width"] - (monitor["width"] % 2)    # even dims for h264
                height = monitor["height"] - (monitor["height"] % 2)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(temp_video_file, fourcc, FPS, (width, height))

                cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) if self.facecam_var.get() else None
                cam_h, cam_w, padding = 180, 240, 20

                # Start the clock only after audio streams are live
                self.start_time = time.time()
                self.paused_duration = 0

                try:
                    while self.is_recording:
                        if self.is_paused:
                            time.sleep(0.05)
                            continue

                        frame = np.array(sct.grab(monitor))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)[:height, :width]
                        frame = np.ascontiguousarray(frame)

                        # Real system cursor
                        self.cursor.draw(frame, monitor["left"], monitor["top"])

                        # Facecam overlay
                        if cam is not None and cam.isOpened():
                            ret, cam_frame = cam.read()
                            if ret:
                                cam_frame = cv2.flip(cam_frame, 1)
                                cam_frame = cv2.resize(cam_frame, (cam_w, cam_h))
                                pos = self.position_var.get()
                                if pos == "Bottom-Right":
                                    y1, x1 = height - cam_h - padding, width - cam_w - padding
                                elif pos == "Bottom-Left":
                                    y1, x1 = height - cam_h - padding, padding
                                elif pos == "Top-Right":
                                    y1, x1 = padding, width - cam_w - padding
                                else:
                                    y1, x1 = padding, padding
                                frame[y1:y1 + cam_h, x1:x1 + cam_w] = cam_frame
                                cv2.rectangle(frame, (x1, y1), (x1 + cam_w, y1 + cam_h), (255, 255, 255), 2)

                        # Write as many frames as real time requires (keeps A/V in sync)
                        elapsed = time.time() - self.start_time - self.paused_duration
                        target = int(elapsed * FPS) + 1
                        while frames_written < target:
                            out.write(frame)
                            frames_written += 1

                        # Pace the loop so we don't starve the audio threads
                        elapsed = time.time() - self.start_time - self.paused_duration
                        sleep_for = frames_written / FPS - elapsed
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                finally:
                    if cam is not None:
                        cam.release()
                    out.release()
        except Exception as e:
            video_error = f"Video capture error: {e}"
            self.is_recording = False

        # Let audio threads finish their last block
        for t in threads:
            t.join(timeout=2)

        # ---- Build audio track aligned to video length ----
        duration = frames_written / FPS
        n_samples = int(duration * SAMPLE_RATE)
        has_audio = False
        tracks = []
        try:
            if mic_frames:
                tracks.append(fit_length(to_stereo(np.concatenate(mic_frames, axis=0)), n_samples))
            if sys_frames:
                tracks.append(fit_length(to_stereo(np.concatenate(sys_frames, axis=0)), n_samples))
            if tracks and n_samples > 0:
                mixed = tracks[0] if len(tracks) == 1 else (tracks[0] * 0.9 + tracks[1] * 0.9)
                mixed = np.clip(mixed, -1.0, 1.0)
                sf.write(temp_audio_file, mixed, SAMPLE_RATE)
                has_audio = True
        except Exception as e:
            errors.append(f"Audio mixing: {e}")

        # ---- Mux video + audio ----
        mux_error = None
        try:
            if frames_written == 0:
                raise RuntimeError("No video frames were captured.")
            video_clip = VideoFileClip(temp_video_file)
            if has_audio:
                audio_clip = AudioFileClip(temp_audio_file)
                final_clip = video_clip.with_audio(audio_clip) if MOVIEPY_V2 else video_clip.set_audio(audio_clip)
                final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac",
                                           fps=FPS, logger=None)
                audio_clip.close()
            else:
                video_clip.write_videofile(output_filename, codec="libx264", fps=FPS, logger=None)
            video_clip.close()
        except Exception as e:
            mux_error = f"Muxing error: {e}"

        shutil.rmtree(tmp_dir, ignore_errors=True)

        all_errors = [e for e in [video_error, mux_error] if e] + errors
        self.root.after(0, self.recording_finished, output_filename, all_errors, mux_error is None and video_error is None)

    def recording_finished(self, filename, errors, success):
        self.is_recording = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Pause", bg="#FF9800")
        self.stop_btn.config(state=tk.DISABLED)
        self.filename_entry.config(state=tk.NORMAL)
        self.timer_label.config(text="00:00:00")
        self.status_label.config(text="Status: Idle", fg="gray")

        if success:
            msg = f"Recording Saved Successfully as:\n{filename}"
            if errors:
                msg += "\n\nWarnings:\n" + "\n".join(errors)
            messagebox.showinfo("Success", msg)
        else:
            messagebox.showerror("Error", "Recording failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    root = tk.Tk()
    app = DiRecorderApp(root)
    root.mainloop()
