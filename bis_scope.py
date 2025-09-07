#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIS Vista EEG "oscilloscope" viewer — folder-autodetect + midline offset + extra SPA fields
"""

from __future__ import annotations
import os
import sys
import math
from pathlib import Path
import csv
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# -------------------- Config --------------------
FS_EEG = 128.0  # Hz, BIS Vista R2A typical
UV_PER_STEP = 1675.42688 / 32767.0  # ≈ 0.0511 µV/step
DEFAULT_WINDOW_S = 10.0
SHOW_SPA_SUMMARY = True  # set False to disable console summary

# -------------------- SPA parsing --------------------

def parse_spa_with_headers(spa_path: Path):
    """
    Parse SPA file capturing header line with column names.
    Returns:
        headers: List[str] or None
        timestamps: List[str]
        BIS: (N,3) float with negatives→NaN
        EMG, SEF, POW, MF, SR, ST: (N,) float with negatives→NaN
    """
    headers: Optional[List[str]] = None
    timestamps: List[str] = []
    bis_cols = [[], [], []]
    sef_vals = []
    mf_vals = []
    pow_vals = []
    emg_vals = []
    sr_vals = []
    st_vals = []

    def _to_float_or_nan(s: str) -> float:
        try:
            v = float(s)
            return math.nan if v < 0 else v
        except Exception:
            return math.nan

    try:
        with spa_path.open("r", encoding="utf-8", errors="replace") as f:
            header1 = f.readline()
            header2 = f.readline()
            headers = [h.strip() for h in header2.strip().split('|')] if header2 else None

            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if not row:
                    continue
                if len(row) < 54:
                    row = row + [''] * (54 - len(row))

                # fixed column indices, python 0-based
                col_t   = 0
                col_bis = [11, 25, 39] #Channel 1, Channel 2, Channel 1 and 2
                col_sef = 36  # spectral edge frequency (all from the averaged channel 1 and 2)
                col_mf  = 37  # median frequency (all from the averaged channel 1 and 2)
                col_pow = 42  # power (all from the averaged channel 1 and 2)
                col_emg = 43  
                col_sr  = 35  # burst suppression ratio (all from the averaged channel 1 and 2)
                col_st  = 48  # suppression time (all from the averaged channel 1 and 2)

                timestamps.append(row[col_t])
                for i, idx in enumerate(col_bis):
                    bis_cols[i].append(_to_float_or_nan(row[idx]))
                sef_vals.append(_to_float_or_nan(row[col_sef]))
                mf_vals.append(_to_float_or_nan(row[col_mf]))
                pow_vals.append(_to_float_or_nan(row[col_pow]))
                emg_vals.append(_to_float_or_nan(row[col_emg]))
                sr_vals.append(_to_float_or_nan(row[col_sr]))
                st_vals.append(_to_float_or_nan(row[col_st]))

        import numpy as _np
        BIS = _np.column_stack(bis_cols) if len(bis_cols[0]) else _np.empty((0, 3))
        SEF = _np.asarray(sef_vals, dtype=float)
        MF  = _np.asarray(mf_vals, dtype=float)
        POW = _np.asarray(pow_vals, dtype=float)
        EMG = _np.asarray(emg_vals, dtype=float)
        SR  = _np.asarray(sr_vals, dtype=float)
        ST  = _np.asarray(st_vals, dtype=float)
        return headers, timestamps, BIS, EMG, SEF, POW, MF, SR, ST
    except FileNotFoundError:
        import numpy as _np
        return None, [], _np.empty((0,3)), _np.array([]), _np.array([]), _np.array([]), _np.array([]), _np.array([]), _np.array([])


def print_spa_summary(headers, timestamps, BIS, EMG, SEF, POW, MF, SR, ST):
    """Console summary of SPA columns and first-non-NaN snapshot."""
    if not SHOW_SPA_SUMMARY:
        return
    print("\\n=== SPA SUMMARY ===")
    if headers:
        print(f"Columns ({len(headers)}): " + ", ".join(headers))
    else:
        print("No header names available.")
    if timestamps:
        print(f"Rows: {len(timestamps)}, start: {timestamps[0]}, end: {timestamps[-1]}")
    else:
        print("No SPA rows parsed.")

    idx = 0
    import numpy as _np
    while idx < len(timestamps):
        parts = []
        if BIS.shape[0] > idx and not _np.all(_np.isnan(BIS[idx])):
            parts.append("BIS=" + "/".join(["—" if _np.isnan(x) else f"{x:.0f}" for x in BIS[idx]]))
        if SEF.size > idx and not _np.isnan(SEF[idx]):
            parts.append(f"SEF={SEF[idx]:.1f}")
        if MF.size > idx and not _np.isnan(MF[idx]):
            parts.append(f"MF={MF[idx]:.1f}")
        if POW.size > idx and not _np.isnan(POW[idx]):
            parts.append(f"POW={POW[idx]:.1f}")
        if EMG.size > idx and not _np.isnan(EMG[idx]):
            parts.append(f"EMG={EMG[idx]:.1f}")
        if SR.size > idx and not _np.isnan(SR[idx]):
            parts.append(f"BSR={SR[idx]:.1f}")
        if ST.size > idx and not _np.isnan(ST[idx]):
            parts.append(f"SuppTime={ST[idx]:.1f}")
        if parts:
            print("First non-NaN snapshot: " + ", ".join(parts))
            break
        idx += 1
    if idx == len(timestamps):
        print("All SPA values appear to be NaN or SPA missing.")
    print("===================\\n")

# -------------------- R2A loading --------------------

def load_r2a_uV(r2a_path: Path) -> np.ndarray:
    """Load two-channel interleaved int16 R2A and scale to microvolts (shape 2 x N, float64)."""
    raw = np.fromfile(r2a_path, dtype=np.int16)
    if raw.size % 2 != 0:
        raw = raw[: raw.size - 1]
    eeg = raw.reshape(-1, 2).T
    return eeg.astype(np.float64) * UV_PER_STEP

def parse_start_time(timestamps: List[str]) -> Optional[datetime]:
    if not timestamps:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%m/%d/%y %H:%M:%S", "%d/%m/%y %H:%M:%S"):
        try:
            return datetime.strptime(timestamps[0].strip(), fmt)
        except Exception:
            continue
    return None

# -------------------- GUI Application --------------------

class EEGScopeApp:
    def __init__(self, root: tk.Tk, r2a_path: Path, spa_path: Optional[Path]):
        self.root = root
        self.root.title(f"BIS Vista EEG Scope — {r2a_path.name}")

        self.EEGuV = load_r2a_uV(r2a_path)
        self.Nsamp = self.EEGuV.shape[1]
        self.total_s = self.Nsamp / FS_EEG

        self.headers = None
        self.timestamps: List[str] = []
        self.BIS = np.empty((0,3))
        self.EMG = np.array([])
        self.SEF = np.array([])
        self.POW = np.array([])
        self.MF  = np.array([])
        self.SR  = np.array([])
        self.ST  = np.array([])
        self.start_dt: Optional[datetime] = None

        if spa_path and spa_path.exists():
            (self.headers, self.timestamps, self.BIS, self.EMG, self.SEF, self.POW,
             self.MF, self.SR, self.ST) = parse_spa_with_headers(spa_path)
            self.start_dt = parse_start_time(self.timestamps)
        else:
            print("Note: Matching SPA not found. Proceeding without SPA values.")
            self.start_dt = None

        self.window_s = DEFAULT_WINDOW_S
        self.current_t = 0.0
        self.playing = False
        self.channel_mode = tk.StringVar(value="both")
        self.scale_uv = tk.DoubleVar(value=200.0)
        self.offset_uv = tk.DoubleVar(value=0.0)

        self._build_layout()
        self._update_plot()
        self._update_status_labels()

    def _build_layout(self):
        ctrl = ttk.Frame(self.root, padding=6)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(ctrl, text="Play", command=self._toggle_play).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Pause", command=self._pause).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="Scale ± (µV):").pack(side=tk.LEFT, padx=(16, 4))
        self.scale_slider = ttk.Scale(ctrl, from_=10, to=1000, orient=tk.HORIZONTAL,
                                      command=lambda e: self._on_scale_change(), length=200)
        self.scale_slider.set(self.scale_uv.get())
        self.scale_slider.pack(side=tk.LEFT, padx=4)
        self.scale_val_label = ttk.Label(ctrl, text=f"{int(self.scale_uv.get())}")
        self.scale_val_label.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="Midline offset (µV):").pack(side=tk.LEFT, padx=(8, 4))
        self.offset_slider = ttk.Scale(ctrl, from_=-1000, to=1000, orient=tk.HORIZONTAL,
                                       command=lambda e: self._on_offset_change(), length=220)
        self.offset_slider.set(self.offset_uv.get())
        self.offset_slider.pack(side=tk.LEFT, padx=4)
        self.offset_val_label = ttk.Label(ctrl, text=f"{int(self.offset_uv.get())}")
        self.offset_val_label.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="Window (s):").pack(side=tk.LEFT, padx=(8, 4))
        self.win_slider = ttk.Scale(ctrl, from_=2, to=30, orient=tk.HORIZONTAL,
                                    command=lambda e: self._on_window_change(), length=160)
        self.win_slider.set(DEFAULT_WINDOW_S)
        self.win_slider.pack(side=tk.LEFT, padx=4)
        self.win_val_label = ttk.Label(ctrl, text=f"{int(DEFAULT_WINDOW_S)} s")
        self.win_val_label.pack(side=tk.LEFT, padx=(4, 12))

        chan_frame = ttk.LabelFrame(self.root, text="Channels", padding=6)
        chan_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0,6))
        ttk.Radiobutton(chan_frame, text="Ch 1", value="ch1",
                        variable=self.channel_mode, command=self._update_plot).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(chan_frame, text="Ch 2", value="ch2",
                        variable=self.channel_mode, command=self._update_plot).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(chan_frame, text="Both", value="both",
                        variable=self.channel_mode, command=self._update_plot).pack(side=tk.LEFT, padx=6)

        fig = Figure(figsize=(10, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Time")
               self.ax.set_ylabel("EEG (µV)")
        self.line1, = self.ax.plot([], [], lw=0.8, label="Ch1")
        self.line2, = self.ax.plot([], [], lw=0.8, label="Ch2")
        self.ax.legend(loc="upper right")
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        ts_frame = ttk.Frame(self.root, padding=6)
        ts_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(ts_frame, text="Position:").pack(side=tk.LEFT)
        self.time_slider = ttk.Scale(ts_frame, from_=0.0, to=max(0.0, self.total_s - self.window_s),
                                     orient=tk.HORIZONTAL, command=lambda e: self._on_time_slider(),
                                     length=800)
        self.time_slider.set(0.0)
        self.time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.pos_label = ttk.Label(ts_frame, text="00:00 / 00:00")
        self.pos_label.pack(side=tk.LEFT, padx=8)

        stat = ttk.LabelFrame(self.root, text="Current Processed Values (SPA)", padding=6)
        stat.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        self.lbl_time = ttk.Label(stat, text="Time: —")
        self.lbl_time.pack(side=tk.LEFT, padx=10)
        self.lbl_bis = ttk.Label(stat, text="BIS: —")
        self.lbl_bis.pack(side=tk.LEFT, padx=10)
        self.lbl_sef = ttk.Label(stat, text="SEF: —")
        self.lbl_sef.pack(side=tk.LEFT, padx=10)
        self.lbl_mf = ttk.Label(stat, text="MF: —")
        self.lbl_mf.pack(side=tk.LEFT, padx=10)
        self.lbl_pow = ttk.Label(stat, text="POW: —")
        self.lbl_pow.pack(side=tk.LEFT, padx=10)
        self.lbl_emg = ttk.Label(stat, text="EMG: —")
        self.lbl_emg.pack(side=tk.LEFT, padx=10)
        self.lbl_bsr = ttk.Label(stat, text="BSR: —")
        self.lbl_bsr.pack(side=tk.LEFT, padx=10)
        self.lbl_st  = ttk.Label(stat, text="Suppression Time: —")
        self.lbl_st.pack(side=tk.LEFT, padx=10)

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._play_loop()

    def _pause(self):
        self.playing = False

    def _on_scale_change(self):
        self.scale_uv.set(float(self.scale_slider.get()))
        self.scale_val_label.config(text=f"{int(self.scale_uv.get())}")
        self._update_plot()

    def _on_offset_change(self):
        self.offset_uv.set(float(self.offset_slider.get()))
        self.offset_val_label.config(text=f"{int(self.offset_uv.get())}")
        self._update_plot()

    def _on_window_change(self):
        self.window_s = max(2.0, min(30.0, float(self.win_slider.get())))
        self.win_val_label.config(text=f"{int(self.window_s)} s")
        self.time_slider.config(to=max(0.0, self.total_s - self.window_s))
        self._update_plot()

    def _on_time_slider(self):
        self.current_t = float(self.time_slider.get())
        self._update_plot()
        self._update_status_labels()

    def _play_loop(self):
        if not self.playing:
            return
        step = 0.05
        self.current_t += step
        end_t = self.total_s - self.window_s
        if self.current_t > end_t:
            self.current_t = end_t
            self.playing = False
        self.time_slider.set(self.current_t)
        self._update_plot()
        self._update_status_labels()
        self.root.after(int(step * 1000), self._play_loop)

    def _update_plot(self):
        t0 = self.current_t
        t1 = t0 + self.window_s
        i0 = max(0, int(t0 * FS_EEG))
        i1 = min(self.EEGuV.shape[1], int(t1 * FS_EEG))
        if i1 <= i0:
            i1 = min(self.EEGuV.shape[1], i0 + 2)

        x = np.linspace(t0, t1, max(2, i1 - i0), endpoint=False)

        mode = self.channel_mode.get()
        y1 = self.EEGuV[0, i0:i1]
        y2 = self.EEGuV[1, i0:i1]

        if mode == "ch1":
            self.line1.set_data(x, y1)
            self.line2.set_data([], [])
        elif mode == "ch2":
            self.line1.set_data([], [])
            self.line2.set_data(x, y2)
        else:
            self.line1.set_data(x, y1)
            self.line2.set_data(x, y2)

        half = float(self.scale_uv.get())
        offset = float(self.offset_uv.get())
        ymin, ymax = offset - half, offset + half
        if ymin >= ymax:
            ymin, ymax = offset - 10.0, offset + 10.0
        self.ax.set_xlim(t0, t1)
        self.ax.set_ylim(ymin, ymax)

        if self.start_dt is not None and self.timestamps:
            abs_t0 = self.start_dt + timedelta(seconds=t0)
            abs_t1 = self.start_dt + timedelta(seconds=t1)
            self.ax.set_xlabel(f"Time ({abs_t0.strftime('%H:%M:%S')} → {abs_t1.strftime('%H:%M:%S')})")
        else:
            self.ax.set_xlabel("Time (s)")

        self.canvas.draw_idle()
        self.pos_label.config(text=f"{self._fmt_time(t0)} / {self._fmt_time(self.total_s)}")

    def _update_status_labels(self):
        if not self.timestamps:
            self.lbl_time.config(text="Time: — (SPA unavailable)")
            for lbl in (self.lbl_bis, self.lbl_sef, self.lbl_mf, self.lbl_pow, self.lbl_emg, self.lbl_bsr, self.lbl_st):
                lbl.config(text=lbl.cget("text").split(':')[0] + ": —")
            return

        sec = int(self.current_t)
        idx = min(max(0, sec), len(self.timestamps) - 1)

        if self.start_dt is not None:
            cur_abs = self.start_dt + timedelta(seconds=sec)
            self.lbl_time.config(text=f"Time: {cur_abs.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            self.lbl_time.config(text=f"Time: {self.timestamps[idx]}")

        bis_str = self._fmt_triplet(self.BIS[idx]) if self.BIS.shape[0] > idx else "—"
        sef = self._safe_val(self.SEF, idx)
        mf  = self._safe_val(self.MF, idx)
        powv = self._safe_val(self.POW, idx)
        emg = self._safe_val(self.EMG, idx)
        bsr = self._safe_val(self.SR, idx)
        st  = self._safe_val(self.ST, idx)

        self.lbl_bis.config(text=f"BIS: {bis_str}")
        self.lbl_sef.config(text=f"SEF: {sef}")
        self.lbl_mf.config(text=f"MF: {mf}")
        self.lbl_pow.config(text=f"POW: {powv}")
        self.lbl_emg.config(text=f"EMG: {emg}")
        self.lbl_bsr.config(text=f"BSR: {bsr}")
        self.lbl_st.config(text=f"Suppression Time: {st}")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = int(max(0, seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    @staticmethod
    def _fmt_triplet(v: np.ndarray) -> str:
        def f(x):
            return "—" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.0f}"
        try:
            return f"{f(v[0])}/{f(v[1])}/{f(v[2])}"
        except Exception:
            return "—"

    @staticmethod
    def _safe_val(arr: np.ndarray, idx: int) -> str:
        try:
            x = arr[idx]
            import math as _m
            return "—" if (x is None or (_m.isnan(x))) else f"{x:.1f}"
        except Exception:
            return "—"

# -------------------- Entry --------------------

def choose_r2a_in_cwd() -> Path:
    """Find R2A files in CWD and prompt user if more than one exists."""
    cwd = Path.cwd()
    r2as = sorted(cwd.glob("*.r2a"))
    if not r2as:
        print("No .r2a files found in the current folder.")
        print("Tip: open a terminal in the study folder and run: python bis_scope.py")
        sys.exit(1)
    if len(r2as) == 1:
        print(f"Found R2A: {r2as[0].name}")
        return r2as[0]

    print("Multiple .r2a files found:")
    for i, p in enumerate(r2as, 1):
        print(f"  [{i}] {p.name}")
    while True:
        sel = input(f"Select 1..{len(r2as)}: ").strip()
        if sel.isdigit():
            k = int(sel)
            if 1 <= k <= len(r2as):
                return r2as[k-1]
        print("Invalid selection. Try again.")

def main():
    if len(sys.argv) >= 2:
        r2a_path = Path(sys.argv[1])
        if r2a_path.is_dir():
            print(f"Using directory: {r2a_path}")
            os.chdir(r2a_path)
            r2a_path = choose_r2a_in_cwd()
        elif r2a_path.suffix.lower() == ".r2a":
            pass
        else:
            print("Argument must be a study folder or an .r2a file path. Falling back to CWD.")
            r2a_path = choose_r2a_in_cwd()
    else:
        r2a_path = choose_r2a_in_cwd()

    spa_path = r2a_path.with_suffix(".spa")
    if not spa_path.exists():
        print(f"Matching SPA not found: {spa_path.name}. Proceeding without SPA.")
        spa_path = None

    if spa_path is not None:
        (headers, timestamps, BIS, EMG, SEF, POW, MF, SR, ST) = parse_spa_with_headers(spa_path)
        print_spa_summary(headers, timestamps, BIS, EMG, SEF, POW, MF, SR, ST)
    else:
        if SHOW_SPA_SUMMARY:
            print("\\n=== SPA SUMMARY ===\\nSPA missing. No variables to display.\\n===================\\n")

    root = tk.Tk()
    try:
        app = EEGScopeApp(root, r2a_path, spa_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load study: {e}")
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
