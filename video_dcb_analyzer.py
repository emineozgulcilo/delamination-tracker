"""
DCB Video Analiz Araci
----------------------
WMC overlay metninden OCR ile P ve delta okur,
compliance yontemiyle a hesaplar,
zaman serileri grafigi gosterir ve CSV kaydeder.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import re, csv, os, sys, threading

# Matplotlib
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# EasyOCR
try:
    import easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ─── Regex ──────────────────────────────────────────────────────────────
_PAT_DISP  = re.compile(r'([+-]?\d+\.?\d+)\s*[\[\(]mm[\]\)]', re.I)
_PAT_FORCE = re.compile(r'([+-]?\d+\.?\d*)\s*[\[\(][Nn][J\]\)]', re.I)

def parse_wmc_line(text):
    """WMC metninden (delta_mm, force_N) dondur."""
    dm = _PAT_DISP.findall(text)
    fm = _PAT_FORCE.findall(text)
    if not dm or not fm:
        return None
    try:
        return float(dm[0]), float(fm[0])
    except Exception:
        return None

# ─── Renkler ─────────────────────────────────────────────────────────────
BG    = "#1a1a2e"
PANEL = "#16213e"
TEXT  = "#eaeaea"
SUB   = "#777"
C1    = "#e94560"
C2    = "#2d7dd2"
C3    = "#27ae60"
C4    = "#f39c12"
FONT  = ("Segoe UI", 10)
FONTB = ("Segoe UI", 10, "bold")
MONO  = ("Consolas", 11)


class DCBVideoAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DCB Video Analiz — WMC OCR")
        self.geometry("1200x740")
        self.minsize(900, 580)
        self.configure(bg=BG)

        self._cap    = None
        self._total  = 0
        self._fps    = 25.0
        self._reader = None
        self._running = False

        # Parametreler
        self._a0_mm   = tk.DoubleVar(value=50.0)
        self._b_mm    = tk.DoubleVar(value=20.0)
        self._h_mm    = tk.DoubleVar(value=2.0)
        self._interval = tk.IntVar(value=5)
        self._strip_pct = tk.IntVar(value=15)

        # Veri
        self._rows = []   # [time_s, delta_mm, P_N, a_mm]

        self._build()

    # ─── UI ──────────────────────────────────────────────────────────────

    def _build(self):
        left = tk.Frame(self, bg=PANEL, width=270)
        left.pack(side="left", fill="y", padx=(6,3), pady=6)
        left.pack_propagate(False)
        self._build_left(left)

        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0,6), pady=6)
        self._build_right(right)

    def _sec(self, p, t):
        tk.Label(p, text=t, font=FONTB, bg=PANEL, fg=C2
                 ).pack(anchor="w", padx=8, pady=(10,0))
        tk.Frame(p, bg="#334", height=1).pack(fill="x", padx=8, pady=2)

    def _row(self, p, label, var, lo, hi, step=1):
        r = tk.Frame(p, bg=PANEL); r.pack(fill="x", padx=8, pady=1)
        tk.Label(r, text=label, font=FONT, bg=PANEL, fg=TEXT,
                 width=20, anchor="w").pack(side="left")
        tk.Spinbox(r, from_=lo, to=hi, increment=step, textvariable=var,
                   width=7, font=MONO, bg="#223", fg=TEXT, relief="flat"
                   ).pack(side="left")

    def _btn(self, p, text, cmd, color=C2, pady=4):
        b = tk.Button(p, text=text, command=cmd, font=FONT, bg=color, fg="white",
                      relief="flat", activebackground=color, cursor="hand2",
                      padx=8, pady=pady, bd=0)
        b.pack(fill="x", padx=8, pady=2)
        return b

    def _build_left(self, p):
        self._sec(p, "1. Video")
        self._btn(p, "Video Yukle (.mp4 / .avi)", self._load, C2)
        self._vid_lbl = tk.Label(p, text="—", font=("Segoe UI", 8),
                                   bg=PANEL, fg=SUB, wraplength=245, justify="left")
        self._vid_lbl.pack(anchor="w", padx=10)

        self._sec(p, "2. Numune Boyutlari")
        self._row(p, "a0  — Bas. catlak (mm)", self._a0_mm, 1, 300, 1)
        self._row(p, "b   — Genislik (mm)",     self._b_mm, 1, 200, 1)
        self._row(p, "h   — Kol kalinligi (mm)",self._h_mm, 0.1, 50, 0.1)

        self._sec(p, "3. Analiz Parametreleri")
        self._row(p, "Her N kare",         self._interval,   1,  100, 1)
        self._row(p, "OCR bant yuksekligi %", self._strip_pct, 5, 30, 1)

        self._sec(p, "4. Baslat")
        if not HAS_OCR:
            tk.Label(p, text="⚠ easyocr yuklu degil!\npip install easyocr",
                     font=FONT, bg=PANEL, fg=C1, wraplength=240
                     ).pack(padx=10, pady=4)
        if not HAS_MPL:
            tk.Label(p, text="⚠ matplotlib yuklu degil!\npip install matplotlib",
                     font=FONT, bg=PANEL, fg=C1, wraplength=240
                     ).pack(padx=10, pady=4)
        self._btn(p, "▶  ANALİZİ BAŞLAT", self._start, C2, pady=8)
        self._btn(p, "⛔  Durdur (ESC)",  self._stop,  C1)
        self.bind("<Escape>", lambda e: self._stop())

        self._pb = ttk.Progressbar(p, maximum=100, mode="determinate")
        self._pb.pack(fill="x", padx=8, pady=4)

        self._sec(p, "Ozet")
        self._sum_lbl = tk.Label(p, text="—", font=("Segoe UI", 9),
                                   bg=PANEL, fg=TEXT, justify="left", wraplength=245)
        self._sum_lbl.pack(anchor="w", padx=10, pady=4)

        self._sec(p, "Kaydet")
        self._btn(p, "CSV Kaydet", self._save_csv, C3)

        self._stat = tk.Label(p, text="Hazir.", font=("Segoe UI", 9, "bold"),
                               bg=PANEL, fg=C4, wraplength=250)
        self._stat.pack(anchor="w", padx=10, pady=6)

    def _build_right(self, p):
        if not HAS_MPL:
            tk.Label(p, text="matplotlib yuklu degil",
                     font=FONTB, bg=BG, fg=C1).pack(expand=True)
            return

        self._fig = Figure(facecolor="#0d1117", tight_layout=True)
        self._ax  = [
            self._fig.add_subplot(3, 1, 1),
            self._fig.add_subplot(3, 1, 2),
            self._fig.add_subplot(3, 1, 3),
        ]
        self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=p)
        self._canvas_mpl.get_tk_widget().pack(fill="both", expand=True)
        self._init_axes()

    def _init_axes(self):
        titles  = ["Catlak Uzunlugu  a  (mm)  vs  Zaman (s)",
                   "Yuk  P  (N)  vs  Zaman (s)",
                   "Yerdeğiştirme  δ  (mm)  vs  Zaman (s)"]
        ylabels = ["a  (mm)", "P  (N)", "δ  (mm)"]
        colors  = ["#27ae60", "#e94560", "#2d7dd2"]
        for ax, title, yl, col in zip(self._ax, titles, ylabels, colors):
            ax.set_facecolor("#0d1117")
            ax.set_title(title, color="#aaa", fontsize=9, pad=3)
            ax.set_ylabel(yl, color=col, fontsize=8)
            ax.set_xlabel("Zaman (s)", color="#666", fontsize=8)
            ax.tick_params(colors="#666", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#333")
        self._canvas_mpl.draw()

    # ─── Video ───────────────────────────────────────────────────────────

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tumu", "*.*")])
        if not path: return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Video acilamadi."); return
        if self._cap: self._cap.release()
        self._cap   = cap
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._rows  = []
        name = os.path.basename(path)
        self._vid_lbl.config(
            text=f"{name}\n{self._total} kare | {self._fps:.1f} fps | "
                 f"{self._total/self._fps:.1f}s")
        self._set_status("Video yuklendi.")

    # ─── Analiz ──────────────────────────────────────────────────────────

    def _start(self):
        if not self._cap:
            messagebox.showwarning("", "Once video yukleyin."); return
        if not HAS_OCR:
            messagebox.showerror("", "easyocr yuklu degil.\npip install easyocr"); return
        if not HAS_MPL:
            messagebox.showerror("", "matplotlib yuklu degil.\npip install matplotlib"); return

        self._rows   = []
        self._running = True
        self._pb["value"] = 0
        self._sum_lbl.config(text="—")
        self._set_status("OCR okuyucu baslatiliyor…")

        # EasyOCR'yi thread'de yükle, sonra döngü başlat
        def _init_and_run():
            if self._reader is None:
                self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self.after(0, self._run_loop, 0)

        threading.Thread(target=_init_and_run, daemon=True).start()

    def _ocr_frame(self, frame):
        """WMC metin bandını OCR ile oku, (delta_mm, P_N) döndür."""
        h = frame.shape[0]
        strip = frame[:int(h * self._strip_pct.get() / 100), :]
        gray  = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        big   = cv2.resize(th, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        results = self._reader.readtext(big, detail=0)
        text    = " ".join(results)
        return parse_wmc_line(text)

    def _run_loop(self, idx):
        if not self._running:
            return

        if idx >= self._total:
            self._finish()
            return

        # Kareyi oku
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if ok:
            parsed = self._ocr_frame(frame)
            if parsed:
                delta_mm, P_N = parsed
                t_s = idx / self._fps
                self._rows.append((t_s, delta_mm, P_N))
                self._update_plot()

        # İlerleme
        pct = int(idx / max(self._total-1, 1) * 100)
        self._pb["value"] = pct
        n_frames_done = len(self._rows)
        self._set_status(
            f"Kare {idx}/{self._total} — {n_frames_done} veri noktasi")

        # Sonraki kare
        self.after(1, self._run_loop, idx + self._interval.get())

    def _finish(self):
        self._running = False
        self._pb["value"] = 100
        n = len(self._rows)
        if n == 0:
            self._set_status("Hic veri bulunamadi! OCR calismiyor olabilir.")
            return

        # Özet
        arr = np.array(self._rows)   # (n, 3): t, delta, P
        t_col = arr[:,0]; d_col = arr[:,1]; p_col = arr[:,2]
        a_col = self._calc_a(d_col, p_col)

        delta_max = d_col.max()
        P_max     = p_col.max()
        a_max     = a_col.max()
        self._sum_lbl.config(
            text=f"Toplam nokta: {n}\n"
                 f"Max δ: {delta_max:.3f} mm\n"
                 f"Max P: {P_max:.2f} N\n"
                 f"Max a: {a_max:.1f} mm\n"
                 f"Sure: {t_col[-1]:.1f} s")
        self._update_plot()
        self._set_status(f"Tamamlandi! {n} nokta. CSV kaydedin.")

    def _calc_a(self, delta_arr, P_arr):
        """Compliance tabanli crack uzunlugu tahmini."""
        a0  = self._a0_mm.get()
        C   = np.where(P_arr > 0.1, delta_arr / P_arr, np.nan)
        # İlk geçerli compliance değeri = C0
        valid = np.where(~np.isnan(C) & (C > 0))[0]
        if len(valid) == 0:
            return np.full_like(delta_arr, a0)
        C0 = C[valid[0]]
        ratio = C / C0
        ratio = np.where(ratio > 0, ratio, np.nan)
        a = a0 * ratio ** (1.0/3.0)
        a = np.where(np.isnan(a), a0, a)
        return a

    def _update_plot(self):
        if not HAS_MPL or not self._rows:
            return
        arr  = np.array(self._rows)
        t    = arr[:,0]
        delta= arr[:,1]
        P    = arr[:,2]
        a    = self._calc_a(delta, P)

        colors = ["#27ae60", "#e94560", "#2d7dd2"]
        data   = [a, P, delta]
        labels = ["a (mm)", "P (N)", "δ (mm)"]

        for ax, col, d, yl in zip(self._ax, colors, data, labels):
            ax.clear()
            ax.set_facecolor("#0d1117")
            ax.plot(t, d, color=col, linewidth=1.2)
            ax.set_ylabel(yl, color=col, fontsize=8)
            ax.set_xlabel("Zaman (s)", color="#666", fontsize=8)
            ax.tick_params(colors="#666", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#333")
            if len(d) > 0:
                ax.set_xlim(0, max(t[-1], 1))

        self._fig.tight_layout(pad=0.8)
        self._canvas_mpl.draw()

    def _stop(self):
        self._running = False
        self._set_status(f"Durduruldu. {len(self._rows)} nokta.")

    # ─── CSV ─────────────────────────────────────────────────────────────

    def _save_csv(self):
        if not self._rows:
            messagebox.showwarning("", "Hic veri yok."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path: return
        arr = np.array(self._rows)
        t_col  = arr[:,0]; d_col = arr[:,1]; p_col = arr[:,2]
        a_col  = self._calc_a(d_col, p_col)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "delta_mm", "P_N", "a_mm"])
            for t, d, p, a in zip(t_col, d_col, p_col, a_col):
                w.writerow([round(t,4), round(d,4), round(p,4), round(a,4)])
        messagebox.showinfo("Kaydedildi", path)

    def _set_status(self, msg):
        self._stat.config(text=msg)
        self.update_idletasks()


if __name__ == "__main__":
    app = DCBVideoAnalyzer()
    app.mainloop()
