"""
ASTM D5528 - Mode I Interlaminar Fracture Toughness Analysis System
Double Cantilever Beam (DCB) Test
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
from video_analyzer import VideoAnalyzer
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import stats
import os

# ─────────────────────────────────────────────────────────────
# Core calculations (ASTM D5528-21)
# ─────────────────────────────────────────────────────────────

def calc_compliance(delta, P):
    """C = δ/P  (mm/N)"""
    return np.array(delta) / np.array(P)

def method_MBT(P, delta, a, b):
    """
    Modified Beam Theory (MBT)
    GIc = 3Pδ / (2b(a+|Δ|))
    Δ from linear fit of C^(1/3) vs a
    """
    P = np.array(P, dtype=float)
    delta = np.array(delta, dtype=float)
    a = np.array(a, dtype=float)
    C = delta / P
    C13 = C ** (1/3)
    slope, intercept, r, _, _ = stats.linregress(a, C13)
    Delta = -intercept / slope   # x-intercept → |Δ|
    GIc = (3 * P * delta) / (2 * b * (a + abs(Delta)))
    return GIc, Delta, slope, intercept, r**2

def method_CC(P, delta, a, b):
    """
    Compliance Calibration (CC)
    log(C) vs log(a) → slope n
    GIc = nPδ / (2ba)
    """
    P = np.array(P, dtype=float)
    delta = np.array(delta, dtype=float)
    a = np.array(a, dtype=float)
    C = delta / P
    logC = np.log(C)
    loga = np.log(a)
    slope, intercept, r, _, _ = stats.linregress(loga, logC)
    n = slope
    GIc = (n * P * delta) / (2 * b * a)
    return GIc, n, slope, intercept, r**2

def method_MCC(P, delta, a, b, h):
    """
    Modified Compliance Calibration (MCC)
    a/h vs C^(1/3) → slope A1
    GIc = 3P²C^(2/3) / (2A1*b*h)
    """
    P = np.array(P, dtype=float)
    delta = np.array(delta, dtype=float)
    a = np.array(a, dtype=float)
    C = delta / P
    C13 = C ** (1/3)
    ah = a / h
    slope, intercept, r, _, _ = stats.linregress(C13, ah)
    A1 = slope
    GIc = (3 * P**2 * C**(2/3)) / (2 * A1 * b * h)
    return GIc, A1, slope, intercept, r**2

def large_deflection_correction(delta, a, h):
    """ASTM D5528 Eq. (A2.1) large deflection correction factor F"""
    t = (delta / (2 * a)) ** 2
    F = 1 - (3/10) * t - (33/280) * t**2
    N = 1 - (3/10) * (h / (2 * a))**2  # ignored when 2h/a<0.4
    return F, N

# ─────────────────────────────────────────────────────────────
# GUI Application
# ─────────────────────────────────────────────────────────────

COLORS = {
    "bg": "#1e2228",
    "panel": "#252b34",
    "accent": "#2d7dd2",
    "accent2": "#e8553e",
    "accent3": "#27ae60",
    "text": "#e8eaed",
    "subtext": "#9aa0a6",
    "border": "#3c4450",
    "entry_bg": "#2d333b",
    "table_header": "#2d7dd2",
    "row_odd": "#252b34",
    "row_even": "#1e2228",
}

FONT_TITLE  = ("Segoe UI", 16, "bold")
FONT_HEADER = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 10)


class ASTMD5528App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ASTM D5528 – Mode I Interlaminar Fracture Toughness Analyzer")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.configure(bg=COLORS["bg"])

        self.specimen_rows = []   # list of {b, h, a0, rows_data}
        self.current_specimen = None  # index into specimen_rows

        self._build_ui()
        self._add_demo_data()

    # ── Layout ────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=COLORS["accent"], height=52)
        top.pack(fill="x", side="top")
        tk.Label(top, text="ASTM D5528", font=("Segoe UI", 14, "bold"),
                 bg=COLORS["accent"], fg="white").pack(side="left", padx=18, pady=12)
        tk.Label(top, text="Mode I Interlaminar Fracture Toughness — DCB Test",
                 font=("Segoe UI", 11), bg=COLORS["accent"], fg="#cce4f7").pack(side="left")

        # Main split
        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=8)

        # Left panel (inputs)
        left = tk.Frame(main, bg=COLORS["panel"], width=390, relief="flat",
                        highlightbackground=COLORS["border"], highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0,6))
        left.pack_propagate(False)

        # Right panel (results + plots)
        right = tk.Frame(main, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=COLORS["panel"])
        f.pack(fill="x", padx=10, pady=(12, 2))
        tk.Label(f, text=title, font=FONT_HEADER,
                 bg=COLORS["panel"], fg=COLORS["accent"]).pack(anchor="w")
        sep = tk.Frame(parent, bg=COLORS["border"], height=1)
        sep.pack(fill="x", padx=10, pady=(2, 6))
        return parent

    def _label_entry(self, parent, label, default="", unit=""):
        row = tk.Frame(parent, bg=COLORS["panel"])
        row.pack(fill="x", padx=14, pady=2)
        tk.Label(row, text=label, font=FONT_BODY, bg=COLORS["panel"],
                 fg=COLORS["text"], width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        e = tk.Entry(row, textvariable=var, font=FONT_MONO,
                     bg=COLORS["entry_bg"], fg=COLORS["text"],
                     insertbackground=COLORS["text"], relief="flat",
                     highlightbackground=COLORS["border"], highlightthickness=1,
                     width=10)
        e.pack(side="left", padx=4)
        if unit:
            tk.Label(row, text=unit, font=FONT_SMALL, bg=COLORS["panel"],
                     fg=COLORS["subtext"]).pack(side="left")
        return var

    # ── Left panel ────────────────────────────────────────────

    def _build_left(self, parent):
        # Scroll frame
        canvas = tk.Canvas(parent, bg=COLORS["panel"], highlightthickness=0)
        sb = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=COLORS["panel"])
        win = canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(
            -1*(e.delta//120), "units"))

        # ── Specimen dimensions ────────────────────────────────
        self._section(inner, "Numune Boyutları")
        self.var_b  = self._label_entry(inner, "Genişlik b",  "20.0", "mm")
        self.var_h  = self._label_entry(inner, "Yarı kalınlık h", "2.0", "mm")
        self.var_a0 = self._label_entry(inner, "Başlangıç çatlak a₀", "50.0", "mm")

        # ── Methods ───────────────────────────────────────────
        self._section(inner, "Hesap Yöntemleri")
        self.var_mbt = tk.BooleanVar(value=True)
        self.var_cc  = tk.BooleanVar(value=True)
        self.var_mcc = tk.BooleanVar(value=True)
        for var, lbl in [(self.var_mbt,"MBT – Modified Beam Theory"),
                          (self.var_cc, "CC – Compliance Calibration"),
                          (self.var_mcc,"MCC – Mod. Compliance Calibration")]:
            cb = tk.Checkbutton(inner, text=lbl, variable=var, font=FONT_BODY,
                                bg=COLORS["panel"], fg=COLORS["text"],
                                selectcolor=COLORS["entry_bg"],
                                activebackground=COLORS["panel"],
                                activeforeground=COLORS["text"])
            cb.pack(anchor="w", padx=14, pady=1)

        # ── Data table ────────────────────────────────────────
        self._section(inner, "Deney Verisi (P, δ, a)")
        hdr = tk.Frame(inner, bg=COLORS["table_header"])
        hdr.pack(fill="x", padx=14, pady=(0,2))
        for h, w in [("#","3"),("P (N)","8"),("δ (mm)","8"),("a (mm)","8")]:
            tk.Label(hdr, text=h, font=("Segoe UI",9,"bold"), bg=COLORS["table_header"],
                     fg="white", width=int(w), anchor="center").pack(side="left", padx=1)

        self.table_frame = tk.Frame(inner, bg=COLORS["panel"])
        self.table_frame.pack(fill="x", padx=14)
        self.data_rows = []   # list of (P_var, d_var, a_var)

        btn_row = tk.Frame(inner, bg=COLORS["panel"])
        btn_row.pack(fill="x", padx=14, pady=4)
        self._btn(btn_row, "+ Satır",   self._add_row,    COLORS["accent"]).pack(side="left", padx=2)
        self._btn(btn_row, "- Satır",   self._del_row,    "#555").pack(side="left", padx=2)
        self._btn(btn_row, "CSV Yükle", self._load_csv,   "#6f42c1").pack(side="left", padx=2)
        self._btn(btn_row, "Temizle",   self._clear_rows, "#555").pack(side="left", padx=2)

        # ── Action buttons ────────────────────────────────────
        self._section(inner, "")
        self._btn(inner, "🎬  VİDEODAN OTOMATİK VERİ ÇEK", self._open_video_analyzer,
                  "#6f42c1", font=("Segoe UI",10,"bold"), pady=8).pack(
                  fill="x", padx=14, pady=(4,2))
        self._btn(inner, "▶  HESAPLA", self._calculate, COLORS["accent"],
                  font=("Segoe UI",11,"bold"), pady=10).pack(fill="x", padx=14, pady=4)
        self._btn(inner, "📊  Grafikleri Kaydet", self._save_plots, COLORS["accent3"]).pack(
            fill="x", padx=14, pady=2)
        self._btn(inner, "📋  Rapor Kaydet (CSV)", self._save_report, "#e09c3a").pack(
            fill="x", padx=14, pady=2)

    def _btn(self, parent, text, cmd, color, font=FONT_BODY, pady=6):
        b = tk.Button(parent, text=text, command=cmd, font=font,
                      bg=color, fg="white", relief="flat", cursor="hand2",
                      activebackground=color, activeforeground="white",
                      padx=10, pady=pady, bd=0)
        return b

    # ── Right panel ───────────────────────────────────────────

    def _build_right(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=COLORS["panel"],
                        foreground=COLORS["text"], padding=[12,6],
                        font=FONT_BODY)
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "white")])

        # Tab 1: P-δ curve
        self.tab_pd = tk.Frame(nb, bg=COLORS["bg"])
        nb.add(self.tab_pd, text="  Yük-Yerdeğiştirme  ")

        # Tab 2: R-curve
        self.tab_rc = tk.Frame(nb, bg=COLORS["bg"])
        nb.add(self.tab_rc, text="  R-Eğrisi (GIc vs a)  ")

        # Tab 3: Calibration
        self.tab_cal = tk.Frame(nb, bg=COLORS["bg"])
        nb.add(self.tab_cal, text="  Kalibrasyon  ")

        # Tab 4: Results
        self.tab_res = tk.Frame(nb, bg=COLORS["bg"])
        nb.add(self.tab_res, text="  Sonuçlar  ")

        # Build figures
        self.fig_pd,  self.ax_pd  = self._make_fig(self.tab_pd)
        self.fig_rc,  self.ax_rc  = self._make_fig(self.tab_rc)
        self.fig_cal, self.axes_cal = self._make_fig_cal(self.tab_cal)
        self._build_results_tab(self.tab_res)

    def _make_fig(self, parent):
        fig = Figure(figsize=(8,5), facecolor=COLORS["bg"])
        ax = fig.add_subplot(111)
        _style_ax(ax)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        toolbar_frame = tk.Frame(parent, bg=COLORS["panel"])
        toolbar_frame.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(canvas, toolbar_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, ax

    def _make_fig_cal(self, parent):
        fig = Figure(figsize=(8,5), facecolor=COLORS["bg"])
        axes = [fig.add_subplot(1,3,i+1) for i in range(3)]
        for ax in axes:
            _style_ax(ax)
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        toolbar_frame = tk.Frame(parent, bg=COLORS["panel"])
        toolbar_frame.pack(side="bottom", fill="x")
        NavigationToolbar2Tk(canvas, toolbar_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, axes

    def _build_results_tab(self, parent):
        # Summary cards
        self.card_frame = tk.Frame(parent, bg=COLORS["bg"])
        self.card_frame.pack(fill="x", padx=10, pady=10)

        # Table
        frame = tk.Frame(parent, bg=COLORS["bg"])
        frame.pack(fill="both", expand=True, padx=10, pady=(0,10))

        cols = ("a (mm)", "P (N)", "δ (mm)", "C (mm/N)",
                "GIc_MBT", "GIc_CC", "GIc_MCC")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings",
                                 height=16)
        style = ttk.Style()
        style.configure("Treeview", background=COLORS["row_odd"],
                        foreground=COLORS["text"], fieldbackground=COLORS["row_odd"],
                        font=FONT_MONO, rowheight=22)
        style.configure("Treeview.Heading", background=COLORS["table_header"],
                        foreground="white", font=("Segoe UI",9,"bold"))
        style.map("Treeview", background=[("selected", COLORS["accent"])])

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, anchor="center")

        self.tree.tag_configure("odd",  background=COLORS["row_odd"])
        self.tree.tag_configure("even", background=COLORS["row_even"])

        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    # ── Data rows ─────────────────────────────────────────────

    def _add_row(self, P="", d="", a=""):
        idx = len(self.data_rows)
        bg = COLORS["row_odd"] if idx%2==0 else COLORS["row_even"]
        row = tk.Frame(self.table_frame, bg=bg)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=str(idx+1), width=3, font=FONT_MONO,
                 bg=bg, fg=COLORS["subtext"]).pack(side="left")
        vars_ = []
        for val, w in [(P,8),(d,8),(a,8)]:
            v = tk.StringVar(value=str(val))
            e = tk.Entry(row, textvariable=v, width=w, font=FONT_MONO,
                         bg=COLORS["entry_bg"], fg=COLORS["text"],
                         insertbackground=COLORS["text"], relief="flat",
                         highlightbackground=COLORS["border"], highlightthickness=1)
            e.pack(side="left", padx=1)
            vars_.append(v)
        self.data_rows.append((vars_[0], vars_[1], vars_[2], row))

    def _del_row(self):
        if self.data_rows:
            *_, row = self.data_rows.pop()
            row.destroy()

    def _clear_rows(self):
        for *_, row in self.data_rows:
            row.destroy()
        self.data_rows.clear()

    def _load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV/Excel","*.csv *.xlsx *.xls"),("All","*.*")])
        if not path:
            return
        try:
            if path.endswith((".xlsx",".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
            # Accept columns named P, delta/d/displacement, a/crack
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if cl in ("p","load","kuvvet","yük"):
                    col_map["P"] = c
                elif cl in ("d","delta","displacement","yerdeğiştirme","deplasman"):
                    col_map["d"] = c
                elif cl in ("a","crack","çatlak","crack length"):
                    col_map["a"] = c
            if len(col_map) < 3:
                # fallback: use first 3 numeric columns
                nums = df.select_dtypes(include=np.number).columns.tolist()
                if len(nums) >= 3:
                    col_map = {"P": nums[0], "d": nums[1], "a": nums[2]}
                else:
                    messagebox.showerror("Hata", "P, δ, a sütunları bulunamadı.")
                    return
            self._clear_rows()
            for _, r in df.iterrows():
                self._add_row(round(float(r[col_map["P"]]),4),
                              round(float(r[col_map["d"]]),4),
                              round(float(r[col_map["a"]]),4))
        except Exception as ex:
            messagebox.showerror("Yükleme Hatası", str(ex))

    # ── Demo data ─────────────────────────────────────────────

    def _add_demo_data(self):
        # GFRP (Glass/Epoxy) DCB test data — representative of DCB/Mode I test
        # Specimen: b=20mm, 2h=4mm, a0=50mm, GIc≈340 J/m²
        demo = [
            (35.8,  5.6,  50.0),
            (34.2,  7.1,  55.0),
            (31.5,  8.6,  60.0),
            (29.6, 10.1,  65.0),
            (27.8, 11.8,  70.0),
            (26.0, 13.7,  75.0),
            (24.5, 15.6,  80.0),
            (23.1, 17.7,  85.0),
            (21.9, 19.9,  90.0),
            (20.7, 22.1,  95.0),
            (19.6, 24.4, 100.0),
        ]
        for P, d, a in demo:
            self._add_row(P, d, a)

    # ── Calculation ───────────────────────────────────────────

    def _get_data(self):
        P, d, a = [], [], []
        for pv, dv, av, _ in self.data_rows:
            try:
                P.append(float(pv.get()))
                d.append(float(dv.get()))
                a.append(float(av.get()))
            except ValueError:
                pass
        return np.array(P), np.array(d), np.array(a)

    def _calculate(self):
        try:
            b  = float(self.var_b.get())  / 1000   # mm → m
            h  = float(self.var_h.get())  / 1000
            a0 = float(self.var_a0.get()) / 1000

            P, delta, a_mm = self._get_data()
            if len(P) < 3:
                messagebox.showwarning("Uyarı", "En az 3 veri noktası gereklidir.")
                return

            # Keep mm for plots, convert to SI for calc
            delta_m = delta / 1000      # mm → m

            # Eğer tüm çatlak uzunlukları aynıysa compliance'tan tahmin et
            # DCB teorisi: C ∝ a³  →  a = a0 * (C/C0)^(1/3)
            a_mm = a_mm.copy().astype(float)
            if np.all(a_mm == a_mm[0]):
                C_raw = delta_m / np.where(P > 0, P, np.nan)
                C0 = C_raw[0]
                if C0 and C0 > 0:
                    a_mm = (a_mm[0]) * (C_raw / C0) ** (1/3)
                    messagebox.showinfo(
                        "Bilgi",
                        "Çatlak uzunluğu (a) tüm satırlarda aynı bulundu.\n"
                        "DCB uyumluluk teorisinden otomatik tahmin edildi:\n"
                        "  a = a₀ × (C/C₀)^(1/3)")

            a = a_mm / 1000        # mm → m

            results = {}
            if self.var_mbt.get():
                GIc, Delta, s, ic, r2 = method_MBT(P, delta_m, a, b)
                results["MBT"] = dict(GIc=GIc, Delta=Delta*1000,
                                       slope=s, intercept=ic, r2=r2)
            if self.var_cc.get():
                GIc, n, s, ic, r2 = method_CC(P, delta_m, a, b)
                results["CC"] = dict(GIc=GIc, n=n,
                                      slope=s, intercept=ic, r2=r2)
            if self.var_mcc.get():
                GIc, A1, s, ic, r2 = method_MCC(P, delta_m, a, b, h)
                results["MCC"] = dict(GIc=GIc, A1=A1,
                                       slope=s, intercept=ic, r2=r2)

            C = delta_m / np.where(P > 0, P, np.nan)   # m/N (SI compliance)

            # Tablo satırlarındaki a değerlerini güncelle (tahmini değerler)
            for i, (_, _, av, _) in enumerate(self.data_rows):
                if i < len(a_mm):
                    av.set(f"{a_mm[i]:.2f}")

            self._plot_pd(P, delta)
            self._plot_rcurve(a_mm, results)
            self._plot_cal(C, a_mm, h*1000, results)
            self._update_results(P, delta, a_mm, C, results)
            self._update_cards(results)

        except Exception as ex:
            messagebox.showerror("Hesap Hatası", str(ex))
            raise

    def _open_video_analyzer(self):
        VideoAnalyzer(self, self._receive_video_data)

    def _receive_video_data(self, P_list, d_list, a_list):
        """Video analizinden gelen veriyi tabloya yükle."""
        self._clear_rows()
        for P, d, a in zip(P_list, d_list, a_list):
            self._add_row(P, d, a)
        messagebox.showinfo("Veri Yüklendi",
            f"{len(P_list)} satır tabloya aktarıldı.\nŞimdi 'HESAPLA' butonuna basın.")

    # ── Plots ─────────────────────────────────────────────────

    def _plot_pd(self, P, delta):
        ax = self.ax_pd
        ax.clear()
        _style_ax(ax)
        ax.plot(delta, P, "o-", color=COLORS["accent"], lw=2,
                ms=5, label="Yük-Yerdeğiştirme")
        ax.set_xlabel("Yerdeğiştirme δ (mm)", color=COLORS["subtext"])
        ax.set_ylabel("Yük P (N)", color=COLORS["subtext"])
        ax.set_title("Yük – Yerdeğiştirme Eğrisi", color=COLORS["text"], fontsize=12)
        ax.legend(frameon=False, labelcolor=COLORS["text"])
        self.fig_pd.tight_layout()
        self.fig_pd.canvas.draw()

    def _plot_rcurve(self, a_mm, results):
        ax = self.ax_rc
        ax.clear()
        _style_ax(ax)
        colors = {"MBT": COLORS["accent"], "CC": COLORS["accent2"],
                  "MCC": COLORS["accent3"]}
        for method, res in results.items():
            GIc = res["GIc"]
            avg = np.mean(GIc)
            ax.plot(a_mm, GIc, "o-", color=colors[method], lw=2, ms=5,
                    label=f"{method}  (ort. {avg:.1f} J/m²)")
        ax.axhline(0, color=COLORS["border"], lw=0.5)
        ax.set_xlabel("Çatlak Uzunluğu a (mm)", color=COLORS["subtext"])
        ax.set_ylabel("G$_{Ic}$ (J/m²)", color=COLORS["subtext"])
        ax.set_title("R-Eğrisi — Delaminasyon Direnç Eğrisi", color=COLORS["text"], fontsize=12)
        ax.legend(frameon=False, labelcolor=COLORS["text"])
        self.fig_rc.tight_layout()
        self.fig_rc.canvas.draw()

    def _plot_cal(self, C, a_mm, h_mm, results):
        axes = self.axes_cal
        for ax in axes:
            ax.clear()
            _style_ax(ax)
        a = a_mm / 1000
        h = h_mm / 1000

        a_m = a_mm / 1000  # SI for fits (matches how methods were called)

        # MBT: C^(1/3) vs a  (a in m, fit in SI)
        ax = axes[0]
        C13 = C ** (1/3)
        ax.scatter(a_mm, C13, color=COLORS["accent"], s=30, zorder=3)
        if "MBT" in results:
            r = results["MBT"]
            afit_m = np.linspace(min(a_m), max(a_m), 100)
            yfit = r["slope"] * afit_m + r["intercept"]
            ax.plot(afit_m * 1000, yfit, "--", color=COLORS["accent"], alpha=0.7,
                    label=f"R²={r['r2']:.4f}")
            ax.set_title("MBT: C¹/³ vs a", color=COLORS["text"], fontsize=10)
            ax.set_xlabel("a (mm)", color=COLORS["subtext"], fontsize=9)
            ax.set_ylabel("C¹/³  (m/N)¹/³", color=COLORS["subtext"], fontsize=9)
            ax.legend(frameon=False, labelcolor=COLORS["text"], fontsize=8)

        # CC: log(C) vs log(a)  (SI)
        ax = axes[1]
        ax.scatter(np.log(a_m), np.log(C), color=COLORS["accent2"], s=30, zorder=3)
        if "CC" in results:
            r = results["CC"]
            xfit = np.linspace(min(np.log(a_m)), max(np.log(a_m)), 100)
            yfit = r["slope"] * xfit + r["intercept"]
            ax.plot(xfit, yfit, "--", color=COLORS["accent2"], alpha=0.7,
                    label=f"n={r['n']:.3f}  R²={r['r2']:.4f}")
            ax.set_title("CC: log(C) vs log(a)", color=COLORS["text"], fontsize=10)
            ax.set_xlabel("log(a [m])", color=COLORS["subtext"], fontsize=9)
            ax.set_ylabel("log(C [m/N])", color=COLORS["subtext"], fontsize=9)
            ax.legend(frameon=False, labelcolor=COLORS["text"], fontsize=8)

        # MCC: a/h vs C^(1/3)
        ax = axes[2]
        ah = a / h
        ax.scatter(C13, ah, color=COLORS["accent3"], s=30, zorder=3)
        if "MCC" in results:
            r = results["MCC"]
            xfit = np.linspace(min(C13), max(C13), 100)
            yfit = r["slope"] * xfit + r["intercept"]
            ax.plot(xfit, yfit, "--", color=COLORS["accent3"], alpha=0.7,
                    label=f"A1={r['A1']:.2f}  R²={r['r2']:.4f}")
            ax.set_title("MCC: a/h vs C¹/³", color=COLORS["text"], fontsize=10)
            ax.set_xlabel("C¹/³", color=COLORS["subtext"], fontsize=9)
            ax.set_ylabel("a/h", color=COLORS["subtext"], fontsize=9)
            ax.legend(frameon=False, labelcolor=COLORS["text"], fontsize=8)

        self.fig_cal.tight_layout(pad=2)
        self.fig_cal.canvas.draw()

    # ── Results tab ───────────────────────────────────────────

    def _update_results(self, P, delta, a_mm, C, results):
        for row in self.tree.get_children():
            self.tree.delete(row)
        n = len(P)
        mbt = results.get("MBT", {}).get("GIc", [None]*n)
        cc  = results.get("CC",  {}).get("GIc", [None]*n)
        mcc = results.get("MCC", {}).get("GIc", [None]*n)

        def fmt(v):
            return f"{v:.2f}" if v is not None else "—"

        for i in range(n):
            tag = "odd" if i%2==0 else "even"
            self.tree.insert("", "end", tags=(tag,), values=(
                f"{a_mm[i]:.2f}",
                f"{P[i]:.2f}",
                f"{delta[i]:.4f}",
                f"{C[i]:.6f}",
                fmt(mbt[i] if hasattr(mbt,"__len__") else None),
                fmt(cc[i]  if hasattr(cc, "__len__") else None),
                fmt(mcc[i] if hasattr(mcc,"__len__") else None),
            ))

    def _update_cards(self, results):
        for w in self.card_frame.winfo_children():
            w.destroy()
        colors = {"MBT": COLORS["accent"], "CC": COLORS["accent2"],
                  "MCC": COLORS["accent3"]}
        for method, res in results.items():
            GIc = res["GIc"]
            avg = np.mean(GIc)
            std = np.std(GIc, ddof=1)
            cv  = 100*std/avg if avg else 0
            card = tk.Frame(self.card_frame, bg=colors[method], padx=16, pady=12)
            card.pack(side="left", padx=6, fill="y")
            tk.Label(card, text=method, font=("Segoe UI",11,"bold"),
                     bg=colors[method], fg="white").pack()
            tk.Label(card, text=f"{avg:.1f} J/m²", font=("Segoe UI",20,"bold"),
                     bg=colors[method], fg="white").pack()
            tk.Label(card, text=f"Ort. G_Ic", font=("Segoe UI",8),
                     bg=colors[method], fg="#ddd").pack()
            tk.Label(card, text=f"σ = {std:.1f}  |  CV = {cv:.1f}%",
                     font=("Segoe UI",9), bg=colors[method], fg="#eee").pack(pady=4)
            # R² info
            r2 = res.get("r2", None)
            if r2:
                tk.Label(card, text=f"Kalibrasyon R² = {r2:.4f}",
                         font=FONT_SMALL, bg=colors[method], fg="#ddd").pack()

    # ── Export ────────────────────────────────────────────────

    def _save_plots(self):
        folder = filedialog.askdirectory(title="Klasör Seç")
        if not folder:
            return
        self.fig_pd.savefig(os.path.join(folder,"yuk_yerdeğiştirme.png"),
                            dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        self.fig_rc.savefig(os.path.join(folder,"R_egrisi.png"),
                            dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        self.fig_cal.savefig(os.path.join(folder,"kalibrasyon.png"),
                             dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        messagebox.showinfo("Kaydedildi", f"Grafikler kaydedildi:\n{folder}")

    def _save_report(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Excel","*.xlsx")])
        if not path:
            return
        rows = []
        for child in self.tree.get_children():
            rows.append(self.tree.item(child)["values"])
        df = pd.DataFrame(rows, columns=["a_mm","P_N","delta_mm","C_mmN",
                                          "GIc_MBT_Jm2","GIc_CC_Jm2","GIc_MCC_Jm2"])
        if path.endswith(".xlsx"):
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False)
        messagebox.showinfo("Kaydedildi", f"Rapor kaydedildi:\n{path}")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(COLORS["panel"])
    ax.figure.patch.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["subtext"], which="both")
    ax.spines[:].set_color(COLORS["border"])
    for sp in ax.spines.values():
        sp.set_color(COLORS["border"])
    ax.grid(True, color=COLORS["border"], linestyle="--", alpha=0.5, linewidth=0.6)


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ASTMD5528App()
    app.mainloop()
