"""
DCB Test Video Otomatik Analiz Modülü
WMC formatı: "WMC frame N Disp, a X.XXX [mm] F X.XX [N]"
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import easyocr
import threading
import re

# ── Lazy OCR reader ──────────────────────────────────────────
_reader = None
def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader

# ── Renk sabitleri ────────────────────────────────────────────
COLORS = {
    "bg": "#1e2228", "panel": "#252b34", "accent": "#2d7dd2",
    "accent2": "#e8553e", "accent3": "#27ae60", "text": "#e8eaed",
    "subtext": "#9aa0a6", "border": "#3c4450", "entry_bg": "#2d333b",
}
FONT     = ("Segoe UI", 10)
FONT_B   = ("Segoe UI", 10, "bold")
FONT_MONO= ("Consolas", 10)


# ── WMC metin satırını parse et ───────────────────────────────
# Örnek: "WMC frame 125 Disp, a 1.234 [mm] F 28.50 [N]"
# veya:  "Disp, a 1.234 [mm] F 28.50 [N]"
# OCR gürültüsüne dayanıklı regex'ler
# "[mm]" veya "(mm]" öncesindeki sayıyı doğrudan hedefle
_PAT_DISP  = re.compile(r'([+-]?\d+\.?\d+)\s*[\[\(]mm[\]\)]', re.I)
# "[N]" veya "(N]" veya "[NJ" öncesindeki sayıyı hedefle
_PAT_FORCE = re.compile(r'([+-]?\d+\.?\d*)\s*[\[\(][Nn][J\]\)]', re.I)

def parse_wmc_line(text):
    """Return (disp_mm, force_N) veya None. OCR gürültüsüne toleranslı."""
    d_matches = _PAT_DISP.findall(text)
    f_matches = _PAT_FORCE.findall(text)
    if d_matches and f_matches:
        try:
            return float(d_matches[-1]), float(f_matches[-1])
        except ValueError:
            pass
    return None


# ── ROI seçici ────────────────────────────────────────────────
class ROISelector(tk.Toplevel):
    def __init__(self, parent, frame_bgr, title="ROI Seç"):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None
        h, w = frame_bgr.shape[:2]
        scale = min(900/w, 600/h, 1.0)
        self._scale = scale
        dw, dh = int(w*scale), int(h*scale)
        img = cv2.cvtColor(cv2.resize(frame_bgr, (dw, dh)), cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        self._c = tk.Canvas(self, width=dw, height=dh, cursor="cross",
                             bg="black", highlightthickness=0)
        self._c.pack()
        self._c.create_image(0, 0, anchor="nw", image=self._photo)
        tk.Label(self, text="Alanı sürükleyerek seçin → bırakınca kaydedilir",
                 bg="#333", fg="white", font=FONT).pack(fill="x")
        self._rect = None; self._start = None
        self._c.bind("<ButtonPress-1>",   self._press)
        self._c.bind("<B1-Motion>",       self._drag)
        self._c.bind("<ButtonRelease-1>", self._release)
        self.grab_set()

    def _press(self, e):
        self._start = (e.x, e.y)
        if self._rect: self._c.delete(self._rect)

    def _drag(self, e):
        if self._rect: self._c.delete(self._rect)
        self._rect = self._c.create_rectangle(*self._start, e.x, e.y,
                                               outline="#e8553e", width=2)
    def _release(self, e):
        x0,y0 = self._start; x1,y1 = e.x, e.y
        s = self._scale
        self.result = (int(min(x0,x1)/s), int(min(y0,y1)/s),
                       int(max(x0,x1)/s), int(max(y0,y1)/s))
        self.destroy()


# ── Ana Video Analiz Penceresi ────────────────────────────────
class VideoAnalyzer(tk.Toplevel):
    def __init__(self, parent, on_data_ready):
        super().__init__(parent)
        self.title("Video Otomatik Analiz — DCB / ASTM D5528")
        self.geometry("1150x720")
        self.configure(bg=COLORS["bg"])

        self._on_data_ready = on_data_ready
        self._cap = None
        self._total_frames = 0
        self._fps = 25.0
        self._frame_cache = {}

        # ROI (manuel mod için)
        self.roi_overlay = None   # video üstündeki text bölgesi
        self.roi_crack   = None   # çatlak bölgesi

        # Parametreler
        self._a0_var      = tk.StringVar(value="50")
        self._px_mm_var   = tk.StringVar(value="")    # px/mm (otomatik veya manuel)
        self._interval    = tk.IntVar(value=50)
        self._mode        = tk.StringVar(value="wmc") # "wmc" veya "roi"

        self._extracted = []
        self._running   = False
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        left = tk.Frame(self, bg=COLORS["panel"], width=320)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)

        right = tk.Frame(self, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(0,8), pady=8)

        self._build_controls(left)
        self._build_preview(right)

    def _section(self, p, text):
        tk.Label(p, text=text, font=FONT_B, bg=COLORS["panel"],
                 fg=COLORS["accent"]).pack(anchor="w", padx=10, pady=(12,0))
        tk.Frame(p, bg=COLORS["border"], height=1).pack(fill="x", padx=10, pady=3)

    def _btn(self, p, text, cmd, color, font=FONT, fill=True):
        b = tk.Button(p, text=text, command=cmd, font=font,
                      bg=color, fg="white", relief="flat", cursor="hand2",
                      activebackground=color, padx=8, pady=5, bd=0)
        if fill:
            b.pack(fill="x", padx=10, pady=2)
        return b

    def _build_controls(self, p):
        # 1. Video
        self._section(p, "1. Video Dosyası")
        self._btn(p, "📂  Video Yükle (.mp4 / .avi / .mov)", self._load_video, COLORS["accent"])
        self._vid_lbl = tk.Label(p, text="Henüz yüklenmedi", font=("Segoe UI",9),
                                  bg=COLORS["panel"], fg=COLORS["subtext"],
                                  wraplength=290, justify="left")
        self._vid_lbl.pack(anchor="w", padx=10, pady=(0,4))

        # 2. Mod
        self._section(p, "2. Veri Okuma Modu")
        modes = [
            ("wmc", "🤖  Otomatik (WMC text overlay)"),
            ("roi", "✏️   Manuel ROI Seç"),
        ]
        for val, lbl in modes:
            tk.Radiobutton(p, text=lbl, variable=self._mode, value=val,
                           font=FONT, bg=COLORS["panel"], fg=COLORS["text"],
                           selectcolor=COLORS["entry_bg"],
                           activebackground=COLORS["panel"],
                           command=self._on_mode_change).pack(anchor="w", padx=14, pady=1)

        # 2b. ROI butonları (manuel mod)
        self._roi_frame = tk.Frame(p, bg=COLORS["panel"])
        self._roi_frame.pack(fill="x")
        for txt, attr, color in [
            ("📝 Text Overlay Bölgesi", "roi_overlay", "#e09c3a"),
            ("〰  Çatlak İzleme Bölgesi", "roi_crack",   "#9b59b6"),
        ]:
            row = tk.Frame(self._roi_frame, bg=COLORS["panel"])
            row.pack(fill="x", padx=10, pady=1)
            tk.Button(row, text=txt, command=lambda a=attr: self._pick_roi(a),
                      font=("Segoe UI",9), bg=color, fg="white", relief="flat",
                      cursor="hand2", padx=6, pady=3, bd=0, width=26).pack(side="left")
            lbl = tk.Label(row, text="—", font=("Segoe UI",8),
                           bg=COLORS["panel"], fg=COLORS["subtext"])
            lbl.pack(side="left", padx=4)
            setattr(self, f"_lbl_{attr}", lbl)

        # 3. Parametreler
        self._section(p, "3. Parametreler")
        for label, var, unit in [
            ("Başlangıç çatlak a₀", self._a0_var, "mm"),
            ("Her N karede bir",     self._interval, "kare"),
        ]:
            row = tk.Frame(p, bg=COLORS["panel"])
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=label, font=FONT, bg=COLORS["panel"],
                     fg=COLORS["text"], width=20, anchor="w").pack(side="left")
            if isinstance(var, tk.IntVar):
                w = tk.Spinbox(row, from_=1, to=500, textvariable=var, width=5,
                               font=FONT_MONO, bg=COLORS["entry_bg"], fg=COLORS["text"])
            else:
                w = tk.Entry(row, textvariable=var, width=7, font=FONT_MONO,
                             bg=COLORS["entry_bg"], fg=COLORS["text"],
                             insertbackground="white", relief="flat")
            w.pack(side="left")
            tk.Label(row, text=unit, font=("Segoe UI",9), bg=COLORS["panel"],
                     fg=COLORS["subtext"]).pack(side="left", padx=3)

        # Crack px/mm (isteğe bağlı)
        row = tk.Frame(p, bg=COLORS["panel"])
        row.pack(fill="x", padx=10, pady=2)
        tk.Label(row, text="Ölçek (px/mm)", font=FONT, bg=COLORS["panel"],
                 fg=COLORS["text"], width=20, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self._px_mm_var, width=7, font=FONT_MONO,
                 bg=COLORS["entry_bg"], fg=COLORS["text"],
                 insertbackground="white", relief="flat").pack(side="left")
        tk.Label(row, text="(boş→video'dan)", font=("Segoe UI",8),
                 bg=COLORS["panel"], fg=COLORS["subtext"]).pack(side="left", padx=2)

        # 4. Çalıştır
        self._section(p, "4. Analiz")
        self._btn(p, "▶  OTOMATİK ANALİZ BAŞLAT", self._start_analysis,
                  COLORS["accent"], font=("Segoe UI",11,"bold"))
        self._btn(p, "⛔  Durdur", self._stop_analysis, COLORS["accent2"])
        self._btn(p, "✅  Veriyi Ana Uygulamaya Aktar", self._export, COLORS["accent3"])

        self._progress = ttk.Progressbar(p, mode="determinate", maximum=100)
        self._progress.pack(fill="x", padx=10, pady=4)
        self._status = tk.Label(p, text="Hazır", font=("Segoe UI",9),
                                 bg=COLORS["panel"], fg=COLORS["subtext"])
        self._status.pack(anchor="w", padx=10)

        self._on_mode_change()

    def _on_mode_change(self):
        show = self._mode.get() == "roi"
        if show:
            self._roi_frame.pack(fill="x")
        else:
            self._roi_frame.pack_forget()

    def _build_preview(self, p):
        self._canvas = tk.Canvas(p, bg="black", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        bot = tk.Frame(p, bg=COLORS["panel"], height=36)
        bot.pack(fill="x")
        tk.Label(bot, text="Kare:", font=FONT, bg=COLORS["panel"],
                 fg=COLORS["text"]).pack(side="left", padx=6)
        self._slider = tk.Scale(bot, from_=0, to=100, orient="horizontal",
                                 bg=COLORS["panel"], fg=COLORS["text"],
                                 highlightthickness=0, troughcolor=COLORS["border"],
                                 activebackground=COLORS["accent"],
                                 command=lambda v: self._show_frame(int(float(v))))
        self._slider.pack(side="left", fill="x", expand=True, padx=4)
        self._frame_lbl = tk.Label(bot, text="0/0", font=FONT_MONO,
                                    bg=COLORS["panel"], fg=COLORS["subtext"], width=10)
        self._frame_lbl.pack(side="left", padx=6)

        log_f = tk.Frame(p, bg=COLORS["panel"], height=130)
        log_f.pack(fill="x")
        log_f.pack_propagate(False)
        self._log = tk.Text(log_f, height=6, font=("Consolas",9),
                             bg="#111", fg="#aaffaa", relief="flat", state="disabled")
        self._log.pack(fill="both", expand=True, padx=2, pady=2)

    # ── Video yükleme ────────────────────────────────────────────

    def _load_video(self):
        path = filedialog.askopenfilename(
            title="Video Seç",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("Tümü","*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Video açılamadı.")
            return
        self._cap = cap
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self._frame_cache.clear()
        self._slider.config(to=max(0, self._total_frames-1))
        name = path.replace("\\","/").split("/")[-1]
        self._vid_lbl.config(text=f"{name}\n{self._total_frames} kare  |  {self._fps:.1f} fps")
        self._log_write(f"Video yüklendi: {name} ({self._total_frames} kare)\n")

        # İlk kareyi göster + WMC kontrolü
        self._show_frame(0)
        frame = self._get_frame(0)
        if frame is not None:
            sample = self._ocr_top_strip(frame)
            if sample and parse_wmc_line(sample):
                self._log_write("WMC text overlay tespit edildi! Otomatik mod seçildi.\n")
                self._mode.set("wmc")
                self._on_mode_change()
            else:
                self._log_write("WMC formatı bulunamadı → Manuel ROI modunu deneyin.\n")

    def _get_frame(self, idx):
        if idx in self._frame_cache:
            return self._frame_cache[idx]
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if ret:
            if len(self._frame_cache) < 200:
                self._frame_cache[idx] = frame
            return frame
        return None

    def _show_frame(self, idx):
        frame = self._get_frame(idx)
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ROI çizgileri
        for roi, color in [(self.roi_overlay,(255,200,0)),(self.roi_crack,(180,80,255))]:
            if roi:
                x0,y0,x1,y1 = roi
                cv2.rectangle(rgb,(x0,y0),(x1,y1),color,2)
        cw = max(self._canvas.winfo_width(), 640)
        ch = max(self._canvas.winfo_height(), 380)
        h,w = rgb.shape[:2]
        scale = min(cw/w, ch/h, 1.0)
        disp = cv2.resize(rgb,(int(w*scale),int(h*scale)))
        from PIL import Image, ImageTk
        photo = ImageTk.PhotoImage(Image.fromarray(disp))
        self._canvas.delete("all")
        self._canvas.create_image(cw//2, ch//2, anchor="center", image=photo)
        self._canvas._photo = photo
        self._frame_lbl.config(text=f"{idx}/{self._total_frames-1}")

    # ── ROI seçimi ────────────────────────────────────────────────

    def _pick_roi(self, attr):
        if not self._cap:
            messagebox.showwarning("Uyarı", "Önce video yükleyin.")
            return
        frame = self._get_frame(int(self._slider.get()))
        if frame is None:
            return
        titles = {
            "roi_overlay": "Text Overlay Bölgesini Seçin (F ve Disp yazısı)",
            "roi_crack":   "Çatlak İzleme Bölgesini Seçin",
        }
        sel = ROISelector(self, frame, titles[attr])
        self.wait_window(sel)
        if sel.result:
            setattr(self, attr, sel.result)
            x0,y0,x1,y1 = sel.result
            getattr(self, f"_lbl_{attr}").config(
                text=f"[{x0},{y0}]→[{x1},{y1}]", fg="#aaffaa")
            self._show_frame(int(self._slider.get()))

    # ── OCR yardımcıları ─────────────────────────────────────────

    def _ocr_top_strip(self, frame):
        """Videonun üst %15'ini OCR'a ver — WMC text satırı için."""
        h = frame.shape[0]
        strip = frame[:int(h*0.15), :]
        gray  = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        # Karartılmış arka plan üzerine beyaz text → threshold tersle
        _, th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        big   = cv2.resize(th, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        reader = get_reader()
        results = reader.readtext(big, detail=0)
        return " ".join(results)

    def _ocr_roi_text(self, frame, roi):
        """Belirli ROI bölgesinden tüm metni oku."""
        x0,y0,x1,y1 = roi
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        big  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        reader = get_reader()
        results = reader.readtext(th, detail=0)
        return " ".join(results)

    def _extract_number(self, text, pattern):
        m = re.search(pattern, text)
        if m:
            s = m.group(1).replace(',','.')
            try: return float(s)
            except: pass
        return None

    # ── Çatlak tespiti ───────────────────────────────────────────

    def _detect_crack_length_px(self, frame):
        """
        Çatlak bölgesinde Canny ile en sağdaki kenar pikselini bul.
        """
        if self.roi_crack is None:
            return None
        x0,y0,x1,y1 = self.roi_crack
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 20, 80)
        cols = np.where(edges.any(axis=0))[0]
        return int(np.max(cols)) if len(cols) > 0 else None

    # ── Ana analiz döngüsü ────────────────────────────────────────

    def _start_analysis(self):
        if not self._cap:
            messagebox.showwarning("Uyari", "Once video yukleyin.")
            return
        self._extracted = []
        self._running   = True
        self._progress.configure(value=0)
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _stop_analysis(self):
        self._running = False
        self._status.config(text="Durduruldu.")

    def _run_analysis(self):
        self._log_write("Analiz baslatiliyor...\n")
        mode     = self._mode.get()
        interval = self._interval.get()
        total    = self._total_frames
        a0       = float(self._a0_var.get())

        try:
            px_mm = float(self._px_mm_var.get())
        except ValueError:
            px_mm = None

        results = []
        last_a  = a0
        last_P  = None
        last_d  = None
        log_buf = []   # UI'ye toplu gönder

        indices = list(range(0, total, interval))
        n_total = len(indices)

        for i, fidx in enumerate(indices):
            if not self._running:
                break

            frame = self._get_frame(fidx)
            if frame is None:
                continue

            P = d = None

            if mode == "wmc":
                text   = self._ocr_top_strip(frame)
                parsed = parse_wmc_line(text)
                if parsed:
                    d, P = parsed
                else:
                    P, d = last_P, last_d
            else:
                if self.roi_overlay:
                    text   = self._ocr_roi_text(frame, self.roi_overlay)
                    parsed = parse_wmc_line(text)
                    if parsed:
                        d, P = parsed
                    else:
                        nums = re.findall(r'[+-]?\d+\.?\d*', text)
                        if len(nums) >= 2:
                            try: P = float(nums[0])
                            except: pass
                            try: d = float(nums[1])
                            except: pass

            crack_px = self._detect_crack_length_px(frame) if self.roi_crack else None
            if crack_px is not None and px_mm:
                a = a0 + crack_px / px_mm
            else:
                a = last_a

            if P is not None and d is not None and P >= 0 and d >= 0:
                results.append((round(P, 3), round(d, 3), round(a, 2)))
                last_a = a; last_P = P; last_d = d
                log_buf.append("  Kare %5d: P=%7.2f N  d=%6.3f mm  a=%.1f mm\n"
                                % (fidx, P, d, a))

            # UI'yi her 10 adımda bir güncelle (event flood'unu önle)
            if i % 10 == 0 or i == n_total - 1:
                pct  = int((i+1)/n_total*100)
                txt  = "Isleniyor: %d/%d kare (%d%%)" % (fidx, total, pct)
                msgs = "".join(log_buf); log_buf.clear()
                self.after(0, lambda p=pct, t=txt, m=msgs: (
                    self._progress.configure(value=p),
                    self._status.config(text=t),
                    self._log_write(m)
                ))

        self._extracted = results
        self.after(0, self._done)

    def _done(self):
        n = len(self._extracted)
        self._status.config(text=f"Tamamlandı — {n} veri noktası")
        self._log_write(f"\nAnaliz tamamlandı: {n} nokta\n")
        self._progress.configure(value=100)
        if n > 0:
            messagebox.showinfo("Bitti",
                f"{n} veri noktası çıkarıldı.\n"
                "'Veriyi Ana Uygulamaya Aktar' butonuna basın.")
        else:
            messagebox.showwarning("Uyarı",
                "Hiç veri çıkarılamadı.\n\n"
                "• WMC modunda: videonun üst kısmında F ve Disp yazıları olmalı.\n"
                "• Manuel modda: ROI bölgelerini seçip tekrar deneyin.")

    def _export(self):
        if not self._extracted:
            messagebox.showwarning("Uyarı", "Önce analizi çalıştırın.")
            return
        P_list = [r[0] for r in self._extracted]
        d_list = [r[1] for r in self._extracted]
        a_list = [r[2] for r in self._extracted]
        self._on_data_ready(P_list, d_list, a_list)
        self.destroy()

    def _log_write(self, text):
        def _do():
            self._log.config(state="normal")
            self._log.insert("end", text)
            self._log.see("end")
            self._log.config(state="disabled")
        self.after(0, _do)
