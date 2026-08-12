"""
DCB Crack Tracker
Mantik: catlak = iki beyaz kol arasindaki SIYAH bolge.
ROI secmek zorunda degilsin — akilli maske otomatik calisir.
Debug modu: algoritminin ne gordugunu goster.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2, numpy as np
from PIL import Image, ImageTk
import csv, os

BG, PANEL, FG, SUB = "#111", "#1a1a2e", "#eaeaea", "#666"
C_GO, C_ST, C_YL, C_BL = "#27ae60", "#e94560", "#f39c12", "#2d7dd2"
FONT  = ("Segoe UI", 10)
FONTB = ("Segoe UI", 10, "bold")
MONO  = ("Consolas", 11)

PALETTE = [(180,80,255),(100,120,255),(60,200,255),
           (80,255,200),(180,255,80),(255,255,60)]

def dot_color(r):
    n=len(PALETTE); i=min(int(r*(n-1)),n-2); t=r*(n-1)-i
    a,b=PALETTE[i],PALETTE[i+1]
    return tuple(int(a[k]*(1-t)+b[k]*t) for k in range(3))


class CrackTracker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DCB Crack Tracker")
        self.geometry("1280x820"); self.minsize(900,600)
        self.configure(bg=BG)

        self._cap=None; self._total=0; self._fps=25.0
        self._idx=0; self._frame=None; self._photo=None

        # ROI (opsiyonel)
        self._roi=None; self._drag0=None

        # Olcek
        self._scale=None; self._spts=[]
        self._mode="idle"

        # Parametreler
        self._bright   = tk.IntVar(value=130)
        self._dark     = tk.IntVar(value=80)
        self._arm_min  = tk.IntVar(value=15)
        self._cr_min   = tk.IntVar(value=3)
        self._cr_max   = tk.IntVar(value=500)
        self._interval = tk.IntVar(value=1)
        self._ref_mm   = tk.DoubleVar(value=10.0)
        self._debug    = tk.BooleanVar(value=False)
        # Oto-maske: WMC video icin
        self._excl_top  = tk.IntVar(value=12)   # ust % (WMC metin)
        self._excl_rx   = tk.IntVar(value=52)   # sag % (logo + aciklama)
        self._excl_ry   = tk.IntVar(value=48)   # alt % (logo alt siniri)

        self._data={}; self._origin=None; self._running=False
        self._img_sc=1.0; self._off_x=0; self._off_y=0

        self._build()
        self.bind("<Escape>",lambda e:self._stop())
        self.bind("<Left>",  lambda e:self._step(-1))
        self.bind("<Right>", lambda e:self._step(1))

    # ══════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════
    def _build(self):
        lf=tk.Frame(self,bg=PANEL,width=280)
        lf.pack(side="left",fill="y",padx=(6,3),pady=6); lf.pack_propagate(False)
        self._build_left(lf)
        rf=tk.Frame(self,bg=BG)
        rf.pack(side="left",fill="both",expand=True,padx=(0,6),pady=6)
        self._build_right(rf)

    def _sec(self,p,t):
        tk.Label(p,text=t,font=FONTB,bg=PANEL,fg=C_BL).pack(anchor="w",padx=8,pady=(10,0))
        tk.Frame(p,bg="#334",height=1).pack(fill="x",padx=8,pady=2)

    def _btn(self,p,t,cmd,bg=C_BL,py=5):
        b=tk.Button(p,text=t,command=cmd,font=FONT,bg=bg,fg="white",
                    relief="flat",cursor="hand2",padx=8,pady=py,bd=0,
                    activebackground=bg)
        b.pack(fill="x",padx=8,pady=2); return b

    def _spin(self,p,lbl,var,lo,hi,step=1,w=5):
        r=tk.Frame(p,bg=PANEL); r.pack(fill="x",padx=8,pady=1)
        tk.Label(r,text=lbl,font=FONT,bg=PANEL,fg=FG,
                 width=22,anchor="w").pack(side="left")
        tk.Spinbox(r,from_=lo,to=hi,increment=step,textvariable=var,
                   width=w,font=MONO,bg="#223",fg=FG,relief="flat").pack(side="left")

    def _build_left(self,p):
        self._sec(p,"1 — Video")
        self._btn(p,"Video Yukle",self._load,C_BL)
        self._lbl_vid=tk.Label(p,text="—",font=("Segoe UI",8),bg=PANEL,fg=SUB,
                                wraplength=255,justify="left")
        self._lbl_vid.pack(anchor="w",padx=10)

        self._sec(p,"2 — ROI  (opsiyonel, daha iyi sonuc verir)")
        self._btn(p,"Numune Uzerinde Kare Ciz",self._cmd_roi,C_YL)
        self._lbl_roi=tk.Label(p,text="ROI: otomatik maske kullaniliyor",
                                font=("Segoe UI",8),bg=PANEL,fg=SUB,wraplength=255)
        self._lbl_roi.pack(anchor="w",padx=10)
        self._btn(p,"ROI Sifirla (oto maskeye don)",self._reset_roi,"#444",3)

        self._sec(p,"3 — Oto-Maske  (ROI yoksa gecerli)")
        self._spin(p,"Ust maske %",    self._excl_top,0,50)
        self._spin(p,"Sag aciklama %", self._excl_rx, 0,100)
        self._spin(p,"Sag aciklama Y%",self._excl_ry, 0,100)

        self._sec(p,"4 — Esikler")
        self._spin(p,"Beyaz kol esigi",    self._bright, 30,255)
        self._spin(p,"Siyah catlak esigi", self._dark,   5, 200)
        self._spin(p,"Kol min yukseklik",  self._arm_min,3, 200)
        self._spin(p,"Catlak min (px)",    self._cr_min, 1, 50)
        self._spin(p,"Catlak max (px)",    self._cr_max, 5, 1000)

        self._sec(p,"5 — Debug / Calistir")
        r=tk.Frame(p,bg=PANEL); r.pack(fill="x",padx=8,pady=2)
        tk.Checkbutton(r,text="Debug modu (ne goruldugu goster)",
                       variable=self._debug,command=self._redraw,
                       font=FONT,bg=PANEL,fg=FG,
                       selectcolor="#333",activebackground=PANEL
                       ).pack(side="left")
        self._spin(p,"Her N kare",self._interval,1,50)

        r2=tk.Frame(p,bg=PANEL); r2.pack(fill="x",padx=8,pady=1)
        tk.Label(r2,text="Olcek ref:",font=FONT,bg=PANEL,fg=FG).pack(side="left")
        tk.Entry(r2,textvariable=self._ref_mm,width=5,font=MONO,
                 bg="#223",fg=FG,relief="flat",insertbackground=FG
                 ).pack(side="left",padx=3)
        tk.Label(r2,text="mm",font=FONT,bg=PANEL,fg=SUB).pack(side="left")
        self._btn(p,"2 Nokta → px/mm",self._cmd_scale,C_YL,3)
        self._lbl_sc=tk.Label(p,text="Olcek: piksel modu",font=MONO,bg=PANEL,fg=SUB)
        self._lbl_sc.pack(anchor="w",padx=10)

        self._btn(p,"▶  OTOMATIK BASLAT",self._start,C_GO,8)
        self._btn(p,"⛔  Durdur  (ESC)",  self._stop, C_ST)
        self._btn(p,"Temizle",            self._clear,"#444")
        self._pb=ttk.Progressbar(p,maximum=100,mode="determinate")
        self._pb.pack(fill="x",padx=8,pady=4)

        self._sec(p,"Sonuc")
        self._lbl_mm=tk.Label(p,text="— mm",font=("Consolas",20,"bold"),
                               bg=PANEL,fg=C_GO); self._lbl_mm.pack(pady=2)
        self._lbl_inf=tk.Label(p,text="",font=("Segoe UI",9),bg=PANEL,fg=SUB)
        self._lbl_inf.pack()

        self._sec(p,"Kaydet")
        self._btn(p,"CSV Kaydet",self._save_csv,C_GO)
        self._btn(p,"PNG Kaydet",self._save_png,C_YL)
        self._lbl_st=tk.Label(p,text="Hazir.",font=("Segoe UI",9,"bold"),
                               bg=PANEL,fg=C_YL,wraplength=260)
        self._lbl_st.pack(anchor="w",padx=10,pady=4)

    def _build_right(self,p):
        self._cv=tk.Canvas(p,bg="black",highlightthickness=0,cursor="crosshair")
        self._cv.pack(fill="both",expand=True)
        self._cv.bind("<Button-1>",        self._c_click)
        self._cv.bind("<B1-Motion>",       self._c_drag)
        self._cv.bind("<ButtonRelease-1>", self._c_release)
        self._cv.bind("<Configure>",       lambda e:self._redraw())
        bot=tk.Frame(p,bg=PANEL,height=34); bot.pack(fill="x",pady=(3,0))
        tk.Label(bot,text="Kare:",font=FONT,bg=PANEL,fg=FG).pack(side="left",padx=6)
        self._slider=tk.Scale(bot,from_=0,to=100,orient="horizontal",
                               bg=PANEL,fg=FG,highlightthickness=0,
                               troughcolor="#0f3460",activebackground=C_BL,
                               command=lambda v:self._goto(int(float(v))))
        self._slider.pack(side="left",fill="x",expand=True)
        self._lbl_fr=tk.Label(bot,text="0/0|0.00s",font=MONO,bg=PANEL,fg=SUB,width=18)
        self._lbl_fr.pack(side="left",padx=8)

    # ══════════════════════════════════════════════════
    # Video
    # ══════════════════════════════════════════════════
    def _load(self):
        path=filedialog.askopenfilename(
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("*","*.*")])
        if not path: return
        cap=cv2.VideoCapture(path)
        if not cap.isOpened(): messagebox.showerror("Hata","Acilamadi."); return
        if self._cap: self._cap.release()
        self._cap=cap; self._total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._slider.config(to=max(0,self._total-1))
        self._data.clear(); self._origin=None; self._roi=None
        nm=os.path.basename(path)
        self._lbl_vid.config(
            text=f"{nm}\n{self._total} kare | {self._fps:.1f}fps | {self._total/self._fps:.1f}s")
        self._lbl_roi.config(text="ROI: otomatik maske kullaniliyor",fg=SUB)
        self._goto(0)
        self._status("Video yuklendi. Debug modunu ac, parametreleri ayarla.")

    def _read(self,idx):
        if not self._cap: return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,f=self._cap.read(); return f if ok else None

    def _goto(self,idx):
        idx=max(0,min(idx,self._total-1))
        self._idx=idx; self._frame=self._read(idx)
        self._slider.set(idx); self._redraw()

    def _step(self,d): self._goto(self._idx+d)

    # ══════════════════════════════════════════════════
    # Canvas
    # ══════════════════════════════════════════════════
    def _to_img(self,cx,cy):
        return (int((cx-self._off_x)/self._img_sc),
                int((cy-self._off_y)/self._img_sc))

    def _c_click(self,e):
        ix,iy=self._to_img(e.x,e.y)
        if self._mode=="roi": self._drag0=(ix,iy)
        elif self._mode=="scale1":
            self._spts=[(ix,iy)]; self._mode="scale2"
            self._status(f"2/2: {self._ref_mm.get():.0f}mm sonuna tiklayin.")
        elif self._mode=="scale2":
            self._spts.append((ix,iy))
            d=np.hypot(self._spts[1][0]-self._spts[0][0],
                       self._spts[1][1]-self._spts[0][1])
            if d>1:
                self._scale=d/self._ref_mm.get()
                self._lbl_sc.config(text=f"Olcek: {self._scale:.3f} px/mm",fg=C_GO)
                self._status(f"Olcek: {self._scale:.3f} px/mm")
            self._mode="idle"

    def _c_drag(self,e):
        if self._mode!="roi" or self._drag0 is None: return
        self._redraw()
        cx0,cy0=(int(self._drag0[0]*self._img_sc+self._off_x),
                  int(self._drag0[1]*self._img_sc+self._off_y))
        self._cv.create_rectangle(cx0,cy0,e.x,e.y,
                                   outline=C_YL,width=2,dash=(4,4))

    def _c_release(self,e):
        if self._mode!="roi" or self._drag0 is None: return
        ix,iy=self._to_img(e.x,e.y); x0,y0=self._drag0
        rx0,rx1=sorted([x0,ix]); ry0,ry1=sorted([y0,iy])
        if rx1-rx0<10 or ry1-ry0<4:
            self._drag0=None; return
        self._roi=(rx0,ry0,rx1,ry1); self._drag0=None; self._mode="idle"
        self._lbl_roi.config(
            text=f"ROI: [{rx0},{ry0}]→[{rx1},{ry1}]  ({rx1-rx0}×{ry1-ry0}px)",
            fg=C_GO)
        self._status("ROI ayarlandi.")
        self._redraw()

    def _cmd_roi(self):
        if not self._cap: messagebox.showwarning("","Once video yukleyin."); return
        self._mode="roi"
        self._status("ONEMLI: Her iki kolu da icine alacak sekilde surukle (ust kol + siyah catlak + alt kol).")

    def _reset_roi(self):
        self._roi=None
        self._lbl_roi.config(text="ROI: otomatik maske kullaniliyor",fg=SUB)
        self._redraw()

    def _cmd_scale(self):
        if not self._cap: messagebox.showwarning("","Once video yukleyin."); return
        self._mode="scale1"; self._spts=[]
        self._status(f"1/2: {self._ref_mm.get():.0f}mm basina tiklayin.")

    # ══════════════════════════════════════════════════
    # Maske
    # ══════════════════════════════════════════════════
    def _make_search_area(self, frame):
        """
        ROI secildiyse onu kullan.
        Yoksa tum kareyi kullan, bilinen gürültü bolgelerini maskelé.
        Donus: (work_region, offset_x, offset_y)
        """
        h,w=frame.shape[:2]
        if self._roi:
            rx0,ry0,rx1,ry1=self._roi
            return frame[ry0:ry1,rx0:rx1], rx0, ry0

        # Oto-maske
        mask=np.ones((h,w),dtype=bool)
        et=self._excl_top.get()
        mask[:int(h*et/100),:]=False          # ust metin
        erx=self._excl_rx.get()
        ery=self._excl_ry.get()
        mask[:int(h*ery/100),int(w*erx/100):]=False   # sag aciklama metni
        mask[int(h*0.62):,int(w*0.50):]=False          # sag-alt logo

        # Maske uygulanmis tam kare (islem icin) + offset
        work=frame.copy()
        work[~mask]=0
        return work, 0, 0

    # ══════════════════════════════════════════════════
    # Catlak tespiti
    # ══════════════════════════════════════════════════
    def _find_tip(self, frame):
        """
        Arama bolgesi icinde her sutunda
        BEYAZ (kol) → SIYAH (catlak) → BEYAZ (kol)
        pattern'i ara.  En sagdaki = crack tip.
        """
        work, ox, oy = self._make_search_area(frame)
        gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        rh,rw=blur.shape

        bright_thr=self._bright.get()
        dark_thr  =self._dark.get()
        arm_min   =self._arm_min.get()
        cr_min    =self._cr_min.get()
        cr_max    =self._cr_max.get()

        found_x=[]; top_y_arr=[]; bot_y_arr=[]

        for x in range(rw):
            col=blur[:,x]
            dark=col<=dark_thr

            dark_px=np.where(dark)[0]
            if len(dark_px)==0: continue

            # Siyah segmentleri bul
            diff=np.diff(dark_px,prepend=dark_px[0]-2)
            segs=np.split(dark_px,np.where(diff>2)[0])

            for seg in segs:
                if len(seg)<cr_min or len(seg)>cr_max: continue
                top=int(seg[0]); bot=int(seg[-1])
                above=(col[:top]>=bright_thr).sum()
                below=(col[bot:]>=bright_thr).sum()
                if above>=arm_min and below>=arm_min:
                    found_x.append(x)
                    top_y_arr.append(top)
                    bot_y_arr.append(bot)
                    break

        if not found_x: return None, None

        tip_x_roi=found_x[-1]
        sm=[i for i,x in enumerate(found_x) if x>=max(0,tip_x_roi-5)]
        top_y=oy+int(np.mean([top_y_arr[i] for i in sm]))
        bot_y=oy+int(np.mean([bot_y_arr[i] for i in sm]))

        # Debug maskeleme icin tum gecerli x'leri sakla
        all_data=list(zip(found_x,[top_y_arr[i] for i in range(len(found_x))],
                                   [bot_y_arr[i] for i in range(len(found_x))]))

        return (ox+tip_x_roi, top_y, bot_y), all_data

    # ══════════════════════════════════════════════════
    # Debug goruntu
    # ══════════════════════════════════════════════════
    def _make_debug(self, frame):
        """
        Gri ikili goruntu:
          Beyaz kol bolgeleri = acik yesil
          Siyah catlak bolgeleri = kirmizi
          Geçerli catlak sutunlari = parlak kirmizi bar
        """
        work,ox,oy=self._make_search_area(frame)
        gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        h,w=frame.shape[:2]

        # Base: orijinal renk soluk
        out=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()
        out=(out*0.35).astype(np.uint8)

        bright_thr=self._bright.get(); dark_thr=self._dark.get()
        arm_min=self._arm_min.get(); cr_min=self._cr_min.get(); cr_max=self._cr_max.get()

        rh,rw=blur.shape
        for x in range(rw):
            col=blur[:,x]
            # Beyaz bolge: yesil
            bright_rows=np.where(col>=bright_thr)[0]
            for r in bright_rows:
                yr=oy+int(r); xr=ox+x
                if 0<=yr<h and 0<=xr<w:
                    out[yr,xr]=(60,200,60)

            # Siyah catlak adaylari: kirmizi
            dark_px=np.where(col<=dark_thr)[0]
            if len(dark_px)==0: continue
            diff=np.diff(dark_px,prepend=dark_px[0]-2)
            segs=np.split(dark_px,np.where(diff>2)[0])
            for seg in segs:
                if len(seg)<cr_min or len(seg)>cr_max: continue
                top=int(seg[0]); bot=int(seg[-1])
                above=(col[:top]>=bright_thr).sum()
                below=(col[bot:]>=bright_thr).sum()
                if above>=arm_min and below>=arm_min:
                    for r in seg:
                        yr=oy+int(r); xr=ox+x
                        if 0<=yr<h and 0<=xr<w:
                            out[yr,xr]=(255,60,60)
        return out

    # ══════════════════════════════════════════════════
    # Cizim — WMC tarzi
    # ══════════════════════════════════════════════════
    def _px_to_mm(self,tx):
        if self._origin is None: return 0.0
        dpx=abs(tx-self._origin[0])
        return dpx/self._scale if self._scale else float(dpx)

    def _draw(self,rgb):
        keys=sorted(self._data); n=len(keys)

        if self._roi:
            rx0,ry0,rx1,ry1=self._roi
            cv2.rectangle(rgb,(rx0,ry0),(rx1,ry1),(80,80,200),1)

        if n==0: return

        top_pts=[(self._data[k][0],self._data[k][1]) for k in keys]
        bot_pts=[(self._data[k][0],self._data[k][2]) for k in keys]

        if n>1:
            for i in range(n-1):
                cv2.line(rgb,top_pts[i],top_pts[i+1],(160,200,255),1,cv2.LINE_AA)
                cv2.line(rgb,bot_pts[i],bot_pts[i+1],(160,200,255),1,cv2.LINE_AA)

        for i,k in enumerate(keys):
            tx,ty_t,ty_b=self._data[k]
            col=dot_color(i/max(n-1,1))
            bgr=(int(col[2]),int(col[1]),int(col[0]))
            cv2.line(rgb,(tx,ty_t),(tx,ty_b),bgr,1)
            cv2.rectangle(rgb,(tx-4,ty_t-4),(tx+4,ty_t+4),bgr,-1)
            cv2.rectangle(rgb,(tx-4,ty_t-4),(tx+4,ty_t+4),(255,255,255),1)
            cv2.rectangle(rgb,(tx-4,ty_b-4),(tx+4,ty_b+4),bgr,-1)
            cv2.rectangle(rgb,(tx-4,ty_b-4),(tx+4,ty_b+4),(255,255,255),1)

        if self._origin:
            ox,ot,ob=self._origin; oy=(ot+ob)//2
            ltx,ltt,ltb=self._data[keys[-1]]; lty=(ltt+ltb)//2
            cv2.line(rgb,(ox,oy),(ltx,lty),(100,255,100),1,cv2.LINE_AA)
            cv2.drawMarker(rgb,(ox,oy),(255,220,0),cv2.MARKER_CROSS,20,2,cv2.LINE_AA)

        if self._idx in self._data:
            tx,ty_t,ty_b=self._data[self._idx]
            mm=self._px_to_mm(tx)
            unit="mm" if self._scale else "px"
            label=f"Cr0 c1|  {mm:.4f}  {unit}"
            fs=0.55
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,fs,1)
            bx=max(0,tx-2); by=ty_b+th+12
            if by>rgb.shape[0]-4: by=ty_t-8
            cv2.rectangle(rgb,(bx,by-th-4),(bx+tw+8,by+4),(20,20,60),-1)
            cv2.rectangle(rgb,(bx,by-th-4),(bx+tw+8,by+4),(120,140,220),1)
            cv2.putText(rgb,label,(bx+4,by),
                        cv2.FONT_HERSHEY_SIMPLEX,fs,(200,220,255),1,cv2.LINE_AA)
            self._lbl_mm.config(text=f"{mm:.4f} {unit}")
            self._lbl_inf.config(text=f"Kare {self._idx} | t={self._idx/self._fps:.2f}s")

    def _redraw(self):
        if self._frame is None: return
        if self._debug.get():
            rgb=self._make_debug(self._frame)
            # Debug basligini ekle
            cv2.putText(rgb,"DEBUG: Yesil=kol  Kirmizi=catlak",
                        (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,100),1,cv2.LINE_AA)
        else:
            rgb=cv2.cvtColor(self._frame,cv2.COLOR_BGR2RGB).copy()
        self._draw(rgb)
        cw=max(self._cv.winfo_width(),640); ch=max(self._cv.winfo_height(),360)
        ih,iw=rgb.shape[:2]; sc=min(cw/iw,ch/ih)
        dw,dh=int(iw*sc),int(ih*sc)
        self._img_sc=sc; self._off_x=(cw-dw)//2; self._off_y=(ch-dh)//2
        self._photo=ImageTk.PhotoImage(
            Image.fromarray(cv2.resize(rgb,(dw,dh),interpolation=cv2.INTER_AREA)))
        self._cv.delete("all")
        self._cv.create_image(self._off_x,self._off_y,anchor="nw",image=self._photo)
        self._lbl_fr.config(text=f"{self._idx}/{self._total-1}|{self._idx/self._fps:.2f}s")
        if self._mode=="roi":
            self._cv.create_text(cw//2,18,
                text="Her iki kolu icine alacak sekilde surukle (ust kol + catlak + alt kol)",
                fill=C_YL,font=FONTB)

    # ══════════════════════════════════════════════════
    # Otomasyon
    # ══════════════════════════════════════════════════
    def _start(self):
        if not self._cap: messagebox.showwarning("","Once video yukleyin."); return
        self._data.clear(); self._origin=None
        self._running=True; self._pb["value"]=0
        self._status("Analiz basliyor...")
        self._goto(0)
        tip,_=self._find_tip(self._frame)
        if tip:
            self._origin=tip; self._data[0]=tip
        else:
            self._status("Ilk karede catlak bulunamadi. "
                         "DEBUG modunu ac, esikleri ayarla.")
        self.after(5,self._loop)

    def _loop(self):
        if not self._running: return
        if self._frame is not None:
            tip,_=self._find_tip(self._frame)
            if tip: self._data[self._idx]=tip
        pct=int(self._idx/max(self._total-1,1)*100)
        self._pb["value"]=pct; self._redraw()
        nxt=self._idx+self._interval.get()
        if nxt>=self._total:
            self._running=False
            self._status(f"Tamamlandi — {len(self._data)} nokta.")
            return
        self._goto(nxt); self.after(1,self._loop)

    def _stop(self):
        self._running=False
        self._status(f"Durduruldu — {len(self._data)} nokta.")

    def _clear(self):
        if messagebox.askyesno("Temizle","Tum olcumler silinsin mi?"):
            self._data.clear(); self._origin=None
            self._lbl_mm.config(text="— mm"); self._lbl_inf.config(text="")
            self._pb["value"]=0; self._redraw()

    # ══════════════════════════════════════════════════
    # Kaydet
    # ══════════════════════════════════════════════════
    def _save_csv(self):
        if not self._data: messagebox.showwarning("","Hic veri yok."); return
        path=filedialog.asksaveasfilename(
            defaultextension=".csv",filetypes=[("CSV","*.csv")])
        if not path: return
        unit="mm" if self._scale else "px"
        with open(path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["frame","time_s","tip_x_px","top_y_px","bot_y_px",
                         f"crack_length_{unit}"])
            for k in sorted(self._data):
                tx,ty_t,ty_b=self._data[k]; mm=self._px_to_mm(tx)
                w.writerow([k,round(k/self._fps,4),tx,ty_t,ty_b,round(mm,4)])
        messagebox.showinfo("Kaydedildi",path)

    def _save_png(self):
        if self._frame is None: return
        path=filedialog.asksaveasfilename(
            defaultextension=".png",filetypes=[("PNG","*.png")])
        if not path: return
        rgb=cv2.cvtColor(self._frame,cv2.COLOR_BGR2RGB).copy()
        self._draw(rgb); Image.fromarray(rgb).save(path)
        messagebox.showinfo("Kaydedildi",path)

    def _status(self,msg):
        self._lbl_st.config(text=msg); self.update_idletasks()


if __name__=="__main__":
    CrackTracker().mainloop()
