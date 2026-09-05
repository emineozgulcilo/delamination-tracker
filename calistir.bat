@echo off
chcp 65001 >nul
cd /d "C:\Users\Süleyman\Desktop\delamınasyon"

echo ================================================
echo   DCB Delamination Tracker
echo ================================================
echo.

if "%~1"=="" (
    echo Video dosyasinin yolunu girin (veya surukle birak):
    set /p VIDEO_PATH="Video: "
) else (
    set VIDEO_PATH=%~1
)

echo.
echo Baslatiliyor...
python tracker\main.py "%VIDEO_PATH%" --roi 100,80,470,260 --pixels-per-mm 4.4

echo.
echo Tamamlandi. Cikmak icin bir tusa basin.
pause >nul
