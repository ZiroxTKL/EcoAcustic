$Python = "C:\Users\leoma\miniconda3\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "`n== Proyecto 2: ejecucion completa ==" -ForegroundColor Cyan
Write-Host "Python usado: $Python" -ForegroundColor Cyan

& $Python src\phase1_dimensionality.py
& $Python src\phase2_clustering.py
& $Python src\phase3_classification.py
& $Python src\phase4_thresholds.py

Write-Host "`nEjecucion finalizada. Revise outputs/ para metricas y figuras." -ForegroundColor Green
