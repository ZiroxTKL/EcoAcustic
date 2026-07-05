# Proyecto 2: Clasificacion de Senales Eco-Acusticas

Este repositorio contiene el codigo modular para las cuatro fases del proyecto.

## Ejecucion

```powershell
python src\phase0_project_context.py
python src\phase1_dimensionality.py --manifold tsne
python src\phase2_clustering.py
python src\phase3_classification.py
python src\phase4_thresholds.py
```
## Salidas principales

- `outputs/phase1`: reduccion de dimensionalidad, metricas y figuras.
- `outputs/phase2`: clustering DBSCAN/GMM y Silhouette.
- `outputs/phase3`: MLP TensorFlow, ensamble, F1-score y matrices.
- `outputs/phase4`: zonas de confianza, incertidumbre y rechazo.
- `report/main.tex`: informe integrado en LaTeX.
