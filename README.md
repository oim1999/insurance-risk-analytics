## Changes
- Initialize DVC for data version control
- Configure local remote storage at `~/dvc-storage/insurance-risk-analytics`
- Track two data versions:
  1. `data/insurance_data.csv` — raw dataset
  2. `data/insurance_data_cleaned.csv` — cleaned dataset
- Add DVC documentation to README
- Add `dvc.yaml` pipeline configuration

## How to Reproduce
```bash
dvc pull
python -c "from src.data_loader import prepare_data; df = prepare_data('data/insurance_data.csv')"