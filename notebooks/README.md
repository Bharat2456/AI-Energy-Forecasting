# Notebooks
Optional space for exploration. A quick way to explore interactively:

```python
import pandas as pd
df = pd.read_csv("../data/energy_clean.csv", index_col=0, parse_dates=True)
df.head(); df["energy_mwh"].resample("W").mean().plot()
```
Install Jupyter with `pip install jupyter`, then run `jupyter notebook` from this folder.
