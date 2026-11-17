# PETE 662 Project 2 — Interactive Nodal Analysis (Carlos Buenrostro)

## Folder layout
```
PETE662_Nodal/
├── pete662_nodal/            calculation package (edit these .py files)
│   ├── inputs.py             WellboreCase (well, tubing, production, fluid inputs)
│   ├── pvt.py                AppendixBPVT (HW5 curves) and CorrelationPVT (Ch. 3-4 correlations)
│   ├── flow.py               superficial velocities, friction factor (Haaland / Chen)
│   ├── hagedorn_brown.py     modified Hagedorn-Brown + Griffith bubble flow
│   ├── beggs_brill.py        Beggs-Brill + Payne correction
│   ├── traverse.py           pressure_traverse, vlp_curve
│   ├── ipr.py                PI, Vogel, generalized Vogel, radial-flow PI with skin
│   └── nodal.py              operating_point, run_sensitivity
├── tests/                    pytest checks (Examples 3-3, 4-2, 4-4, 7-8, 7-9, HW5/Exam 1 regression)
├── 01_Validation_and_Nodal.ipynb
├── 02_PVT_Correlations.ipynb
├── app.py                    Streamlit application
├── run_app.bat               Windows double-click launcher
├── .streamlit/config.toml    app theme
├── conftest.py
└── requirements.txt
```

## Setup (once)
```
pip install -r requirements.txt
```

## Running the application
```
pip install -r requirements.txt      (once)
streamlit run app.py                 (from this folder)
```
or double-click `run_app.bat` on Windows. The app opens in the browser at http://localhost:8501.

* **Sidebar:** load an example case, then edit reservoir/IPR, well and tubing, production and fluid
  inputs, PVT model, VLP correlation and friction factor.
* **Nodal analysis tab:** IPR/VLP plot, operating rate and pwf for each correlation, no-flow and
  unstable-crossing warnings.
* **Pressure profile tab:** pressure and holdup vs. depth with flow regime, at any rate.
* **Sensitivity tab:** vary tubing ID, wellhead pressure, GOR, WOR, reservoir pressure, skin,
  permeability or J; overlay plots and summary table.
* **Data & report tab:** IPR/VLP tables and a downloadable example-case report (inputs + outputs).

## Working in Jupyter
Open `01_Validation_and_Nodal.ipynb` from this folder (Jupyter must be started in, or
navigated to, `PETE662_Nodal` so `import pete662_nodal` works). The notebook uses
`%autoreload 2`, so saved edits to the `.py` files take effect on the next cell run.

## Running the tests
From this folder:
```
python -m pytest -q
```

## Changes relative to the HW5 / Exam 1 notebooks
* Same equations and the same traverse algorithm; inputs are passed as a `WellboreCase`
  instead of globals, so one input can be varied for sensitivity runs.
* **Griffith bubble-flow friction corrected** to the book form (in-situ liquid velocity,
  Eqs. 7-111 to 7-116). Rates that never reach bubble flow reproduce the notebooks exactly;
  rates that do change by at most a few psi (e.g. 2254.31 -> 2256.68 psia at 2000 STB/d, HW5).
* Nodal solver picks the stable (highest-rate, VLP-rising) crossing and reports
  "no natural flow" when the VLP is above the IPR everywhere.

## PVT models
Choose with `WellboreCase(..., pvt_model="appendix_b")` (default, reproduces the homework) or
`pvt_model="correlations"`:
* Vasquez-Beggs Rs and Bo (PPS Eqs. 3-8 to 3-15), oil density (3-17, 3-18)
* Beggs-Robinson oil viscosity, Vasquez-Beggs above pb (3-19 to 3-27)
* Standing pseudocriticals (4-8, 4-9), Dranchuk-Abou-Kassem Z (4-5, 4-6)
* Lee-Gonzalez-Eakin gas viscosity (4-23 to 4-26); liquid mixing (3-29, 3-31)

Validated against Example 3-3 (Rs, Bo, Bg, oil density, viscosities, liquid mixture).
With correlations the produced-fluid bubble point is where Rs = producing GOR;
`pb`/`Rsb` then only feed the IPR.

