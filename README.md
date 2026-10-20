# Green-Hydrogen-TEA: Techno-Economic Analysis & LCOH Optimization

Techno-Economic Analysis (TEA) and Levelized Cost of Hydrogen (LCOH) optimization framework for grid-scale green hydrogen production systems powered by intermittent renewables (solar PV and wind).

## Features
- **LCOH Dynamic Modeling:** Calculates Net Present Value (NPV), Internal Rate of Return (IRR), and Levelized Cost of Hydrogen ($/kg H2) accounting for electrolyzer capital expenditure (CAPEX), stack replacement schedules, and operating expenses (OPEX).
- **Renewable Intermittency & Capacity Sizing:** Simulates hourly power generation profiles to optimize the capacity ratio between renewable generation, battery energy storage systems (BESS), and proton exchange membrane (PEM) electrolyzers.
- **Electrolyzer Degradation Curves:** Models membrane thinning, catalyst dissolution, and overpotential drift under fluctuating power regimes.

## Usage
```bash
git clone https://github.com/dhrubchatterjee16-lang/GREEN_HYDROGEN_generation-.git
cd GREEN_HYDROGEN_generation-
pip install -r requirements.txt
python run_lcoh_optimization.py --solar_mw 100 --wind_mw 50 --electrolyzer_mw 80
```

*(For electrochemical cell physics and dynamic cell polarization modeling, see [green-hydrogen-generation](https://github.com/dhrubchatterjee16-lang/green-hydrogen-generation)).*
