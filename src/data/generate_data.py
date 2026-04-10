"""
Generates synthetic solar plant operational + revenue data.
Simulates 2 years of daily records for a 50MW solar farm.
Includes a simulated price-shock drift event starting at day 500.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def generate_solar_data(n_days: int=730, drift_start_day: int=500, seed: int=42) -> pd.DataFrame:

    np.random.seed(seed)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

    # Weather features
    # GHI: Global Horizontal Irradiance (W/m^2) - seasonal + noise
    day_of_year = np.array([d.timetuple().tm_yday for d in dates]) # 2022: 1 to 366, 2023: 1 to 366
    seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365) # cyclical seaonal factor (offset by 80 days to align peaks within summer time)
    ghi = 400 + 350 * seasonal_factor + np.random.normal(0, 60, n_days) # base + seasonal factor + noise
    ghi = np.clip(ghi, 50, 1000) # keep the values within a realistic range -> (50,1000)

    # Temperature
    temperature = 15 + 12 * seasonal_factor + np.random.normal(0, 4, n_days) # base + scaled seasonal factor + noise

    # Cloud cover (0,1)
    cloud_cover = np.clip(np.random.beta(2, 3, n_days), 0, 1) # beta distribution to mathematically keep value between 0 and 1

    # System features
    # Effective GHI after cloud adjustment
    effective_ghi = ghi * (1 - 0.7 * cloud_cover)

    # Temperature derating (panels lose ~0.4% effeciency per degree above 25 celsus)
    temp_derating = 1 - 0.004 * np.maximum(0, temperature - 25)

    # Panel degradation (linear ~0.5% per year)
    degradation = 1 - 0.005 * (np.arange(n_days) / 365)

    # System capacity factor (actual generation / max possible)
    capacity_factor = (effective_ghi / 1000) * temp_derating * degradation
    capacity_factor = np.clip(capacity_factor, 0, 1)

    # Generate (MWh/day) for a 50 MW plant
    plant_capacity_mw = 50
    hours_of_peak = 5.5 # average peak sun hours
    generation_mwh = plant_capacity_mw * capacity_factor * hours_of_peak
    generation_mwh = generation_mwh + np.random.normal(0, 5, n_days)
    generation_mwh = np.maximum(0, generation_mwh)

    # Market / Financial features
    # PPA price ($/MWh) - stable contract price
    ppa_price = np.full(n_days, 45.0) + np.random.normal(0, 1, n_days)

    # Spot market price ($/MPh) - volatile
    # drift event: price shock starting at drift_start_day
    spot_price = 35 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 8, n_days)
    # Simulate a 30% price drop (e.g. grid saturation event)
    drift_mask = np.arange(n_days) >= drift_start_day
    spot_price[drift_mask] = spot_price[drift_mask] * 0.70
    spot_price = np.maximum(0, spot_price)

    # REC price ($MWh)
    rec_price = 8 + np.random.normal(0, 1.5, n_days)
    rec_price[drift_mask] = rec_price[drift_mask] * 0.85  # RECs also drop with oversupply

    # Operational features
    # Curtailment fraction (0-1): fraction of generation that gets curtailed
    curtailment = np.random.beta(1, 20, n_days)
    curtailment[drift_mask] += np.random.beta(2, 10, len(curtailment[drift_mask]))
    curtailment = np.clip(curtailment, 0, 0.3)

    # Available generation after curtailment
    available_mwh = generation_mwh * (1 - curtailment)

    # Revenue calculation
    revenue = (
        0.7 * available_mwh * ppa_price +
        0.3 * available_mwh * spot_price +
        available_mwh * rec_price
    )
    revenue = np.maximum(0, revenue)

    # Assemble DataFrame
    df = pd.DataFrame({
        "date": dates,
        "ghi": ghi.round(2),
        "temperature": temperature.round(2),
        "cloud_cover": cloud_cover.round(4),
        "effective_ghi": effective_ghi.round(2),
        "capacity_factor": capacity_factor.round(4),
        "generation_mwh": generation_mwh.round(2),
        "ppa_price": ppa_price.round(2),
        "spot_price": spot_price.round(2),
        "rec_price": rec_price.round(2),
        "curtailment_fraction": curtailment.round(4),
        "available_mwh": available_mwh.round(2),
        "daily_revenue_usd": revenue.round(2),
        "is_drift_period": drift_mask.astype(int),  # label for monitoring
    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/solar_data.csv")
    parser.add_argument("--n-days", type=int, default=730)
    parser.add_argument("--drift-start", type=int, default=500)
    args = parser.parse_args()

    df = generate_solar_data(n_days=args.n_days, drift_start_day=args.drift_start)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} days of data → {args.output}")
    print(f"Revenue stats: mean=${df['daily_revenue_usd'].mean():,.0f}, "
          f"std=${df['daily_revenue_usd'].std():,.0f}")
    print(f"Drift period starts at row {df['is_drift_period'].idxmax()} "
          f"({df.loc[df['is_drift_period']==1,'date'].min().date()})")

    

