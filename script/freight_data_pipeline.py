#!/usr/bin/env python3
"""Leakage safe data preparation for the Freight Rate Prediction Challenge.

The script keeps source fields intact and adds model-ready fields for:
  * weight sign correction and missingness flags
  * date aware market/quote imputation
  * canonical city coordinates and route geometry
  * distance sanity checks
  * calendar, equipment, weight, quote, and route features
  * target rate per mile diagnostics when the target is present

Example:
    python freight_data_pipeline.py --input-dir upload --output-dir prepared_freight

The built in coordinate table is a transparent starting point. For production,
replace it with coordinates returned by a verified geocoder and keep the same
city to state disambiguation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TARGET_COLUMN = "posted_rate"
BASE_DATE = pd.Timestamp("2025-01-01")
EARTH_RADIUS_MILES = 3958.7613


# Approximate city-center coordinates. The city/state choices are explicit so
# ambiguous names such as Columbia, Jackson, and Washington do not get mapped
# to the wrong place by an automatic geocoder.
CITY_COORDS: dict[str, dict[str, Any]] = {
    "Albany": {"state": "NY", "lat": 42.6526, "lon": -73.7562, "region": "Northeast"},
    "Albuquerque": {"state": "NM", "lat": 35.0844, "lon": -106.6504, "region": "West"},
    "Allentown": {"state": "PA", "lat": 40.6023, "lon": -75.4714, "region": "Northeast"},
    "Amarillo": {"state": "TX", "lat": 35.2220, "lon": -101.8313, "region": "Southwest"},
    "Atlanta": {"state": "GA", "lat": 33.7490, "lon": -84.3880, "region": "Southeast"},
    "Austin": {"state": "TX", "lat": 30.2672, "lon": -97.7431, "region": "Southwest"},
    "Bakersfield": {"state": "CA", "lat": 35.3733, "lon": -119.0187, "region": "West"},
    "Baltimore": {"state": "MD", "lat": 39.2904, "lon": -76.6122, "region": "Northeast"},
    "Baton Rouge": {"state": "LA", "lat": 30.4515, "lon": -91.1871, "region": "Southeast"},
    "Birmingham": {"state": "AL", "lat": 33.5186, "lon": -86.8104, "region": "Southeast"},
    "Boston": {"state": "MA", "lat": 42.3601, "lon": -71.0589, "region": "Northeast"},
    "Buffalo": {"state": "NY", "lat": 42.8864, "lon": -78.8784, "region": "Northeast"},
    "Charleston": {"state": "SC", "lat": 32.7765, "lon": -79.9311, "region": "Southeast"},
    "Charlotte": {"state": "NC", "lat": 35.2271, "lon": -80.8431, "region": "Southeast"},
    "Chattanooga": {"state": "TN", "lat": 35.0456, "lon": -85.3097, "region": "Southeast"},
    "Chicago": {"state": "IL", "lat": 41.8781, "lon": -87.6298, "region": "Midwest"},
    "Cincinnati": {"state": "OH", "lat": 39.1031, "lon": -84.5120, "region": "Midwest"},
    "Columbia": {"state": "SC", "lat": 34.0007, "lon": -81.0348, "region": "Southeast"},
    "Corpus Christi": {"state": "TX", "lat": 27.8006, "lon": -97.3964, "region": "Southwest"},
    "Dallas": {"state": "TX", "lat": 32.7767, "lon": -96.7970, "region": "Southwest"},
    "Dayton": {"state": "OH", "lat": 39.7589, "lon": -84.1916, "region": "Midwest"},
    "Detroit": {"state": "MI", "lat": 42.3314, "lon": -83.0458, "region": "Midwest"},
    "El Paso": {"state": "TX", "lat": 31.7619, "lon": -106.4850, "region": "Southwest"},
    "Fort Wayne": {"state": "IN", "lat": 41.0793, "lon": -85.1394, "region": "Midwest"},
    "Fresno": {"state": "CA", "lat": 36.7378, "lon": -119.7871, "region": "West"},
    "Grand Rapids": {"state": "MI", "lat": 42.9634, "lon": -85.6681, "region": "Midwest"},
    "Green Bay": {"state": "WI", "lat": 44.5133, "lon": -88.0133, "region": "Midwest"},
    "Greensboro": {"state": "NC", "lat": 36.0726, "lon": -79.7920, "region": "Southeast"},
    "Harrisburg": {"state": "PA", "lat": 40.2732, "lon": -76.8867, "region": "Northeast"},
    "Hartford": {"state": "CT", "lat": 41.7658, "lon": -72.6734, "region": "Northeast"},
    "Houston": {"state": "TX", "lat": 29.7604, "lon": -95.3698, "region": "Southwest"},
    "Indianapolis": {"state": "IN", "lat": 39.7684, "lon": -86.1581, "region": "Midwest"},
    "Jackson": {"state": "MS", "lat": 32.2988, "lon": -90.1848, "region": "Southeast"},
    "Jacksonville": {"state": "FL", "lat": 30.3322, "lon": -81.6557, "region": "Southeast"},
    "Kansas City": {"state": "MO", "lat": 39.0997, "lon": -94.5786, "region": "Midwest"},
    "Knoxville": {"state": "TN", "lat": 35.9606, "lon": -83.9207, "region": "Southeast"},
    "Laredo": {"state": "TX", "lat": 27.5036, "lon": -99.5075, "region": "Southwest"},
    "Las Vegas": {"state": "NV", "lat": 36.1699, "lon": -115.1398, "region": "West"},
    "Lexington": {"state": "KY", "lat": 38.0406, "lon": -84.5037, "region": "Southeast"},
    "Little Rock": {"state": "AR", "lat": 34.7465, "lon": -92.2896, "region": "Southeast"},
    "Los Angeles": {"state": "CA", "lat": 34.0522, "lon": -118.2437, "region": "West"},
    "Louisville": {"state": "KY", "lat": 38.2527, "lon": -85.7585, "region": "Southeast"},
    "Lubbock": {"state": "TX", "lat": 33.5779, "lon": -101.8552, "region": "Southwest"},
    "Madison": {"state": "WI", "lat": 43.0731, "lon": -89.4012, "region": "Midwest"},
    "Memphis": {"state": "TN", "lat": 35.1495, "lon": -90.0490, "region": "Southeast"},
    "Milwaukee": {"state": "WI", "lat": 43.0389, "lon": -87.9065, "region": "Midwest"},
    "Mobile": {"state": "AL", "lat": 30.6954, "lon": -88.0399, "region": "Southeast"},
    "Montgomery": {"state": "AL", "lat": 32.3668, "lon": -86.3000, "region": "Southeast"},
    "Nashville": {"state": "TN", "lat": 36.1627, "lon": -86.7816, "region": "Southeast"},
    "New Orleans": {"state": "LA", "lat": 29.9511, "lon": -90.0715, "region": "Southeast"},
    "New York": {"state": "NY", "lat": 40.7128, "lon": -74.0060, "region": "Northeast"},
    "Norfolk": {"state": "VA", "lat": 36.8508, "lon": -76.2859, "region": "Southeast"},
    "Oklahoma City": {"state": "OK", "lat": 35.4676, "lon": -97.5164, "region": "Southwest"},
    "Philadelphia": {"state": "PA", "lat": 39.9526, "lon": -75.1652, "region": "Northeast"},
    "Phoenix": {"state": "AZ", "lat": 33.4484, "lon": -112.0740, "region": "West"},
    "Providence": {"state": "RI", "lat": 41.8240, "lon": -71.4128, "region": "Northeast"},
    "Raleigh": {"state": "NC", "lat": 35.7796, "lon": -78.6382, "region": "Southeast"},
    "Reno": {"state": "NV", "lat": 39.5296, "lon": -119.8138, "region": "West"},
    "Richmond": {"state": "VA", "lat": 37.5407, "lon": -77.4360, "region": "Southeast"},
    "Salt Lake City": {"state": "UT", "lat": 40.7608, "lon": -111.8910, "region": "West"},
    "San Antonio": {"state": "TX", "lat": 29.4241, "lon": -98.4936, "region": "Southwest"},
    "San Diego": {"state": "CA", "lat": 32.7157, "lon": -117.1611, "region": "West"},
    "San Francisco": {"state": "CA", "lat": 37.7749, "lon": -122.4194, "region": "West"},
    "Savannah": {"state": "GA", "lat": 32.0809, "lon": -81.0912, "region": "Southeast"},
    "Shreveport": {"state": "LA", "lat": 32.5252, "lon": -93.7502, "region": "Southeast"},
    "St. Louis": {"state": "MO", "lat": 38.6270, "lon": -90.1994, "region": "Midwest"},
    "Syracuse": {"state": "NY", "lat": 43.0481, "lon": -76.1474, "region": "Northeast"},
    "Tampa": {"state": "FL", "lat": 27.9506, "lon": -82.4572, "region": "Southeast"},
    "Toledo": {"state": "OH", "lat": 41.6528, "lon": -83.5379, "region": "Midwest"},
    "Tucson": {"state": "AZ", "lat": 32.2226, "lon": -110.9747, "region": "West"},
    "Tulsa": {"state": "OK", "lat": 36.1540, "lon": -95.9928, "region": "Southwest"},
    "Washington": {"state": "DC", "lat": 38.9072, "lon": -77.0369, "region": "Northeast"},
}


CITY_LAT = {city: value["lat"] for city, value in CITY_COORDS.items()}
CITY_LON = {city: value["lon"] for city, value in CITY_COORDS.items()}
CITY_STATE = {city: value["state"] for city, value in CITY_COORDS.items()}
CITY_REGION = {city: value["region"] for city, value in CITY_COORDS.items()}


US_HOLIDAYS_2025 = pd.to_datetime(
    [
        "2025-01-01",
        "2025-01-20",
        "2025-02-17",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
        "2025-09-01",
        "2025-10-13",
        "2025-11-11",
        "2025-11-27",
        "2025-12-25",
    ]
)


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize spacing without changing the intended city spelling."""

    return (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def haversine_miles(
    lat1: Iterable[float] | pd.Series,
    lon1: Iterable[float] | pd.Series,
    lat2: Iterable[float] | pd.Series,
    lon2: Iterable[float] | pd.Series,
) -> pd.Series:
    """Vectorized great circle distance in miles."""

    a1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    o1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    a2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    o2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
    delta_a = a2 - a1
    delta_o = o2 - o1
    h = np.sin(delta_a / 2.0) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(delta_o / 2.0) ** 2
    result = 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    return pd.Series(result, index=getattr(lat1, "index", None), dtype="float64")


def bearing_components(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return sine/cosine of the initial route bearing."""

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    bearing = np.arctan2(
        np.sin(dl) * np.cos(p2),
        np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl),
    )
    return pd.Series(np.sin(bearing), index=lat1.index), pd.Series(np.cos(bearing), index=lat1.index)


def normalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce source columns to predictable types and normalize text spacing."""

    df = frame.copy()
    for column in ["pickup", "delivery", "equipment"]:
        if column in df.columns:
            df[column] = clean_text(df[column])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        TARGET_COLUMN,
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _daily_median_reference(reference: pd.DataFrame | None, column: str) -> pd.Series:
    if reference is None or "date" not in reference.columns or column not in reference.columns:
        return pd.Series(dtype="float64")
    ref = normalize_input(reference)
    return ref.groupby("date")[column].median()


@dataclass
class FreightFeaturePipeline:
    """Fit training-only imputers and transform any compatible input frame."""

    outlier_low_rpm: float = 0.5
    outlier_high_rpm: float = 6.0
    weight_medians_by_equipment: dict[str, float] = field(default_factory=dict)
    global_weight_median: float = 0.0
    global_market_median: float = 0.0
    global_quote_median: float = 0.0
    train_daily_market: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    train_daily_quote: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    fitted: bool = False

    def fit(self, train_frame: pd.DataFrame) -> "FreightFeaturePipeline":
        train = normalize_input(train_frame)
        weight_abs = train["weight"].abs() if "weight" in train.columns else pd.Series(dtype="float64")
        if "equipment" in train.columns:
            medians = pd.DataFrame({"equipment": train["equipment"], "weight_abs": weight_abs}).groupby(
                "equipment"
            )["weight_abs"].median()
            self.weight_medians_by_equipment = {
                str(key): float(value) for key, value in medians.dropna().items()
            }
        self.global_weight_median = float(weight_abs.median()) if weight_abs.notna().any() else 0.0
        self.global_market_median = (
            float(train["market_index"].median())
            if "market_index" in train.columns and train["market_index"].notna().any()
            else 0.0
        )
        self.global_quote_median = (
            float(train["quote_signal"].median())
            if "quote_signal" in train.columns and train["quote_signal"].notna().any()
            else 2.05
        )
        self.train_daily_market = _daily_median_reference(train, "market_index")
        self.train_daily_quote = _daily_median_reference(train, "quote_signal")
        self.fitted = True
        return self

    def _fill_market_or_quote(
        self,
        df: pd.DataFrame,
        column: str,
        external_reference: pd.DataFrame | None,
    ) -> pd.Series:
        if column not in df.columns:
            raw = pd.Series(np.nan, index=df.index, dtype="float64")
        else:
            raw = pd.to_numeric(df[column], errors="coerce")

        # A same-date median is valid for batch scoring because it uses only
        # covariates, not the target. For December, pass validation.csv here.
        daily_external = _daily_median_reference(external_reference, column)
        daily_local = _daily_median_reference(df, column)
        daily_train = self.train_daily_market if column == "market_index" else self.train_daily_quote

        filled = raw.copy()
        if len(daily_external):
            filled = filled.fillna(df["date"].map(daily_external))
        filled = filled.fillna(df["date"].map(daily_local))
        filled = filled.fillna(df["date"].map(daily_train))
        fallback = self.global_market_median if column == "market_index" else self.global_quote_median
        return filled.fillna(fallback).astype("float64")

    def _add_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
            if column not in df.columns:
                df[column] = np.nan
            df[column] = pd.to_numeric(df[column], errors="coerce")

        for side in ["pickup", "delivery"]:
            df[f"{side}_lat_real"] = df[side].map(CITY_LAT).astype("float64")
            df[f"{side}_lon_real"] = df[side].map(CITY_LON).astype("float64")
            df[f"{side}_state"] = df[side].map(CITY_STATE).astype("string")
            df[f"{side}_region"] = df[side].map(CITY_REGION).astype("string")
            df[f"{side}_coordinate_lookup_missing_flag"] = df[f"{side}_lat_real"].isna().astype("int8")

        df["raw_coordinate_available_flag"] = (
            df[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all(axis=1)
        ).astype("int8")
        df["pickup_coordinate_error_miles"] = haversine_miles(
            df["pickup_lat"],
            df["pickup_lon"],
            df["pickup_lat_real"],
            df["pickup_lon_real"],
        )
        df["delivery_coordinate_error_miles"] = haversine_miles(
            df["delivery_lat"],
            df["delivery_lon"],
            df["delivery_lat_real"],
            df["delivery_lon_real"],
        )
        df["coordinate_error_flag"] = (
            (df["pickup_coordinate_error_miles"] > 50.0)
            | (df["delivery_coordinate_error_miles"] > 50.0)
        ).astype("int8")
        df.loc[df["raw_coordinate_available_flag"] == 0, "coordinate_error_flag"] = np.nan

        df["source_haversine_miles"] = haversine_miles(
            df["pickup_lat"],
            df["pickup_lon"],
            df["delivery_lat"],
            df["delivery_lon"],
        )
        df["real_haversine_miles"] = haversine_miles(
            df["pickup_lat_real"],
            df["pickup_lon_real"],
            df["delivery_lat_real"],
            df["delivery_lon_real"],
        )
        df["real_lat_delta"] = df["delivery_lat_real"] - df["pickup_lat_real"]
        df["real_lon_delta"] = df["delivery_lon_real"] - df["pickup_lon_real"]
        df["real_lat_abs"] = df["real_lat_delta"].abs()
        df["real_lon_abs"] = df["real_lon_delta"].abs()
        df["same_region_flag"] = (df["pickup_region"] == df["delivery_region"]).astype("int8")
        df["same_city_flag"] = (df["pickup"] == df["delivery"]).astype("int8")
        df["region_pair"] = df["pickup_region"].astype("string") + "_to_" + df["delivery_region"].astype("string")

        bearing_sin, bearing_cos = bearing_components(
            df["pickup_lat_real"],
            df["pickup_lon_real"],
            df["delivery_lat_real"],
            df["delivery_lon_real"],
        )
        df["bearing_sin"] = bearing_sin
        df["bearing_cos"] = bearing_cos
        return df

    def _add_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "distance" not in df.columns:
            df["distance"] = np.nan
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df["distance_original"] = df["distance"]
        df["distance_log1p"] = np.log1p(df["distance"].clip(lower=0))
        df["distance_sqrt"] = np.sqrt(df["distance"].clip(lower=0))
        df["real_haversine_log1p"] = np.log1p(df["real_haversine_miles"].clip(lower=0))
        df["distance_real_gap"] = df["distance_original"] - df["real_haversine_miles"]
        df["distance_real_ratio"] = df["distance_original"] / (df["real_haversine_miles"] + 1.0)
        df["distance_impossible_flag"] = (
            df["distance_original"] < df["real_haversine_miles"]
        ).astype("int8")
        df.loc[df["distance_original"].isna() | df["real_haversine_miles"].isna(), "distance_impossible_flag"] = np.nan
        df["distance_band"] = pd.cut(
            df["distance_original"],
            bins=[-np.inf, 300, 600, 1000, 1500, 2200, np.inf],
            labels=["short", "regional", "medium", "long", "very_long", "cross_country"],
        ).astype("string")
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = df["date"]
        df["date_invalid_flag"] = dates.isna().astype("int8")
        df["year"] = dates.dt.year.astype("Int64")
        df["month"] = dates.dt.month.astype("Int64")
        df["day_of_month"] = dates.dt.day.astype("Int64")
        df["day_of_week"] = dates.dt.dayofweek.astype("Int64")
        df["week_of_year"] = dates.dt.isocalendar().week.astype("Int64")
        df["day_of_year"] = dates.dt.dayofyear.astype("Int64")
        df["day_index"] = (dates - BASE_DATE).dt.days
        df["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype("int8")
        df["is_holiday"] = dates.dt.normalize().isin(US_HOLIDAYS_2025).astype("int8")
        df["days_to_christmas"] = (pd.Timestamp("2025-12-25") - dates).dt.days
        df["days_from_thanksgiving"] = (dates - pd.Timestamp("2025-11-27")).dt.days
        df["day_of_year_sin"] = np.sin(2.0 * np.pi * dates.dt.dayofyear / 365.25)
        df["day_of_year_cos"] = np.cos(2.0 * np.pi * dates.dt.dayofyear / 365.25)
        df["day_of_week_sin"] = np.sin(2.0 * np.pi * dates.dt.dayofweek / 7.0)
        df["day_of_week_cos"] = np.cos(2.0 * np.pi * dates.dt.dayofweek / 7.0)
        return df

    def _add_numeric_and_target_features(
        self,
        df: pd.DataFrame,
        external_reference: pd.DataFrame | None,
    ) -> pd.DataFrame:
        raw_weight = pd.to_numeric(df["weight"], errors="coerce") if "weight" in df.columns else pd.Series(np.nan, index=df.index)
        df["weight_negative_flag"] = (raw_weight < 0).astype("int8")
        df["weight_missing_flag"] = raw_weight.isna().astype("int8")
        weight_abs = raw_weight.abs()
        equipment_median = df["equipment"].map(self.weight_medians_by_equipment)
        df["weight_clean"] = weight_abs.fillna(equipment_median).fillna(self.global_weight_median)
        df["weight_capacity_ratio"] = df["weight_clean"] / 47500.0
        df["weight_over_30000_flag"] = (df["weight_clean"] > 30000).astype("int8")
        df["weight_over_40000_flag"] = (df["weight_clean"] > 40000).astype("int8")
        df["weight_band"] = pd.cut(
            df["weight_clean"],
            bins=[-np.inf, 20000, 25000, 30000, 35000, 40000, np.inf],
            labels=["light", "moderate", "medium", "heavy", "very_heavy", "near_capacity"],
        ).astype("string")

        if "market_index" not in df.columns:
            df["market_index"] = np.nan
        if "quote_signal" not in df.columns:
            df["quote_signal"] = np.nan
        df["market_index"] = pd.to_numeric(df["market_index"], errors="coerce")
        df["quote_signal"] = pd.to_numeric(df["quote_signal"], errors="coerce")
        df["market_index_missing_flag"] = df["market_index"].isna().astype("int8")
        df["quote_signal_missing_flag"] = df["quote_signal"].isna().astype("int8")
        df["market_index_clean"] = self._fill_market_or_quote(df, "market_index", external_reference)
        df["quote_signal_clean"] = self._fill_market_or_quote(df, "quote_signal", external_reference)
        df["market_index_centered"] = df["market_index_clean"] - self.global_market_median
        df["market_index_squared"] = df["market_index_centered"] ** 2
        df["quote_centered"] = df["quote_signal_clean"] - 2.05
        df["quote_extremeness"] = df["quote_centered"].abs()
        df["quote_extremeness_squared"] = df["quote_centered"] ** 2
        df["market_x_distance"] = df["market_index_clean"] * df["distance_log1p"]
        df["quote_x_equipment"] = df["quote_extremeness"] * df["equipment"].map(
            {"Dry Van": 0.0, "Flatbed": 1.0, "Reefer": 2.0}
        ).fillna(0.0)

        if TARGET_COLUMN in df.columns:
            target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
            df["rate_per_mile"] = target / df["distance_original"].replace(0, np.nan)
            df["log_posted_rate"] = np.log1p(target.clip(lower=0))
            df["target_low_rpm_flag"] = (df["rate_per_mile"] < self.outlier_low_rpm).astype("int8")
            df["target_high_rpm_flag"] = (df["rate_per_mile"] > self.outlier_high_rpm).astype("int8")
            df["target_outlier_flag"] = (
                (df["rate_per_mile"] < self.outlier_low_rpm)
                | (df["rate_per_mile"] > self.outlier_high_rpm)
            ).astype("int8")
        return df

    def transform(
        self,
        frame: pd.DataFrame,
        external_daily_reference: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Call fit(train_frame) before transform(...).")
        df = normalize_input(frame)
        required = ["pickup", "delivery", "equipment", "date"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = self._add_coordinates(df)
        df = self._add_distance_features(df)
        df = self._add_calendar_features(df)
        df = self._add_numeric_and_target_features(df, external_daily_reference)
        df["route_id"] = df["pickup"].astype("string") + "__" + df["delivery"].astype("string")
        df["route_equipment_id"] = df["route_id"] + "__" + df["equipment"].astype("string")
        # CSVs should contain ISO dates, while the in-memory frame stays typed.
        if "date" in df.columns:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def profile_frame(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    df = normalize_input(frame)
    result: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_load_ids": int(df["load_id"].duplicated().sum()) if "load_id" in df.columns else None,
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "date_min": _json_value(df["date"].min()) if "date" in df.columns else None,
        "date_max": _json_value(df["date"].max()) if "date" in df.columns else None,
        "unique_pickup_cities": int(df["pickup"].nunique()) if "pickup" in df.columns else None,
        "unique_delivery_cities": int(df["delivery"].nunique()) if "delivery" in df.columns else None,
        "negative_weight_rows": int((df["weight"] < 0).sum()) if "weight" in df.columns else None,
        "target_outliers": None,
    }
    if TARGET_COLUMN in df.columns:
        rpm = df[TARGET_COLUMN] / df["distance"].replace(0, np.nan)
        result["target_outliers"] = {
            "low_rpm_below_0_5": int((rpm < 0.5).sum()),
            "high_rpm_above_6": int((rpm > 6.0).sum()),
            "posted_rate_below_200": int((df[TARGET_COLUMN] < 200).sum()),
            "posted_rate_above_10000": int((df[TARGET_COLUMN] > 10000).sum()),
        }
    all_cities = set(df.get("pickup", pd.Series(dtype="string")).dropna()) | set(
        df.get("delivery", pd.Series(dtype="string")).dropna()
    )
    result["unknown_cities"] = sorted(str(city) for city in all_cities if city not in CITY_COORDS)
    return result


def locate_december_file(input_dir: Path) -> Path:
    candidates = [
        input_dir / "december-chart-inputs(1).csv",
        input_dir / "december-chart-inputs.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the December inputs CSV.")


def prepare_files(
    input_dir: Path,
    output_dir: Path,
    train_name: str = "train-test.csv",
    validation_name: str = "validation.csv",
    december_name: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = input_dir / train_name
    validation_path = input_dir / validation_name
    december_path = input_dir / december_name if december_name else locate_december_file(input_dir)

    train_raw = pd.read_csv(train_path)
    validation_raw = pd.read_csv(validation_path)
    december_raw = pd.read_csv(december_path)

    pipeline = FreightFeaturePipeline().fit(train_raw)
    train_prepared = pipeline.transform(train_raw)
    validation_prepared = pipeline.transform(validation_raw)
    # The validation file supplies covariate-only daily signals for the
    # December template. It contains no target, so this does not leak labels.
    december_prepared = pipeline.transform(december_raw, external_daily_reference=validation_raw)

    train_output = output_dir / "prepared_train.csv"
    validation_output = output_dir / "prepared_validation.csv"
    december_output = output_dir / "prepared_december.csv"
    report_output = output_dir / "data_quality_report.json"

    train_prepared.to_csv(train_output, index=False)
    validation_prepared.to_csv(validation_output, index=False)
    december_prepared.to_csv(december_output, index=False)

    report = {
        "pipeline": {
            "outlier_low_rpm": pipeline.outlier_low_rpm,
            "outlier_high_rpm": pipeline.outlier_high_rpm,
            "coordinate_count": len(CITY_COORDS),
            "weight_imputation": "equipment median, then training global median",
            "market_quote_imputation": "same-date median, then training date median, then training global median",
            "december_covariate_source": "validation.csv date-level median",
            "original_distance_preserved": True,
        },
        "input_profiles": [
            profile_frame(train_raw, "train-test.csv"),
            profile_frame(validation_raw, "validation.csv"),
            profile_frame(december_raw, december_path.name),
        ],
        "output_files": {
            "prepared_train": train_output.name,
            "prepared_validation": validation_output.name,
            "prepared_december": december_output.name,
        },
    }
    report_output.write_text(json.dumps(report, indent=2, default=_json_value), encoding="utf-8")

    return {
        "train": train_output,
        "validation": validation_output,
        "december": december_output,
        "report": report_output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_freight"))
    parser.add_argument("--train-name", default="train-test.csv")
    parser.add_argument("--validation-name", default="validation.csv")
    parser.add_argument("--december-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = prepare_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_name=args.train_name,
        validation_name=args.validation_name,
        december_name=args.december_name,
    )
    for label, path in paths.items():
        print(f"{label}: {path.resolve()}")


if __name__ == "__main__":
    main()
