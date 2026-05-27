"""
ЧАСТЬ АРТУША: население, расстояния и расчёт индекса транспортной доступности

Задача файла:
1. Получить сырые buildings/stops от парсера Ильи
2. Оценить население микрорайона через площадь жилой застройки
3. Посчитать расстояние от каждого жилого здания до ближайшей остановки
4. Сформировать итоговую строку по каждому микрорайону
5. Сформировать общий CSV microdistrict_accessibility_results.csv
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from ilya_transport_parser import (
    CITY_CONFIG,
    stable_seed,
    geocode_district,
    fetch_osm_objects_for_bbox,
)


@dataclass
class ParsedDistrict:
    city: str
    district: str
    data_source: str
    bbox: Tuple[float, float, float, float]
    buildings_count: int
    stops_count: int
    routes_count: int
    residential_area_m2: float
    population_estimate: float
    avg_distance_m: float
    p90_distance_m: float
    far_population_share_500m: float
    accessibility_index: float
    stops_per_10k: float
    routes_per_10k: float
    recommendation: str


def nearest_distances(buildings: pd.DataFrame, stops: pd.DataFrame) -> np.ndarray:
    """Считает расстояние от каждого здания до ближайшей остановки."""
    if buildings.empty or stops.empty:
        return np.full(len(buildings), np.inf)

    b = buildings[["x_m", "y_m"]].to_numpy()
    s = stops[["x_m", "y_m"]].to_numpy()

    if HAS_SCIPY:
        tree = cKDTree(s)
        dists, _ = tree.query(b, k=1)
        return dists

    # Медленный fallback, если scipy не установлен.
    dists = []
    for bx, by in b:
        diff = s - np.array([bx, by])
        dists.append(np.sqrt((diff * diff).sum(axis=1)).min())
    return np.array(dists)


def recommendation_from_index(far_share: float) -> str:
    """Текстовая рекомендация по уровню проблемности района."""
    if far_share >= 0.65:
        return "Приоритет: добавить остановки и продлить/запустить 1–2 маршрута"
    if far_share >= 0.35:
        return "Средний приоритет: увеличить частоту и проверить покрытие остановками"
    return "Низкий приоритет: поддерживать текущую сеть и мониторить рост застройки"


def fallback_district_metrics(city: str, district: str, city_population: int, total_districts: int) -> ParsedDistrict:
    """Демо-режим, если Nominatim/Overpass не доступны.

    Данные не являются реальными. Они нужны, чтобы пайплайн и графики работали
    даже без интернета.
    """
    rng = np.random.default_rng(stable_seed(city + district))
    typology = stable_seed(district) % 3

    if typology == 0:
        buildings_count = int(rng.integers(250, 850))
        stops_count = int(rng.integers(3, 12))
        routes_count = int(rng.integers(1, 6))
        avg_distance = float(rng.uniform(650, 1200))
    elif typology == 1:
        buildings_count = int(rng.integers(700, 1800))
        stops_count = int(rng.integers(12, 36))
        routes_count = int(rng.integers(5, 18))
        avg_distance = float(rng.uniform(280, 650))
    else:
        buildings_count = int(rng.integers(500, 1300))
        stops_count = int(rng.integers(6, 22))
        routes_count = int(rng.integers(3, 12))
        avg_distance = float(rng.uniform(430, 850))

    residential_area = float(buildings_count * rng.uniform(75, 160))
    pop = city_population / total_districts * rng.uniform(0.45, 1.65)
    p90 = avg_distance * rng.uniform(1.35, 1.85)

    far_share = 1 / (1 + math.exp(-(avg_distance - 500) / 150))
    far_share = float(np.clip(far_share + rng.normal(0, 0.06), 0, 1))

    stops_per_10k = stops_count / pop * 10_000
    routes_per_10k = routes_count / pop * 10_000

    return ParsedDistrict(
        city=city,
        district=district,
        data_source="synthetic_fallback",
        bbox=(0, 0, 0, 0),
        buildings_count=buildings_count,
        stops_count=stops_count,
        routes_count=routes_count,
        residential_area_m2=residential_area,
        population_estimate=pop,
        avg_distance_m=avg_distance,
        p90_distance_m=p90,
        far_population_share_500m=far_share,
        accessibility_index=far_share,
        stops_per_10k=stops_per_10k,
        routes_per_10k=routes_per_10k,
        recommendation=recommendation_from_index(far_share),
    )


def analyze_district_real(city: str, district: str, city_population: int, total_districts: int) -> ParsedDistrict:
    """Реальный расчёт показателей микрорайона по OSM-данным."""
    bbox = geocode_district(city, district)
    if bbox is None:
        raise RuntimeError("Cannot geocode district")

    buildings, stops, routes_count = fetch_osm_objects_for_bbox(bbox)
    if buildings.empty or stops.empty:
        raise RuntimeError("No buildings or stops found")

    dists = nearest_distances(buildings, stops)
    total_area = buildings["area_m2"].sum()

    # Сначала используем площадь жилой застройки как вес населения.
    # После обработки всех районов население масштабируется до общей численности города.
    population_estimate = float(total_area)

    far_share = float((dists > 500).mean())
    avg_distance = float(np.mean(dists))
    p90 = float(np.percentile(dists, 90))

    stops_per_10k = float(stops.shape[0] / max(population_estimate, 1) * 10_000)
    routes_per_10k = float(routes_count / max(population_estimate, 1) * 10_000)

    return ParsedDistrict(
        city=city,
        district=district,
        data_source="osm_real",
        bbox=bbox,
        buildings_count=int(buildings.shape[0]),
        stops_count=int(stops.shape[0]),
        routes_count=int(routes_count),
        residential_area_m2=float(total_area),
        population_estimate=population_estimate,
        avg_distance_m=avg_distance,
        p90_distance_m=p90,
        far_population_share_500m=far_share,
        accessibility_index=far_share,
        stops_per_10k=stops_per_10k,
        routes_per_10k=routes_per_10k,
        recommendation=recommendation_from_index(far_share),
    )


def run_accessibility_analysis(mode: str = "auto") -> pd.DataFrame:
    """Основной расчёт по всем микрорайонам Оренбурга и Тюмени."""
    records: List[ParsedDistrict] = []

    for city, cfg in CITY_CONFIG.items():
        districts = cfg["microdistricts"]
        for district in districts:
            if mode in {"auto", "real"}:
                try:
                    rec = analyze_district_real(city, district, cfg["population"], len(districts))
                    records.append(rec)
                    print(f"OK real: {city} — {district}")
                    continue
                except Exception as e:
                    if mode == "real":
                        print(f"FAILED real: {city} — {district}: {e}")
                        continue

            rec = fallback_district_metrics(city, district, cfg["population"], len(districts))
            records.append(rec)
            print(f"OK fallback: {city} — {district}")

    df = pd.DataFrame([r.__dict__ for r in records])

    # Масштабируем оценку населения внутри каждого города до общей численности.
    for city, cfg in CITY_CONFIG.items():
        mask = df["city"] == city
        total = df.loc[mask, "population_estimate"].sum()
        if total > 0:
            df.loc[mask, "population_estimate"] = df.loc[mask, "population_estimate"] / total * cfg["population"]
            df.loc[mask, "stops_per_10k"] = df.loc[mask, "stops_count"] / df.loc[mask, "population_estimate"] * 10_000
            df.loc[mask, "routes_per_10k"] = df.loc[mask, "routes_count"] / df.loc[mask, "population_estimate"] * 10_000

    return df


if __name__ == "__main__":
    df = run_accessibility_analysis(mode="fallback")
    print(df.head())
