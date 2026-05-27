"""
Парсер и аналитика транспортной доступности микрорайонов Оренбурга и Тюмени.

Что делает скрипт:
1. Берёт полный список микрорайонов Оренбурга и Тюмени.
2. В real-режиме геокодирует каждый микрорайон через Nominatim.
3. По найденному bbox запрашивает через Overpass:
   - жилые здания,
   - остановки общественного транспорта,
   - route relations общественного транспорта.
4. Считает расстояние от каждого жилого здания до ближайшей остановки.
5. Агрегирует показатели по микрорайонам:
   - количество зданий,
   - количество остановок,
   - количество маршрутов,
   - средняя и 90-процентильная дистанция до остановки,
   - доля населения дальше 500 м от остановки,
   - индекс транспортной недоступности.
6. Строит графики для презентации.

Важно:
- В средах без доступа к Nominatim/Overpass скрипт автоматически переходит в
  fallback-режим и генерирует воспроизводимые демо-данные. Такие графики нужны
  для проверки пайплайна и макета презентации, но финальные выводы лучше делать
  после запуска real-режима на компьютере с интернетом.

Запуск:
    python code/district_transport_parser.py --mode auto
    python code/district_transport_parser.py --mode real
    python code/district_transport_parser.py --mode fallback

Зависимости:
    pandas numpy matplotlib requests shapely pyproj scipy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    from shapely.geometry import Polygon, Point
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer
    HAS_GEO = True
except Exception:
    HAS_GEO = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


ORENBURG_MICRODISTRICTS = [
    "Авиагородок", "Аренда", "Беловка", "Берды", "Восточный", "2-й Восточный",
    "Заречный", "Звёздный", "Карачи", "Красный городок", "Красный посад",
    "Кузнечный", "Имени Куйбышева", "Кушкуль", "Маяк", "Малая земля",
    "Новая слободка", "Новостройка", "Октябрьский", "Овчинный городок",
    "Подмаячный (Маяк)", "Пороховые", "Пристанционный", "Пугачи", "Ренда",
    "Ростоши", "Северный", "Ситцовка", "Степной", "Сырейная площадь",
    "Форштадт", "Хлебный городок", "Южный",
]

TYUMEN_MICRODISTRICTS = [
    "Антипино", "Березняковский", "Быкова", "Верхний Бор", "Войновка",
    "Воронина", "Гилёва", "Дорожный", "Зайкова", "Казарово", "Княжева",
    "Комарово", "Копытова", "Матмасы", "Метелева", "Мыс", "Новорощино",
    "Парфёновский", "Плеханова", "Посёлок Мелиораторов", "Посёлок Механизаторов",
    "Рощино", "Суходольский", "Тараскуль", "Тарманы", "Труфанова", "Утешево",
]

CITY_CONFIG = {
    "Оренбург": {
        "population": 536_515,
        "bbox": (51.65, 54.90, 51.90, 55.35),  # south, west, north, east
        "microdistricts": ORENBURG_MICRODISTRICTS,
    },
    "Тюмень": {
        "population": 872_077,
        "bbox": (57.05, 65.20, 57.30, 65.80),
        "microdistricts": TYUMEN_MICRODISTRICTS,
    },
}


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


def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def project_lonlat_to_meters(lons: Iterable[float], lats: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(list(lons), list(lats))
    return np.array(xs), np.array(ys)


def polygon_area_m2(coords: List[Tuple[float, float]]) -> float:
    if not HAS_GEO or len(coords) < 3:
        return 0.0
    poly = Polygon(coords)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    poly_m = shapely_transform(transformer.transform, poly)
    return float(abs(poly_m.area))


def polygon_centroid(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    if HAS_GEO and len(coords) >= 3:
        poly = Polygon(coords)
        c = poly.centroid
        return float(c.x), float(c.y)
    lon = np.mean([c[0] for c in coords])
    lat = np.mean([c[1] for c in coords])
    return float(lon), float(lat)


def geocode_district(city: str, district: str, timeout: int = 12) -> Optional[Tuple[float, float, float, float]]:
    if not HAS_REQUESTS:
        return None
    query_variants = [
        f"{district}, {city}, Россия",
        f"микрорайон {district}, {city}, Россия",
        f"{district} район, {city}, Россия",
    ]
    headers = {"User-Agent": "transport-accessibility-student-project/1.0"}
    for q in query_variants:
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "json", "limit": 1, "polygon_geojson": 0},
                headers=headers,
                timeout=timeout,
            )
            if r.status_code != 200:
                continue
            data = r.json()
            if not data:
                continue
            item = data[0]
            # Nominatim boundingbox order: south, north, west, east
            bb = item.get("boundingbox")
            if not bb or len(bb) != 4:
                continue
            south, north, west, east = map(float, bb)
            # Reject country/region-scale results. Microdistrict bbox should be compact.
            if abs(north - south) > 0.15 or abs(east - west) > 0.25:
                continue
            return (south, west, north, east)
        except Exception:
            continue
        finally:
            time.sleep(1.0)
    return None


def overpass_query(query: str, timeout: int = 180) -> Optional[Dict]:
    if not HAS_REQUESTS:
        return None
    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": "transport-accessibility-student-project/1.0"},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_osm_objects_for_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    south, west, north, east = bbox
    buildings_query = f"""
    [out:json][timeout:180];
    (
      way["building"~"apartments|residential|house|detached|dormitory"]({south},{west},{north},{east});
      relation["building"~"apartments|residential|house|detached|dormitory"]({south},{west},{north},{east});
    );
    out geom;
    """
    stops_query = f"""
    [out:json][timeout:180];
    (
      node["highway"="bus_stop"]({south},{west},{north},{east});
      node["public_transport"="platform"]({south},{west},{north},{east});
      node["amenity"="bus_station"]({south},{west},{north},{east});
      node["railway"="tram_stop"]({south},{west},{north},{east});
    );
    out;
    """
    routes_query = f"""
    [out:json][timeout:180];
    (
      relation["type"="route"]["route"~"bus|trolleybus|tram|share_taxi"]({south},{west},{north},{east});
    );
    out tags;
    """
    buildings_json = overpass_query(buildings_query)
    stops_json = overpass_query(stops_query)
    routes_json = overpass_query(routes_query)
    if buildings_json is None or stops_json is None:
        raise RuntimeError("Overpass request failed")

    building_records = []
    for el in buildings_json.get("elements", []):
        geom = el.get("geometry")
        if not geom:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom]
        if len(coords) < 3:
            continue
        lon, lat = polygon_centroid(coords)
        area = polygon_area_m2(coords)
        if area <= 5:
            continue
        x, y = project_lonlat_to_meters([lon], [lat])
        building_records.append({"id": el.get("id"), "lon": lon, "lat": lat, "x_m": x[0], "y_m": y[0], "area_m2": area})

    stop_records = []
    seen = set()
    for el in stops_json.get("elements", []):
        if el.get("type") != "node" or "lon" not in el or "lat" not in el:
            continue
        key = (round(float(el["lon"]), 6), round(float(el["lat"]), 6))
        if key in seen:
            continue
        seen.add(key)
        x, y = project_lonlat_to_meters([el["lon"]], [el["lat"]])
        stop_records.append({"id": el.get("id"), "lon": el["lon"], "lat": el["lat"], "x_m": x[0], "y_m": y[0]})

    route_ids = set()
    if routes_json is not None:
        for el in routes_json.get("elements", []):
            if el.get("type") == "relation":
                route_ids.add(el.get("id"))

    return pd.DataFrame(building_records), pd.DataFrame(stop_records), len(route_ids)


def nearest_distances(buildings: pd.DataFrame, stops: pd.DataFrame) -> np.ndarray:
    if buildings.empty or stops.empty:
        return np.full(len(buildings), np.inf)
    b = buildings[["x_m", "y_m"]].to_numpy()
    s = stops[["x_m", "y_m"]].to_numpy()
    if HAS_SCIPY:
        tree = cKDTree(s)
        dists, _ = tree.query(b, k=1)
        return dists
    # Fallback for small dataframes
    dists = []
    for bx, by in b:
        diff = s - np.array([bx, by])
        dists.append(np.sqrt((diff * diff).sum(axis=1)).min())
    return np.array(dists)


def fallback_district_metrics(city: str, district: str, city_population: int, total_districts: int) -> ParsedDistrict:
    rng = np.random.default_rng(stable_seed(city + district))
    # Different typologies: peripheral, dense, mixed. Stable deterministic based on hash.
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
    # Allocate around equal district population but with deterministic spread.
    pop = city_population / total_districts * rng.uniform(0.45, 1.65)
    p90 = avg_distance * rng.uniform(1.35, 1.85)
    far_share = 1 / (1 + math.exp(-(avg_distance - 500) / 150))
    far_share = float(np.clip(far_share + rng.normal(0, 0.06), 0, 1))
    stops_per_10k = stops_count / pop * 10_000
    routes_per_10k = routes_count / pop * 10_000
    if far_share >= 0.65:
        rec = "Приоритет: добавить остановки и продлить/запустить 1–2 маршрута"
    elif far_share >= 0.35:
        rec = "Средний приоритет: увеличить частоту и проверить покрытие остановками"
    else:
        rec = "Низкий приоритет: поддерживать текущую сеть и мониторить рост застройки"
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
        recommendation=rec,
    )


def analyze_district_real(city: str, district: str, city_population: int, total_districts: int) -> ParsedDistrict:
    bbox = geocode_district(city, district)
    data_source = "osm_real"
    if bbox is None:
        # fallback to city bbox with tiny random crop around deterministic pseudo location,
        # but mark clearly that it is not a real district boundary.
        raise RuntimeError("Cannot geocode district")
    buildings, stops, routes_count = fetch_osm_objects_for_bbox(bbox)
    if buildings.empty or stops.empty:
        raise RuntimeError("No buildings or stops found")
    dists = nearest_distances(buildings, stops)
    total_area = buildings["area_m2"].sum()
    # This estimates population within one district from building area. The final post-processing
    # rescales all districts to city population.
    pop_raw = total_area
    population_estimate = float(pop_raw)
    far_share = float((dists > 500).mean())
    avg_distance = float(np.mean(dists))
    p90 = float(np.percentile(dists, 90))
    stops_per_10k = float(stops.shape[0] / max(population_estimate, 1) * 10_000)
    routes_per_10k = float(routes_count / max(population_estimate, 1) * 10_000)
    if far_share >= 0.65:
        rec = "Приоритет: добавить остановки и продлить/запустить 1–2 маршрута"
    elif far_share >= 0.35:
        rec = "Средний приоритет: увеличить частоту и проверить покрытие остановками"
    else:
        rec = "Низкий приоритет: поддерживать текущую сеть и мониторить рост застройки"
    return ParsedDistrict(
        city=city,
        district=district,
        data_source=data_source,
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
        recommendation=rec,
    )


def run_analysis(mode: str) -> pd.DataFrame:
    records: List[ParsedDistrict] = []
    for city, cfg in CITY_CONFIG.items():
        districts = cfg["microdistricts"]
        for d in districts:
            if mode in {"auto", "real"}:
                try:
                    rec = analyze_district_real(city, d, cfg["population"], len(districts))
                    records.append(rec)
                    print(f"OK real: {city} — {d}")
                    continue
                except Exception as e:
                    if mode == "real":
                        print(f"FAILED real: {city} — {d}: {e}")
                        continue
            rec = fallback_district_metrics(city, d, cfg["population"], len(districts))
            records.append(rec)
            print(f"OK fallback: {city} — {d}")
    df = pd.DataFrame([r.__dict__ for r in records])

    # If real data were used, rescale population estimates within each city so all districts sum to city population.
    for city, cfg in CITY_CONFIG.items():
        mask = df["city"] == city
        total = df.loc[mask, "population_estimate"].sum()
        if total > 0:
            df.loc[mask, "population_estimate"] = df.loc[mask, "population_estimate"] / total * cfg["population"]
            df.loc[mask, "stops_per_10k"] = df.loc[mask, "stops_count"] / df.loc[mask, "population_estimate"] * 10_000
            df.loc[mask, "routes_per_10k"] = df.loc[mask, "routes_count"] / df.loc[mask, "population_estimate"] * 10_000
    return df


def save_barh(df: pd.DataFrame, city: str, metric: str, title: str, xlabel: str, path: Path, top_n: Optional[int] = None) -> None:
    sub = df[df["city"] == city].sort_values(metric, ascending=True)
    if top_n:
        sub = sub.tail(top_n)
    plt.figure(figsize=(12, max(6, 0.32 * len(sub))))
    bars = plt.barh(sub["district"], sub[metric])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Микрорайон")
    if metric in {"accessibility_index", "far_population_share_500m"}:
        plt.axvline(0.35, linestyle="--", linewidth=1, label="Средний риск")
        plt.axvline(0.65, linestyle="--", linewidth=1, label="Высокий риск")
        plt.legend(loc="lower right")
    for b in bars:
        v = b.get_width()
        label = f"{v:.2f}" if v < 10 else f"{v:.0f}"
        plt.text(v + (sub[metric].max() * 0.01 if sub[metric].max() else 0.01), b.get_y() + b.get_height()/2, label, va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_graphs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "microdistricts").mkdir(exist_ok=True)
    (out_dir / "cities").mkdir(exist_ok=True)

    for city in df["city"].unique():
        save_barh(
            df, city, "accessibility_index",
            f"{city}: индекс транспортной недоступности по микрорайонам",
            "Индекс: доля населения дальше 500 м от остановки",
            out_dir / "microdistricts" / f"{city}_microdistrict_accessibility_index.png",
        )
        save_barh(
            df, city, "avg_distance_m",
            f"{city}: среднее расстояние до ближайшей остановки",
            "Среднее расстояние, м",
            out_dir / "microdistricts" / f"{city}_avg_distance_to_stop.png",
        )
        save_barh(
            df, city, "stops_per_10k",
            f"{city}: плотность остановок на 10 тыс. жителей",
            "Остановок на 10 тыс. жителей",
            out_dir / "microdistricts" / f"{city}_stops_per_10k.png",
        )
        plt.figure(figsize=(8, 4))
        vals = df[df["city"] == city]["accessibility_index"]
        plt.hist(vals, bins=10, edgecolor="black")
        plt.title(f"{city}: распределение индекса недоступности")
        plt.xlabel("Индекс недоступности")
        plt.ylabel("Количество микрорайонов")
        plt.tight_layout()
        plt.savefig(out_dir / "microdistricts" / f"{city}_accessibility_histogram.png", dpi=180)
        plt.close()

    # Top underserved across both cities
    top = df.sort_values("accessibility_index", ascending=True).tail(18)
    plt.figure(figsize=(12, 8))
    labels = top["city"] + " — " + top["district"]
    plt.barh(labels, top["accessibility_index"])
    plt.title("Топ проблемных микрорайонов по индексу транспортной недоступности")
    plt.xlabel("Индекс: доля населения дальше 500 м от остановки")
    plt.axvline(0.65, linestyle="--", linewidth=1, label="Высокий риск")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "microdistricts" / "top_underserved_microdistricts.png", dpi=180)
    plt.close()

    # City comparison
    city_summary = df.groupby("city").agg(
        mean_index=("accessibility_index", "mean"),
        median_index=("accessibility_index", "median"),
        mean_avg_distance=("avg_distance_m", "mean"),
        high_risk_districts=("accessibility_index", lambda s: int((s >= 0.65).sum())),
        districts=("district", "count"),
        stops_per_10k=("stops_count", "sum"),
        population=("population_estimate", "sum"),
    ).reset_index()
    city_summary["stops_per_10k"] = city_summary["stops_per_10k"] / city_summary["population"] * 10_000
    city_summary.to_csv(out_dir.parent / "results" / "city_summary.csv", index=False)

    metrics = [
        ("mean_index", "Средний индекс недоступности", "Индекс"),
        ("mean_avg_distance", "Среднее расстояние до остановки по городу", "Метры"),
        ("high_risk_districts", "Количество микрорайонов высокого риска", "Количество"),
        ("stops_per_10k", "Остановки на 10 тыс. жителей", "Остановок/10 тыс."),
    ]
    for metric, title, ylabel in metrics:
        plt.figure(figsize=(6, 4))
        bars = plt.bar(city_summary["city"], city_summary[metric])
        plt.title(title)
        plt.ylabel(ylabel)
        for b in bars:
            v = b.get_height()
            label = f"{v:.2f}" if v < 10 else f"{v:.0f}"
            plt.text(b.get_x()+b.get_width()/2, v, label, ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(out_dir / "cities" / f"city_{metric}.png", dpi=180)
        plt.close()

    # Improvement scenario: add stops/routes for high-risk top 12
    scenario = df.sort_values("accessibility_index", ascending=False).head(12).copy()
    scenario["after_index"] = np.maximum(scenario["accessibility_index"] * 0.68 - 0.04, 0)
    scenario["label"] = scenario["city"] + " — " + scenario["district"]
    x = np.arange(len(scenario))
    width = 0.38
    plt.figure(figsize=(13, 6))
    plt.bar(x - width/2, scenario["accessibility_index"], width, label="До улучшений")
    plt.bar(x + width/2, scenario["after_index"], width, label="После: +2 остановки и продление маршрута")
    plt.xticks(x, scenario["label"], rotation=45, ha="right")
    plt.ylabel("Индекс недоступности")
    plt.title("Сценарий улучшения для самых проблемных микрорайонов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "microdistricts" / "improvement_scenario_top12.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "real", "fallback"], default="auto")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    graphs_dir = project_dir / "graphs"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)

    df = run_analysis(args.mode)
    df.to_csv(results_dir / "microdistrict_accessibility_results.csv", index=False)
    generate_graphs(df, graphs_dir)

    with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "note": "Rows with data_source=synthetic_fallback are generated fallback data, not real OSM measurements.",
            "cities": list(CITY_CONFIG.keys()),
            "districts_total": int(df.shape[0]),
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {results_dir}")
    print(f"Saved graphs to {graphs_dir}")


if __name__ == "__main__":
    main()
