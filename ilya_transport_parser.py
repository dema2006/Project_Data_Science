"""
ЧАСТЬ ИЛЬИ: сбор и парсинг транспортной инфраструктуры

Задача файла:
1. Хранить полный список микрорайонов Оренбурга и Тюмени
2. Геокодировать микрорайоны через Nominatim
3. Парсить через Overpass API:
   - жилые здания
   - остановки общественного транспорта
   - маршрутные relation public transport
4. Возвращать сырые таблицы buildings/stops/routes для дальнейшего анализа

Запуск отдельно нужен только для проверки парсинга. Обычно этот файл вызывается
из файла Артуша и файла Дёмы.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    from shapely.geometry import Polygon
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer
    HAS_GEO = True
except Exception:
    HAS_GEO = False


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
        "bbox": (51.65, 54.90, 51.90, 55.35),
        "microdistricts": ORENBURG_MICRODISTRICTS,
    },
    "Тюмень": {
        "population": 872_077,
        "bbox": (57.05, 65.20, 57.30, 65.80),
        "microdistricts": TYUMEN_MICRODISTRICTS,
    },
}


def stable_seed(text: str) -> int:
    """Нужен для воспроизводимого fallback-режима."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def project_lonlat_to_meters(lons: Iterable[float], lats: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Переводит координаты lon/lat в метры, чтобы можно было считать расстояния."""
    if not HAS_GEO:
        # Грубый fallback: 1 градус широты примерно 111 км.
        return np.array(list(lons)) * 111_000, np.array(list(lats)) * 111_000
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(list(lons), list(lats))
    return np.array(xs), np.array(ys)


def polygon_area_m2(coords: List[Tuple[float, float]]) -> float:
    """Считает площадь полигона здания в квадратных метрах."""
    if not HAS_GEO or len(coords) < 3:
        return 0.0
    poly = Polygon(coords)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    poly_m = shapely_transform(transformer.transform, poly)
    return float(abs(poly_m.area))


def polygon_centroid(coords: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Возвращает центр здания по его геометрии."""
    if HAS_GEO and len(coords) >= 3:
        poly = Polygon(coords)
        c = poly.centroid
        return float(c.x), float(c.y)
    lon = np.mean([c[0] for c in coords])
    lat = np.mean([c[1] for c in coords])
    return float(lon), float(lat)


def geocode_district(city: str, district: str, timeout: int = 12) -> Optional[Tuple[float, float, float, float]]:
    """Находит bbox микрорайона через Nominatim.

    Возвращает bbox в формате: south, west, north, east.
    Если район не найден или найден слишком большой объект, возвращает None.
    """
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
            bb = item.get("boundingbox")
            if not bb or len(bb) != 4:
                continue

            # Nominatim: south, north, west, east
            south, north, west, east = map(float, bb)

            # Отсекаем слишком крупные результаты: область, город целиком и т.п.
            if abs(north - south) > 0.15 or abs(east - west) > 0.25:
                continue
            return (south, west, north, east)
        except Exception:
            continue
        finally:
            time.sleep(1.0)
    return None


def overpass_query(query: str, timeout: int = 180) -> Optional[Dict]:
    """Выполняет запрос к Overpass API."""
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
    """Парсит здания, остановки и количество маршрутов внутри bbox микрорайона."""
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
        building_records.append({
            "id": el.get("id"),
            "lon": lon,
            "lat": lat,
            "x_m": float(x[0]),
            "y_m": float(y[0]),
            "area_m2": area,
        })

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
        stop_records.append({
            "id": el.get("id"),
            "lon": el["lon"],
            "lat": el["lat"],
            "x_m": float(x[0]),
            "y_m": float(y[0]),
        })

    route_ids = set()
    if routes_json is not None:
        for el in routes_json.get("elements", []):
            if el.get("type") == "relation":
                route_ids.add(el.get("id"))

    return pd.DataFrame(building_records), pd.DataFrame(stop_records), len(route_ids)


if __name__ == "__main__":
    # Быстрая проверка, что парсер запускается.
    city = "Оренбург"
    district = "Степной"
    bbox = geocode_district(city, district)
    print(f"bbox для {city} — {district}: {bbox}")
    if bbox:
        buildings, stops, routes_count = fetch_osm_objects_for_bbox(bbox)
        print(f"Здания: {len(buildings)}, остановки: {len(stops)}, маршруты: {routes_count}")
