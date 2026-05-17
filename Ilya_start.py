from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from urllib3.util.retry import Retry


@dataclass
class ParserConfig:
    top_cities_url: str = "https://www.r-statistics.ru/po-chislennosti"
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    out_dir: str = "data"
    top_n: int = 20
    timeout: int = 180
    sleep_sec: float = 1.0
    user_agent: str = (
        "UrbanMetricsConsultingStudentParser/1.0 "
        "(educational data science project; contact: student)"
    )


class TextCleaner:
    @staticmethod
    def clean_text(value: object) -> Optional[str]:
        if pd.isna(value):
            return None

        text = str(value)
        text = text.replace("\xa0", " ")
        text = re.sub(r"\[[^\]]*]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" \n\t\r.,;")

        if not text or text.lower() == "nan":
            return None

        return text

    @staticmethod
    def normalize_city_name(value: object) -> Optional[str]:
        text = TextCleaner.clean_text(value)

        if not text:
            return None

        text = text.replace("—", "-")
        text = text.replace("–", "-")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    @staticmethod
    def parse_int(value: object) -> Optional[int]:
        if pd.isna(value):
            return None

        text = str(value)
        text = text.replace("\xa0", " ")
        text = re.sub(r"\[[^\]]*]", "", text)
        text = re.sub(r"[^\d]", "", text)

        if not text:
            return None

        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def slug(value: str) -> str:
        value = value.lower()
        value = value.replace("ё", "е")
        value = value.replace("—", "-")
        value = value.replace("–", "-")
        value = re.sub(r"[^a-zа-я0-9]+", "_", value)
        value = value.strip("_")
        return value


class HttpClient:
    def __init__(self, config: ParserConfig):
        self.config = config
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            }
        )

        return session

    def get_html(self, url: str) -> str:
        logging.info("Загружаю страницу: %s", url)

        response = self.session.get(url, timeout=self.config.timeout)
        response.raise_for_status()

        time.sleep(self.config.sleep_sec)

        logging.info("Страница загружена: %s символов", len(response.text))

        return response.text

    def get_json(self, url: str, params: dict) -> dict | list:
        logging.info("GET JSON: %s", url)

        response = self.session.get(
            url,
            params=params,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        time.sleep(self.config.sleep_sec)

        return response.json()

    def post_overpass(self, query: str) -> dict:
        logging.info("Отправляю запрос в Overpass API")

        response = self.session.post(
            self.config.overpass_url,
            data={"data": query},
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        time.sleep(self.config.sleep_sec)

        data = response.json()

        logging.info(
            "Overpass вернул элементов: %s",
            len(data.get("elements", [])),
        )

        return data


class FileStorage:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)

        self.raw_dir = self.out_dir / "raw"
        self.processed_dir = self.out_dir / "processed"
        self.final_dir = self.out_dir / "final"

        for folder in [self.raw_dir, self.processed_dir, self.final_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def save_text(self, text: str, folder: str, filename: str) -> Path:
        path = self._get_folder(folder) / filename
        path.write_text(text, encoding="utf-8")
        logging.info("Файл сохранен: %s", path)
        return path

    def save_json(self, data: dict | list, folder: str, filename: str) -> Path:
        path = self._get_folder(folder) / filename
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("JSON сохранен: %s", path)
        return path

    def save_csv(self, df: pd.DataFrame, folder: str, filename: str) -> Path:
        path = self._get_folder(folder) / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logging.info("CSV сохранен: %s", path)
        return path

    def _get_folder(self, folder: str) -> Path:
        if folder == "raw":
            return self.raw_dir

        if folder == "processed":
            return self.processed_dir

        if folder == "final":
            return self.final_dir

        raise ValueError(f"Неизвестная папка: {folder}")


class TopCitiesParser:
    def __init__(self, config: ParserConfig, client: HttpClient, storage: FileStorage):
        self.config = config
        self.client = client
        self.storage = storage

    def parse_top_cities(self) -> pd.DataFrame:
        html = self.client.get_html(self.config.top_cities_url)

        self.storage.save_text(
            html,
            "raw",
            "top_cities_source_page_transport.html",
        )

        tables = pd.read_html(io.StringIO(html))
        raw_table = self._find_city_table(tables)

        self.storage.save_csv(
            raw_table,
            "raw",
            "top_cities_raw_transport.csv",
        )

        df = self._clean_city_table(raw_table)
        df = df.head(self.config.top_n).reset_index(drop=True)

        self.storage.save_csv(
            df,
            "processed",
            "top_20_russian_cities_transport.csv",
        )

        return df

    def _find_city_table(self, tables: list[pd.DataFrame]) -> pd.DataFrame:
        best_score = -1
        best_table = None

        for table in tables:
            if table.empty:
                continue

            columns_text = " ".join([str(c).lower() for c in table.columns])
            sample_text = " ".join(
                table.fillna("").astype(str).head(10).values.flatten()
            ).lower()

            full_text = columns_text + " " + sample_text

            score = 0

            if "город" in full_text:
                score += 2

            if "население" in full_text:
                score += 2

            if "регион" in full_text:
                score += 1

            if "федеральный" in full_text:
                score += 1

            if len(table) >= 20:
                score += 1

            if score > best_score:
                best_score = score
                best_table = table

        if best_table is None or best_score < 3:
            raise RuntimeError("Не удалось найти таблицу с городами.")

        return best_table.copy()

    def _clean_city_table(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self._normalize_col(c) for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)

        city_col = self._find_col(df, ["город", "название"])
        population_col = self._find_col(df, ["население"])
        region_col = self._find_col(df, ["регион", "субъект"], required=False)
        district_col = self._find_col(
            df,
            ["федеральный округ", "федеральный"],
            required=False,
        )
        timezone_col = self._find_col(
            df,
            ["часовой пояс", "часовой"],
            required=False,
        )

        result = pd.DataFrame()

        result["rank"] = range(1, len(df) + 1)
        result["city"] = df[city_col].apply(TextCleaner.normalize_city_name)
        result["population"] = df[population_col].apply(TextCleaner.parse_int)

        if region_col:
            result["region"] = df[region_col].apply(TextCleaner.clean_text)
        else:
            result["region"] = None

        if district_col:
            result["federal_district"] = df[district_col].apply(TextCleaner.clean_text)
        else:
            result["federal_district"] = None

        if timezone_col:
            result["timezone"] = df[timezone_col].apply(TextCleaner.clean_text)
        else:
            result["timezone"] = None

        result = result.dropna(subset=["city", "population"])
        result["population"] = result["population"].astype(int)

        bad_exact_values = {
            "город",
            "название",
            "итого",
            "всего",
            "население",
        }

        result = result[
            ~result["city"]
            .astype(str)
            .str.lower()
            .str.strip()
            .isin(bad_exact_values)
        ]

        result = result.reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        return result

    @staticmethod
    def _normalize_col(col: object) -> str:
        if isinstance(col, tuple):
            col = " ".join(str(x) for x in col if str(x) != "nan")

        col = str(col)
        col = col.replace("\xa0", " ")
        col = re.sub(r"\s+", " ", col)
        col = col.strip().lower()

        return col

    @staticmethod
    def _find_col(
        df: pd.DataFrame,
        keywords: list[str],
        required: bool = True,
    ) -> Optional[str]:
        for col in df.columns:
            col_lower = str(col).lower()

            if any(keyword in col_lower for keyword in keywords):
                return col

        if required:
            raise RuntimeError(
                f"Не найдена колонка по ключам {keywords}. "
                f"Доступные колонки: {list(df.columns)}"
            )

        return None


class CitySelector:
    @staticmethod
    def choose_city(top_cities: pd.DataFrame) -> dict:
        print("\nВыбери город для анализа транспортной инфраструктуры:\n")

        for _, row in top_cities.iterrows():
            population = f"{int(row['population']):,}".replace(",", " ")
            print(f"{int(row['rank']):>2}. {row['city']} — {population} чел.")

        while True:
            value = input("\nВведи номер города из списка: ").strip()

            if not value.isdigit():
                print("Нужно ввести число.")
                continue

            number = int(value)

            if number < 1 or number > len(top_cities):
                print(f"Нужно выбрать число от 1 до {len(top_cities)}.")
                continue

            selected = top_cities.iloc[number - 1].to_dict()

            print(f"\nВыбран город: {selected['city']}\n")

            return selected


class CityBoundaryResolver:
    def __init__(self, config: ParserConfig, client: HttpClient, storage: FileStorage):
        self.config = config
        self.client = client
        self.storage = storage

    def resolve_city(self, city: str) -> dict:
        params = {
            "q": f"{city}, Россия",
            "format": "json",
            "limit": 5,
            "addressdetails": 1,
            "polygon_geojson": 0,
        }

        data = self.client.get_json(self.config.nominatim_url, params=params)

        slug = TextCleaner.slug(city)

        self.storage.save_json(
            data,
            "raw",
            f"{slug}_nominatim_candidates.json",
        )

        if not data:
            raise RuntimeError(f"Nominatim не нашел город: {city}")

        best = self._choose_best_candidate(data, city)

        osm_type = best.get("osm_type")
        osm_id = int(best.get("osm_id"))

        if osm_type == "relation":
            overpass_area_id = 3600000000 + osm_id
        elif osm_type == "way":
            overpass_area_id = 2400000000 + osm_id
        else:
            raise RuntimeError(
                f"Для города {city} найден osm_type={osm_type}. "
                f"Для Overpass area нужен relation или way."
            )

        bbox = best.get("boundingbox", [])

        result = {
            "city": city,
            "display_name": best.get("display_name"),
            "osm_type": osm_type,
            "osm_id": osm_id,
            "overpass_area_id": overpass_area_id,
            "lat": float(best.get("lat")),
            "lon": float(best.get("lon")),
            "boundingbox": bbox,
        }

        self.storage.save_json(
            result,
            "processed",
            f"{slug}_city_boundary.json",
        )

        return result

    @staticmethod
    def _choose_best_candidate(candidates: list[dict], city: str) -> dict:
        city_lower = city.lower()

        for item in candidates:
            display = item.get("display_name", "").lower()
            osm_type = item.get("osm_type")

            if city_lower in display and osm_type == "relation":
                return item

        for item in candidates:
            if item.get("osm_type") in {"relation", "way"}:
                return item

        return candidates[0]


class OverpassTransportParser:
    def __init__(self, config: ParserConfig, client: HttpClient, storage: FileStorage):
        self.config = config
        self.client = client
        self.storage = storage

    def parse_transport_stops(self, city_info: dict) -> pd.DataFrame:
        city = city_info["city"]
        area_id = city_info["overpass_area_id"]
        slug = TextCleaner.slug(city)

        query = f"""
        [out:json][timeout:{self.config.timeout}];
        area(id:{area_id})->.searchArea;
        (
          node["highway"="bus_stop"](area.searchArea);
          node["public_transport"="platform"](area.searchArea);
          node["public_transport"="stop_position"](area.searchArea);
          node["railway"="tram_stop"](area.searchArea);
          node["railway"="station"](area.searchArea);
          node["railway"="halt"](area.searchArea);
          node["railway"="subway_entrance"](area.searchArea);
          node["amenity"="bus_station"](area.searchArea);
        );
        out body;
        """

        data = self.client.post_overpass(query)

        self.storage.save_json(
            data,
            "raw",
            f"{slug}_transport_stops_overpass_raw.json",
        )

        rows = []

        for element in data.get("elements", []):
            tags = element.get("tags", {})

            if element.get("type") != "node":
                continue

            lat = element.get("lat")
            lon = element.get("lon")

            if lat is None or lon is None:
                continue

            rows.append(
                {
                    "city": city,
                    "osm_type": element.get("type"),
                    "osm_id": element.get("id"),
                    "name": tags.get("name"),
                    "lat": lat,
                    "lon": lon,
                    "transport_type": self._classify_transport_type(tags),
                    "highway": tags.get("highway"),
                    "public_transport": tags.get("public_transport"),
                    "railway": tags.get("railway"),
                    "amenity": tags.get("amenity"),
                    "bus": tags.get("bus"),
                    "tram": tags.get("tram"),
                    "trolleybus": tags.get("trolleybus"),
                    "subway": tags.get("subway"),
                    "train": tags.get("train"),
                    "operator": tags.get("operator"),
                    "network": tags.get("network"),
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            logging.warning("Остановки транспорта не найдены.")
            return df

        df = df.drop_duplicates(subset=["osm_type", "osm_id"])
        df = df.reset_index(drop=True)

        self.storage.save_csv(
            df,
            "processed",
            f"{slug}_transport_stops_processed.csv",
        )

        return df

    def parse_transport_routes(self, city_info: dict) -> pd.DataFrame:
        city = city_info["city"]
        area_id = city_info["overpass_area_id"]
        slug = TextCleaner.slug(city)

        query = f"""
        [out:json][timeout:{self.config.timeout}];
        area(id:{area_id})->.searchArea;
        (
          relation["type"="route"]["route"~"bus|tram|trolleybus|subway|train"](area.searchArea);
        );
        out tags;
        """

        data = self.client.post_overpass(query)

        self.storage.save_json(
            data,
            "raw",
            f"{slug}_transport_routes_overpass_raw.json",
        )

        rows = []

        for element in data.get("elements", []):
            tags = element.get("tags", {})

            rows.append(
                {
                    "city": city,
                    "osm_type": element.get("type"),
                    "osm_id": element.get("id"),
                    "name": tags.get("name"),
                    "ref": tags.get("ref"),
                    "route": tags.get("route"),
                    "operator": tags.get("operator"),
                    "network": tags.get("network"),
                    "from": tags.get("from"),
                    "to": tags.get("to"),
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            logging.warning("Маршруты транспорта не найдены.")
            return df

        df = df.drop_duplicates(subset=["osm_type", "osm_id"])
        df = df.reset_index(drop=True)

        self.storage.save_csv(
            df,
            "processed",
            f"{slug}_transport_routes_processed.csv",
        )

        return df

    @staticmethod
    def _classify_transport_type(tags: dict) -> str:
        highway = tags.get("highway")
        public_transport = tags.get("public_transport")
        railway = tags.get("railway")
        amenity = tags.get("amenity")

        if amenity == "bus_station":
            return "bus_station"

        if railway == "subway_entrance":
            return "subway_entrance"

        if railway == "tram_stop":
            return "tram_stop"

        if railway in {"station", "halt"}:
            if tags.get("station") == "subway" or tags.get("subway") == "yes":
                return "metro_station"
            return "railway_station"

        if highway == "bus_stop":
            return "bus_stop"

        if public_transport == "stop_position":
            return "stop_position"

        if public_transport == "platform":
            if tags.get("tram") == "yes":
                return "tram_platform"
            if tags.get("bus") == "yes":
                return "bus_platform"
            if tags.get("trolleybus") == "yes":
                return "trolleybus_platform"
            return "public_transport_platform"

        return "other_transport_point"


class OverpassDistrictParser:
    def __init__(self, config: ParserConfig, client: HttpClient, storage: FileStorage):
        self.config = config
        self.client = client
        self.storage = storage

    def parse_districts(self, city_info: dict) -> pd.DataFrame:
        city = city_info["city"]
        area_id = city_info["overpass_area_id"]
        slug = TextCleaner.slug(city)

        query = f"""
        [out:json][timeout:{self.config.timeout}];
        area(id:{area_id})->.searchArea;
        (
          relation["boundary"="administrative"]["admin_level"~"8|9|10|11"](area.searchArea);
        );
        out tags geom;
        """

        data = self.client.post_overpass(query)

        self.storage.save_json(
            data,
            "raw",
            f"{slug}_district_boundaries_overpass_raw.json",
        )

        rows = []

        for element in data.get("elements", []):
            tags = element.get("tags", {})

            name = tags.get("name")

            if not name:
                continue

            if self._is_bad_district_name(name):
                continue

            geometry = self._relation_to_geometry(element)

            if geometry is None or geometry.is_empty:
                continue

            area_km2 = GeoUtils.geometry_area_km2(geometry)

            rows.append(
                {
                    "city": city,
                    "osm_type": element.get("type"),
                    "osm_id": element.get("id"),
                    "district": self._clean_district_name(name),
                    "admin_level": tags.get("admin_level"),
                    "official_status": tags.get("official_status"),
                    "area_km2_osm": round(area_km2, 2) if area_km2 else None,
                    "geometry_wkt": geometry.wkt,
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            logging.warning("Районы через Overpass не найдены.")
            return df

        df = df.drop_duplicates(subset=["district"])
        df = df.sort_values(["admin_level", "district"]).reset_index(drop=True)

        self.storage.save_csv(
            df.drop(columns=["geometry_wkt"]),
            "processed",
            f"{slug}_district_boundaries_processed.csv",
        )

        self.storage.save_json(
            df.to_dict(orient="records"),
            "processed",
            f"{slug}_district_boundaries_with_geometry.json",
        )

        return df

    @staticmethod
    def _relation_to_geometry(element: dict) -> Optional[Polygon | MultiPolygon]:
        members = element.get("members", [])

        polygons = []

        for member in members:
            geometry = member.get("geometry")

            if not geometry:
                continue

            coords = []

            for point in geometry:
                lat = point.get("lat")
                lon = point.get("lon")

                if lat is None or lon is None:
                    continue

                coords.append((lon, lat))

            if len(coords) < 4:
                continue

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            try:
                polygon = Polygon(coords)

                if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
                    polygons.append(polygon)
            except Exception:
                continue

        if not polygons:
            return None

        try:
            merged = unary_union(polygons)

            if merged.is_empty:
                return None

            return merged
        except Exception:
            return None

    @staticmethod
    def _is_bad_district_name(name: str) -> bool:
        text = name.lower()

        bad_words = [
            "город",
            "городской округ",
            "муниципальный округ",
            "поселение",
            "сельское поселение",
            "район области",
            "муниципальное образование",
        ]

        return any(word in text for word in bad_words)

    @staticmethod
    def _clean_district_name(name: str) -> str:
        text = TextCleaner.clean_text(name) or name

        text = text.replace("район", "")
        text = text.replace("Район", "")
        text = text.replace("административный", "")
        text = text.replace("Административный", "")
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" —-.,;")

        return text


class GeoUtils:
    @staticmethod
    def geometry_area_km2(geometry: Polygon | MultiPolygon) -> Optional[float]:
        try:
            if isinstance(geometry, Polygon):
                return GeoUtils._polygon_area_km2(geometry)

            if isinstance(geometry, MultiPolygon):
                return sum(GeoUtils._polygon_area_km2(poly) for poly in geometry.geoms)

            return None
        except Exception:
            return None

    @staticmethod
    def _polygon_area_km2(poly: Polygon) -> float:
        coords = list(poly.exterior.coords)

        if len(coords) < 3:
            return 0.0

        avg_lat = sum(lat for lon, lat in coords) / len(coords)

        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(math.radians(avg_lat))

        projected = [
            (lon * km_per_deg_lon, lat * km_per_deg_lat)
            for lon, lat in coords
        ]

        area = 0.0

        for i in range(len(projected) - 1):
            x1, y1 = projected[i]
            x2, y2 = projected[i + 1]
            area += x1 * y2 - x2 * y1

        return abs(area) / 2


class TransportDistrictAggregator:
    def __init__(self, storage: FileStorage):
        self.storage = storage

    def aggregate(
        self,
        city: str,
        stops: pd.DataFrame,
        routes: pd.DataFrame,
        districts: pd.DataFrame,
    ) -> pd.DataFrame:
        slug = TextCleaner.slug(city)

        if stops.empty:
            logging.warning("Нет остановок для агрегации.")
            return pd.DataFrame()

        if districts.empty:
            logging.warning("Нет районов для агрегации.")
            return pd.DataFrame()

        district_geometries = self._load_district_geometries(districts)

        assigned_stops = []

        for _, stop in stops.iterrows():
            point = Point(float(stop["lon"]), float(stop["lat"]))

            district_name = self._find_point_district(point, district_geometries)

            row = stop.to_dict()
            row["district"] = district_name

            assigned_stops.append(row)

        stops_with_district = pd.DataFrame(assigned_stops)

        self.storage.save_csv(
            stops_with_district,
            "processed",
            f"{slug}_transport_stops_with_district.csv",
        )

        known_stops = stops_with_district.dropna(subset=["district"])

        if known_stops.empty:
            logging.warning("Не удалось распределить остановки по районам.")
            return pd.DataFrame()

        base = (
            known_stops
            .groupby("district")
            .agg(
                total_transport_points=("osm_id", "count"),
                named_transport_points=("name", lambda x: x.notna().sum()),
            )
            .reset_index()
        )

        type_pivot = (
            known_stops
            .pivot_table(
                index="district",
                columns="transport_type",
                values="osm_id",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )

        result = base.merge(type_pivot, on="district", how="left")

        district_area = districts[["district", "area_km2_osm"]].copy()
        result = result.merge(district_area, on="district", how="left")

        result["transport_points_per_km2"] = (
            result["total_transport_points"] / result["area_km2_osm"]
        ).round(2)

        route_summary = self._make_route_summary(routes)
        result["city_route_count_total"] = route_summary["total_routes"]
        result["city_bus_routes"] = route_summary["bus_routes"]
        result["city_tram_routes"] = route_summary["tram_routes"]
        result["city_trolleybus_routes"] = route_summary["trolleybus_routes"]
        result["city_subway_routes"] = route_summary["subway_routes"]
        result["city_train_routes"] = route_summary["train_routes"]

        result["transport_rank"] = (
            result["total_transport_points"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        result["density_rank_transport"] = (
            result["transport_points_per_km2"]
            .rank(method="dense", ascending=False)
            .astype("Int64")
        )

        result = result.sort_values("transport_rank").reset_index(drop=True)

        self.storage.save_csv(
            result,
            "final",
            f"{slug}_transport_by_district_final.csv",
        )

        return result

    @staticmethod
    def _load_district_geometries(
        districts: pd.DataFrame,
    ) -> list[tuple[str, Polygon | MultiPolygon]]:
        result = []

        for _, row in districts.iterrows():
            wkt = row.get("geometry_wkt")

            if not wkt:
                continue

            try:
                from shapely import wkt as shapely_wkt

                geom = shapely_wkt.loads(wkt)
                result.append((row["district"], geom))
            except Exception:
                continue

        return result

    @staticmethod
    def _find_point_district(
        point: Point,
        district_geometries: list[tuple[str, Polygon | MultiPolygon]],
    ) -> Optional[str]:
        for district_name, geometry in district_geometries:
            try:
                if geometry.contains(point) or geometry.touches(point):
                    return district_name
            except Exception:
                continue

        return None

    @staticmethod
    def _make_route_summary(routes: pd.DataFrame) -> dict:
        if routes.empty or "route" not in routes.columns:
            return {
                "total_routes": 0,
                "bus_routes": 0,
                "tram_routes": 0,
                "trolleybus_routes": 0,
                "subway_routes": 0,
                "train_routes": 0,
            }

        return {
            "total_routes": int(len(routes)),
            "bus_routes": int((routes["route"] == "bus").sum()),
            "tram_routes": int((routes["route"] == "tram").sum()),
            "trolleybus_routes": int((routes["route"] == "trolleybus").sum()),
            "subway_routes": int((routes["route"] == "subway").sum()),
            "train_routes": int((routes["route"] == "train").sum()),
        }


class CityLevelTransportSummary:
    def __init__(self, storage: FileStorage):
        self.storage = storage

    def make_summary(
        self,
        city: str,
        city_info: dict,
        stops: pd.DataFrame,
        routes: pd.DataFrame,
    ) -> pd.DataFrame:
        slug = TextCleaner.slug(city)

        if stops.empty:
            stop_counts = {}
            total_stops = 0
        else:
            stop_counts = stops["transport_type"].value_counts().to_dict()
            total_stops = len(stops)

        if routes.empty:
            route_counts = {}
            total_routes = 0
        else:
            route_counts = routes["route"].value_counts().to_dict()
            total_routes = len(routes)

        row = {
            "city": city,
            "osm_type": city_info.get("osm_type"),
            "osm_id": city_info.get("osm_id"),
            "overpass_area_id": city_info.get("overpass_area_id"),
            "total_transport_points": total_stops,
            "total_routes": total_routes,
            "bus_stops": stop_counts.get("bus_stop", 0),
            "bus_platforms": stop_counts.get("bus_platform", 0),
            "tram_stops": stop_counts.get("tram_stop", 0),
            "tram_platforms": stop_counts.get("tram_platform", 0),
            "metro_stations": stop_counts.get("metro_station", 0),
            "subway_entrances": stop_counts.get("subway_entrance", 0),
            "railway_stations": stop_counts.get("railway_station", 0),
            "bus_stations": stop_counts.get("bus_station", 0),
            "bus_routes": route_counts.get("bus", 0),
            "tram_routes": route_counts.get("tram", 0),
            "trolleybus_routes": route_counts.get("trolleybus", 0),
            "subway_routes": route_counts.get("subway", 0),
            "train_routes": route_counts.get("train", 0),
        }

        df = pd.DataFrame([row])

        self.storage.save_csv(
            df,
            "final",
            f"{slug}_transport_city_summary_final.csv",
        )

        return df


class TransportAnalysisPipeline:
    def __init__(self, config: ParserConfig):
        self.config = config
        self.client = HttpClient(config)
        self.storage = FileStorage(config.out_dir)

        self.top_parser = TopCitiesParser(config, self.client, self.storage)
        self.boundary_resolver = CityBoundaryResolver(config, self.client, self.storage)
        self.transport_parser = OverpassTransportParser(config, self.client, self.storage)
        self.district_parser = OverpassDistrictParser(config, self.client, self.storage)
        self.aggregator = TransportDistrictAggregator(self.storage)
        self.city_summary = CityLevelTransportSummary(self.storage)

    def run(self, city_number: Optional[int] = None) -> None:
        top_cities = self.top_parser.parse_top_cities()

        if city_number is None:
            selected = CitySelector.choose_city(top_cities)
        else:
            if city_number < 1 or city_number > len(top_cities):
                raise ValueError(f"city_number должен быть от 1 до {len(top_cities)}")

            selected = top_cities.iloc[city_number - 1].to_dict()
            print(f"\nВыбран город: {selected['city']}\n")

        city = selected["city"]
        slug = TextCleaner.slug(city)

        city_info = self.boundary_resolver.resolve_city(city)

        stops = self.transport_parser.parse_transport_stops(city_info)
        routes = self.transport_parser.parse_transport_routes(city_info)
        districts = self.district_parser.parse_districts(city_info)

        city_summary = self.city_summary.make_summary(
            city=city,
            city_info=city_info,
            stops=stops,
            routes=routes,
        )

        district_summary = pd.DataFrame()

        if not districts.empty:
            district_summary = self.aggregator.aggregate(
                city=city,
                stops=stops,
                routes=routes,
                districts=districts,
            )

        result = {
            "project": "Urban Metrics Consulting",
            "part": "Ilya transport infrastructure analysis",
            "selected_city": selected,
            "city_info": city_info,
            "stops_rows": int(len(stops)),
            "routes_rows": int(len(routes)),
            "districts_rows": int(len(districts)),
            "has_district_transport_summary": not district_summary.empty,
            "parsed_at": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(self.config),
        }

        self.storage.save_json(
            result,
            "final",
            f"{slug}_transport_parse_result.json",
        )

        self._print_report(
            city=city,
            selected=selected,
            stops=stops,
            routes=routes,
            districts=districts,
            city_summary=city_summary,
            district_summary=district_summary,
        )

    @staticmethod
    def _print_report(
        city: str,
        selected: dict,
        stops: pd.DataFrame,
        routes: pd.DataFrame,
        districts: pd.DataFrame,
        city_summary: pd.DataFrame,
        district_summary: pd.DataFrame,
    ) -> None:
        print("\nПарсинг транспортной инфраструктуры завершен.\n")

        print("Выбранный город:")
        print(f"- город: {city}")
        print(
            f"- население из топ-таблицы: "
            f"{int(selected['population']):,}".replace(",", " ")
        )

        if selected.get("region"):
            print(f"- регион: {selected['region']}")

        print("\nОбщий результат по городу:")
        print(city_summary.to_string(index=False))

        if not stops.empty:
            print("\nТипы транспортных точек:")
            print(
                stops["transport_type"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "transport_type", "transport_type": "count"})
                .to_string(index=False)
            )

        if not routes.empty:
            print("\nМаршруты по типам:")
            print(
                routes["route"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "route", "route": "count"})
                .to_string(index=False)
            )

        print("\nРайоны OSM:")
        print(f"- найдено районов: {len(districts)}")

        if district_summary.empty:
            print("\nИтоговая таблица по районам не создана.")
            print("Причина: районы не найдены или остановки не удалось привязать к районам.")
            print("Но городская транспортная сводка и таблица остановок сохранены.")
        else:
            print("\nТоп районов по количеству транспортных точек:")
            cols = [
                "district",
                "total_transport_points",
                "transport_points_per_km2",
            ]
            print(district_summary[cols].head(10).to_string(index=False))

        print("\nГлавные файлы:")
        print("- data/final/<город>_transport_city_summary_final.csv")
        print("- data/final/<город>_transport_by_district_final.csv")
        print("- data/processed/<город>_transport_stops_processed.csv")
        print("- data/processed/<город>_transport_routes_processed.csv")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Парсер транспортной инфраструктуры города через OpenStreetMap."
    )

    parser.add_argument(
        "--out",
        default="data",
        help="Папка для сохранения данных.",
    )

    parser.add_argument(
        "--top",
        default=20,
        type=int,
        help="Сколько крупнейших городов показывать.",
    )

    parser.add_argument(
        "--city-number",
        default=None,
        type=int,
        help="Номер города. Если не указать, появится интерактивный выбор.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    config = ParserConfig(
        out_dir=args.out,
        top_n=args.top,
    )

    pipeline = TransportAnalysisPipeline(config)
    pipeline.run(city_number=args.city_number)


if __name__ == "__main__":
    main()