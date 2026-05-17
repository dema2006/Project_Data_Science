from __future__ import annotations

import argparse
import io
import json
import logging
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
from urllib3.util.retry import Retry


@dataclass
class ParserConfig:
    top_cities_url: str = "https://www.r-statistics.ru/po-chislennosti"
    wiki_base_url: str = "https://ru.wikipedia.org/wiki/"
    out_dir: str = "data"
    top_n: int = 20
    timeout: int = 30
    sleep_sec: float = 1.0
    user_agent: str = (
        "UrbanMetricsConsultingStudentParser/1.0 "
        "(educational data science project)"
    )


class TextCleaner:
    @staticmethod
    def clean_text(value: object) -> Optional[str]:
        if pd.isna(value):
            return None

        text = str(value)
        text = text.replace("\xa0", " ")
        text = text.replace("—", "-")
        text = text.replace("–", "-")
        text = text.replace("-", "-")
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
        text = text.replace("-", "-")
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
    def parse_float(value: object) -> Optional[float]:
        if pd.isna(value):
            return None

        text = str(value)
        text = text.replace("\xa0", " ")
        text = re.sub(r"\[[^\]]*]", "", text)
        text = text.replace(",", ".")
        text = re.sub(r"[^0-9.]", "", text)

        if not text:
            return None

        parts = text.split(".")

        if len(parts) > 2:
            text = parts[0] + "." + "".join(parts[1:])

        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def slug(value: str) -> str:
        value = value.lower()
        value = value.replace("ё", "е")
        value = value.replace("—", "-")
        value = value.replace("–", "-")
        value = value.replace("-", "-")
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
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
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

    def save_csv(self, df: pd.DataFrame, folder: str, filename: str) -> Path:
        path = self._get_folder(folder) / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logging.info("CSV сохранен: %s", path)
        return path

    def save_json(self, data: dict | list, folder: str, filename: str) -> Path:
        path = self._get_folder(folder) / filename
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("JSON сохранен: %s", path)
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
            "top_cities_source_page.html",
        )

        tables = pd.read_html(io.StringIO(html))
        raw_table = self._find_city_table(tables)

        self.storage.save_csv(
            raw_table,
            "raw",
            "top_cities_raw.csv",
        )

        df = self._clean_city_table(raw_table)
        df = df.head(self.config.top_n).reset_index(drop=True)

        self.storage.save_csv(
            df,
            "processed",
            "top_20_russian_cities.csv",
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
        print("\nВыбери город для анализа:\n")

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


class CityWikiProfileParser:
    def __init__(self, config: ParserConfig, client: HttpClient, storage: FileStorage):
        self.config = config
        self.client = client
        self.storage = storage

    def parse_city_profile(self, city: str) -> dict:
        url = self._city_wiki_url(city)
        html = self.client.get_html(url)

        slug = TextCleaner.slug(city)

        self.storage.save_text(
            html,
            "raw",
            f"{slug}_city_wiki_page.html",
        )

        soup = BeautifulSoup(html, "html.parser")
        title = self._get_heading(soup)

        profile = {
            "city": city,
            "wiki_title": title,
            "wiki_url": url,
            "population_from_wiki": None,
            "area_km2_from_wiki": None,
            "density_from_wiki": None,
            "founded": None,
            "subject": None,
        }

        infobox = self._find_infobox(soup)

        if infobox is None:
            logging.warning("Не найдена карточка города на wiki-странице.")
            self.storage.save_json(
                profile,
                "processed",
                f"{slug}_city_profile.json",
            )
            return profile

        raw_info = self._parse_infobox(infobox)

        profile["population_from_wiki"] = self._find_value_as_int(
            raw_info,
            ["население"],
        )

        profile["area_km2_from_wiki"] = self._find_value_as_float(
            raw_info,
            ["площадь"],
        )

        profile["density_from_wiki"] = self._find_value_as_float(
            raw_info,
            ["плотность"],
        )

        profile["founded"] = self._find_value_as_text(
            raw_info,
            ["основан", "первое упоминание"],
        )

        profile["subject"] = self._find_value_as_text(
            raw_info,
            ["субъект", "регион"],
        )

        self.storage.save_json(
            profile,
            "processed",
            f"{slug}_city_profile.json",
        )

        return profile

    def _city_wiki_url(self, city: str) -> str:
        title = city.replace(" ", "_")
        return self.config.wiki_base_url + quote(title)

    @staticmethod
    def _get_heading(soup: BeautifulSoup) -> Optional[str]:
        heading = soup.find(id="firstHeading")

        if heading:
            return TextCleaner.clean_text(heading.get_text(" "))

        return None

    @staticmethod
    def _find_infobox(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        tables = soup.find_all("table")

        for table in tables:
            classes = " ".join(table.get("class", []))

            if "infobox" in classes or "карточка" in classes:
                return table

        for table in tables[:10]:
            text = table.get_text(" ").lower()

            if "население" in text and "площадь" in text:
                return table

        return None

    @staticmethod
    def _parse_infobox(table: BeautifulSoup) -> dict[str, str]:
        result = {}

        for row in table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")

            if th is None or td is None:
                continue

            key = TextCleaner.clean_text(th.get_text(" "))
            value = TextCleaner.clean_text(td.get_text(" "))

            if key and value:
                result[key.lower()] = value

        return result

    @staticmethod
    def _find_value_as_int(data: dict[str, str], keys: list[str]) -> Optional[int]:
        text = CityWikiProfileParser._find_raw_value(data, keys)
        return TextCleaner.parse_int(text)

    @staticmethod
    def _find_value_as_float(data: dict[str, str], keys: list[str]) -> Optional[float]:
        text = CityWikiProfileParser._find_raw_value(data, keys)
        return TextCleaner.parse_float(text)

    @staticmethod
    def _find_value_as_text(data: dict[str, str], keys: list[str]) -> Optional[str]:
        return CityWikiProfileParser._find_raw_value(data, keys)

    @staticmethod
    def _find_raw_value(data: dict[str, str], keys: list[str]) -> Optional[str]:
        for source_key, value in data.items():
            if any(key in source_key for key in keys):
                return value

        return None


class AdminDivisionParser:
    ADMIN_TITLES = {
        "Москва": "Административно-территориальное_деление_Москвы",
        "Санкт-Петербург": "Административно-территориальное_деление_Санкт-Петербурга",
        "Новосибирск": "Административное_деление_Новосибирска",
        "Екатеринбург": "Административное_деление_Екатеринбурга",
        "Нижний Новгород": "Административное_деление_Нижнего_Новгорода",
        "Казань": "Административное_деление_Казани",
        "Челябинск": "Административное_деление_Челябинска",
        "Омск": "Административное_деление_Омска",
        "Самара": "Административное_деление_Самары",
        "Ростов-на-Дону": "Административное_деление_Ростова-на-Дону",
        "Уфа": "Административное_деление_Уфы",
        "Волгоград": "Административное_деление_Волгограда",
        "Пермь": "Административное_деление_Перми",
        "Красноярск": "Административное_деление_Красноярска",
        "Воронеж": "Административное_деление_Воронежа",
        "Саратов": "Административное_деление_Саратова",
        "Краснодар": "Административное_деление_Краснодара",
        "Тольятти": "Административное_деление_Тольятти",
        "Ижевск": "Административное_деление_Ижевска",
        "Ульяновск": "Административное_деление_Ульяновска",
    }

    DISTRICT_HINTS = [
        "район",
        "округ",
        "центральный",
        "ленинский",
        "советский",
        "кировский",
        "промышленный",
        "адмиралтейский",
        "выборгский",
        "калининский",
        "приморский",
        "невский",
        "железнодорожный",
        "красноглинский",
        "куйбышевский",
        "октябрьский",
        "самарский",
        "пролетарский",
        "автозаводский",
        "московский",
        "верх-исетский",
        "орджоникидзевский",
    ]

    def __init__(self, config: ParserConfig, client: HttpClient, storage: FileStorage):
        self.config = config
        self.client = client
        self.storage = storage

    def parse_admin_division(self, city: str) -> Optional[pd.DataFrame]:
        title = self.ADMIN_TITLES.get(city)

        if not title:
            logging.warning("Для города нет заготовленного wiki-title: %s", city)
            return None

        url = self.config.wiki_base_url + quote(title)
        html = self.client.get_html(url)

        if self._page_does_not_exist(html):
            logging.warning("Страница административного деления не найдена: %s", city)
            return None

        slug = TextCleaner.slug(city)

        self.storage.save_text(
            html,
            "raw",
            f"{slug}_admin_division_page.html",
        )

        try:
            tables = pd.read_html(io.StringIO(html))
        except ValueError:
            logging.warning("На странице административного деления нет таблиц.")
            return None

        table = self._find_admin_table(tables)

        if table is None:
            logging.warning("Не удалось найти таблицу районов для города: %s", city)
            return None

        self.storage.save_csv(
            table,
            "raw",
            f"{slug}_admin_division_raw.csv",
        )

        try:
            clean_df = self._clean_admin_table(table, city)
        except Exception as error:
            logging.warning("Таблица районов найдена, но очистка не удалась.")
            logging.warning("Причина: %s", error)
            return None

        if clean_df.empty:
            logging.warning("После очистки таблица районов пустая.")
            return None

        self.storage.save_csv(
            clean_df,
            "processed",
            f"{slug}_districts_processed.csv",
        )

        return clean_df

    @staticmethod
    def _page_does_not_exist(html: str) -> bool:
        text = BeautifulSoup(html, "html.parser").get_text(" ").lower()

        bad_signs = [
            "страницы с таким названием не существует",
            "вы можете создать её",
            "нет статьи с таким названием",
        ]

        return any(sign in text for sign in bad_signs)

    def _find_admin_table(self, tables: list[pd.DataFrame]) -> Optional[pd.DataFrame]:
        best_score = -1
        best_table = None

        for table in tables:
            if table.empty:
                continue

            table_copy = table.copy()
            table_copy.columns = [self._normalize_col(c) for c in table_copy.columns]

            columns_text = " ".join([str(c).lower() for c in table_copy.columns])
            sample_text = " ".join(
                table_copy.fillna("").astype(str).head(15).values.flatten()
            ).lower()

            full_text = columns_text + " " + sample_text

            score = 0

            if "район" in full_text or "округ" in full_text:
                score += 3

            if "население" in full_text:
                score += 3

            if "площад" in full_text:
                score += 2

            if "плотность" in full_text:
                score += 1

            if len(table_copy) >= 3:
                score += 1

            if len(table_copy) <= 100:
                score += 1

            if score > best_score:
                best_score = score
                best_table = table_copy

        if best_table is None or best_score < 3:
            return None

        return best_table.copy()

    def _clean_admin_table(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self._normalize_col(c) for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)

        name_col = self._detect_name_col(df)
        population_col = self._detect_population_col(df, exclude=[name_col])
        area_col = self._detect_area_col(df, exclude=[name_col, population_col])
        density_col = self._detect_density_col(
            df,
            exclude=[name_col, population_col, area_col],
        )

        if name_col is None:
            logging.warning("Не найдена колонка с названием района.")
            logging.warning("Доступные колонки: %s", list(df.columns))
            return pd.DataFrame()

        if population_col is None:
            logging.warning("Не найдена колонка с населением районов.")
            logging.warning("Доступные колонки: %s", list(df.columns))
            return pd.DataFrame()

        result = pd.DataFrame()

        result["city"] = city
        result["district"] = df[name_col].apply(self._clean_district_name)
        result["population"] = df[population_col].apply(TextCleaner.parse_int)

        if area_col:
            result["area_km2"] = df[area_col].apply(TextCleaner.parse_float)
        else:
            result["area_km2"] = None

        if density_col:
            result["density_people_per_km2"] = df[density_col].apply(
                TextCleaner.parse_float
            )
        else:
            result["density_people_per_km2"] = None

        result = result.dropna(subset=["district"])
        result = result[result["district"].astype(str).str.len() > 1]

        result = result[
            ~result["district"]
            .astype(str)
            .str.lower()
            .str.contains(
                "итого|всего|город|население|площадь|плотность|примечание",
                regex=True,
                na=False,
            )
        ]

        result = result.drop_duplicates(subset=["district"])

        result["population"] = pd.to_numeric(result["population"], errors="coerce")
        result = result.dropna(subset=["population"])
        result["population"] = result["population"].astype(int)

        result["area_km2"] = pd.to_numeric(result["area_km2"], errors="coerce")
        result["density_people_per_km2"] = pd.to_numeric(
            result["density_people_per_km2"],
            errors="coerce",
        )

        has_area = result["area_km2"].notna() & (result["area_km2"] > 0)
        need_density = result["density_people_per_km2"].isna()

        result.loc[has_area & need_density, "density_people_per_km2"] = (
            result.loc[has_area & need_density, "population"]
            / result.loc[has_area & need_density, "area_km2"]
        ).round(1)

        if result.empty:
            return result

        result["population_share_pct"] = (
            result["population"] / result["population"].sum() * 100
        ).round(2)

        result["population_rank"] = (
            result["population"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        if result["density_people_per_km2"].notna().any():
            result["density_rank"] = (
                result["density_people_per_km2"]
                .rank(method="dense", ascending=False)
                .astype("Int64")
            )
        else:
            result["density_rank"] = None

        result = result.sort_values("population_rank").reset_index(drop=True)

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

    def _detect_name_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            col_lower = str(col).lower()

            if "район" in col_lower or "округ" in col_lower:
                return col

        best_col = None
        best_score = -1

        for col in df.columns:
            values = df[col].head(40).fillna("").astype(str).tolist()
            text = " ".join(values).lower()

            score = 0

            for word in self.DISTRICT_HINTS:
                if word in text:
                    score += 1

            numeric_count = df[col].apply(TextCleaner.parse_int).notna().sum()
            text_count = df[col].fillna("").astype(str).str.len().gt(2).sum()

            if text_count > numeric_count:
                score += 1

            if score > best_score:
                best_score = score
                best_col = col

        if best_score > 0:
            return best_col

        return None

    def _detect_population_col(
        self,
        df: pd.DataFrame,
        exclude: list[Optional[str]],
    ) -> Optional[str]:
        exclude_set = {x for x in exclude if x is not None}

        for col in df.columns:
            if col in exclude_set:
                continue

            col_lower = str(col).lower()

            if "население" in col_lower or "жителей" in col_lower:
                return col

        best_col = None
        best_score = -1

        for col in df.columns:
            if col in exclude_set:
                continue

            parsed = df[col].apply(TextCleaner.parse_int)
            values = parsed.dropna()

            if values.empty:
                continue

            median_value = values.median()
            max_value = values.max()
            count_value = len(values)

            score = 0

            if count_value >= 3:
                score += 1

            if median_value >= 10_000:
                score += 3

            if max_value >= 50_000:
                score += 2

            if score > best_score:
                best_score = score
                best_col = col

        if best_score > 0:
            return best_col

        return None

    def _detect_area_col(
        self,
        df: pd.DataFrame,
        exclude: list[Optional[str]],
    ) -> Optional[str]:
        exclude_set = {x for x in exclude if x is not None}

        for col in df.columns:
            if col in exclude_set:
                continue

            col_lower = str(col).lower()

            if "площад" in col_lower:
                return col

        best_col = None
        best_score = -1

        for col in df.columns:
            if col in exclude_set:
                continue

            parsed = df[col].apply(TextCleaner.parse_float)
            values = parsed.dropna()

            if values.empty:
                continue

            median_value = values.median()
            max_value = values.max()
            count_value = len(values)

            score = 0

            if count_value >= 3:
                score += 1

            if 1 <= median_value <= 500:
                score += 3

            if max_value <= 2000:
                score += 1

            if score > best_score:
                best_score = score
                best_col = col

        if best_score > 2:
            return best_col

        return None

    def _detect_density_col(
        self,
        df: pd.DataFrame,
        exclude: list[Optional[str]],
    ) -> Optional[str]:
        exclude_set = {x for x in exclude if x is not None}

        for col in df.columns:
            if col in exclude_set:
                continue

            col_lower = str(col).lower()

            if "плотность" in col_lower:
                return col

        return None

    @staticmethod
    def _clean_district_name(value: object) -> Optional[str]:
        text = TextCleaner.clean_text(value)

        if not text:
            return None

        text = re.sub(r"^\d+\s*", "", text)
        text = re.sub(r"\([^)]*\)", "", text)

        text = text.replace("район", "")
        text = text.replace("Район", "")
        text = text.replace("городской округ", "")
        text = text.replace("Городской округ", "")
        text = text.replace("округ", "")
        text = text.replace("Округ", "")

        text = re.sub(r"\s+", " ", text)
        text = text.strip(" —-.,;")

        if not text:
            return None

        return text


class CityAnalysisPipeline:
    def __init__(self, config: ParserConfig):
        self.config = config
        self.client = HttpClient(config)
        self.storage = FileStorage(config.out_dir)

        self.top_parser = TopCitiesParser(config, self.client, self.storage)
        self.profile_parser = CityWikiProfileParser(config, self.client, self.storage)
        self.admin_parser = AdminDivisionParser(config, self.client, self.storage)

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

        profile = self.profile_parser.parse_city_profile(city)
        districts = self.admin_parser.parse_admin_division(city)

        final_result = {
            "project": "Urban Metrics Consulting",
            "selected_city": selected,
            "city_profile": profile,
            "has_districts_table": districts is not None,
            "parsed_at": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(self.config),
        }

        if districts is not None:
            final_result["districts_rows"] = len(districts)
            final_result["districts_columns"] = list(districts.columns)

            self.storage.save_csv(
                districts,
                "final",
                f"{slug}_district_population_final.csv",
            )

        self.storage.save_json(
            final_result,
            "final",
            f"{slug}_city_parse_result.json",
        )

        self._print_report(city, selected, profile, districts)

    @staticmethod
    def _print_report(
        city: str,
        selected: dict,
        profile: dict,
        districts: Optional[pd.DataFrame],
    ) -> None:
        print("\nПарсинг завершен.\n")

        print("Выбранный город:")
        print(f"- город: {city}")
        print(
            f"- население из топ-таблицы: "
            f"{int(selected['population']):,}".replace(",", " ")
        )

        if selected.get("region"):
            print(f"- регион: {selected['region']}")

        if selected.get("federal_district"):
            print(f"- федеральный округ: {selected['federal_district']}")

        if selected.get("timezone"):
            print(f"- часовой пояс: {selected['timezone']}")

        print("\nПрофиль города с wiki-страницы:")
        print(f"- wiki title: {profile.get('wiki_title')}")
        print(f"- население: {profile.get('population_from_wiki')}")
        print(f"- площадь, км²: {profile.get('area_km2_from_wiki')}")
        print(f"- плотность: {profile.get('density_from_wiki')}")
        print(f"- субъект: {profile.get('subject')}")

        if districts is None:
            print("\nТаблицу районов автоматически получить не удалось или не удалось нормально очистить.")
            print("Но общий профиль города сохранен.")
            print("Смотри файлы в папках data/raw, data/processed и data/final.")
            return

        print("\nТаблица районов получена.")
        print(f"- строк: {len(districts)}")
        print(f"- колонок: {len(districts.columns)}")

        print("\nТоп районов по населению:")

        cols = ["district", "population"]

        if "area_km2" in districts.columns:
            cols.append("area_km2")

        if "density_people_per_km2" in districts.columns:
            cols.append("density_people_per_km2")

        print(districts[cols].head(10).to_string(index=False))

        print("\nГлавный итоговый файл:")
        print("data/final/<город>_district_population_final.csv")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Парсер выбора города из топ-20 городов России."
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

    pipeline = CityAnalysisPipeline(config)
    pipeline.run(city_number=args.city_number)


if __name__ == "__main__":
    main()