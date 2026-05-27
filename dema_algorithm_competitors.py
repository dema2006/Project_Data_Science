"""
ЧАСТЬ ДЁМЫ: алгоритм, графики, сценарий улучшений и анализ конкурентов

Задача файла:
1. Запускать общий пайплайн по районам
2. Строить графики по микрорайонам и городам
3. Формировать сценарий улучшения для проблемных районов
4. Делать конкурентный анализ через снапшот данных госзакупок

Запуск всего проекта:
    python code/dema_algorithm_competitors.py --mode auto --project-dir .
    python code/dema_algorithm_competitors.py --mode real --project-dir .
    python code/dema_algorithm_competitors.py --mode fallback --project-dir .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from artush_population_index import run_accessibility_analysis
from ilya_transport_parser import CITY_CONFIG


def save_barh(df: pd.DataFrame, city: str, metric: str, title: str, xlabel: str, path: Path, top_n: Optional[int] = None) -> None:
    """Горизонтальная диаграмма по микрорайонам."""
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

    max_value = sub[metric].max() if len(sub) else 0
    for b in bars:
        v = b.get_width()
        label = f"{v:.2f}" if v < 10 else f"{v:.0f}"
        plt.text(v + (max_value * 0.01 if max_value else 0.01), b.get_y() + b.get_height() / 2, label, va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_transport_graphs(df: pd.DataFrame, out_dir: Path, results_dir: Path) -> None:
    """Графики для анализа районов и городов."""
    (out_dir / "microdistricts").mkdir(parents=True, exist_ok=True)
    (out_dir / "cities").mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

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

    # Топ проблемных районов
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

    # Сравнение городов
    city_summary = df.groupby("city").agg(
        mean_index=("accessibility_index", "mean"),
        median_index=("accessibility_index", "median"),
        mean_avg_distance=("avg_distance_m", "mean"),
        high_risk_districts=("accessibility_index", lambda s: int((s >= 0.65).sum())),
        districts=("district", "count"),
        stops_total=("stops_count", "sum"),
        population=("population_estimate", "sum"),
    ).reset_index()
    city_summary["stops_per_10k"] = city_summary["stops_total"] / city_summary["population"] * 10_000
    city_summary.to_csv(results_dir / "city_summary.csv", index=False)

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
            plt.text(b.get_x() + b.get_width() / 2, v, label, ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(out_dir / "cities" / f"city_{metric}.png", dpi=180)
        plt.close()

    # Сценарий улучшения для 12 худших микрорайонов
    scenario = df.sort_values("accessibility_index", ascending=False).head(12).copy()
    scenario["after_index"] = np.maximum(scenario["accessibility_index"] * 0.68 - 0.04, 0)
    scenario["label"] = scenario["city"] + " — " + scenario["district"]

    x = np.arange(len(scenario))
    width = 0.38
    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, scenario["accessibility_index"], width, label="До улучшений")
    plt.bar(x + width / 2, scenario["after_index"], width, label="После: +2 остановки и продление маршрута")
    plt.xticks(x, scenario["label"], rotation=45, ha="right")
    plt.ylabel("Индекс недоступности")
    plt.title("Сценарий улучшения для самых проблемных микрорайонов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "microdistricts" / "improvement_scenario_top12.png", dpi=180)
    plt.close()


def annotate_bars(bars) -> None:
    for b in bars:
        v = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if v >= 1 else f"{v:.1f}", ha="center", va="bottom")


COMPETITORS = [
    {
        "company": "SIMETRA",
        "legal_name": "ООО Симетра Групп",
        "contracts_total": 54,
        "contracts_completed": 51,
        "contracts_failed": 1,
        "law": "44-ФЗ",
        "years_on_market": 15,
    },
    {
        "company": "STEP",
        "legal_name": "ООО СТП",
        "contracts_total": 43,
        "contracts_completed": 40,
        "contracts_failed": 3,
        "law": "44-ФЗ",
        "years_on_market": 9,
    },
    {
        "company": "Новая компания",
        "legal_name": "наш проект",
        "contracts_total": 0,
        "contracts_completed": 0,
        "contracts_failed": 0,
        "law": "—",
        "years_on_market": 0,
    },
]

SIMETRA_BY_YEAR = [
    {"year": 2024, "contracts": 4, "amount_mln_rub": 75.457751},
    {"year": 2025, "contracts": 5, "amount_mln_rub": 57.2},
    {"year": 2026, "contracts": 1, "amount_mln_rub": 1.251542},
]


def generate_competitor_graphs(out_dir: Path, results_dir: Path) -> None:
    """Конкурентный анализ через данные госзакупок."""
    out = out_dir / "competitors"
    out.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(COMPETITORS)
    sim = pd.DataFrame(SIMETRA_BY_YEAR)

    df.to_csv(results_dir / "competitor_procurement_snapshot.csv", index=False)
    sim.to_csv(results_dir / "simetra_procurement_by_year.csv", index=False)

    plt.figure(figsize=(7, 4))
    bars = plt.bar(df["company"], df["contracts_total"])
    plt.title("Конкуренты: количество госконтрактов")
    plt.ylabel("Контракты, шт.")
    annotate_bars(bars)
    plt.tight_layout()
    plt.savefig(out / "competitors_total_contracts.png", dpi=180)
    plt.close()

    x = range(len(df))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar([i - width / 2 for i in x], df["contracts_completed"], width, label="Исполнены")
    plt.bar([i + width / 2 for i in x], df["contracts_failed"], width, label="Проблемные/прекращены")
    plt.xticks(list(x), df["company"])
    plt.title("Исполнение контрактов конкурентов")
    plt.ylabel("Контракты, шт.")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "competitors_contract_execution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    bars = plt.bar(sim["year"].astype(str), sim["amount_mln_rub"])
    plt.title("SIMETRA: сумма поставок по годам")
    plt.ylabel("млн ₽")
    for b in bars:
        v = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out / "simetra_procurement_amount_by_year.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    bars = plt.bar(sim["year"].astype(str), sim["contracts"])
    plt.title("SIMETRA: число поставок по годам")
    plt.ylabel("Контракты, шт.")
    annotate_bars(bars)
    plt.tight_layout()
    plt.savefig(out / "simetra_procurement_contracts_by_year.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "real", "fallback"], default="auto")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    results_dir = project_dir / "results"
    graphs_dir = project_dir / "graphs"
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Артуш считает таблицу индексов по районам
    df = run_accessibility_analysis(args.mode)
    df.to_csv(results_dir / "microdistrict_accessibility_results.csv", index=False)

    # 2. Дёма строит графики и сценарий улучшений
    generate_transport_graphs(df, graphs_dir, results_dir)
    generate_competitor_graphs(graphs_dir, results_dir)

    # 3. Метаданные запуска
    with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "note": "Rows with data_source=synthetic_fallback are generated fallback data, not real OSM measurements.",
                "cities": list(CITY_CONFIG.keys()),
                "districts_total": int(df.shape[0]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved results to {results_dir}")
    print(f"Saved graphs to {graphs_dir}")


if __name__ == "__main__":
    main()
