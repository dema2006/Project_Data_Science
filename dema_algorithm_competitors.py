from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from artush_population_index import run_accessibility_analysis


def save_barh(
    df: pd.DataFrame,
    city: str,
    metric: str,
    title: str,
    xlabel: str,
    path: Path,
    top_n: Optional[int] = None
) -> None:
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

    for bar in bars:
        value = bar.get_width()
        label = f"{value:.2f}" if value < 10 else f"{value:.0f}"
        offset = max_value * 0.01 if max_value else 0.01

        plt.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def generate_transport_graphs(
    df: pd.DataFrame,
    graphs_dir: Path,
    results_dir: Path
) -> None:
    microdistricts_dir = graphs_dir / "microdistricts"
    cities_dir = graphs_dir / "cities"

    microdistricts_dir.mkdir(parents=True, exist_ok=True)
    cities_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for city in df["city"].unique():
        save_barh(
            df=df,
            city=city,
            metric="accessibility_index",
            title=f"{city}: индекс транспортной недоступности по микрорайонам",
            xlabel="Индекс: доля населения дальше 500 м от остановки",
            path=microdistricts_dir / f"{city}_microdistrict_accessibility_index.png"
        )

        save_barh(
            df=df,
            city=city,
            metric="avg_distance_m",
            title=f"{city}: среднее расстояние до ближайшей остановки",
            xlabel="Среднее расстояние, м",
            path=microdistricts_dir / f"{city}_avg_distance_to_stop.png"
        )

        save_barh(
            df=df,
            city=city,
            metric="stops_per_10k",
            title=f"{city}: плотность остановок на 10 тыс. жителей",
            xlabel="Остановок на 10 тыс. жителей",
            path=microdistricts_dir / f"{city}_stops_per_10k.png"
        )

        values = df[df["city"] == city]["accessibility_index"]

        plt.figure(figsize=(8, 4))
        plt.hist(values, bins=10, edgecolor="black")
        plt.title(f"{city}: распределение индекса недоступности")
        plt.xlabel("Индекс недоступности")
        plt.ylabel("Количество микрорайонов")
        plt.tight_layout()
        plt.savefig(microdistricts_dir / f"{city}_accessibility_histogram.png", dpi=180)
        plt.close()

    top_districts = df.sort_values("accessibility_index", ascending=True).tail(18)
    labels = top_districts["city"] + " — " + top_districts["district"]

    plt.figure(figsize=(12, 8))
    plt.barh(labels, top_districts["accessibility_index"])
    plt.title("Топ проблемных микрорайонов по индексу транспортной недоступности")
    plt.xlabel("Индекс: доля населения дальше 500 м от остановки")
    plt.axvline(0.65, linestyle="--", linewidth=1, label="Высокий риск")
    plt.legend()
    plt.tight_layout()
    plt.savefig(microdistricts_dir / "top_underserved_microdistricts.png", dpi=180)
    plt.close()

    city_summary = (
        df.groupby("city")
        .agg(
            mean_index=("accessibility_index", "mean"),
            median_index=("accessibility_index", "median"),
            mean_avg_distance=("avg_distance_m", "mean"),
            high_risk_districts=("accessibility_index", lambda s: int((s >= 0.65).sum())),
            districts=("district", "count"),
            stops_total=("stops_count", "sum"),
            population=("population_estimate", "sum")
        )
        .reset_index()
    )

    city_summary["stops_per_10k"] = (
        city_summary["stops_total"] / city_summary["population"] * 10_000
    )

    city_summary.to_csv(results_dir / "city_summary.csv", index=False)

    city_metrics = [
        ("mean_index", "Средний индекс недоступности", "Индекс"),
        ("mean_avg_distance", "Среднее расстояние до остановки по городу", "Метры"),
        ("high_risk_districts", "Количество микрорайонов высокого риска", "Количество"),
        ("stops_per_10k", "Остановки на 10 тыс. жителей", "Остановок/10 тыс.")
    ]

    for metric, title, ylabel in city_metrics:
        plt.figure(figsize=(6, 4))
        bars = plt.bar(city_summary["city"], city_summary[metric])

        plt.title(title)
        plt.ylabel(ylabel)

        for bar in bars:
            value = bar.get_height()
            label = f"{value:.2f}" if value < 10 else f"{value:.0f}"

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                label,
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        plt.savefig(cities_dir / f"city_{metric}.png", dpi=180)
        plt.close()

    scenario = df.sort_values("accessibility_index", ascending=False).head(12).copy()
    scenario["after_index"] = np.maximum(scenario["accessibility_index"] * 0.68 - 0.04, 0)
    scenario["label"] = scenario["city"] + " — " + scenario["district"]

    x = np.arange(len(scenario))
    width = 0.38

    plt.figure(figsize=(13, 6))
    plt.bar(
        x - width / 2,
        scenario["accessibility_index"],
        width,
        label="До улучшений"
    )
    plt.bar(
        x + width / 2,
        scenario["after_index"],
        width,
        label="После: +2 остановки и продление маршрута"
    )

    plt.xticks(x, scenario["label"], rotation=45, ha="right")
    plt.ylabel("Индекс недоступности")
    plt.title("Сценарий улучшения для самых проблемных микрорайонов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(microdistricts_dir / "improvement_scenario_top12.png", dpi=180)
    plt.close()


def annotate_bars(bars) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.0f}" if value >= 1 else f"{value:.1f}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            label,
            ha="center",
            va="bottom"
        )


COMPETITORS = [
    {
        "company": "SIMETRA",
        "legal_name": "ООО Симетра Групп",
        "contracts_total": 54,
        "contracts_completed": 51,
        "contracts_failed": 1,
        "law": "44-ФЗ",
        "years_on_market": 15
    },
    {
        "company": "STEP",
        "legal_name": "ООО СТП",
        "contracts_total": 43,
        "contracts_completed": 40,
        "contracts_failed": 3,
        "law": "44-ФЗ",
        "years_on_market": 9
    },
    {
        "company": "Новая компания",
        "legal_name": "наш проект",
        "contracts_total": 0,
        "contracts_completed": 0,
        "contracts_failed": 0,
        "law": "—",
        "years_on_market": 0
    }
]


SIMETRA_BY_YEAR = [
    {
        "year": 2024,
        "contracts": 4,
        "amount_mln_rub": 75.457751
    },
    {
        "year": 2025,
        "contracts": 5,
        "amount_mln_rub": 57.2
    },
    {
        "year": 2026,
        "contracts": 1,
        "amount_mln_rub": 1.251542
    }
]


def generate_competitor_graphs(
    graphs_dir: Path,
    results_dir: Path
) -> None:
    competitors_dir = graphs_dir / "competitors"

    competitors_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    competitors_df = pd.DataFrame(COMPETITORS)
    simetra_df = pd.DataFrame(SIMETRA_BY_YEAR)

    competitors_df.to_csv(
        results_dir / "competitor_procurement_snapshot.csv",
        index=False
    )

    simetra_df.to_csv(
        results_dir / "simetra_procurement_by_year.csv",
        index=False
    )

    plt.figure(figsize=(7, 4))
    bars = plt.bar(
        competitors_df["company"],
        competitors_df["contracts_total"]
    )

    plt.title("Конкуренты: количество госконтрактов")
    plt.ylabel("Контракты, шт.")
    annotate_bars(bars)
    plt.tight_layout()
    plt.savefig(competitors_dir / "competitors_total_contracts.png", dpi=180)
    plt.close()

    x = np.arange(len(competitors_df))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(
        x - width / 2,
        competitors_df["contracts_completed"],
        width,
        label="Исполнены"
    )
    plt.bar(
        x + width / 2,
        competitors_df["contracts_failed"],
        width,
        label="Проблемные/прекращены"
    )

    plt.xticks(x, competitors_df["company"])
    plt.title("Исполнение контрактов конкурентов")
    plt.ylabel("Контракты, шт.")
    plt.legend()
    plt.tight_layout()
    plt.savefig(competitors_dir / "competitors_contract_execution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    bars = plt.bar(
        simetra_df["year"].astype(str),
        simetra_df["amount_mln_rub"]
    )

    plt.title("SIMETRA: сумма поставок по годам")
    plt.ylabel("млн ₽")

    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(competitors_dir / "simetra_procurement_amount_by_year.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    bars = plt.bar(
        simetra_df["year"].astype(str),
        simetra_df["contracts"]
    )

    plt.title("SIMETRA: число поставок по годам")
    plt.ylabel("Контракты, шт.")
    annotate_bars(bars)
    plt.tight_layout()
    plt.savefig(competitors_dir / "simetra_procurement_contracts_by_year.png", dpi=180)
    plt.close()


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results_dir = project_dir / "results"
    graphs_dir = project_dir / "graphs"

    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    df = run_accessibility_analysis("auto")

    df.to_csv(
        results_dir / "microdistrict_accessibility_results.csv",
        index=False
    )

    generate_transport_graphs(
        df=df,
        graphs_dir=graphs_dir,
        results_dir=results_dir
    )

    generate_competitor_graphs(
        graphs_dir=graphs_dir,
        results_dir=results_dir
    )

    print(f"Результаты сохранены в папку: {results_dir}")
    print(f"Графики сохранены в папку: {graphs_dir}")


if __name__ == "__main__":
    main()
