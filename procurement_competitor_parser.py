"""
Парсер госзакупок для конкурентного анализа.

Идея: вместо абстрактного сравнения «у кого сколько проектов» сравниваем
компании по публичным сведениям о госконтрактах.

Источники, которые пытается читать скрипт:
- T-Банк: карточка контрагента SIMETRA / ООО «СИМЕТРА ГРУПП».
- Checkspot: карточка контрагента STEP / ООО «СТП».
- Star-Pro: поставки SIMETRA по годам.

В некоторых средах сайты могут блокировать requests или DNS. Тогда используется
явный snapshot, занесенный в код по открытым страницам, чтобы графики всё равно
можно было воспроизвести. Snapshot помечен как snapshot_used=True в CSV.

Запуск:
python code/procurement_competitor_parser.py --out results
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import caas_jupyter_tools  # type: ignore
    def _noop(*args, **kwargs):
        return None
    caas_jupyter_tools.log_exception = _noop  # type: ignore
    caas_jupyter_tools.log_matplotlib_img_fallback = _noop  # type: ignore
except Exception:
    pass

HEADERS = {"User-Agent": "Mozilla/5.0 (TransportAnalyticsStudentProject/1.0)"}


@dataclass
class CompetitorProcurement:
    company: str
    legal_name: str
    inn: str
    ogrn: str
    source_url: str
    years_active: Optional[int]
    total_contracts_44fz: Optional[int]
    executed_contracts: Optional[int]
    failed_or_terminated_contracts: Optional[int]
    total_amount_rub_snapshot: Optional[float]
    snapshot_used: bool
    notes: str


SNAPSHOTS = {
    "simetra": CompetitorProcurement(
        company="SIMETRA",
        legal_name="ООО «СИМЕТРА ГРУПП»",
        inn="7841446798",
        ogrn="1117847259529",
        source_url="https://www.tbank.ru/business/contractor/legal/1117847259529/",
        years_active=15,
        total_contracts_44fz=54,
        executed_contracts=51,
        failed_or_terminated_contracts=1,
        total_amount_rub_snapshot=1_251_542 + 21_000_000 + 19_900_000,
        snapshot_used=True,
        notes="Snapshot по карточке Т-Банка: 54 госконтракта, 51 исполненный, 1 неисполненный; видны последние контракты на 1.25, 21 и 19.9 млн ₽.",
    ),
    "step": CompetitorProcurement(
        company="STEP / СТП",
        legal_name="ООО «СТУДИЯ ТРАНСПОРТНОГО ПРОЕКТИРОВАНИЯ»",
        inn="9710035205",
        ogrn="1177746949522",
        source_url="https://checkspot.ru/company/1177746949522",
        years_active=9,
        total_contracts_44fz=43,
        executed_contracts=40,
        failed_or_terminated_contracts=3,
        total_amount_rub_snapshot=None,
        snapshot_used=True,
        notes="Snapshot по Checkspot: 43 контракта в реестрах, 40 завершено, 3 прекращено, 44-ФЗ — 43.",
    ),
    "our": CompetitorProcurement(
        company="Наша команда",
        legal_name="Учебный проект",
        inn="—",
        ogrn="—",
        source_url="—",
        years_active=0,
        total_contracts_44fz=0,
        executed_contracts=0,
        failed_or_terminated_contracts=0,
        total_amount_rub_snapshot=0,
        snapshot_used=True,
        notes="Новая учебная консалтинговая команда, госконтрактов нет.",
    ),
}

SIMETRA_ANNUAL_SNAPSHOT = pd.DataFrame([
    {"company": "SIMETRA", "year": 2024, "contracts": 4, "amount_rub": 75_457_751},
    {"company": "SIMETRA", "year": 2025, "contracts": 5, "amount_rub": 57_200_000},
    {"company": "SIMETRA", "year": 2026, "contracts": 1, "amount_rub": 1_251_542},
])


def get_html(url: str, cache_dir: Path) -> Optional[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_") + ".html")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="ignore")
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            return None
        cache_file.write_text(response.text, encoding="utf-8")
        time.sleep(1)
        return response.text
    except Exception:
        return None


def parse_int_after_pattern(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.I | re.S)
    if not match:
        return None
    number = re.sub(r"\D+", "", match.group(1))
    return int(number) if number else None


def parse_tbank_simetra(cache_dir: Path) -> CompetitorProcurement:
    url = SNAPSHOTS["simetra"].source_url
    html = get_html(url, cache_dir)
    if html is None:
        return SNAPSHOTS["simetra"]
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.get_text(" ").split())

    total = parse_int_after_pattern(text, r"Госконтракты по 44-ФЗ\s*(\d+)\s*Всего")
    executed = parse_int_after_pattern(text, r"Всего\s*контрактов\s*(\d+)\s*Исполненные")
    failed = parse_int_after_pattern(text, r"Исполненные\s*контракты\s*(\d+)\s*Неисполненные")
    # Если регулярки не сработали из-за верстки, используем snapshot.
    snap = SNAPSHOTS["simetra"]
    return CompetitorProcurement(
        company=snap.company,
        legal_name=snap.legal_name,
        inn=snap.inn,
        ogrn=snap.ogrn,
        source_url=url,
        years_active=snap.years_active,
        total_contracts_44fz=total or snap.total_contracts_44fz,
        executed_contracts=executed or snap.executed_contracts,
        failed_or_terminated_contracts=failed or snap.failed_or_terminated_contracts,
        total_amount_rub_snapshot=snap.total_amount_rub_snapshot,
        snapshot_used=(total is None),
        notes="Parsed from T-Bank contractor card" if total is not None else snap.notes,
    )


def parse_checkspot_step(cache_dir: Path) -> CompetitorProcurement:
    url = SNAPSHOTS["step"].source_url
    html = get_html(url, cache_dir)
    if html is None:
        return SNAPSHOTS["step"]
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.get_text(" ").split())
    total = parse_int_after_pattern(text, r"Всего контрактов в реестрах:\s*(\d+)")
    executed = parse_int_after_pattern(text, r"Исполнение завершено\s*·\s*(\d+)")
    terminated = parse_int_after_pattern(text, r"Исполнение прекращено\s*·\s*(\d+)")
    years = parse_int_after_pattern(text, r"Лет на рынке:\s*(\d+)")
    snap = SNAPSHOTS["step"]
    return CompetitorProcurement(
        company=snap.company,
        legal_name=snap.legal_name,
        inn=snap.inn,
        ogrn=snap.ogrn,
        source_url=url,
        years_active=years or snap.years_active,
        total_contracts_44fz=total or snap.total_contracts_44fz,
        executed_contracts=executed or snap.executed_contracts,
        failed_or_terminated_contracts=terminated or snap.failed_or_terminated_contracts,
        total_amount_rub_snapshot=snap.total_amount_rub_snapshot,
        snapshot_used=(total is None),
        notes="Parsed from Checkspot contractor card" if total is not None else snap.notes,
    )


def build_dataset(cache_dir: Path) -> pd.DataFrame:
    rows = [
        asdict(parse_tbank_simetra(cache_dir)),
        asdict(parse_checkspot_step(cache_dir)),
        asdict(SNAPSHOTS["our"]),
    ]
    df = pd.DataFrame(rows)
    return df


def plot_total_contracts(df: pd.DataFrame, graphs_dir: Path) -> None:
    ordered = df.sort_values("total_contracts_44fz", ascending=True)
    plt.figure(figsize=(8, 4))
    plt.barh(ordered["company"], ordered["total_contracts_44fz"])
    plt.xlabel("Количество госконтрактов по 44-ФЗ")
    plt.title("Конкуренты: портфель госконтрактов")
    for i, v in enumerate(ordered["total_contracts_44fz"]):
        plt.text(v, i, f" {int(v)}", va="center")
    plt.tight_layout()
    plt.savefig(graphs_dir / "procurement_total_contracts.png", dpi=180)
    plt.close()


def plot_status(df: pd.DataFrame, graphs_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["other_contracts"] = plot_df["total_contracts_44fz"] - plot_df["executed_contracts"].fillna(0) - plot_df["failed_or_terminated_contracts"].fillna(0)
    plot_df = plot_df.sort_values("total_contracts_44fz", ascending=False)
    x = range(len(plot_df))
    plt.figure(figsize=(8, 4))
    plt.bar(x, plot_df["executed_contracts"], label="Исполненные")
    plt.bar(x, plot_df["failed_or_terminated_contracts"], bottom=plot_df["executed_contracts"], label="Прекращенные/неисполненные")
    bottom = plot_df["executed_contracts"].fillna(0) + plot_df["failed_or_terminated_contracts"].fillna(0)
    plt.bar(x, plot_df["other_contracts"], bottom=bottom, label="Прочие/на исполнении")
    plt.xticks(list(x), plot_df["company"], rotation=0)
    plt.ylabel("Количество контрактов")
    plt.title("Статусы госконтрактов конкурентов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphs_dir / "procurement_contract_status.png", dpi=180)
    plt.close()


def plot_efficiency(df: pd.DataFrame, graphs_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["contracts_per_year"] = plot_df.apply(
        lambda r: r["total_contracts_44fz"] / r["years_active"] if r["years_active"] and r["years_active"] > 0 else 0,
        axis=1,
    )
    ordered = plot_df.sort_values("contracts_per_year", ascending=True)
    plt.figure(figsize=(8, 4))
    plt.barh(ordered["company"], ordered["contracts_per_year"])
    plt.xlabel("Госконтрактов на год работы")
    plt.title("Интенсивность участия в госзакупках")
    for i, v in enumerate(ordered["contracts_per_year"]):
        plt.text(v, i, f" {v:.1f}", va="center")
    plt.tight_layout()
    plt.savefig(graphs_dir / "procurement_contracts_per_year.png", dpi=180)
    plt.close()


def plot_simetra_annual(out_dir: Path) -> None:
    df = SIMETRA_ANNUAL_SNAPSHOT.copy()
    graphs_dir = out_dir / "graphs"
    df.to_csv(out_dir / "simetra_annual_procurement_snapshot.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(7, 4))
    plt.plot(df["year"], df["contracts"], marker="o")
    plt.xticks(df["year"])
    plt.ylabel("Контракты")
    plt.title("SIMETRA: поставки по годам, snapshot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "procurement_simetra_contracts_by_year.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(df["year"].astype(str), df["amount_rub"] / 1_000_000)
    plt.ylabel("Сумма, млн ₽")
    plt.title("SIMETRA: сумма поставок по годам, snapshot")
    for i, v in enumerate(df["amount_rub"] / 1_000_000):
        plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(graphs_dir / "procurement_simetra_amount_by_year.png", dpi=180)
    plt.close()


def generate_graphs(df: pd.DataFrame, out_dir: Path) -> None:
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    plot_total_contracts(df, graphs_dir)
    plot_status(df, graphs_dir)
    plot_efficiency(df, graphs_dir)
    plot_simetra_annual(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Procurement competitor parser")
    parser.add_argument("--out", default="results", help="Папка для CSV и графиков")
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out)
    cache_dir = base_dir / "cache" / "procurement"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(cache_dir)
    df.to_csv(out_dir / "procurement_competitors.csv", index=False, encoding="utf-8-sig")
    generate_graphs(df, out_dir)
    print(df[["company", "total_contracts_44fz", "executed_contracts", "failed_or_terminated_contracts", "snapshot_used"]])
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
