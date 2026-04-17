import asyncio
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple, Union

import pandas as pd


async def load_data(path: str) -> pd.DataFrame:
    """
    Асинхронно загружает данные из Excel файла.

    Args:
        path: Путь к Excel файлу

    Returns:
        DataFrame с очищенными названиями колонок, а также удалены пробелы и приведены к нижнему регистру
    """

    df = await asyncio.to_thread(pd.read_excel, path, engine='openpyxl')
    df.columns = df.columns.str.strip().str.lower()

    return df


async def load_multiple_files(files: List[str]) -> pd.DataFrame:
    """
    Параллельно загружает список файлов и объединяет их в один DataFrame.

    Args:
        files: Список путей к файлам

    Returns:
        Объединенный DataFrame
    """

    tasks = [load_data(file) for file in files]
    dfs = await asyncio.gather(*tasks)

    return pd.concat(dfs, ignore_index=True)


def parse_date(value: Union[str, datetime, float]) -> Optional[datetime]:
    """
    Преобразует строку с датой в объект datetime, пробуя несколько форматов.

    Args:
        value: Значение даты для парсинга (строка, datetime или NaN)

    Returns:
        Объект datetime или None, если парсинг не удался
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    formats = [
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%b %d, %Y",
        "%B %d, %Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            pass

    return None


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует указанные колонки с датами из строк в datetime.

    Args:
        df: Входной DataFrame с колонками дат

    Returns:
        DataFrame с преобразованными колонками дат
    """

    date_columns = [
        "install_date",
        "warranty_until",
        "last_calibration_date",
        "last_service_date"
    ]

    for col in date_columns:
        df[col] = df[col].apply(parse_date)

    return df


def normalize_status(status: Union[str, float]) -> str:
    """
    Приводит статусы оборудования к стандартизированным категориям.

    Args:
        status: Исходное значение статуса для нормализации

    Returns:
        Нормализованная строка статуса
    """

    if pd.isna(status):
        return "unknown"

    status = str(status).strip().lower()

    mapping = {
        "ok": "operational",
        "working": "operational",
        "op": "operational",
        "maintenance": "maintenance_scheduled",
        "maint_sched": "maintenance_scheduled",
        "planned": "planned_installation",
        "scheduled_install": "planned_installation",
        "broken": "faulty",
        "error": "faulty"
    }

    return mapping.get(status, status)


def normalize_status_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет нормализацию статусов к колонке status.

    Args:
        df: Входной DataFrame с колонкой status

    Returns:
        DataFrame с нормализованными значениями статусов
    """

    df["status"] = df["status"].apply(normalize_status)

    return df


def clean_uptime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает и преобразует колонку процента аптайма в числовой формат.

    Args:
        df: Входной DataFrame с колонкой uptime_pct

    Returns:
        DataFrame с очищенной колонкой uptime_pct в числовом формате
    """

    df["uptime_pct"] = (
        df["uptime_pct"]
        .astype(str)
        .str.replace(",", ".")
    )

    df["uptime_pct"] = pd.to_numeric(df["uptime_pct"], errors="coerce")

    return df


def check_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверяет согласованность дат (дата калибровки не может быть раньше даты установки).

    Args:
        df: Входной DataFrame с колонками дат

    Returns:
        DataFrame с некорректными датами калибровки, замененными на None
    """

    mask = (df["install_date"].notna()) & (df["last_calibration_date"].notna())
    invalid = mask & (df["last_calibration_date"] < df["install_date"])
    df.loc[invalid, "last_calibration_date"] = None

    return df


def filter_warranty(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделяет устройства на группы по гарантии (в гарантии и вне гарантии).

    Args:
        df: Входной DataFrame с колонкой warranty_until

    Returns:
        Кортеж из двух DataFrame: (устройства_в_гарантии, устройства_вне_гарантии)
    """

    today = datetime.today()
    in_warranty = df[df["warranty_until"] >= today]
    out_warranty = df[df["warranty_until"] < today]

    return in_warranty, out_warranty


def clinics_with_problems(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует количество проблем по клиникам и сортирует по убыванию.

    Args:
        df: Входной DataFrame с данными о клиниках и проблемах

    Returns:
        DataFrame с ID клиники, названием и общим количеством проблем
    """

    result = (
        df.groupby(["clinic_id", "clinic_name"])
        .agg({
            "issues_reported_12mo": "sum"
        })
        .sort_values("issues_reported_12mo", ascending=False)
    )

    return result


def calibration_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает отчет со сроками калибровки для всех устройств.

    Args:
        df: Входной DataFrame с данными о калибровке устройств

    Returns:
        DataFrame с ключевыми колонками для отчета
    """

    report = df[[
        "device_id",
        "clinic_name",
        "model",
        "last_calibration_date"
    ]]

    return report


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает сводную таблицу с проблемами и аптаймом по клиникам и моделям.

    Args:
        df: Входной DataFrame с данными о клиниках и моделях

    Returns:
        Сводная таблица с суммой проблем и средним аптаймом
    """

    pivot = pd.pivot_table(
        df,
        index=["clinic_name", "model"],
        values=[
            "issues_reported_12mo",
            "uptime_pct"
        ],
        aggfunc={
            "issues_reported_12mo": "sum",
            "uptime_pct": "mean"
        }
    )

    return pivot


async def save_excel_async(df: pd.DataFrame, folder: str, name: str) -> None:
    """
    Асинхронно сохраняет DataFrame в Excel файл.

    Args:
        df: DataFrame для сохранения
        folder: Целевая папка
        name: Имя файла
    """

    path = os.path.join(folder, name)
    await asyncio.to_thread(df.to_excel, path)


def sync_main(files: List[str], output_folder: str) -> None:
    """
    Синхронная версия выполнения программы.

    Args:
        files: Список исходных файлов
        output_folder: Папка для сохранения результатов
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_time = time.time()

    dfs = [pd.read_excel(f, engine='openpyxl') for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower()

    df = convert_dates(df)
    df = normalize_status_column(df)
    df = clean_uptime(df)
    df = check_dates(df)

    in_w, out_w = filter_warranty(df)
    clinics = clinics_with_problems(df)
    calib = calibration_report(df)
    summary = summary_table(df)

    in_w.to_excel(os.path.join(output_folder, "in_warranty.xlsx"))
    out_w.to_excel(os.path.join(output_folder, "out_warranty.xlsx"))
    clinics.to_excel(os.path.join(output_folder, "clinics.xlsx"))
    calib.to_excel(os.path.join(output_folder, "calibration.xlsx"))
    summary.to_excel(os.path.join(output_folder, "summary.xlsx"))

    print(f"СИНХРОННО: {round(time.time() - start_time, 2)} сек")


async def async_main(files: List[str], output_folder: str) -> None:
    """
    Асинхронная версия выполнения программы.

    Args:
        files: Список исходных файлов
        output_folder: Папка для сохранения результатов
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_time = time.time()

    df = await load_multiple_files(files)

    df = convert_dates(df)
    df = normalize_status_column(df)
    df = clean_uptime(df)
    df = check_dates(df)

    in_w, out_w = filter_warranty(df)
    clinics = clinics_with_problems(df)
    calib = calibration_report(df)
    summary = summary_table(df)

    await asyncio.gather(
        save_excel_async(in_w, output_folder, "in_warranty.xlsx"),
        save_excel_async(out_w, output_folder, "out_warranty.xlsx"),
        save_excel_async(clinics, output_folder, "clinics.xlsx"),
        save_excel_async(calib, output_folder, "calibration.xlsx"),
        save_excel_async(summary, output_folder, "summary.xlsx")
    )

    print(f"АСИНХРОННО: {round(time.time() - start_time, 2)} сек")


if __name__ == "__main__":
    file_list = [f"medical_diagnostic_devices_{i}.xlsx" for i in range(1, 11)]
    existing_files = [
        f for f in file_list
        if os.path.exists(f) and os.path.getsize(f) > 0
    ]

    if existing_files:
        sync_main(existing_files, "results_sync")
        asyncio.run(async_main(existing_files, "results_async"))
    else:
        print("Ошибка: Корректные файлы не найдены.")