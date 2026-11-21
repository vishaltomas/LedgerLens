from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(ROOT, ".env"))

    finance_bench_url: str
    finance_bench_dataset: str
    finance_bench_config: str
    finance_bench_split: str
    finance_bench_pdf_base_url: str
    request_row_length: int = 100

settings = Settings()