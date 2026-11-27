import requests
import pandas as pd
from pathlib import Path
import os
import time

from config import settings

__PATH__ = Path(__file__).resolve()
DATA_PATH = Path(__PATH__.parent.parent, "data")


# To paginate request
def paginate(offset, length, file_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal offset
            max_tries = 3
            num_rows_total = offset + 1
            request_tries = 0
            dataset = []
            backoff = 0.1
            while offset < num_rows_total:
                try:
                    data, num_rows_total = func(*args, **kwargs, offset=offset, length=length)
                    print(f"Fetched {offset + length} records")
                    dataset.extend(data)
                    offset += length
                except Exception as e:
                    request_tries += 1
                    print(e, "Error in fetching records")
                    if request_tries > max_tries:
                        print(f"Http calls {request_tries} got exhausted after maximum tries")
                        break
                    time.sleep(backoff)
                    backoff *= request_tries
            # write the pandas dataframe to a file
            pd.DataFrame(dataset).to_parquet(file_name)

        return wrapper

    return decorator


# Get FinanceBench datasets
@paginate(
    offset=0, length=settings.request_row_length, file_name=Path(DATA_PATH, "financebench.parquet")
)
def get_finance_bench_data(offset, length):
    """
    Get dataset from FinanceBench
    """
    records = []
    data = requests.get(
        settings.finance_bench_url,
        params={
            "dataset": settings.finance_bench_dataset,
            "config": settings.finance_bench_config,
            "split": settings.finance_bench_split,
            "offset": offset,
            "length": length,
        },
    )
    data_json = data.json()
    for item in data_json["rows"]:
        records.append(item["row"])
    num_rows_total = int(data_json["num_rows_total"])
    return records, num_rows_total


# Get PDFs of all the sec filings
def get_sec_filings_pdfs(fresh_download=False):
    """
    Get all the pdfs listed in the FinanceBench Dataset
    """
    file_path = Path(DATA_PATH, "financebench.parquet")
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} is not found")
    df = pd.read_parquet(file_path)
    doc_names = df["doc_name"].unique().tolist()
    print(f"There are {len(doc_names)} to be downloaded.")
    for doc_name in doc_names:
        doc_link = f"{settings.finance_bench_pdf_base_url}/{doc_name}.pdf"
        pdf_path = Path(DATA_PATH, "pdf", doc_name + ".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if not fresh_download and os.path.exists(pdf_path):
            print(f"File {pdf_path} already exists, Trying to fetch next file")
            continue
        with open(pdf_path, mode="wb") as pdf:
            res = requests.get(doc_link)
            if res.status_code != 200:
                print(f"Issues in fetching the file: {doc_name}.pdf from the doc_link")
                continue
            pdf.write(res.content)
            print(f"File: {doc_name}.pdf is written")


if __name__ == "__main__":
    get_finance_bench_data()
    get_sec_filings_pdfs()
