from airflow.sdk import dag, task
from airflow.models import Variable
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from pendulum import datetime, now
import logging

# Buckets (note: no s3:// prefix for S3Hook/S3KeySensor)
RAW_BUCKET = "raw-data"
STAGING_BUCKET = "staging-data"
ARCHIVE_BUCKET = "archive-data"

# Configurable batch handling parameters via Airflow Variables
MAX_BATCH_SIZE = int(Variable.get("CREDITCARD_MAX_BATCH_SIZE", default_var=10))
BATCH_WINDOW = int(Variable.get("CREDITCARD_BATCH_WINDOW", default_var=60))  # seconds
CHUNK_SIZE = int(Variable.get("CREDITCARD_CHUNK_SIZE", default_var=4))      # files per task

# Helper function for chunking
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@dag(
    start_date=datetime(2025, 9, 18),
    schedule=None,
    catchup=False,
    tags=["minio", "s3sensor", "creditcardfraud", "staging"],
)
def creditcardfrauddetection_dag():

    # Wait until at least one CSV file is present in the raw-data bucket
    wait_for_file = S3KeySensor(
        task_id="wait_for_new_file",
        bucket_key="*.csv",
        bucket_name=RAW_BUCKET,
        aws_conn_id="minio_s3_conn",
        wildcard_match=True,
        poke_interval=30,
        timeout=60 * 60,
)

    @task
    def stage_file():
        """Move raw files → staging bucket"""
        hook = S3Hook(aws_conn_id="minio_s3_conn")
        keys = hook.list_keys(bucket_name=RAW_BUCKET, prefix="", delimiter="")

        staged_files = []
        for key in keys or []:
            if key.endswith(".csv"):
                staging_key = key
                logging.info(f"Staging {RAW_BUCKET}/{key} → {STAGING_BUCKET}/{staging_key}")

                # Copy then delete
                hook.copy_object(
                    source_bucket_key=key,
                    dest_bucket_key=staging_key,
                    source_bucket_name=RAW_BUCKET,
                    dest_bucket_name=STAGING_BUCKET,
                )
                hook.delete_objects(bucket=RAW_BUCKET, keys=[key])

                staged_files.append({"path": staging_key, "timestamp": now().int_timestamp})
        return staged_files

    @task
    def collect_staged_files(staged_files: list):
        cutoff = now().int_timestamp - BATCH_WINDOW
        eligible = [f["path"] for f in staged_files if f["timestamp"] <= cutoff]

        if not eligible:
            logging.warning("No files ready yet (inside batch window)")
            return []

        limited = eligible[:MAX_BATCH_SIZE]
        logging.info(f"Collected {len(limited)} files for batch: {limited}")
        return limited

    @task
    def create_chunks(file_paths: list):
        return list(chunk_list(file_paths, CHUNK_SIZE))

    @task
    def process_file_batch(file_batch: list):
        processed = []
        for file_path in file_batch:
            logging.info(f"Processing {file_path}")
            processed.append(file_path)
        return processed

    @task
    def transform_data(processed_files: list):
        return [f"{f}_transformed" for f in processed_files]

    @task
    def merge_transformed(chunks: list):
        total_records = sum(len(chunk) for chunk in chunks)
        return f"Merged {len(chunks)} chunks into {total_records} records"

    @task
    def cleanup(processed_batches: list):
        """Move processed files → archive bucket"""
        hook = S3Hook(aws_conn_id="minio_s3_conn")

        for batch in processed_batches:
            for f in batch:
                archive_key = f
                logging.info(f"Archiving {STAGING_BUCKET}/{f} → {ARCHIVE_BUCKET}/{archive_key}")

                # Copy then delete
                hook.copy_object(
                    source_bucket_key=f,
                    dest_bucket_key=archive_key,
                    source_bucket_name=STAGING_BUCKET,
                    dest_bucket_name=ARCHIVE_BUCKET,
                )
                hook.delete_objects(bucket=STAGING_BUCKET, keys=[f])

        return "Archived files"

    # DAG flow
    staged = stage_file()
    staged.set_upstream(wait_for_file)  # ensure sensor runs first

    files = collect_staged_files(staged)
    chunks = create_chunks(files)
    processed = process_file_batch.expand(file_batch=chunks)
    transformed = transform_data.expand(processed_files=processed)
    merged = merge_transformed(transformed)
    cleanup(processed)


dag_instance = creditcardfrauddetection_dag()
