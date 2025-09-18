
"""
fraud_simulator_minio.py

Generates a Fraud-Detection-Handbook style simulated transaction dataset,
splits it into CSV batches and (optionally) uploads to MinIO (S3-compatible).

Usage example:
  python fraud_simulator_minio.py --n-customers 500 --n-terminals 1000 --nb-days 30 \
    --start-date 2018-04-01 --r 5 --batch-size 5000 --bucket fraud-sim \
    --minio-endpoint http://localhost:9000 --minio-access minioadmin \
    --minio-secret minioadmin --upload
"""

import os
import argparse
import uuid
import random
import math
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

# boto3 is required for uploading to MinIO
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    boto3 = None

# -------------------------
# Utilities / Generators
# -------------------------
def generate_customer_profiles_table(n_customers: int, random_state: int = 0) -> pd.DataFrame:
    np.random.seed(random_state)
    ids = np.arange(n_customers)
    x = np.random.uniform(0, 100, n_customers)
    y = np.random.uniform(0, 100, n_customers)
    mean_amount = np.random.uniform(5, 100, n_customers)
    std_amount = mean_amount / 2.0
    mean_nb_tx_per_day = np.random.uniform(0, 4, n_customers)

    df = pd.DataFrame({
        "CUSTOMER_ID": ids,
        "x_customer_id": x,
        "y_customer_id": y,
        "mean_amount": mean_amount,
        "std_amount": std_amount,
        "mean_nb_tx_per_day": mean_nb_tx_per_day
    })
    return df

def generate_terminal_profiles_table(n_terminals: int, random_state: int = 0) -> pd.DataFrame:
    np.random.seed(random_state + 1)
    ids = np.arange(n_terminals)
    x = np.random.uniform(0, 100, n_terminals)
    y = np.random.uniform(0, 100, n_terminals)
    df = pd.DataFrame({
        "TERMINAL_ID": ids,
        "x_terminal_id": x,
        "y_terminal_id": y
    })
    return df

def associate_terminals(customers: pd.DataFrame, terminals: pd.DataFrame, r: float = 5.0,
                        use_kdtree: bool = False) -> pd.DataFrame:
    """
    For each customer, compute list of terminal IDs within radius r and store in 'available_terminals'.
    If no terminals are within r, available_terminals will be an empty list.
    For large n, you may want to install scikit-learn and set use_kdtree=True for speed.
    """
    term_xy = terminals[["x_terminal_id", "y_terminal_id"]].values.astype(float)
    cust_xy = customers[["x_customer_id", "y_customer_id"]].values.astype(float)
    n_customers = customers.shape[0]
    avail = []

    # Try KDTree if user requested and it's installed
    if use_kdtree:
        try:
            from sklearn.neighbors import BallTree
            tree = BallTree(term_xy, leaf_size=40, metric='euclidean')
            # for each customer, query terminals within radius r
            ind = tree.query_radius(cust_xy, r=r)
            avail = [list(arr.astype(int)) for arr in ind]
            customers = customers.copy()
            customers["available_terminals"] = avail
            return customers
        except Exception:
            # fallback to plain approach
            pass

    # plain but chunked approach (safe for moderate sizes)
    customers = customers.copy()
    avail = []
    for cx, cy in cust_xy:
        dx = term_xy[:, 0] - cx
        dy = term_xy[:, 1] - cy
        d2 = dx * dx + dy * dy
        inds = np.where(d2 <= r * r)[0]
        avail.append(list(inds.astype(int)))
    customers["available_terminals"] = avail
    return customers

def generate_transactions_table_for_customer(customer_row: pd.Series,
                                             terminals: pd.DataFrame,
                                             window_minutes: int = 15,
                                             baseline_fraud_rate: float = 0.005,
                                             p_high_amount: float = 0.5,
                                             p_geo_fraud: float = 0.5,
                                             p_burst: float = 0.02) -> pd.DataFrame:
    """
    Simulate transactions for ONE customer within a recent time window (e.g., last 15 minutes).
    Includes fraud scenarios:
      - random baseline fraud
      - high-amount anomaly
      - geographic anomaly
      - burst (a sequence of fraudulent transactions in quick succession)
    """
    cust_id = int(customer_row["CUSTOMER_ID"])
    mean_amount = float(customer_row["mean_amount"])
    std_amount = float(customer_row["std_amount"])
    mean_nb_tx_per_day = float(customer_row["mean_nb_tx_per_day"])
    available_terminals: list[int] = customer_row["available_terminals"]

    # Define the time window
    start_dt = pd.Timestamp.now() - pd.Timedelta(minutes=window_minutes)
    nb_seconds = window_minutes * 60

    # Determine number of transactions for this customer (Poisson around mean_nb_tx_per_day)
    nb_tx = np.random.poisson(mean_nb_tx_per_day)
    txs = []

    for _ in range(nb_tx):
        # random timestamp within window
        seconds_offset = int(np.random.uniform(0, nb_seconds))
        tx_dt = start_dt + pd.Timedelta(seconds=seconds_offset)

        terminal_id = int(random.choice(available_terminals)) if available_terminals else int(random.randint(0, len(terminals)-1))
        # amount: normal around mean (clipped to >0.1)
        amount = max(0.1, float(np.random.normal(mean_amount, std_amount)))

        # Default label: non-fraud
        is_fraud = 1 if random.random() < baseline_fraud_rate else 0

        # Scenario 1: high-amount anomaly
        if amount > 4 * mean_amount and random.random() < p_high_amount:
            is_fraud = 1

        # Scenario 2: geo fraud (customer uses random far terminal)
        elif random.random() < (baseline_fraud_rate * 10):
            terminal_id = int(random.randint(0, len(terminals)-1))
            if available_terminals and terminal_id in available_terminals and len(terminals) > 1:
                terminal_id = (terminal_id + int(len(terminals) // 2)) % len(terminals)
            if random.random() < p_geo_fraud:
                is_fraud = 1

        # Scenario 3: bursting fraud
        if random.random() < p_burst:
            burst_count = random.randint(2, 4)
            for b in range(burst_count):
                bt = tx_dt + pd.Timedelta(seconds=b * random.randint(5, 20))
                burst_amount = round(max(0.1, np.random.uniform(mean_amount * 2.0, mean_amount * 8.0)), 2)
                txs.append({
                    "TRANSACTION_ID": str(uuid.uuid4()),
                    "TX_DATETIME": bt,
                    "CUSTOMER_ID": cust_id,
                    "TERMINAL_ID": int(terminal_id),
                    "TX_AMOUNT": burst_amount,
                    "TX_TIME_SECONDS": int((bt - start_dt).total_seconds()),
                    "TX_TIME_DAYS": 0,
                    "TX_FRAUD": 1
                })
            continue

        txs.append({
            "TRANSACTION_ID": str(uuid.uuid4()),
            "TX_DATETIME": tx_dt,
            "CUSTOMER_ID": cust_id,
            "TERMINAL_ID": int(terminal_id),
            "TX_AMOUNT": round(amount, 2),
            "TX_TIME_SECONDS": int((tx_dt - start_dt).total_seconds()),
            "TX_TIME_DAYS": 0,
            "TX_FRAUD": int(is_fraud)
        })

    if len(txs) == 0:
        return pd.DataFrame(columns=["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID",
                                     "TX_AMOUNT", "TX_TIME_SECONDS", "TX_TIME_DAYS", "TX_FRAUD"])
    
    df_tx = pd.DataFrame(txs)
    return df_tx


def generate_dataset(n_customers=500, n_terminals=1000,
                     window_minutes=15, start_date=None, r=5.0,
                     seed=0, use_kdtree=False, show_progress=True):
    """
    Generates customers, terminals, associates terminals, and transactions
    for a recent time window (e.g., last `window_minutes` minutes).
    """

    #  Generate profiles
    customers = generate_customer_profiles_table(n_customers, random_state=seed)
    terminals = generate_terminal_profiles_table(n_terminals, random_state=seed)
    customers = associate_terminals(customers, terminals, r=r, use_kdtree=use_kdtree)

    #  Generate transactions for all customers
    tx_list = []
    iterator = customers.itertuples(index=False)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(list(iterator), desc="Generating customers' transactions")

    for c in iterator:
        c_row = pd.Series(index=customers.columns, data=list(c))
        df_c = generate_transactions_table_for_customer(
            c_row,
            terminals,
            window_minutes=window_minutes
        )
        if not df_c.empty:
            tx_list.append(df_c)

    #  Aggregate and sort transactions
    if tx_list:
        transactions = pd.concat(tx_list, ignore_index=True)
        transactions = transactions.sort_values("TX_DATETIME").reset_index(drop=True)
    else:
        transactions = pd.DataFrame(columns=["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", 
                                             "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS", 
                                             "TX_TIME_DAYS", "TX_FRAUD"])
    
    return customers, terminals, transactions


# -------------------------
# MinIO / S3 helper
# -------------------------
def init_s3_client(endpoint: str, access_key: str, secret_key: str, region_name: str = "us-east-1"):
    if boto3 is None:
        raise RuntimeError("boto3 is required for MinIO uploads. `pip install boto3`")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name
    )
    return s3

def ensure_bucket(s3_client, bucket_name: str):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        try:
            s3_client.create_bucket(Bucket=bucket_name)
        except Exception as exc:
            # possible error when running MinIO with custom config; still try to continue
            print(f"Warning: could not create bucket {bucket_name}: {exc}")

# -------------------------
# Batch creation & upload
# -------------------------
def save_and_upload_batches(transactions: pd.DataFrame, batch_size: int, out_dir: str,
                            s3_client=None, bucket: str = None, compress: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    total = len(transactions)
    n_batches = max(1, math.ceil(total / batch_size))  # ensure at least 1 batch
    created_files = []

    for bi, start in enumerate(range(0, max(total, 1), batch_size), start=1):
        batch_df = transactions.iloc[start:start + batch_size]
        
        # If transactions are empty, create an empty dataframe with correct columns
        if batch_df.empty:
            batch_df = pd.DataFrame(columns=["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID",
                                             "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS",
                                             "TX_TIME_DAYS", "TX_FRAUD"])
        
        ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        filename = f"transactions_{ts}_batch{bi:03d}.csv"
        path = os.path.join(out_dir, filename)

        if compress:
            path += ".gz"
            batch_df.to_csv(path, index=False, compression="gzip")
        else:
            batch_df.to_csv(path, index=False)

        created_files.append(path)
        print(f"Created batch file {path} ({len(batch_df)} rows)")

        if s3_client and bucket:
            key = os.path.basename(path)
            try:
                s3_client.upload_file(path, bucket, key)
                print(f"  Uploaded {key} -> s3://{bucket}/{key}")
            except Exception as e:
                print(f"  Upload failed for {key}: {e}")

    return created_files

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-customers", type=int, default=500, help="Number of customers")
    p.add_argument("--n-terminals", type=int, default=1000, help="Number of terminals")
    p.add_argument("--nb-days", type=int, default=30, help="Number of days to simulate")
    p.add_argument("--start-date", type=str, default="2018-04-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--r", type=float, default=5.0, help="Radius for available terminals")
    p.add_argument("--seed", type=int, default=0, help="Random seed (reproducible)")
    p.add_argument("--batch-size", type=int, default=5000, help="CSV batch size (rows per file)")
    p.add_argument("--out-dir", type=str, default="data/out_batches", help="Local output directory")
    p.add_argument("--compress", action="store_true", help="Compress CSVs with gzip")
    p.add_argument("--upload", action="store_true", help="Upload batches to MinIO/S3")
    p.add_argument("--minio-endpoint", type=str, default="http://localhost:9000")
    p.add_argument("--minio-access", type=str, default="minioadmin")
    p.add_argument("--minio-secret", type=str, default="minioadmin")
    p.add_argument("--bucket", type=str, default="fraud-simulated-data")
    p.add_argument("--use-kdtree", action="store_true", help="Use sklearn BallTree for terminal association (faster for large inputs)")
    p.add_argument("--recent-timestamps", action="store_true",
                   help="Generate transactions within the last 10 minutes instead of historical days")
    p.add_argument("--window-minutes", type=int, default=None,
               help="Simulate transactions within the last N minutes instead of historical days")

    return p.parse_args()



def main():
    args = parse_args()
    print("Generating dataset with parameters:", vars(args))

    customers, terminals, transactions = generate_dataset(
        n_customers=args.n_customers,
        n_terminals=args.n_terminals,
        start_date=args.start_date,
        r=args.r,
        seed=args.seed,
        use_kdtree=args.use_kdtree,
        show_progress=True
    )

    print(f"Generated: {len(customers)} customers, {len(terminals)} terminals, {len(transactions)} transactions")
    if len(transactions) == 0:
        print("No transactions generated (check mean_nb_tx_per_day parameter). Exiting.")
        return

    # Basic stats
    fraud_pct = transactions["TX_FRAUD"].mean() * 100
    print(f"Overall fraud rate in generated data: {fraud_pct:.4f}%")

    # Initialize S3 / MinIO client if requested
    s3_client = None
    if args.upload:
        try:
            s3_client = init_s3_client(args.minio_endpoint, args.minio_access, args.minio_secret)
            ensure_bucket(s3_client, args.bucket)
            print(f"Connected to MinIO/S3 endpoint {args.minio_endpoint}, bucket '{args.bucket}'")
        except Exception as exc:
            print(f"Warning: could not initialize S3 client: {exc}")
            s3_client = None

    # Save and (maybe) upload
    files = save_and_upload_batches(transactions, args.batch_size, args.out_dir,
                                    s3_client=s3_client, bucket=args.bucket if args.upload else None,
                                    compress=args.compress)

    print("Done. Created files:")
    for f in files:
        print(" -", f)

if __name__ == "__main__":
    main()
