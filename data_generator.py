"""
Synthetic data generator for the mule account detection platform.

Produces three tables:
  - accounts.csv      : account metadata
  - transactions.csv  : transfers between accounts
  - complaints.csv    : cybercrime complaints filed by victims

Injected fraud patterns:
  - Fan-in / fan-out mule hubs (many victims pay in, funds quickly forwarded out)
  - Rapid pass-through chains (layering across several intermediary accounts)
  - Dormant accounts that suddenly activate with high-value bursts
  - Cash-out sinks (accounts that receive mule proceeds and withdraw)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


@dataclass
class GeneratorConfig:
    n_normal_accounts: int = 400
    n_fraud_rings: int = 6
    mules_per_ring: int = 4
    victims_per_ring: int = 15
    n_normal_transactions: int = 4000
    start_date: datetime = datetime(2026, 1, 1)
    days: int = 90
    seed: int = SEED


def _random_time(rng: random.Random, start: datetime, days: int) -> datetime:
    return start + timedelta(seconds=rng.randint(0, days * 86400))


def generate(config: GeneratorConfig | None = None, out_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    cfg = config or GeneratorConfig()
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    accounts: list[dict] = []
    transactions: list[dict] = []
    complaints: list[dict] = []
    tx_id = 0

    def add_account(acc_id: str, kind: str, opened: datetime) -> None:
        accounts.append({
            "account_id": acc_id,
            "holder_name": f"Holder_{acc_id}",
            "account_type": kind,
            "opened_on": opened.date().isoformat(),
            "true_label": "mule" if kind == "mule" else ("cashout" if kind == "cashout" else "normal"),
        })

    def add_tx(src: str, dst: str, amount: float, ts: datetime, channel: str) -> None:
        nonlocal tx_id
        tx_id += 1
        transactions.append({
            "tx_id": f"T{tx_id:06d}",
            "sender": src,
            "receiver": dst,
            "amount": round(float(amount), 2),
            "timestamp": ts.isoformat(timespec="seconds"),
            "channel": channel,
        })

    # Normal population
    normal_ids = [f"AC{i:05d}" for i in range(cfg.n_normal_accounts)]
    for acc in normal_ids:
        opened = cfg.start_date - timedelta(days=rng.randint(60, 2000))
        add_account(acc, "normal", opened)

    channels = ["UPI", "NEFT", "IMPS", "CARD"]
    for _ in range(cfg.n_normal_transactions):
        src, dst = rng.sample(normal_ids, 2)
        amount = float(np_rng.lognormal(mean=6.5, sigma=0.9))
        add_tx(src, dst, amount, _random_time(rng, cfg.start_date, cfg.days), rng.choice(channels))

    # Fraud rings
    mule_counter = 0
    cashout_counter = 0
    for ring in range(cfg.n_fraud_rings):
        mules = []
        for _ in range(cfg.mules_per_ring):
            acc = f"MU{mule_counter:05d}"
            mule_counter += 1
            # Mule accounts were opened recently and were mostly dormant before the burst
            opened = cfg.start_date + timedelta(days=rng.randint(-20, 40))
            add_account(acc, "mule", opened)
            mules.append(acc)

        cashout = f"CO{cashout_counter:05d}"
        cashout_counter += 1
        add_account(cashout, "cashout", cfg.start_date - timedelta(days=rng.randint(200, 900)))

        victims = []
        for v in range(cfg.victims_per_ring):
            vid = f"VI{ring:02d}{v:03d}"
            add_account(vid, "normal", cfg.start_date - timedelta(days=rng.randint(60, 2000)))
            victims.append(vid)

        burst_start = cfg.start_date + timedelta(days=rng.randint(10, cfg.days - 15))

        # Fan-in: victims pay mules in a short window
        for victim in victims:
            mule = rng.choice(mules)
            amount = float(np_rng.uniform(5000, 60000))
            ts = burst_start + timedelta(minutes=rng.randint(0, 60 * 24 * 3))
            add_tx(victim, mule, amount, ts, rng.choice(["UPI", "IMPS"]))
            complaints.append({
                "complaint_id": f"CMP{len(complaints) + 1:05d}",
                "victim_account": victim,
                "reported_account": mule,
                "amount_lost": round(amount, 2),
                "complaint_date": (ts + timedelta(hours=rng.randint(2, 96))).date().isoformat(),
                "category": rng.choice(["UPI fraud", "Investment scam", "Phishing", "Job scam"]),
            })

        # Rapid fan-out: mules forward most funds to intermediaries and then to cashout
        intermediaries = [f"IN{ring:02d}{i:03d}" for i in range(3)]
        for inter in intermediaries:
            add_account(inter, "normal", cfg.start_date - timedelta(days=rng.randint(100, 1000)))

        for mule in mules:
            inflow = sum(
                t["amount"] for t in transactions if t["receiver"] == mule
            )
            remaining = inflow * 0.92
            for _ in range(2):
                inter = rng.choice(intermediaries)
                share = remaining / 2
                ts = burst_start + timedelta(days=rng.randint(0, 4), hours=rng.randint(0, 20))
                add_tx(mule, inter, share, ts, "IMPS")
            # Intermediaries pass funds to cashout quickly
            for inter in intermediaries:
                fwd = rng.uniform(0.2, 0.5) * remaining
                ts = burst_start + timedelta(days=rng.randint(3, 7), hours=rng.randint(0, 20))
                add_tx(inter, cashout, fwd, ts, "NEFT")

        # Mules also show a few small "camouflage" transfers to normal accounts
        for mule in mules:
            for _ in range(3):
                peer = rng.choice(normal_ids)
                add_tx(mule, peer, float(np_rng.uniform(200, 900)),
                       _random_time(rng, cfg.start_date, cfg.days), "UPI")

    # Dormant-then-active accounts (noise that a naive threshold would flag)
    for i in range(10):
        acc = f"DO{i:05d}"
        add_account(acc, "normal", cfg.start_date - timedelta(days=rng.randint(500, 1500)))
        peer = rng.choice(normal_ids)
        add_tx(peer, acc, float(np_rng.uniform(2000, 8000)), _random_time(rng, cfg.start_date, cfg.days), "NEFT")

    accounts_df = pd.DataFrame(accounts).drop_duplicates("account_id")
    transactions_df = pd.DataFrame(transactions).sort_values("timestamp").reset_index(drop=True)
    complaints_df = pd.DataFrame(complaints)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    accounts_df.to_csv(out / "accounts.csv", index=False)
    transactions_df.to_csv(out / "transactions.csv", index=False)
    complaints_df.to_csv(out / "complaints.csv", index=False)

    return {"accounts": accounts_df, "transactions": transactions_df, "complaints": complaints_df}


if __name__ == "__main__":
    tables = generate(out_dir=Path(__file__).resolve().parent.parent / "data")
    for name, df in tables.items():
        print(f"{name}: {len(df)} rows")
