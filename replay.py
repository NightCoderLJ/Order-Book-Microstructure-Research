"""
replay.py -- drives ingest.py + book.py over a full LOBSTER trading day.

Takes in: config.yaml (data file paths, checkpoint interval, chunk size).

Produces: a fully reconstructed OrderBook plus a stats dict (rows read,
per-event-type counts, invariant violation counts, wall-clock throughput).
Run directly (`python replay.py`) this prints the events/sec benchmark
number and the event-count reconciliation for the full day.
"""

import time

import yaml

from book import OrderBook
from ingest import iter_events


def load_config(path: str = "config.yaml") -> dict:
    """Loads the small set of named assumptions this pipeline depends on."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_full_day(config: dict):
    """Streams the configured message file through a fresh OrderBook and
    returns (book, stats): the reconstructed book plus rows read, per-type
    event counts, invariant violation counts, and wall-clock throughput.
    """
    book = OrderBook(checkpoint_interval=config["book"]["checkpoint_interval"])
    chunk_size = config["ingest"]["chunk_size"]

    rows_read = 0
    start = time.perf_counter()

    for chunk in iter_events(config["data"]["message_file"], chunk_size=chunk_size):
        for event in chunk:
            book.apply_event(event)
            rows_read += 1

    elapsed = time.perf_counter() - start

    stats = {
        "rows_read": rows_read,
        "events_applied": book.events_applied,
        "event_type_counts": dict(book.event_type_counts),
        "violations": dict(book.violations),
        "elapsed_seconds": elapsed,
        "events_per_second": rows_read / elapsed if elapsed > 0 else float("inf"),
    }
    return book, stats


def main():
    config = load_config()
    _, stats = run_full_day(config)

    print(f"rows read:            {stats['rows_read']:,}")
    print(f"events applied:       {stats['events_applied']:,}")
    print("event type counts:")
    for event_type, count in sorted(stats["event_type_counts"].items()):
        print(f"  {event_type:<16} {count:,}")
    print(f"invariant violations: {stats['violations'] or 'none'}")
    print(f"elapsed:              {stats['elapsed_seconds']:.2f}s")
    print(f"throughput:           {stats['events_per_second']:,.0f} events/sec")


if __name__ == "__main__":
    main()
