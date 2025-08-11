import os
import pandas as pd


def ensure_final_results_csv() -> None:
    # No-op in production: plots are served from DB only. Keep for legacy compatibility.
    return


if __name__ == "__main__":
    ensure_final_results_csv()


