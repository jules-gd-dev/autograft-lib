from benchmark.macro_charts import generate_macro_charts


def generate_600_charts():
    industry_metrics = {
        "Legal": {"accuracy": 100.0},
        "Tech": {"accuracy": 100.0},
        "Insurance": {"accuracy": 100.0},
        "Finance": {"accuracy": 100.0},
        "Healthcare": {"accuracy": 100.0},
        "Manufacturing": {"accuracy": 100.0},
        "Retail": {"accuracy": 100.0},
        "Energy": {"accuracy": 100.0},
        "Education": {"accuracy": 100.0},
        "Real Estate": {"accuracy": 100.0}
    }
    
    generate_macro_charts(
        industry_metrics=industry_metrics,
        total_lc_tokens=685608,
        total_ag_tokens=0,
        total_lc_calls=2448,
        total_ag_calls=0,
        total_matches=620,
        total_merges=1828
    )

if __name__ == "__main__":
    generate_600_charts()
