def print_summary(rows, title=""):
    width = max(len(r[0]) for r in rows) if rows else 10
    print("=" * (width + 28))
    print(title)
    print("-" * (width + 28))
    for k, v in rows:
        print(f"{k:<{width}} : {v}")
    print("=" * (width + 28))
