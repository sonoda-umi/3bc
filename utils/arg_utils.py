import re


def parse_gen_range(gen_range_str: str) -> tuple[int, int]:
    gen_range_str = gen_range_str.strip()

    # Strict pattern: digits-digits (no spaces, no negatives unless you want them)
    if not re.fullmatch(r"\d+-\d+", gen_range_str):
        raise ValueError(f"Invalid format: '{gen_range_str}'. Expected 'start-end' (e.g., '10-100').")

    start_str, end_str = gen_range_str.split("-")

    try:
        start = int(start_str)
        end = int(end_str)
    except ValueError:
        raise ValueError(f"Non-integer values in range: '{gen_range_str}'")

    if start > end:
        raise ValueError(f"Invalid range: start ({start}) must be <= end ({end})")

    max_size = 1_000_000
    if end - start + 1 > max_size:
        raise ValueError(f"Range too large: {end - start + 1} elements (max {max_size})")

    return start, end + 1
