from benchmarks.utils import (
    Extractor,
)

__all__ = [
    "parse_mdout_column",
    "parse_mdout_rows",
    "read_first_field_int",
]


def parse_mdout_rows(mdout_path, columns, *, int_columns=("step",)):
    return Extractor.parse_mdout_rows(
        mdout_path, columns=columns, int_columns=int_columns
    )


def parse_mdout_column(mdout_path, column_name):
    rows = parse_mdout_rows(mdout_path, [column_name], int_columns=())
    return [row[column_name] for row in rows]


def read_first_field_int(path):
    return Extractor.read_first_field_int(path)
