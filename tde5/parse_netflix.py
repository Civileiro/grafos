import csv
from types import TracebackType
from typing import Iterable, Self


class NetflixCsvReader:
    def __init__(self: Self, filename: str) -> None:
        self.file = open(filename, "r", newline="")
        self.reader = csv.DictReader(self.file)

    def close(self: Self) -> None:
        self.file.close()

    def __enter__(self: Self) -> Self:
        self.file.__enter__()
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return self.file.__exit__(exc_type, exc_value, traceback)

    def __iter__(self: Self) -> Self:
        return self

    def __next__(self: Self) -> tuple[set[str], set[str]]:
        row = next(self.reader)
        diretores = set(self.normalize_names(row["director"]))
        atores = set(self.normalize_names(row["cast"]))
        return diretores, atores

    def normalize_names(self: Self, names: str, sep: str = ", ") -> Iterable[str]:
        for name in names.split(sep):
            # Durante o processo de construção, todos os nomes devem
            # ser padronizados em letras maiúsculas e sem espaços em
            # branco no início e no final da string. Entradas do
            # conjunto de dados onde o nome do diretor e/ou nome do
            # elenco estão vazias, devem ser ignoradas.
            normalized = name.upper().strip()
            if normalized == "":
                continue
            yield normalized


if __name__ == "__main__":
    with NetflixCsvReader("netflix_amazon_disney_titles.csv") as reader:
        for diretores, atores in reader:
            print(f"{diretores = }")
            print(f"{atores = }")
