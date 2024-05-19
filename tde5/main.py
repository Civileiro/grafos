import time
from typing import Type

from grafo import Grafo, GrafoNetflixDirected, GrafoNetflixUndirected


class Part1:
    """
    1) Construção dos dois grafos solicitados (direcionado e não-direcionado). Para cada grafo,
    retorne a quantidade de vértices e arestas. Durante o processo de construção, todos os nomes devem
    ser padronizados em letras maiúsculas e sem espaços em branco no início e no final da string. Entradas
    do conjunto de dados onde o nome do diretor e/ou nome do elenco estão vazias, devem ser ignoradas.
    """

    @staticmethod
    def perform(dataset_path: str) -> tuple[Grafo, Grafo]:
        print("Criando grafo direcionado...")
        t0 = time.perf_counter()
        grafo_direcionado = GrafoNetflixDirected(dataset_path)
        t = time.perf_counter() - t0
        print(f"Grafo criado em {t} segundos")
        print(f"{grafo_direcionado.ordem = }")
        print(f"{grafo_direcionado.tamanho = }")

        print("\nCriando grafo não-direcionado...")
        t0 = time.perf_counter()
        grafo_nao_direcionado = GrafoNetflixUndirected(dataset_path)
        t = time.perf_counter() - t0
        print(f"Grafo criado em {t} segundos")
        print(f"{grafo_nao_direcionado.ordem = }")
        print(f"{grafo_nao_direcionado.tamanho = }")

        return grafo_direcionado, grafo_nao_direcionado


def perform_module(module: Type, perform_inputs):
    module_attrs = vars(module)
    if "__doc__" in module_attrs and module_attrs["__doc__"]:
        print("\n", module_attrs["__doc__"].strip(), sep="")
    return_result = None
    for function in module_attrs.values():
        if not isinstance(function, staticmethod) or not function.__name__.startswith(
            "perform"
        ):
            continue
        if function.__doc__:
            print("\n", function.__doc__.strip(), sep="")
        res = function(**perform_inputs)
        if function.__name__ == "perform":
            return_result = res
    print()
    return return_result


if __name__ == "__main__":
    gdir, gundir = perform_module(
        Part1,
        perform_inputs={"dataset_path": "./netflix_amazon_disney_titles.csv"},
    )
