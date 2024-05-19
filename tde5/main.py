from typing import Type

from grafo import Grafo, GrafoNetflixDirected, GrafoNetflixUndirected
from timing import TimingPrinter


class Part1:
    """
    1) Construção dos dois grafos solicitados (direcionado e não-direcionado). Para cada grafo,
    retorne a quantidade de vértices e arestas. Durante o processo de construção, todos os nomes devem
    ser padronizados em letras maiúsculas e sem espaços em branco no início e no final da string. Entradas
    do conjunto de dados onde o nome do diretor e/ou nome do elenco estão vazias, devem ser ignoradas.
    """

    @staticmethod
    def perform(dataset_path: str) -> tuple[Grafo, Grafo]:

        with TimingPrinter("criar grafo direcionado"):
            grafo_direcionado = GrafoNetflixDirected.from_file(dataset_path)
        print(f"{grafo_direcionado.ordem = }")
        print(f"{grafo_direcionado.tamanho = }")

        print()
        with TimingPrinter("criar grafo não direcionado"):
            grafo_nao_direcionado = GrafoNetflixUndirected.from_file(dataset_path)
        print(f"{grafo_nao_direcionado.ordem = }")
        print(f"{grafo_nao_direcionado.tamanho = }")

        return grafo_direcionado, grafo_nao_direcionado


class Part2:
    """
    2) Função para a identificação de componentes. Para o grafo direcionado, utilize essa função
    para contar a quantidade de componentes fortemente conectadas. Para o grafo não-direcionado,
    retorne a quantidade de componentes conectadas.
    """

    @staticmethod
    def perform(
        grafo_direcionado: Grafo,
        grafo_nao_direcionado: Grafo,
    ):
        with TimingPrinter("contar componentes do grafo direcionado"):
            count = grafo_direcionado.count_fully_connected_components()
        print(f"{count = }")
        print()
        with TimingPrinter("contar componentes do grafo não direcionado"):
            count = grafo_nao_direcionado.count_fully_connected_components()
        print(f"{count = }")


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
    grafos = {"grafo_direcionado": gdir, "grafo_nao_direcionado": gundir}
    perform_module(Part2, perform_inputs=grafos)
