from typing import Type

from grafo import Grafo, GrafoDirected, GrafoUndirected
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
            grafo_direcionado = GrafoDirected.from_file(dataset_path)
        print(f"{grafo_direcionado.ordem = }")
        print(f"{grafo_direcionado.tamanho = }")

        print()
        with TimingPrinter("criar grafo não direcionado"):
            grafo_nao_direcionado = GrafoUndirected.from_file(dataset_path)
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


class Part3:
    """
    3) Função que recebe como entrada um vértice X (por exemplo, BOB ODENKIRK) e retorna a
    Árvore Geradora Mínima da componente que contêm o vértice X, bem como o custo total da árvore
    (i.e., a soma dos pesos das arestas da árvore).
    """

    @staticmethod
    def perform(
        grafo_direcionado: Grafo,
        grafo_nao_direcionado: GrafoUndirected,
    ):
        def uma_funcao(g: GrafoUndirected, x: str) -> tuple[Grafo, int]:
            with TimingPrinter("encontrando componente"):
                [scc] = g.kosaraju_scc(target=x)

            with TimingPrinter("criando subgrafo"):
                scc_graph = g.subset(scc)
            with TimingPrinter("criando msp"):
                msp = scc_graph.kruskal_msp()
            with TimingPrinter("computando soma dos pesos"):
                total_cost = sum(msp.edges().values())
            return msp, total_cost

        target = "BOB ODENKIRK"
        # não existe arvore geradora minima para grafos direcionados?
        # with TimingPrinter(f"executando função no grafo direcionado para {target!r}"):
        #     agm, custo_total = uma_funcao(grafo_direcionado, target)
        # print(f"{agm.ordem = }")
        # print(f"{agm.tamanho = }")
        # print(f"{custo_total = }")

        with TimingPrinter(
            f"executando função no grafo não direcionado para {target!r}"
        ):
            agm, custo_total = uma_funcao(grafo_nao_direcionado, target)
        print(f"{agm.ordem = }")
        print(f"{agm.tamanho = }")
        print(f"{custo_total = }")


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
    perform_module(Part3, perform_inputs=grafos)
