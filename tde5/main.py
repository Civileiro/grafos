from heapq import nlargest
from typing import Type

import numpy as np
from matplotlib import pyplot as plt

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


class Part4:
    """
    4) Função que calcula a Centralidade de Grau (Degree Centrality) de um vértice. Utilize essa
    função para gerar gráficos de histograma com a distribuição de graus dos vértices de cada grafo.
    """

    @staticmethod
    def plot_graph_graus(graus: dict[str, int], savefile: str):
        graus_array = np.fromiter(graus.values(), dtype=np.int32)
        media = float(np.mean(graus_array))

        plt.figure()
        plt.title("Distribuição de graus")
        plt.hist(graus_array, color="skyblue", edgecolor="black", bins=20)
        plt.axvline(x=media, color="red", linestyle="dashed", label=f"Média: {media}")
        plt.xlabel("Grau")
        plt.ylabel("Frequência")
        plt.legend()
        plt.savefig(savefile, dpi=300)

    @staticmethod
    def perform(
        grafo_direcionado: Grafo,
        grafo_nao_direcionado: Grafo,
    ):
        with TimingPrinter("calcular graus do grafo direcionado"):
            graus = grafo_direcionado.graus()
        FILE = "histograma_graus_dir.png"
        with TimingPrinter("gerando grafico dos graus do grafo direcionado"):
            Part4.plot_graph_graus(graus, FILE)
        print(f"grafico salvo em {FILE}")

        with TimingPrinter("calcular graus do grafo não direcionado"):
            graus = grafo_nao_direcionado.graus()
        FILE = "histograma_graus_não_dir.png"
        with TimingPrinter("gerando grafico dos graus do grafo não direcionado"):
            Part4.plot_graph_graus(graus, FILE)
        print(f"grafico salvo em {FILE}")


def plot_graph_top_vertex_values(graus: dict[str, int], savefile: str):
    x = nlargest(10, graus, key=graus.__getitem__)
    y = [graus[node] for node in x]

    plt.figure()
    plt.title("Top 10 graus")
    plt.bar(x=x, height=y)
    plt.xlabel("Nome vértice")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Grau")
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    print(f"grafico salvo em {savefile}")


class Part5:
    """
    5) Apresente um gráfico de barras que mostra os top-10 vértices com os maiores valores de
    Degree Centrality.
    """

    @staticmethod
    def perform(
        grafo_direcionado: Grafo,
        grafo_nao_direcionado: Grafo,
    ):
        with TimingPrinter("calculando graus do grafo direcionado"):
            graus = grafo_direcionado.graus()
        with TimingPrinter("gerando grafico dos top 10 graus do grafo direcionado"):
            plot_graph_top_vertex_values(graus, "top10_graus_dir.png")

        with TimingPrinter("calculando graus do grafo não direcionado"):
            graus = grafo_nao_direcionado.graus()
        with TimingPrinter("gerando grafico dos top 10 graus do grafo não direcionado"):
            plot_graph_top_vertex_values(graus, "top10_graus_não_dir.png")


class Part6:
    """
    6) Função que calcula a Centralidade de Intermediação (Betweenness Centrality) de um vértice.
    Utilize essa função para gerar um gráfico de barras que mostra os top-10 vértices com os maiores
    valores.
    """

    @staticmethod
    def perform(
        grafo_direcionado: Grafo,
        grafo_nao_direcionado: Grafo,
    ):
        with TimingPrinter(
            "calculando centralidades de intermediação do grafo direcionado"
        ):
            centralities = grafo_direcionado.betweenness_centralities()
        with TimingPrinter("gerando grafico dos top 10 graus do grafo direcionado"):
            plot_graph_top_vertex_values(centralities, "top10_graus_dir.png")

        with TimingPrinter(
            "calculando centralidades de intermediação do grafo não direcionado"
        ):
            centralities = grafo_nao_direcionado.betweenness_centralities()
        with TimingPrinter("gerando grafico dos top 10 graus do grafo não direcionado"):
            plot_graph_top_vertex_values(centralities, "top10_graus_não_dir.png")


class Part7:
    """
    7) Função que calcula a Centralidade de Proximidade (Closeness Centrality) de um vértice.
    Utilize essa função para gerar um gráfico de barras que mostra os top-10 vértices com os maiores
    valores.
    """

    @staticmethod
    def perform(
        grafo_direcionado: Grafo,
        grafo_nao_direcionado: Grafo,
    ):
        with TimingPrinter(
            "calculando centralidades de proximidade do grafo direcionado"
        ):
            centralities = grafo_direcionado.closeness_centralities()
        with TimingPrinter("gerando grafico dos top 10 graus do grafo direcionado"):
            plot_graph_top_vertex_values(centralities, "top10_graus_dir.png")

        with TimingPrinter(
            "calculando centralidades de proximidade do grafo não direcionado"
        ):
            centralities = grafo_nao_direcionado.closeness_centralities()
        with TimingPrinter("gerando grafico dos top 10 graus do grafo não direcionado"):
            plot_graph_top_vertex_values(centralities, "top10_graus_não_dir.png")


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
    try:
        gdir, gundir = perform_module(
            Part1,
            perform_inputs={"dataset_path": "./netflix_amazon_disney_titles.csv"},
        )
        grafos = {"grafo_direcionado": gdir, "grafo_nao_direcionado": gundir}
        perform_module(Part2, perform_inputs=grafos)
        perform_module(Part3, perform_inputs=grafos)
        perform_module(Part4, perform_inputs=grafos)
        perform_module(Part5, perform_inputs=grafos)
        perform_module(Part6, perform_inputs=grafos)
        perform_module(Part7, perform_inputs=grafos)
    except NotImplementedError as e:
        print(f"função {e} não foi implementada ainda... parando programa")
