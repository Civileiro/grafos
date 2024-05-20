import os
from email.parser import HeaderParser
from typing import (
    Any,
    Callable,
    Type,
    TypeVar,
)
from grafo import Grafo

P = TypeVar("P")


def find_pop(lst: list[P], pred: Callable[[P], bool]) -> P | None:
    """
    Acha e remove o primeiro elemento da lista
    para o qual o predicado eh verdade
    """
    index = next((i for i, t in enumerate(lst) if pred(t)), None)
    if index is None:
        return None
    return lst.pop(index)


class Email:
    header_parser = HeaderParser()

    @classmethod
    def parse_mail_list(cls, field: str | None) -> list[str]:
        if not field:
            return []
        return [mail.strip() for mail in field.split(",")]

    @classmethod
    def parse_email_from_to(cls, email_path: str) -> tuple[list[str], list[str]]:
        with open(email_path, "r") as f:
            # f.read()
            email = cls.header_parser.parse(f)

        from_ = cls.parse_mail_list(email.get("From", None))
        to = cls.parse_mail_list(email.get("To", None))

        return from_, to


class FileTree:
    @classmethod
    def map_files_rec(cls, path: str, func: Callable[[str], Any]):
        for basepath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(basepath, filename)
                func(filepath)


class Part1:
    """
    1) Construção dos dois grafos solicitados (direcionado e não-direcionado). Para cada grafo,
    retorne a quantidade de vértices e arestas. Durante o processo de construção, todos os nomes devem
    ser padronizados em letras maiúsculas e sem espaços em branco no início e no final da string. Entradas
    do conjunto de dados onde o nome do diretor e/ou nome do elenco estão vazias, devem ser ignoradas.
    """

    @staticmethod
    def perform(dataset_path: str) -> Grafo[str]:
        g: Grafo[str] = Grafo()

        def add_email_to_graph(email_path: str):
            from_, to = Email.parse_email_from_to(email_path)
            for from_email in from_:

                for to_email in to:

                    ar = g.get_aresta(from_email, to_email)
                    if not ar:
                        g.adiciona_aresta(from_email, to_email, 1)
                        continue
                    ar.weight += 1

        FileTree.map_files_rec(dataset_path, add_email_to_graph)

        graph_file = "grafo_emails.txt"
        # with open(graph_file, "w") as f:
        # g.imprime_lista_adjacencias(f)
        print(f"Grafo salvo em {graph_file!r}")

        return g


class Part2:
    """
    2) Implemente métodos/funções para extrair as seguintes informações
    gerais do grafo construído:
    """

    @staticmethod
    def perform_a(graph: Grafo[str]):
        """
        a. O número de vértices do grafo (ordem)
        """
        print(f"{graph.ordem() = }")

    @staticmethod
    def perform_b(graph: Grafo[str]):
        """
        b. O número de arestas do grafo (tamanho);
        """
        print(f"{graph.tamanho() = }")

    @staticmethod
    def perform_c(graph: Grafo[str]):
        """
        c. Os 20 indivíduos que possuem maior grau de saída e os valores
        correspondentes (de maneira ordenada e decrescente de acordo com o grau);
        """
        graus = sorted(
            ((graph.grau_saida(u), u) for u in graph.vertices()), reverse=True
        )
        for grau_saida, individuo in graus[:20]:
            print(f"{individuo}: {grau_saida = }")

    # @staticmethod
    # def perform_d(graph: Grafo[str]):
    #     """
    #     d. Os 20 indivíduos que possuem maior grau de entrada e os valores
    #     correspondentes (de maneira ordenada e decrescente de acordo com o grau);
    #     """
    #     graus = sorted(
    #         ((graph.grau_entrada(u), u) for u in graph.vertices()), reverse=True
    #     )
    #     for grau_entrada, individuo in graus[:20]:
    #         print(f"{individuo}: {grau_entrada = }")


class Part3:
    """
    3) Implemente uma função que verifica se o grafo é Euleriano, retornando
    true ou false. Caso a resposta seja false, a sua função deve informar ao usuário quais
    condições não foram satisfeitas.
    """

    @staticmethod
    def euleriano(graph: Grafo[str]) -> bool:
        # um grafo direcionado é Euleriano se ele for conexo e
        # todos os vertices possuirem grau de saida e entrada iguais
        for vertice in graph.vertices():
            if graph.grau_entrada(vertice) == graph.grau_saida(vertice):
                continue
            print(
                f"O {vertice = !r} possui graus de entrada e saida diferentes "
                f"({graph.grau_entrada(vertice) = }, {graph.grau_saida(vertice) = })"
            )
            return False
        # se a condição anterior passou então é suficiente checar apenas
        # se é possivel chegar em todos os vertices a partir de um vertice
        # qualquer para que o grafo seja conexo
        v = next((v for v in graph.vertices()), None)
        if v is None:
            # o grafo é vazio
            return True
        if graph.nodes_in_range(v) != graph.ordem():
            print("O grafo não é conexo")
            return False

        return True

    @staticmethod
    def perform(graph: Grafo[str]):
        print(f"{Part3.euleriano(graph) = }")


class Part4:
    """
    4) Implemente um método que percorre o grafo em LARGURA e verifica
    se um indivíduo X pode alcançar um indivíduo Y retornando a sequência de vértices
    explorados durante a etapa de busca (vértices visitados).
    """

    @staticmethod
    def perform(graph: Grafo[str]):
        print("graph.bfs('daniel.muschar@enron.com', 'james.derrick@enron.com')")
        source = graph.adiciona_vertice("daniel.muschar@enron.com")
        dest = graph.adiciona_vertice("james.derrick@enron.com")
        path, explored = graph.bfs(source, dest)
        print(f"{path = }")
        print(f"{len(explored) = }")


class Part5:
    """
    5) Implemente um método que retorne uma lista com todos os vértices que
    estão localizados até uma distância D de um vértice N, em que D é a soma dos
    pesos ao longo do caminho entre dois vértices.
    """

    @staticmethod
    def perform(graph: Grafo[str]):
        node = graph.adiciona_vertice("donna.lowry@enron.com")
        print(f"{len(graph.nodes_in_range(node, 5)) = }")


class Part6:
    """
    6) Implemente um método que encontre qual é o maior caminho mínimo
    entre qualquer par de vértices do grafo (i.e., diâmetro do grafo), retornando o valor e
    o caminho encontrado.
    """

    @staticmethod
    def perform(graph: Grafo[str]):
        print(f"{graph.diametro() = }")


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
    g = perform_module(
        Part1,
        perform_inputs={"dataset_path": "./amostra"},
    )
    perform_module(Part2, perform_inputs={"graph": g})
    perform_module(Part3, perform_inputs={"graph": g})
    perform_module(Part4, perform_inputs={"graph": g})
    perform_module(Part5, perform_inputs={"graph": g})
    perform_module(Part6, perform_inputs={"graph": g})
