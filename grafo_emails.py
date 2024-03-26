from collections import deque
import os
from dataclasses import dataclass
from email.parser import HeaderParser
from typing import IO, Any, Callable, Generic, Iterable, Self, Type, TypeVar
from heapq import heappop, heappush

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


T = TypeVar("T")
W = TypeVar("W")


@dataclass
class Aresta(Generic[T, W]):
    target_node: T
    weight: W


# Grafo ponderado e direcionado
class Grafo(Generic[T]):
    default_weight = 1

    def __init__(self: Self):
        self._ordem = 0
        self._tamanho = 0
        self._adj_lists: dict[T, list[Aresta[T, int]]] = {}

    def ordem(self):
        return self._ordem

    def tamanho(self):
        return self._tamanho

    def vertices(self) -> Iterable[T]:
        return self._adj_lists.keys()

    def existe_vertice(self, u: T) -> bool:
        return u in self.vertices()

    def adiciona_vertice(self, u: T) -> bool:
        if self.existe_vertice(u):
            return False
        self._adj_lists[u] = []
        self._ordem += 1
        return True

    def remove_vertice(self, u: T) -> bool:
        if not self.existe_vertice(u):
            return False
        del self._adj_lists[u]
        self._ordem -= 1
        # remover ligacoes de outros vertices
        for v in self.vertices():
            self._remove_aresta_unchecked(v, u)
        return True

    def adiciona_aresta(self, u: T, v: T, peso: int = default_weight) -> bool:
        # nao adicionar arestas duplicadas
        if self.tem_aresta(u, v):
            return False
        # criar vertices caso nao existam
        self.adiciona_vertice(u)
        self.adiciona_vertice(v)

        self._adj_lists[u].append(Aresta(target_node=v, weight=peso))
        self._tamanho += 1

        return True

    def get_aresta(self, u: T, v: T) -> Aresta | None:
        if not self.existe_vertice(u) or not self.existe_vertice(v):
            return None
        ar = next((ar for ar in self._adj_lists[u] if ar.target_node == v), None)
        return ar

    # def atualiza_aresta_direcionado(
    #     self, u: T, v: T, peso: int = default_weight
    # ) -> int | None:
    #     ar = self.get_aresta_direcionado(u, v)
    #     if ar is None:
    #         self.adiciona_aresta(u, v, peso)
    #         return None
    #     old_weight = ar.weight
    #     ar.weight = peso
    #     return old_weight
    #
    # def atualiza_aresta(self, u: T, v: T, peso: int = default_weight) -> int | None:
    #     old_weight = self.atualiza_aresta_direcionado(u, v, peso)
    #     if not self.direcionado:
    #         self.atualiza_aresta_direcionado(v, u, peso)
    #     return old_weight

    def _remove_aresta_unchecked(self, u: T, v: T) -> bool:
        ar = find_pop(self._adj_lists[u], lambda ar: ar.target_node == v)
        self._tamanho -= 1
        return ar is not None

    def remove_aresta(self, u: T, v: T) -> bool:
        if not self.tem_aresta(u, v):
            return False
        return self._remove_aresta_unchecked(u, v)

    def tem_aresta(self, u: T, v: T) -> bool:
        if not self.existe_vertice(u) or not self.existe_vertice(v):
            return False
        for ar in self._adj_lists[u]:
            if ar.target_node == v:
                return True
        return False

    def grau_entrada(self, u: T) -> int:
        if not self.existe_vertice(u):
            return 0
        grau = 0
        for v in self.vertices():
            if self.tem_aresta(v, u):
                grau += 1
        return grau

    def grau_entrada_all(self) -> dict[T, int]:
        graus = {u: 0 for u in self.vertices()}
        for u in self.vertices():
            for ar in self._adj_lists[u]:
                graus[ar.target_node] += 1
        return graus

    def grau_saida(self, u: T) -> int:
        if not self.existe_vertice(u):
            return 0
        return len(self._adj_lists[u])

    def grau(self, u: T) -> int:
        return self.grau_entrada(u) + self.grau_saida(u)

    def get_peso(self, u: T, v: T) -> int | None:
        for ar in self._adj_lists[u]:
            if ar.target_node == v:
                return ar.weight
        return None

    def retorna_adjacentes(self, u: T) -> list[T]:
        if not self.existe_vertice(u):
            return []
        return [ar.target_node for ar in self._adj_lists[u]]

    def get_max_arestas(self) -> int:
        n = len(self._adj_lists)
        return n * n

    def imprime_lista_adjacencias(self, file: IO[str] | None):
        def write(s: str):
            if file:
                file.write(s)
            else:
                print(s, end="")

        for u in self.vertices():
            write(f"{u}: ")
            first = True
            for ar in self._adj_lists[u]:
                if not first:
                    write(" -> ")
                first = False
                write(f"({ar.target_node!r}, {ar.weight})")
            write("\n")

    def conexo(self) -> bool:
        # TODO:
        return True

    def bfs(self, start: T, end: T) -> list[T]:
        """
        Busca em largura para achar uma caminho de start ate end
        retorna uma lista vazia caso um caminho nao tenha sido encotrado
        """
        if not self.existe_vertice(start) or not self.existe_vertice(end):
            return []

        seen: set[T] = set([start])
        to_visit: deque[T] = deque([start])
        backtrace: dict[T, T | None] = {start: None}

        # breadth first search
        while node := to_visit.popleft():
            if node == end:
                break
            for next in self.retorna_adjacentes(node):
                if next in seen:
                    continue
                seen.add(next)
                backtrace[next] = node
                to_visit.append(next)

        # nao foi encontrado um caminho ate o alvo
        if end not in seen:
            return []

        # reconstruir caminho encontrado
        path_node = end
        path = [end]
        while prev_node := backtrace[path_node]:
            path.append(prev_node)
            path_node = prev_node
        path.reverse()

        return path

    def nodes_in_range(self, source: T, radius: int) -> set[T]:
        """
        retorna todos os vertices dentro de uma "distancia" da fonte
        usa algoritmo de dijkstra modificado
        """
        if not self.existe_vertice(source):
            return set()
        # vertices que ja foram visitados
        visited: set[T] = set()
        # distancia minima encontrada ate cada vertice
        scores: dict[T, int] = {}
        # vertices a serem visitados
        visit_next: list[tuple[int, T]] = []

        def visit_pop():
            try:
                return heappop(visit_next)
            except IndexError:
                return None

        def visit_push(score, node):
            heappush(visit_next, (score, node))

        visit_push(0, source)
        scores[source] = 0

        while next := visit_pop():
            score, node = next
            if score > radius:
                break
            if node in visited:
                continue
            visited.add(node)
            for edge in self._adj_lists[node]:
                next = edge.target_node
                next_score = score + edge.weight
                if next in visited:
                    continue
                if next not in scores or (next in scores and next_score < scores[next]):
                    scores[next] = next_score
                    visit_push(next_score, next)

        return visited


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
    1) A partir das mensagens de e-mail da base, construa um grafo
    direcionado considerando o remetente e o(s) destinatários de cada mensagem. O
    grafo deve ser ponderado, considerando a frequência com que um remetente envia
    uma mensagem para um destinatário. O grafo também deve ser rotulado,
    considerando como rótulo o endereço de e-mail de cada usuário. Para demonstrar a
    criação do grafo, você deve salvar toda a lista de adjacências em um arquivo texto.
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
        with open(graph_file, "w") as f:
            g.imprime_lista_adjacencias(f)
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

    @staticmethod
    def perform_d(graph: Grafo[str]):
        """
        d. Os 20 indivíduos que possuem maior grau de entrada e os valores
        correspondentes (de maneira ordenada e decrescente de acordo com o grau);
        """
        graus = sorted(
            ((grau, u) for u, grau in graph.grau_entrada_all().items()), reverse=True
        )
        for grau_entrada, individuo in graus[:20]:
            print(f"{individuo}: {grau_entrada = }")


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
        if not graph.conexo():
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
        print(f"{graph.bfs('daniel.muschar@enron.com', 'james.derrick@enron.com') = }")


class Part5:
    """
    5) Implemente um método que retorne uma lista com todos os vértices que
    estão localizados até uma distância D de um vértice N, em que D é a soma dos
    pesos ao longo do caminho entre dois vértices.
    """

    @staticmethod
    def perform(graph: Grafo[str]):
        print(f"{len(graph.nodes_in_range('donna.lowry@enron.com', 3)) = }")


class Part6:
    """
    6) Implemente um método que encontre qual é o maior caminho mínimo
    entre qualquer par de vértices do grafo (i.e., diâmetro do grafo), retornando o valor e
    o caminho encontrado.
    """

    @staticmethod
    def perform(graph: Grafo[str]):
        ...


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
