from dataclasses import dataclass
from typing import (
    IO,
    Callable,
    Generic,
    Iterable,
    Self,
    TypeVar,
)
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


A = TypeVar("A")


def path_from_pred(pred: dict[A, A | None], dest: A) -> list[A]:
    # nao foi encontrado um caminho ate o alvo
    if dest not in pred:
        return []

    path_node = dest
    path = [dest]
    while prev_node := pred[path_node]:
        path_node = prev_node
        path.append(path_node)
    path.reverse()
    return path


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

    def adiciona_vertice(self, u: T) -> T:
        if self.existe_vertice(u):
            return u
        self._adj_lists[u] = []
        self._ordem += 1
        return u

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

    def tem_aresta(self, u: T, v: T) -> bool:
        if not self.existe_vertice(u) or not self.existe_vertice(v):
            return False
        for ar in self._adj_lists[u]:
            if ar.target_node == v:
                return True
        return False

    def tem_aresta_undirected(self, u: T, v: T) -> bool:
        return self.tem_aresta(u, v) or self.tem_aresta(v, u)

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

    def dijkstra(
        self,
        source: T,
        dest: T | None,
        cost: Callable[[int], int] = lambda weight: weight,
    ) -> tuple[dict[T, int], dict[T, T | None]]:
        """
        retorna a menor distancia de start ate todos os vertices ate encontrar end
        """
        if not self.existe_vertice(source):
            return {}, {}
        # distancia minima encontrada ate cada vertice
        distances: dict[T, int] = {}
        # vertices a serem visitados
        visit_next: list[tuple[int, T, T | None]] = []
        # predecessores
        predecessors: dict[T, T | None] = {source: None}

        def visit_pop():
            try:
                return heappop(visit_next)
            except IndexError:
                return None

        def visit_push(score, node, pred):
            heappush(visit_next, (score, node, pred))

        visit_push(0, source, None)

        while next := visit_pop():
            dist, node, pred = next
            if node in distances:
                continue
            distances[node] = dist
            predecessors[node] = pred
            if node == dest:
                break
            for edge in self._adj_lists[node]:
                next = edge.target_node
                if next in distances:
                    continue
                next_score = dist + cost(edge.weight)
                visit_push(next_score, next, node)

        return distances, predecessors

    def bfs(self, source: T, dest: T) -> tuple[list[T], set[T]]:
        """
        Busca em largura para achar uma caminho de source ate dest
        retorna uma lista vazia caso um caminho nao tenha sido encotrado
        """

        distances, pred = self.dijkstra(source, dest, lambda _: 1)
        explored = {vert for vert in distances}

        # reconstruir caminho encontrado
        path = path_from_pred(pred, dest)

        return path, explored

    def nodes_in_range(self, source: T, radius: int | None = None) -> set[T]:
        """
        retorna todos os vertices dentro de uma "distancia" da fonte
        usa algoritmo de dijkstra modificado
        """
        if not self.existe_vertice(source):
            return set()
        # vertices que ja foram visitados
        visited: set[T] = set()
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

        while next := visit_pop():
            score, node = next
            if radius is not None and score > radius:
                break
            if node in visited:
                continue
            for edge in self._adj_lists[node]:
                next = edge.target_node
                next_score = score + edge.weight
                if next in visited:
                    continue
                visit_push(next_score, next)
            visited.add(node)

        return visited

    def diametro(self) -> tuple[int, list[T]]:
        """
        calcula o maior caminho minimo entre quaisquer 2 vertices do grafo
        """
        max_path_len = 0
        max_dest = None
        max_pred: dict[T, T | None] = {}
        # calcular todos os caminhos minimos entre todos os vertices
        for v in self.vertices():
            smallest_paths_from_v, pred = self.dijkstra(v, dest=None)
            v_max_dest = max(
                smallest_paths_from_v, key=smallest_paths_from_v.__getitem__
            )
            v_max_path = smallest_paths_from_v[v_max_dest]
            # salver possivel maior caminho minimo
            if v_max_path > max_path_len:
                max_path_len = v_max_path
                max_dest = v_max_dest
                max_pred = pred

        if max_dest is None:
            return 0, []
        # calcular caminho da maior caminhada minima
        path = path_from_pred(max_pred, max_dest)

        return max_path_len, path

    def is_cyclic(self) -> bool:
        # tarjans algorithm
        to_visit = 0
        visiting = 1
        visited = 2
        statuses = {vert: to_visit for vert in self.vertices()}
        pred: dict[T, T | None] = {}
        while True:
            start_node = next(
                (k for k in statuses.keys() if statuses[k] == to_visit), None
            )
            if start_node is None:
                return False
            pred[start_node] = None
            curr_node: T | None = start_node
            while curr_node:
                # print(f"{curr_node = }")
                node = curr_node
                statuses[node] = visiting
                is_leaf = True
                for ar in self._adj_lists[node]:
                    if statuses[ar.target_node] == visiting:
                        return True
                    if (
                        ar.target_node == pred[node]
                        or statuses[ar.target_node] == visited
                    ):
                        continue
                    is_leaf = False
                    curr_node = ar.target_node
                    pred[ar.target_node] = node
                    break
                if is_leaf:
                    curr_node = pred[node]
                    print(node)
                    statuses[node] = visited


g: Grafo[str] = Grafo()
g.adiciona_aresta("A", "F")
g.adiciona_aresta("A", "B")
g.adiciona_aresta("B", "H")
g.adiciona_aresta("D", "C")
g.adiciona_aresta("D", "E")
g.adiciona_aresta("D", "H")
g.adiciona_aresta("E", "I")
g.adiciona_aresta("G", "A")
g.adiciona_aresta("G", "B")
g.adiciona_aresta("G", "C")
g.adiciona_aresta("J", "E")
g.adiciona_aresta("I", "C")
print(f"{g.is_cyclic() = }")
