from abc import ABC, abstractmethod
from itertools import product
from typing import Iterable, Self

from parse_netflix import NetflixCsvReader


class Grafo(ABC):
    def __init__(self: Self):
        self._adj: dict[str, dict[str, int]] = {}
        self._tamanho = 0

    @property
    def ordem(self: Self) -> int:
        return len(self._adj)

    @property
    def tamanho(self: Self) -> int:
        return self._tamanho

    def vertices(self) -> Iterable[str]:
        return self._adj.keys()

    def existe_vertice(self, u: str) -> bool:
        return u in self._adj

    def adiciona_vertice(self, u: str) -> str:
        if self.existe_vertice(u):
            return u
        self._adj[u] = {}
        return u

    @abstractmethod
    def adiciona_aresta(self, u: str, v: str) -> bool: ...

    def get_peso(self, u: str, v: str) -> int | None:
        if not self.existe_vertice(u) or not self.existe_vertice(v):
            return None
        return self._adj[u].get(v, None)

    def tem_aresta(self, u: str, v: str) -> bool:
        if not self.existe_vertice(u):
            return False
        return v in self._adj[u]

    def neighbors(self, u: str) -> Iterable[str]:
        return self._adj[u].keys()

    def count_fully_connected_components(self: Self) -> int:
        return len(self.kosaraju_scc())

    def kosaraju_scc(self) -> list[set[str]]:
        stack: list[str] = []
        visited: set[str] = set()

        # 1. Run DFS on G, storing the [visiting/finished] times of every node
        for start in self.vertices():
            if start in visited:
                continue
            self.dfs_visit(start, visited, stack)

        # 2. Compute Gt (the transpose graph)
        transposed = self.transposed()

        # 3. Run DFS on Gt considering the decreasing order of finished time of G
        visited.clear()
        sccs = []
        while stack:
            start = stack.pop()
            if start in visited:
                continue
            scc = transposed.collect_connected(start, visited)
            sccs.append(scc)

        return sccs

    def dfs_visit(self, start: str, visited: set[str], stack: list[str]):
        to_visit = [start]
        pred: dict[str, str | None] = {start: None}
        while to_visit:
            node = to_visit.pop()
            if node in visited:
                continue
            visited.add(node)
            has_neigh = False
            for next in self.neighbors(node):
                if next in visited:
                    continue
                has_neigh = True
                to_visit.append(next)
                pred[next] = node
            if has_neigh:
                continue
            stack.append(node)
            curr: str | None = pred[node]
            while curr:
                if any(n not in visited for n in self.neighbors(curr)):
                    break
                stack.append(curr)
                curr = pred[curr]

    def transposed(self) -> Self:
        transposed = self.__class__()
        for node in self.vertices():
            for neighbor in self.neighbors(node):
                transposed.adiciona_aresta(neighbor, node)
        return transposed

    def collect_connected(self, start: str, visited: set[str]) -> set[str]:
        scc: set[str] = set()
        to_visit = {start}
        while to_visit:
            node = to_visit.pop()
            if node in visited:
                continue
            scc.add(node)
            visited.add(node)
            for neighbor in self.neighbors(node):
                if neighbor in visited:
                    continue
                to_visit.add(neighbor)

        return scc


class GrafoNetflixDirected(Grafo):

    @classmethod
    def from_file(cls, filename: str) -> Self:
        self = cls()
        with NetflixCsvReader(filename) as reader:
            for diretores, atores in reader:
                for diretor in diretores:
                    for ator in atores:
                        self.adiciona_aresta(ator, diretor)
        return self

    def adiciona_aresta(self, u: str, v: str) -> bool:
        # nao adicionar arestas duplicadas
        # apenas incrementar peso
        if self.tem_aresta(u, v):
            self._adj[u][v] += 1
            return False
        # criar vertices caso nao existam
        self.adiciona_vertice(u)
        self.adiciona_vertice(v)

        self._adj[u][v] = 1
        self._tamanho += 1

        return True


class GrafoNetflixUndirected(Grafo):

    @classmethod
    def from_file(cls, filename: str) -> Self:
        self = cls()
        with NetflixCsvReader(filename) as reader:
            for _, atores in reader:
                for ator1, ator2 in product(atores, atores):
                    self.adiciona_aresta(ator1, ator2)
        return self

    def adiciona_aresta(self, u: str, v: str) -> bool:
        if u == v:
            return False
        # nao adicionar arestas duplicadas
        # apenas incrementar peso
        if self.tem_aresta(u, v):
            self._adj[u][v] += 1
            self._adj[v][u] += 1
            return False
        # criar vertices caso nao existam
        self.adiciona_vertice(u)
        self.adiciona_vertice(v)

        self._adj[u][v] = 1
        self._adj[v][u] = 1
        self._tamanho += 1

        return True

    def count_fully_connected_components(self: Self) -> int:
        visited: set[str] = set()
        num_components = 0
        for start in self.vertices():
            if start in visited:
                continue
            num_components += 1
            to_visit = {start}
            while to_visit:
                node = to_visit.pop()
                visited.add(node)
                for neighbor in self.neighbors(node):
                    if neighbor in visited:
                        continue
                    to_visit.add(neighbor)

        return num_components
