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


class GrafoNetflixDirected(Grafo):

    def __init__(self, filename: str):
        super().__init__()
        with NetflixCsvReader(filename) as reader:
            for diretores, atores in reader:
                for diretor in diretores:
                    for ator in atores:
                        self.adiciona_aresta(ator, diretor)

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
    def __init__(self, filename: str):
        super().__init__()
        with NetflixCsvReader(filename) as reader:
            for _, atores in reader:
                for ator1, ator2 in product(atores, atores):
                    self.adiciona_aresta(ator1, ator2)

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
