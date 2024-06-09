from abc import ABC, abstractmethod
from collections import defaultdict
from heapq import heapify, heappop, heappush
import heapq
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
    def adiciona_aresta(self, u: str, v: str, peso: int | None = None) -> bool: ...

    def get_peso(self, u: str, v: str) -> int | None:
        if not self.existe_vertice(u) or not self.existe_vertice(v):
            return None
        return self._adj[u].get(v, None)

    def get_peso_unsafe(self, u: str, v: str) -> int:
        return self._adj[u][v]

    def tem_aresta(self, u: str, v: str) -> bool:
        if not self.existe_vertice(u):
            return False
        return v in self._adj[u]

    def neighbors(self, u: str) -> Iterable[str]:
        return self._adj[u].keys()

    @abstractmethod
    def graus(self, subset: set[str] | None = None) -> dict[str, int]: ...

    def count_fully_connected_components(self: Self) -> int:
        return len(self.kosaraju_scc())

    def kosaraju_scc(self, target: str | None = None) -> list[set[str]]:
        stack: list[str] = []
        visited: set[str] = set()

        # 1. Run DFS on G, storing the [visiting/finished] times of every node
        if target:
            self.dfs_visit(target, visited, stack)
        else:
            for start in self.vertices():
                if start in visited:
                    continue
                self.dfs_visit(start, visited, stack)

        # 2. Compute Gt (the transpose graph)
        transposed = self.transposed(subset=visited if target else None)

        # 3. Run DFS on Gt considering the decreasing order of finished time of G
        visited.clear()
        sccs = []
        while stack:
            start = stack.pop()
            if start in visited:
                continue
            scc = transposed.collect_connected(start, visited)
            if target and target in scc:
                sccs = [scc]
                break
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

    def transposed(self, subset: set[str] | None = None) -> Self:
        transposed = self.__class__()
        nodes = subset if subset is not None else self.vertices()
        for node in nodes:
            transposed.adiciona_vertice(node)
            for neighbor in self.neighbors(node):
                if subset is not None and neighbor not in subset:
                    continue
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

    def subset(self, vertices: set[str]) -> Self:
        subset = self.__class__()
        for node in vertices:
            subset.adiciona_vertice(node)
            for neighbor in self.neighbors(node):
                if neighbor not in vertices:
                    continue
                peso = self.get_peso(node, neighbor)
                subset.adiciona_aresta(node, neighbor, peso=peso)
        return subset

    @abstractmethod
    def betweenness_centralities(self) -> dict[str, float]: ...

    def _betweenness_centralities(self, directed) -> dict[str, float]:
        CB: dict[str, float] = defaultdict(float)

        for s in self.vertices():
            delta: dict[str, float] = defaultdict(float)
            prev: dict[str, list[str]] = defaultdict(list)
            sigma = defaultdict(int)
            dist: dict[str, int] = defaultdict(int)

            sigma[s] = 1
            dist[s] = 0

            Q = [(0, s)]
            S = []

            while Q:
                curr_cost, u = heapq.heappop(Q)
                S.append(u)

                for v in self.neighbors(u):
                    weight = self.get_peso_unsafe(u, v)
                    cost = curr_cost + weight
                    if v not in dist or dist[v] > cost:
                        dist[v] = cost
                        heapq.heappush(Q, (cost, v))
                        prev[v] = []

                    if dist[v] == cost:
                        sigma[v] += sigma[u]
                        prev[v].append(u)

            while S:
                v = S.pop()

                for u in prev[v]:
                    delta[u] += sigma[u] / sigma[v] * (1 + delta[v])

                    if u != s:
                        CB[v] += delta[v]

        if not directed:
            for v in CB:
                CB[v] /= 2

        return CB

    def dijkstra(self, start: str) -> dict[str, int]:
        distances = {start: 0}
        pq = [(0, start)]

        while pq:
            curr_cost, node = heapq.heappop(pq)
            if curr_cost > distances[node]:
                continue

            for next in self.neighbors(node):
                weight = self.get_peso_unsafe(node, next)
                distance = curr_cost + weight

                if next not in distances or distance < distances[next]:
                    distances[next] = distance
                    heapq.heappush(pq, (distance, next))

        return distances

    def closeness_centralities(self) -> dict[str, float]:
        res = {}

        for u in self.vertices():
            distances = self.dijkstra(u)
            total = sum(1 / d for d in distances.values() if d != 0)
            reachable = len(distances)

            res[u] = (reachable - 1) * total

        return res


class GrafoDirected(Grafo):

    @classmethod
    def from_file(cls, filename: str) -> Self:
        self = cls()
        with NetflixCsvReader(filename) as reader:
            for diretores, atores in reader:
                for ator in atores:
                    self.adiciona_vertice(ator)
                for diretor in diretores:
                    self.adiciona_vertice(diretor)
                    for ator in atores:
                        self.adiciona_aresta(ator, diretor)
        return self

    def adiciona_aresta(self, u: str, v: str, peso: int | None = None) -> bool:
        # nao adicionar arestas duplicadas
        # apenas incrementar peso
        if self.tem_aresta(u, v):
            if peso is None:
                self._adj[u][v] += 1
            else:
                self._adj[u][v] = peso
            return False
        # criar vertices caso nao existam
        self.adiciona_vertice(u)
        self.adiciona_vertice(v)

        self._adj[u][v] = 1 if peso is None else peso
        self._tamanho += 1

        return True

    def graus(self, subset: set[str] | None = None) -> dict[str, int]:
        graus: dict[str, int] = defaultdict(int)
        for node in self.vertices():
            for neighbor in self.neighbors(node):
                if subset is None or node in subset:
                    graus[node] += 1
                if subset is None or neighbor in subset:
                    graus[neighbor] += 1
        return graus

    def betweenness_centralities(self) -> dict[str, float]:
        return super()._betweenness_centralities(directed=True)


class GrafoUndirected(Grafo):

    @classmethod
    def from_file(cls, filename: str) -> Self:
        self = cls()
        with NetflixCsvReader(filename) as reader:
            for _, atores in reader:
                for ator1, ator2 in product(atores, atores):
                    if ator1 == ator2:
                        self.adiciona_vertice(ator1)
                    else:
                        self.adiciona_aresta(ator1, ator2)
        return self

    def adiciona_aresta(self, u: str, v: str, peso: int | None = None) -> bool:
        # nao adicionar arestas duplicadas
        # apenas incrementar peso
        if self.tem_aresta(u, v):
            if peso is None:
                self._adj[u][v] += 1
                self._adj[v][u] += 1
            else:
                self._adj[u][v] = peso
                self._adj[v][u] = peso
            return False
        # criar vertices caso nao existam
        self.adiciona_vertice(u)
        self.adiciona_vertice(v)

        self._adj[u][v] = 1 if peso is None else peso
        self._adj[v][u] = 1 if peso is None else peso
        self._tamanho += 1

        return True

    def graus(self, subset: set[str] | None = None) -> dict[str, int]:
        nodes = subset if subset is not None else self.vertices()
        return {node: sum(1 for _ in self.neighbors(node)) for node in nodes}

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

    def edges(self) -> dict[tuple[str, str], int]:
        edges = {}
        for node in self.vertices():
            for neighbor in self.neighbors(node):
                k1, k2 = sorted((node, neighbor))
                k = k1, k2
                if k in edges:
                    continue
                edges[k] = self.get_peso_unsafe(node, neighbor)

        return edges

    def kruskal_msp(self) -> Self:
        msp = self.__class__()
        start = next(iter(self.vertices()), None)
        if start is None:
            return msp
        msp.adiciona_vertice(start)
        edges = [
            (self.get_peso_unsafe(start, neighbor), start, neighbor)
            for neighbor in self.neighbors(start)
        ]
        heapify(edges)
        while msp.ordem != self.ordem:
            peso, node_internal, node_external = heappop(edges)
            if msp.existe_vertice(node_external):
                continue
            msp.adiciona_aresta(node_internal, node_external, peso=peso)
            for neighbor in self.neighbors(node_external):
                if msp.existe_vertice(neighbor):
                    continue
                peso = self.get_peso_unsafe(node_external, neighbor)
                heappush(edges, (peso, node_external, neighbor))
        return msp

    def betweenness_centralities(self) -> dict[str, float]:
        return super()._betweenness_centralities(directed=False)
