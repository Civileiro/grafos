"""
Resolução de Problemas com Grafos
TDE 4 - Implementação da Codificação de Huffman
Gabriel Prost Gomes Pereira
https://youtu.be/_UabBj4wXGQ
Python 3.11

python huffman.py # imprime testes padroes

python huffman.py <FILE> # imprime testes feitos no arquivo
"""

import heapq
from collections import Counter
from dataclasses import dataclass, field
from sys import argv
from typing import Self

# bits will be stored as a string as we are just testing the theoretical compression
# if you want an actually compact representation then use a library like bitarray
HuffmanCode = str


@dataclass(order=True)
class HuffmanBinaryTree:
    frequency: int
    # the remaining fields wont be used for comparison when in a priority queue
    # https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
    token: int = field(default=0, compare=False)
    children: tuple[Self, Self] | None = field(default=None, compare=False)

    @classmethod
    def leaf(cls, token: int, frequency: int) -> Self:
        return cls(token=token, frequency=frequency)

    @classmethod
    def join(cls, left: Self, right: Self) -> Self:
        children = (left, right)
        frequency = left.frequency + right.frequency
        return cls(frequency=frequency, children=children)

    @classmethod
    def from_bytes(cls, s: bytes) -> Self:
        # (edge case) if theres nothing then just return anything
        if s == b"":
            return cls.leaf(0, 0)
        # init priority queue with token frequency leaves
        frequencies = Counter(s)
        pq = [cls.leaf(token, frequency) for token, frequency in frequencies.items()]
        heapq.heapify(pq)
        # keep joining smallest trees
        while len(pq) > 1:
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            joined = cls.join(left, right)
            heapq.heappush(pq, joined)
        # end of algorithm, all elements were joined into a single tree
        return pq[0]

    def encode(self, s: bytes) -> HuffmanCode:
        # encode tokens by mapping each one with out code map
        huffman_code = self.code_map()
        return "".join(huffman_code[token] for token in s)

    def code_map(self, walked_path="") -> dict[int, HuffmanCode]:
        # stop recursion when we're a leaf
        if self.children is None:
            # (edge case) if we have a token but path is empty, then the
            # entire encoding has just one token (or empty)
            if not walked_path:
                # but we have to set path to something so encoding isnt empty
                walked_path = "0"
            return {self.token: walked_path}

        left, right = self.children
        # recursive DFS
        left_codes = left.code_map(walked_path + "0")
        right_codes = right.code_map(walked_path + "1")
        return left_codes | right_codes

    def decode(self, encoded: HuffmanCode) -> bytes:
        # decode tokens by walking down the huffman tree bit by bit
        curr_node = self
        decoded = bytearray()
        for bit in encoded:
            # (edge case) if current is None, then the entire tree is just this one node
            if curr_node.children is None:
                decoded.append(curr_node.token)
                continue

            # walk down the tree depending on the bit
            left, right = curr_node.children
            match bit:
                case "0":
                    curr_node = left
                case "1":
                    curr_node = right
                case _:
                    raise RuntimeError("this is not a bit!!???!!?")

            # if we have not reached an end, do nothing
            if curr_node.children is not None:
                continue

            # otherwise we are at an end, append the reached token and
            # return to the beggining of the tree
            decoded.append(curr_node.token)
            curr_node = self

        return decoded


class CompressionStats:
    def __init__(self, raw: bytes, encoded: HuffmanCode):
        self.encoded_len_bits = len(encoded)
        # byte * 8 = bits
        self.raw_len_bits = len(raw) * 8
        if self.encoded_len_bits == 0:
            self.compression_rate = float("Inf")
            self.space_saving = 1.0
        else:
            self.compression_rate = self.raw_len_bits / self.encoded_len_bits
            self.space_saving = 1 - 1 / self.compression_rate


def test_encoding(input: bytes):
    huffman = HuffmanBinaryTree.from_bytes(input)

    print("\n")
    if len(input) < 1000:
        print(f"{input = !r}")
    else:
        print(f"{len(input) = } bytes")
    print(f"{huffman.code_map() = }")

    encoded = huffman.encode(input)
    stats = CompressionStats(input, encoded)

    print(f"{stats.raw_len_bits = } bits")
    print(f"{stats.encoded_len_bits = } bits")
    print(f"{stats.compression_rate = :.3}")
    print(f"{stats.space_saving = :.2%}")

    # test if decoding return original bytes
    print(f"{huffman.decode(encoded) == input = }")


def default_tests():
    test_encoding(b"")
    test_encoding(b"\0\0\0\0\0")
    test_encoding(b"aaaaaaaaaaaaaaaaa")
    test_encoding(b"Huffman gosta de batata e de estudar")
    test_encoding(
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer quis varius augue, sit amet euismod risus. Maecenas aliquam diam non dui iaculis vehicula. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin magna ex, luctus a nibh efficitur, interdum malesuada purus. Mauris id augue vitae diam viverra accumsan. Donec consectetur sapien eget ipsum tincidunt posuere. Mauris dignissim iaculis ornare. Morbi rhoncus eget mauris at suscipit. Mauris lacinia condimentum orci ac convallis. Donec accumsan lobortis lacus, non tempor quam sagittis non. Aenean lobortis erat vitae laoreet dapibus. Curabitur in enim est. Fusce tempus, mauris at scelerisque venenatis, dui felis sodales odio, eu vulputate lorem ipsum non odio. Suspendisse vitae libero quis urna porttitor facilisis. Nunc vulputate augue iaculis nibh commodo porta. "
    )


if __name__ == "__main__":
    if len(argv) == 1:
        default_tests()
        exit(0)
    file_name = argv[1]
    with open(file_name, "rb") as f:
        test_encoding(f.read())
