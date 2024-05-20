import time


class TimingPrinter:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        print(f"trabalhando em {self.task_name!r}...")
        self.t = time.perf_counter()

    def __exit__(self, *args):
        duration = time.perf_counter() - self.t
        print(f"{self.task_name!r} terminou em {duration} segundos")
