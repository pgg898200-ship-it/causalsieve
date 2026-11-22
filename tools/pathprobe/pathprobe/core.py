"""Exact single-task duration counterfactuals in a finish-to-start DAG."""

from collections import deque
from collections.abc import Iterable, Mapping


def _duration(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("duration must be a nonnegative integer")
    return int(value)


class DurationIndex:
    """Index immutable task durations and finish-to-start dependency edges.

    A task starts as soon as all predecessors finish. All roots start at zero
    and all sinks must finish. Queries change exactly one task on this fixed
    graph and do not mutate the index. Use one integer time unit throughout.
    """

    def __init__(
        self, durations: Mapping[str, int], edges: Iterable[tuple[str, str]]
    ) -> None:
        if not isinstance(durations, Mapping):
            raise ValueError("durations must be a mapping of task names to durations")
        names = list(durations)
        if any(not isinstance(name, str) for name in names):
            raise ValueError("task names must be strings")
        self._ids = {name: i for i, name in enumerate(names)}
        self._weights = [_duration(durations[name]) for name in names]

        n = len(names)
        source, sink = n, n + 1
        children = [[] for _ in range(n + 2)]
        indegree = [0] * (n + 2)
        if isinstance(edges, (str, bytes)):
            raise ValueError("edges must be an iterable of task pairs")
        try:
            edge_iterator = iter(edges)
        except TypeError as error:
            raise ValueError("edges must be an iterable of task pairs") from error
        for edge in edge_iterator:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError("each edge must be a pair of task names")
            parent, child = edge
            if not isinstance(parent, str) or not isinstance(child, str):
                raise ValueError("edge endpoints must be task names")
            if parent not in self._ids or child not in self._ids:
                raise ValueError(f"edge references an unknown task: {edge!r}")
            u, v = self._ids[parent], self._ids[child]
            children[u].append(v)
            indegree[v] += 1

        if not n:
            self._makespan = 0
            self._through = []
            self._bypass = []
            return

        for v in range(n):
            if not indegree[v]:
                children[source].append(v)
                indegree[v] = 1
            if not children[v]:
                children[v].append(sink)
                indegree[sink] += 1

        weights = self._weights + [0, 0]
        prefix = [-1] * (n + 2)
        prefix[source] = 0
        previous = [-1] * (n + 2)
        order = []
        ready = deque([source])
        while ready:
            u = ready.popleft()
            order.append(u)
            for v in children[u]:
                length = prefix[u] + weights[v]
                if length > prefix[v]:
                    prefix[v] = length
                    previous[v] = u
                indegree[v] -= 1
                if not indegree[v]:
                    ready.append(v)
        if len(order) != n + 2:
            raise ValueError("dependency graph contains a cycle")

        suffix = [0] * (n + 2)
        for u in reversed(order):
            if children[u]:
                suffix[u] = weights[u] + max(suffix[v] for v in children[u])
        self._makespan = prefix[sink]
        self._through = [prefix[u] + suffix[u] - weights[u] for u in range(n)]

        # Tasks outside this particular longest path already have a bypass of
        # makespan length. Only the tasks on the chosen path need a search.
        on_path = bytearray(n + 2)
        u = previous[sink]
        while u != source:
            on_path[u] = 1
            u = previous[u]
        path = []
        before = [0] * (n + 2)
        for u in order:
            before[u] = len(path)
            if on_path[u]:
                path.append(u)

        # Any path avoiding x crosses the topological position of x on an
        # edge u->v that strictly straddles it. The best such path on that
        # edge has length prefix[u] + suffix[v]. Record each edge's length
        # for the interval of chosen path vertices strictly between u and v.
        size = 1 << (len(path) - 1).bit_length()
        best = [-1] * (2 * size)
        for u in order:
            start = before[u] + on_path[u]
            for v in children[u]:
                stop = before[v]
                if start >= stop:
                    continue
                left, right = start + size, stop + size
                candidate = prefix[u] + suffix[v]
                while left < right:
                    if left & 1:
                        if candidate > best[left]:
                            best[left] = candidate
                        left += 1
                    if right & 1:
                        right -= 1
                        if candidate > best[right]:
                            best[right] = candidate
                    left //= 2
                    right //= 2
        for i in range(1, size):
            value = best[i]
            if value > best[2 * i]:
                best[2 * i] = value
            if value > best[2 * i + 1]:
                best[2 * i + 1] = value
        self._bypass = [self._makespan] * n
        for i, u in enumerate(path):
            self._bypass[u] = best[size + i]

    @property
    def makespan(self) -> int:
        """Completion time with the original durations."""
        return self._makespan

    def predict(self, task: str, new_duration: int) -> int:
        """Return completion time if this one task uses new_duration."""
        i = self._ids[task]
        changed = self._through[i] - self._weights[i] + _duration(new_duration)
        return max(self._bypass[i], changed)

    def profile(self, task: str) -> dict[str, int | None]:
        """Describe the longest paths through and around a task.

        bypass is None when every complete source-to-sink path uses the task.
        max_saving is the reduction in completion time if its duration is zero.
        """
        i = self._ids[task]
        duration = self._weights[i]
        through = self._through[i]
        bypass = self._bypass[i]
        return {
            "duration": duration,
            "through": through,
            "bypass": None if bypass < 0 else bypass,
            "max_saving": self._makespan - max(bypass, through - duration),
        }
