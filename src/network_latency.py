# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
"""Event-driven latency model for cloud-edge-end model transmissions."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, Mapping, Sequence, Tuple


Path = Sequence[int]
Link = Tuple[int, int]


@dataclass(frozen=True)
class _Flow:
    kind: str
    key: Hashable
    path: Tuple[int, ...]


def simulate_model_forwarding_latency(
    *,
    group_access_pairs: Iterable[Tuple[int, int]],
    source_owner: Mapping[int, int] = None,
    path_finder: Callable[[int, int], Path],
    link_rates,
    model_size: float,
    cloud_sync: bool = False,
) -> Dict[str, object]:
    """Schedule only the shared group-model forwarding phase with FCFS.

    One model flow is created for each distinct ``(group, physical ES)``
    pair.  This phase is independent of the current aggregation owner under
    the protocol used here: the previous owner sends the group model directly
    to every physical access ES that currently serves that group's clients.
    """
    pairs = sorted(
        {(int(group_id), int(physical_es))
         for group_id, physical_es in group_access_pairs}
    )
    if model_size < 0:
        raise ValueError("model_size must be non-negative")
    if not cloud_sync and source_owner is None:
        raise ValueError("source_owner is required outside a cloud-sync round")

    theta: Dict[Tuple[int, int], float] = {}
    link_available: Dict[Link, float] = {}
    trace = []
    events = []
    sequence = 0

    def rate_for(link: Link) -> float:
        u, v = link
        try:
            rate = float(link_rates[u][v])
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"missing rate for ES link {link}") from exc
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"rate for ES link {link} must be positive, got {rate}")
        return rate

    def enqueue(flow: _Flow, arrival: float, hop_index: int = 0):
        nonlocal sequence
        heapq.heappush(
            events,
            (float(arrival), tuple(flow.key), sequence, flow, hop_index),
        )
        sequence += 1

    for group_id, physical_es in pairs:
        if cloud_sync:
            theta[(group_id, physical_es)] = 0.0
            continue
        if group_id not in source_owner:
            raise KeyError(f"missing previous owner for group {group_id}")
        model_source = int(source_owner[group_id])
        if model_source == physical_es:
            theta[(group_id, physical_es)] = 0.0
            continue
        path = tuple(int(node) for node in path_finder(model_source, physical_es))
        if not path or path[0] != model_source or path[-1] != physical_es:
            raise ValueError(
                f"invalid path for {model_source}->{physical_es}: {path}"
            )
        flow = _Flow(
            kind="down",
            key=(group_id, model_source, physical_es),
            path=path,
        )
        enqueue(flow, 0.0)

    while events:
        arrival, _, _, flow, hop_index = heapq.heappop(events)
        link = (flow.path[hop_index], flow.path[hop_index + 1])
        start = max(arrival, link_available.get(link, 0.0))
        finish = start + 8.0 * model_size / rate_for(link)
        link_available[link] = finish
        trace.append(
            {
                "kind": "down",
                "flow": flow.key,
                "link": link,
                "arrival": arrival,
                "start": start,
                "finish": finish,
            }
        )
        if hop_index + 1 < len(flow.path) - 1:
            enqueue(flow, finish, hop_index + 1)
            continue
        group_id, _, physical_es = flow.key
        theta[(int(group_id), int(physical_es))] = finish

    return {
        "theta": theta,
        "trace": trace,
    }


def simulate_round_latency(
    *,
    client_ids: Iterable[int],
    physical_owner: Mapping[int, int],
    logical_owner: Mapping[int, int],
    download_latency: Mapping[int, float],
    compute_latency: Mapping[int, float],
    upload_latency: Mapping[int, float],
    path_finder: Callable[[int, int], Path],
    link_rates,
    model_size: float,
    cloud_sync: bool = False,
    client_group: Mapping[int, int] = None,
    source_owner: Mapping[int, int] = None,
) -> Dict[str, object]:
    """Simulate one round with per-directed-link FCFS queues.

    Link rates are specified in Mbps and model_size is specified in MB, so a
    one-hop service time is 8*model_size/rate.  A group's model is downloaded
    from the ES that aggregated it in the
    previous round.  Its trained client updates are sent to the ES selected to
    aggregate the current round.  Model forwarding is shared by clients of the
    same group attached to the same physical ES.  All model and update flows
    share the same directed-link FCFS queues.
    """

    clients = sorted(int(client_id) for client_id in client_ids)
    if model_size < 0:
        raise ValueError("model_size must be non-negative")

    theta: Dict[Tuple[int, int, int], float] = {}
    q: Dict[int, float] = {}
    a: Dict[int, float] = {}
    completion: Dict[int, float] = {}
    down_clients: Dict[Tuple[int, int, int], list[int]] = {}
    group_logical_owner: Dict[int, int] = {}
    link_available: Dict[Link, float] = {}
    trace = []

    # (arrival, kind priority, stable key, sequence, flow, hop index)
    events = []
    sequence = 0

    def stable_key(flow: _Flow):
        if flow.kind == "down":
            return tuple(int(value) for value in flow.key)
        return (int(flow.key),)

    def enqueue(flow: _Flow, arrival: float, hop_index: int = 0):
        nonlocal sequence
        priority = {"down": 0, "update": 1}[flow.kind]
        heapq.heappush(
            events,
            (
                float(arrival),
                priority,
                stable_key(flow),
                sequence,
                flow,
                hop_index,
            ),
        )
        sequence += 1

    def make_flow(kind: str, key: Hashable, src: int, dst: int) -> _Flow:
        path = tuple(int(node) for node in path_finder(int(src), int(dst)))
        if not path or path[0] != src or path[-1] != dst:
            raise ValueError(f"invalid path for {src}->{dst}: {path}")
        return _Flow(kind=kind, key=key, path=path)

    def rate_for(link: Link) -> float:
        u, v = link
        try:
            rate = float(link_rates[u][v])
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"missing rate for ES link {link}") from exc
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"rate for ES link {link} must be positive, got {rate}")
        return rate

    def schedule_client(client_id: int, model_ready: float):
        cid = int(client_id)
        q[cid] = float(model_ready) + float(download_latency[cid])
        a[cid] = q[cid] + float(compute_latency[cid]) + float(upload_latency[cid])
        src = int(physical_owner[cid])
        dst = int(logical_owner[cid])
        if src == dst:
            completion[cid] = a[cid]
            return
        flow = make_flow("update", cid, src, dst)
        enqueue(flow, a[cid])

    # A down-forwarding flow is identified by (group, previous owner,
    # physical access ES).  Different group models are never merged.
    for cid in clients:
        if cid not in physical_owner or cid not in logical_owner:
            raise KeyError(f"missing physical/logical ES assignment for client {cid}")
        gid = int(client_group[cid]) if client_group is not None else int(logical_owner[cid])
        current_owner = int(logical_owner[cid])
        old_logical = group_logical_owner.setdefault(gid, current_owner)
        if old_logical != current_owner:
            raise ValueError(f"group {gid} has multiple logical owners")
        model_source = (
            int(source_owner[gid])
            if source_owner is not None and gid in source_owner
            else current_owner
        )
        key = (gid, model_source, int(physical_owner[cid]))
        down_clients.setdefault(key, []).append(cid)

    for key in sorted(down_clients):
        _, model_source, physical_es = key
        if cloud_sync or model_source == physical_es:
            theta[key] = 0.0
            for cid in down_clients[key]:
                schedule_client(cid, 0.0)
            continue
        flow = make_flow("down", key, model_source, physical_es)
        enqueue(flow, 0.0)

    while events:
        arrival, _, _, _, flow, hop_index = heapq.heappop(events)
        link = (flow.path[hop_index], flow.path[hop_index + 1])
        start = max(arrival, link_available.get(link, 0.0))
        # model_size is MB while link_rates are Mbps.
        finish = start + 8.0 * model_size / rate_for(link)
        link_available[link] = finish
        trace.append(
            {
                "kind": flow.kind,
                "flow": flow.key,
                "link": link,
                "arrival": arrival,
                "start": start,
                "finish": finish,
            }
        )

        if hop_index + 1 < len(flow.path) - 1:
            enqueue(flow, finish, hop_index + 1)
            continue

        if flow.kind == "down":
            key = flow.key
            theta[key] = finish
            for cid in down_clients[key]:
                schedule_client(cid, finish)
        else:
            completion[int(flow.key)] = finish

    missing = set(clients) - set(completion)
    if missing:
        raise RuntimeError(f"latency simulation did not complete clients: {sorted(missing)}")

    return {
        "latency": max(completion.values(), default=0.0),
        "theta": theta,
        "q": q,
        "a": a,
        "T": completion,
        "trace": trace,
    }
