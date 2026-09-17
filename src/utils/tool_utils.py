# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
import torch
import math
import numpy as np
import random
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

DEFAULT_SYSTEM_TRACE_SLOTS = 1000 * 4
DEFAULT_MC_UPLINK_MIN_MBPS = 4.0
DEFAULT_MC_UPLINK_MAX_MBPS = 40.0
DEFAULT_MC_DOWNLINK_MULTIPLIER = 4.0


def _required_system_slots(options):
    return int(options['round_num']) * int(options['edge_epoch'])


def _system_trace_slots(options):
    slots = int(options.get('system_trace_slots', DEFAULT_SYSTEM_TRACE_SLOTS))
    required_slots = _required_system_slots(options)
    if slots < required_slots:
        raise ValueError(
            "system_trace_slots={} is smaller than the {} slots required by "
            "round_num={} and edge_epoch={}".format(
                slots,
                required_slots,
                options['round_num'],
                options['edge_epoch'],
            )
        )
    return slots


def _numeric_tag(value):
    return format(float(value), '.12g').replace('-', 'm').replace('.', 'p')


def _client_rate_profile(options):
    uplink_min = float(
        options.get('mc_uplink_min_mbps', DEFAULT_MC_UPLINK_MIN_MBPS)
    )
    uplink_max = float(
        options.get('mc_uplink_max_mbps', DEFAULT_MC_UPLINK_MAX_MBPS)
    )
    downlink_multiplier = float(
        options.get(
            'mc_downlink_multiplier',
            DEFAULT_MC_DOWNLINK_MULTIPLIER,
        )
    )
    if uplink_min <= 0 or uplink_max < uplink_min:
        raise ValueError(
            'MC uplink rates require 0 < min <= max, got {}--{} Mbps'.format(
                uplink_min, uplink_max
            )
        )
    if downlink_multiplier <= 0:
        raise ValueError('mc_downlink_multiplier must be positive')
    return {
        'uplink_min_mbps': uplink_min,
        'uplink_max_mbps': uplink_max,
        'downlink_multiplier': downlink_multiplier,
        'downlink_min_mbps': uplink_min * downlink_multiplier,
        'downlink_max_mbps': uplink_max * downlink_multiplier,
    }


def defaultSystemParameterPath(options, save_dir='data/system_heter'):
    """Return the shared trace path for one physical system configuration."""
    client_rates = _client_rate_profile(options)
    v_min = float(options.get('es_link_rate_min', 500.0))
    v_max = float(options.get('es_link_rate_max', 500.0))
    if v_min == v_max:
        rate_tag = _numeric_tag(v_min)
    else:
        rate_tag = '{}_{}'.format(_numeric_tag(v_min), _numeric_tag(v_max))
    filename = (
        'parameters_m{}_e{}_slots{}_u{}to{}_d{}to{}_v{}_seed{}.npy'
    ).format(
        int(options['num_of_clients']),
        int(options['num_of_edges']),
        _system_trace_slots(options),
        _numeric_tag(client_rates['uplink_min_mbps']),
        _numeric_tag(client_rates['uplink_max_mbps']),
        _numeric_tag(client_rates['downlink_min_mbps']),
        _numeric_tag(client_rates['downlink_max_mbps']),
        rate_tag,
        int(options.get('system_seed', 2025)),
    )
    return os.path.join(save_dir, filename)


def _validate_reusable_parameters(params, options, source):
    slots = _system_trace_slots(options)
    num_clients = int(options['num_of_clients'])
    num_edges = int(options['num_of_edges'])
    required = ('cpu_frequency', 'U', 'D', 'V', 'transmit_power')
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(
            "{} is missing system parameter fields: {}".format(
                source, ', '.join(missing)
            )
        )

    for key in ('cpu_frequency', 'U', 'D'):
        values = np.asarray(params[key])
        if values.shape != (slots, num_clients):
            raise ValueError(
                "{} field {} has shape {}, expected ({}, {})".format(
                    source, key, values.shape, slots, num_clients
                )
            )

    rates = np.asarray(params['V'])
    if rates.shape != (slots, num_edges, num_edges):
        raise ValueError(
            "{} field V has shape {}, expected ({}, {}, {})".format(
                source, rates.shape, slots, num_edges, num_edges
            )
        )

    power = np.asarray(params['transmit_power'])
    if power.shape != (num_clients,):
        raise ValueError(
            "{} field transmit_power has shape {}, expected ({},)".format(
                source, power.shape, num_clients
            )
        )


def paraGeneration(options, save_dir="data/system_heter/", save_filename="parameters.npy"):
    system_slots = _system_trace_slots(options)
    num_clients = options['num_of_clients']
    num_edges = options['num_of_edges']
    system_seed = int(options.get('system_seed', 2025))
    client_rates = _client_rate_profile(options)
    rng = np.random.RandomState(system_seed)

    print(f"Current working directory: {os.getcwd()}")

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    # CPU clock speed for every client in every edge-round slot.
    # Shape: [system slot][client].
    cpu_frequency = [
        np.round(rng.uniform(0.1, 5, size=num_clients), 1)
        for _ in range(system_slots)
    ]

    # bandwidth_ul = [np.round(np.random.uniform(1, 5, size=num_clients), 1) for _ in range(round_num * edge_round)]
    # # print(bandwidth_ul)
    # ratio_dl_ul = 10.0
    # bandwidth_dl = [np.round(ratio_dl_ul * arr, 1) for arr in bandwidth_ul]

    # Client uplink/downlink rates are stored in Mbps.  Generate the uplink in
    # MB/s first to preserve the existing 0.1 MB/s granularity, then convert
    # it to Mbps.  Downlink is a fixed multiple of the corresponding uplink.
    client_uplink_mb_per_s = [
        np.round(
            rng.uniform(
                client_rates['uplink_min_mbps'] / 8.0,
                client_rates['uplink_max_mbps'] / 8.0,
                size=num_clients,
            ),
            1,
        )
        for _ in range(system_slots)
    ]
    U = [u_array * 8.0 for u_array in client_uplink_mb_per_s]
    D = [u_array * client_rates['downlink_multiplier'] for u_array in U]

    transmit_power = [
        np.round(rng.uniform(1, 10), 1)
        for _ in range(num_clients)
    ]

    # ES-to-ES rates V[r][e][f], in Mbps.  Rates are generated symmetrically,
    # while the latency simulator keeps a separate FCFS queue per direction.
    # The ES adjacency matrix masks out non-links when paths are selected.
    v_min = float(options.get('es_link_rate_min', 500.0))
    v_max = float(options.get('es_link_rate_max', 500.0))
    if v_min <= 0 or v_max < v_min:
        raise ValueError("ES link rates require 0 < es_link_rate_min <= es_link_rate_max")
    V = []
    for _ in range(system_slots):
        rates = np.zeros((num_edges, num_edges), dtype=float)
        for e in range(num_edges):
            for f in range(e + 1, num_edges):
                rate = np.round(rng.uniform(v_min, v_max), 1)
                rates[e, f] = rate
                rates[f, e] = rate
        V.append(rates)

    params = {
        'cpu_frequency':  cpu_frequency,
        # 'bandwidth_ul':   bandwidth_ul,
        # 'bandwidth_dl':   bandwidth_dl,
        # 'ratio_dl_ul':    ratio_dl_ul,
        'U': U,
        'D': D,
        'V': V,
        'transmit_power': transmit_power
    }

    save_path = os.path.join(save_dir, save_filename)
    temp_path = "{}.tmp-{}".format(save_path, os.getpid())
    try:
        with open(temp_path, 'wb') as handle:
            np.save(handle, params)
        os.replace(temp_path, save_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    print(
        "System parameters generated once: {} slots, seed {}, {}".format(
            system_slots, system_seed, save_path
        )
    )
    return save_path


def ensureSystemParameters(
    options,
    save_dir="data/system_heter/",
    save_filename="parameters.npy",
):
    """Reuse a compatible system trace, generating it only when absent."""
    save_path = os.path.join(save_dir, save_filename)
    if os.path.isfile(save_path):
        params = np.load(save_path, allow_pickle=True).item()
        _validate_reusable_parameters(params, options, save_path)
        print("Reusing system parameters: {}".format(save_path))
        return save_path
    return paraGeneration(options, save_dir=save_dir, save_filename=save_filename)


if __name__ == "__main__":
    # Example usage
    options = {
        'num_of_clients': 5,
        'num_of_edges': 2,
        'round_num': 10,
        'edge_epoch': 3,
        'system_trace_slots': DEFAULT_SYSTEM_TRACE_SLOTS,
        'system_seed': 2025,
    }
    paraGeneration(options)
