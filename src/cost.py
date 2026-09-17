# HFL-M3 -- https://github.com/kt4ngw/HFL-M3
# Copyright (c) 2026 Jian Tang. Academic use only; see LICENSE and cite the HFL-M3 paper.
from src.network_latency import simulate_round_latency


class Cost(object):
    def __init__(self):
        self.accumulated_energy = 0
        self.accumulated_latency = 0


    def get_latency_sum(self, selected_clients, round_i, system_params):
        waiting_time = 0
        latency_sum = 0
        latency_local_sum = 0
        latency_upload_sum = 0
        local = [0 for _ in range(len(selected_clients))]
        upload = [0 for _ in range(len(selected_clients))]
        download = [0 for _ in range(len(selected_clients))]
        for i in range(len(selected_clients)):
            local[i] = selected_clients[i].getLocalDelay(round_i, system_params)
            upload[i] = selected_clients[i].getUploadDelay(round_i, system_params)
            download[i] = selected_clients[i].get_downmodel_latency(round_i, system_params)
            if local[i] + upload[i] + download[i] > latency_sum:
                latency_sum = local[i] + upload[i] + download[i]
        # print(local)
        # print(upload)
        # print(download)
        return latency_sum

    def get_routed_latency(
        self,
        selected_clients,
        round_i,
        system_params,
        physical_owner,
        logical_owner,
        path_finder,
        model_size,
        cloud_sync=False,
        client_group=None,
        source_owner=None,
    ):
        """Return the paper's routed round latency and its timing details."""
        if 'V' not in system_params:
            raise KeyError("system parameter 'V' (ES-to-ES link rates) is required")

        clients_by_id = {client.idx: client for client in selected_clients}
        client_ids = sorted(physical_owner)
        missing = set(client_ids) - set(clients_by_id)
        if missing:
            raise KeyError(f"unknown clients in physical_owner: {sorted(missing)}")

        download = {
            cid: clients_by_id[cid].get_downmodel_latency(round_i, system_params)
            for cid in client_ids
        }
        compute = {
            cid: clients_by_id[cid].getLocalDelay(round_i, system_params)
            for cid in client_ids
        }
        upload = {
            cid: clients_by_id[cid].getUploadDelay(round_i, system_params)
            for cid in client_ids
        }

        return simulate_round_latency(
            client_ids=client_ids,
            physical_owner=physical_owner,
            logical_owner=logical_owner,
            download_latency=download,
            compute_latency=compute,
            upload_latency=upload,
            path_finder=path_finder,
            link_rates=system_params['V'][round_i],
            model_size=model_size,
            cloud_sync=cloud_sync,
            client_group=client_group,
            source_owner=source_owner,
        )



    # def get_energy_sum(self, selected_clients, bandwidth_allocation_result, round_i):
    #     energy_sum = 0
    #     energy_local_sum = 0
    #     energy_upload_sum = 0
    #     local = [0 for _ in range(len(selected_clients))]
    #     upload = [0 for _ in range(len(selected_clients))]
    #     for i in range(len(selected_clients)):
    #         local[i] = selected_clients[i].getLocalEngery(round_i)
    #         upload[i] = selected_clients[i].getUploadEngery(round_i, bandwidth_allocation_result[i])
    #         energy_sum += (local[i] + upload[i])
    #         energy_local_sum += local[i]
    #         energy_upload_sum += upload[i]
    #     return (energy_sum, energy_local_sum, energy_upload_sum)




class ClientAttr(object):
    def __init__(self, cpu_frequency, B, transmit_power):
        self.cpu_frequency = cpu_frequency
        self.bandwidth = B
        self.transmit_power = transmit_power

    def get_client_attr(self, id):
        return {
            "cpu_frequency": self.cpu_frequency[id],
            "B": self.bandwidth,
            "transmit_power": self.transmit_power[id],
        }
