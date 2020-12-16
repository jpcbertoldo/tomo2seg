from dataclasses import dataclass
from typing import ClassVar

import yaml

from tomo2seg.logger import logger


HOSTS_CONFIGS_YAML = "/home/users/jcasagrande/projects/tomo2seg/data/hosts.yml"


@dataclass
class Host:
    hostname: str
    analyse_parallel_nprocs: int
    gpu_max_memory_factor: float


class HostsConfigsError(Exception):
    pass


def get_hosts():

    logger.debug(f"Getting host configs from {HOSTS_CONFIGS_YAML}")

    try:
        with open(HOSTS_CONFIGS_YAML, "r") as f:
            hosts_configs_ = yaml.load(f, Loader=yaml.FullLoader)

    except FileNotFoundError as ex:
        logger.exception(ex)
        logger.warning(f"Please create a yaml file at {HOSTS_CONFIGS_YAML}.")
        raise HostsConfigsError(f"FileNotFound {HOSTS_CONFIGS_YAML=}")

    assert hosts_configs_ is not None
    assert "hosts" in hosts_configs_, f"{hosts_configs_.keys()}"

    hosts_ = hosts_configs_["hosts"]

    assert isinstance(hosts_, dict), f"{type(hosts_)=}"

    for hostname, host in hosts_.items():

        assert isinstance(hostname, str), f"{type(hostname)=}"
        assert isinstance(host, dict), f"{hostname=} {type(host)=}"
        assert hostname == host["hostname"]

        hosts_[hostname] = Host(**host)

    return hosts_


hosts = get_hosts()


if __name__ == "__main__":

    print(hosts)
