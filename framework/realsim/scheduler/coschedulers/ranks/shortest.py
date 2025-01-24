from abc import ABC
from .ranks import RanksCoscheduler
from numpy.random import seed, randint
from time import time_ns
from math import inf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../../"
)))

from realsim.jobs.jobs import Job
from realsim.scheduler.coschedulers.ranks.random import RandomRanksCoscheduler


class ShortestJobFirstCoscheduler(RandomRanksCoscheduler, ABC):

    name = "Shortest Job First Co-Scheduler"
    description = """Co-scheduling using ranks architecture to favor job avg speeedup and then slowdown"""

    def waiting_queue_reorder(self, job: Job) -> float:
        return -float(job.remaining_time)

