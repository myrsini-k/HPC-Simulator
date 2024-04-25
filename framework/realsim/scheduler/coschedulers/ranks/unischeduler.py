import os
import sys
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')
))

from realsim.jobs.jobs import Job
from realsim.jobs.utils import deepcopy_list
from .balancing import BalancingRanksCoscheduler

from numpy import average as avg
import math

class UniSceduler(BalancingRanksCoscheduler):
    """We try to provide at every checkpoint an execution list whose average
    speedup is higher than 1. We try to distribute the higher speedup candidates
    among the checkpoints.
    """

    name = "UNICEF Balancing Ranks Co-Scheduler"
    description = "My implementation on balancing ranks coscheduler"

    def waiting_queue_order(self, job: Job) -> float:
        rank_r = self.ranks[job.job_id] / len(self.cluster.waiting_queue)
        cores_r = self.cluster.free_cores / job.num_of_processes
        w = job.waiting_time
        p = job.remaining_time
        term = math.log2(float(job.queued_time) + 0.01) * p
         
        return -(w / term)