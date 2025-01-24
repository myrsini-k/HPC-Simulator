from abc import ABC
from .ranks import RanksCoscheduler
from numpy.random import seed, randint
from time import time_ns
from math import inf
import os
import sys
from realsim.jobs.utils import deepcopy_list

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../../"
)))

from realsim.jobs.jobs import Job
from realsim.scheduler.coschedulers.ranks.random import RandomRanksCoscheduler


class AntiPopularFirstRanksCoscheduler(RandomRanksCoscheduler, ABC):

    name = "Reverse Popularity Co-Scheduler"
    description = """Co-scheduling using ranks architecture to favor job avg speeedup and then slowdown"""

    def waiting_queue_reorder(self, job: Job) -> float:
        # seed(time_ns() % (2 ** 32))
        # return float(randint(len(self.cluster.waiting_queue)))
	    # return 1.0
        return  - self.ranks[job.job_id] / len(self.cluster.waiting_queue)
    

    def deploy(self) -> bool:

        deployed = False

        # Update the rank of each job before scheduling them
        self.update_ranks()

        waiting_queue = deepcopy_list(self.cluster.waiting_queue[:self.queue_depth])
        waiting_queue.sort(key=lambda job: self.waiting_queue_reorder(job),
                           reverse=True)

        while waiting_queue != []:

            # Remove from the waiting queue
            job = self.pop(waiting_queue)

            if self.ranks[job.job_id]< 0.3*len(self.cluster.waiting_queue):
                if self.compact_allocation(job):
                    deployed = True
                    self.after_deployment()
            else:
                # Colocate
                if self.allocation(job, self.cluster.half_socket_allocation):
                    deployed = True
                    self.after_deployment()
                else:
                    break

        return deployed
