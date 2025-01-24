import os
import sys

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../../"
)))

from realsim.jobs.jobs import Job
from realsim.jobs.utils import deepcopy_list
from realsim.scheduler.coschedulers.ranks.ranks import RanksCoscheduler
from realsim.cluster.host import Host


class TwoStepsCoscheduler(RanksCoscheduler):

    name = "TwoSteps Co-Scheduler"
    description = """Co-scheduler that tries to fill the ''holes'' 
    in the HPC system's resources created by the allocation of jobs inside"""

    # def host_alloc_condition(self, hostname, job):
    #     worst_speedup = super().host_alloc_condition(hostname, job)
    #     idx = int(hostname.replace("host", "")) + 1
    #     return worst_speedup/idx


    # def host_alloc_condition(self, hostname: str, job: Job) -> float:
    #     return float(self.cluster.hosts[hostname].state != Host.IDLE)

    def get_index(self, job: Job) -> int:
        positions = [index for index, value in enumerate(self.cluster.second_queue) if value.job_id == job.job_id]
        # print("positions:", positions)
        # print(self.cluster.second_queue)
        return positions[0]
    
    def waiting_queue_reorder(self, job: Job) -> float:
        # The job that is closer to cover the gaps is more preferrable
        sys_free_cores = self.cluster.get_idle_cores()
        if sys_free_cores > 0:
            diff = sys_free_cores - job.num_of_processes
            if diff > 0:
                factor0 = 1 - (diff/sys_free_cores)
            elif diff == 0:
                factor0 = 1
            else:
                factor0 = -1
        else:
            factor0 = 1

        # factor1 = ((job.job_id + 1) / len(self.cluster.waiting_queue))
        
        factor1 = self.get_index(job) / len(self.cluster.waiting_queue)


        # return factor0 / factor1
        return factor0+factor1
    
    def second_queue_reorder(self, job: Job) -> float:
        return job.remaining_time
    


    # def coloc_condition(self, hostname: str, job: Job) -> float:
    #     return float(self.cluster.hosts[hostname].state != Host.IDLE)


    def deploy(self) -> bool:

        deployed = False

        # Update the rank of each job before scheduling them
        self.update_ranks()
        self.cluster.second_queue = deepcopy_list(self.cluster.waiting_queue[:self.queue_depth])
        self.cluster.second_queue.sort(key=lambda job: self.second_queue_reorder(job), reverse=True)

        waiting_queue = deepcopy_list(self.cluster.waiting_queue[:self.queue_depth])
        waiting_queue.sort(key=lambda job: self.waiting_queue_reorder(job),
                           reverse=True)

        while waiting_queue != []:

            # Remove from the waiting queue
            job = self.pop(waiting_queue)

            # Colocate
            if self.allocation(job, self.cluster.half_socket_allocation):
                deployed = True
                self.after_deployment()
            else:
                break

        return deployed