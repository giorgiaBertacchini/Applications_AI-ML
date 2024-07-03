#from simulation.sim_system import SimSystem
from simulation.sim_system_2 import SimSystem


class PushSystem(SimSystem):

    def job_manager(self, job):
        self.jobs.append(job)
        self.env.process(job.main())
