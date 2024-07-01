from simulation.sim_system import SimSystem


class PushSystem(SimSystem):

    def job_manager(self, job):
        self.jobs.append(job)
        self.env.process(job.main())
