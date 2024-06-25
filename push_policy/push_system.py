from simulation.sim_system import SimSystem


class PushSystem(SimSystem):
    def job_manager(self, end_event, job):
        self.jobs.append(job)
        self.env.process(job.main())
        self.env.process(self.job_finished(end_event))