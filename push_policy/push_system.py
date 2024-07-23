from simulation.sim_system import SimSystem


class PushSystem(SimSystem):

    def job_manager(self, job):
        """
        Manages the addition of a new job to the simulation system and initiates its processing.
        :param job: The job object to be managed.
        :returns: None
        """

        self.jobs.append(job)
        self.env.process(job.main())
