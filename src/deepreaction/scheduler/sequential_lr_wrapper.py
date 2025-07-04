from torch.optim.lr_scheduler import SequentialLR

class SequentialLRWrapper(SequentialLR):
    """
    A wrapper around SequentialLR that recursively instantiates any schedulers or 
    scheduler factories with a shared optimizer passed to this wrapper optimizer.
    """

    def __init__(self, optimizer, schedulers, milestones, **kwargs):
        """
        Initialize the SequentialLR and pass the optimizer to each scheduler
        or scheduler factory.
        
        Args:
            optimizer (Optimizer): The optimizer to be used with the schedulers.
            schedulers (list): A list of scheduler instances or factory functions that return scheduler instances.
            milestones (list): A list of milestones for the SequentialLR.
            **kwargs: Additional keyword arguments to be passed to the SequentialLR constructor.
        """
        schedulers = [
            s(optimizer=optimizer) if callable(s) else s
            for s in schedulers
        ]
        super().__init__(optimizer, schedulers=schedulers, milestones=milestones, **kwargs)