"""FSDP worker whose actor is TameActor.

verl's ActorRolloutRefWorker.init_model resolves DataParallelPPOActor from verl.workers.actor when it runs, so
importing this module inside the worker process (Ray unpickles the class below by reference) is enough to swap it in.
The reference policy goes through the same class, but TameActor only applies the penalty when it owns an optimizer.
"""

import verl.workers.actor as _actor_pkg
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .actor import TameActor

_actor_pkg.DataParallelPPOActor = TameActor


class TameActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Inherits every RPC from verl unchanged."""
