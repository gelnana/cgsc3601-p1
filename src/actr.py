from collections import deque
from dataclasses import dataclass
import math

from python_actr import *
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from rps import Dyad, Move


class Agent(ACTR):
    # stores what opponent played as single value.
    # e.g. 'rock', 'paper', 'scissors', or 'none'
    vision = Buffer()

    # vision but persistent, doesn't get flushed.
    # used in recall cases, where opponent previous 
    # move needs to readily accessed.
    context = Buffer()
    
    # holds results of declative memory queries, matched in agents of select strategies
    # matched on `move:rock/paper/scissors`, `outcome:win/!win`, or `opponent:rock/paper/scissors`
    retrieval = Buffer()
    
    # state machine for controlling agent action:
    # - waiting : idle, once stimuli detected in vision hands off to `retrieving`
    # - retrieving : seen opponent move, in process of making choice
    goal = Buffer()

    dm = Memory(retrieval, threshold=-10)
    
    # declarative memory parameters
    DMBaseLevel(dm)
    DMNoise(dm, noise=0.25)

    # production rule parameters
    # basically kept at default, but made available just in case
    production_time = 0.05
    production_time_sd = 0.01
    production_threshold = -20

    def __init__(self):
        super().__init__()
        self.vision.set('idle')
        self.Move = Move
        self.move = Move.rock
        self.goal.set('waiting')
        self._opponent_move = None

    def play(self, move):
        # score before caching the new move
        self._store()
        self._opponent_move = move
        self.context.set(move)
        self.vision.set(move)

    def perceive(vision='?x', goal='waiting'):
        self.vision.clear()
        self.dm.request('move:? outcome:?')
        self.goal.set('retrieving')

    def forget(goal='retrieving', retrieval=None):
        self.move = self.random.choice(list(self.Move))
        self.goal.set('waiting')
        self.opponent.play(self.move)

    def _store(self):
        # write results to decl. memory
        if self._opponent_move in Move.__members__:
            outcome = self.move.outcome(Move[self._opponent_move])
            self.dm.add(f'move:{self.move.name} outcome:{outcome} opponent:{self._opponent_move}')

    def flush(self):
        self._store()
        self._opponent_move = None


class DMReward(MemorySubModule):
    def __init__(self, memory, key='outcome', decay=None, **values):
        super().__init__(memory)
        self.key = key
        self.decay = decay
        self.values = values

    def activation(self, chunk):
        base = self.values.get(chunk.get(self.key), 0.0)
        if self.decay is not None and hasattr(chunk, 'creation'):
            age = max(self.now() - chunk.creation, 0.0)
            base *= math.exp(-self.decay * age)
        return base


def fresh_dm(noise=0.25, threshold=-10, associate_weight=1.0, associate_decay=0.5):
    dm = Memory(Agent.retrieval, threshold=threshold)
    DMBaseLevel(dm)
    DMNoise(dm, noise=noise)
    DMAssociate(dm, Agent.context, weight=associate_weight, decay=associate_decay)
    return dm


def configure(agent: Model, noise: float = None, reward: float = None) -> Model:
    for adaptor in agent.dm.adaptors:
        if noise is not None and isinstance(adaptor, DMNoise):
            adaptor.noise = noise
        if reward is not None and isinstance(adaptor, DMReward):
            adaptor.values = {key: math.copysign(reward, value) for key, value in adaptor.values.items()}
    return agent


@dataclass
class Fit:
    # paper: https://doi.org/10.48550/arXiv.2410.18055
    # grid search tuning, paper suggests using Powell's method to find local minima
    # - but I don't understand it completely, and a simple grid should be just fine
    # for our purposes.
    NOISE_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
    REWARD_GRID = (0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4)
    FIT_REPEATS = 25 # run X simulations per parameter pair
    FIT_STATS = ("win_stay_rate", "lose_stay_rate", "counter_rate")

    model_cls: type
    noise: float = 0.25
    reward: float = 0.2
    log_lik: float = -math.inf

    def __call__(self) -> Model:
        return configure(self.model_cls(), noise=self.noise, reward=self.reward)

    @classmethod
    def search(cls, model_cls: type, frame: pd.DataFrame, target) -> "Fit":
        from stats import Profile

        best_fit = cls(model_cls)
        for noise in cls.NOISE_GRID:
            for reward in cls.REWARD_GRID:
                fit = cls(model_cls, noise, reward)
                samples = Profile.sample(fit, frame, repeats=cls.FIT_REPEATS)
                log_lik, n_contrib = 0.0, 0
                
                # score how closely the simulated behaviors match the target behavior
                for stat in cls.FIT_STATS:
                    target_value = getattr(target, stat)
                    if pd.isna(target_value):
                        continue
                    
                    # collect valid simulation results for this behavior
                    values = np.array([getattr(s, stat) for s in samples])
                    values = values[~np.isnan(values)]
                    
                    # skip if there isn't enough variance in the results to estimate a curve
                    if len(values) < 2 or np.ptp(values) == 0:
                        continue
                    
                    # estimate how likely the target value is given our simulated results 
                    density = gaussian_kde(values)(target_value)[0]
                    if density <= 0:
                        continue

                    # sum up log-density (where tall hills = higher score)
                    # use log so that tiny values don't multiply to 0
                    log_lik += math.log(density)
                    n_contrib += 1
                # select winner only when atleast one behaviour was scored.
                if n_contrib and log_lik > best_fit.log_lik:
                    best_fit = cls(model_cls, noise, reward, log_lik)
        return best_fit

    @classmethod
    def select(cls, model_classes, frame: pd.DataFrame, target) -> "Fit":
        # pick candidate model that aligns strongest with participant's calibration
        fits = [cls.search(model_cls, frame, target) for model_cls in model_classes]
        return max(fits, key=lambda fit: fit.log_lik)


class API(Model):
    def __init__(self, dyad: Dyad):
        super().__init__()
        self.history = []
        # load the opponents move into a dequeue
        self._queue = deque(
            Move(int(move)).name if pd.notna(move) else 'none'
            for move in dyad.partner_move
        )

    def play(self, move):
        # write opponent move to log, then queue next
        self.history.append(move)
        self.emit_next()

    def emit_next(self):
        if self._queue:
            self.opponent.play(self._queue.popleft())


class Session(Model):
    def __init__(self, player: Model, opponent: Model):
        super().__init__()
        self.player = player
        self.opponent = opponent
        self.__pair__(self.player, self.opponent)
        # initialize queue using model w/ available next move
        for model in (player, opponent):
            emit_next = getattr(model, 'emit_next', None)
            if callable(emit_next):
                emit_next()

    def __pair__(self, player, opponent):
        player.opponent = opponent
        opponent.opponent = player


def actr_simulate(agent: Model, group: pd.DataFrame) -> pd.Series:
    dyad = Dyad(group)
    api = API(dyad)
    session = Session(agent, api)
    session.run()
    # clear & store the final turn's outcome to memory
    flush = getattr(agent, 'flush', None)
    if callable(flush):
        flush()
    # return log of model's actions aligned to original trial rows
    return pd.Series([move.value for move in api.history], index=dyad.self_move.index)
