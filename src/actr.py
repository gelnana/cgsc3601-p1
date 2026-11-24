from collections import deque

from python_actr import *
import pandas as pd

from rps import Dyad, Move


class Agent(ACTR):
    vision = Buffer()
    retrieval = Buffer()
    goal = Buffer()
    dm = Memory(retrieval, threshold=-10)
    DMBaseLevel(dm)
    DMNoise(dm, noise=0.25)

    def __init__(self):
        super().__init__()
        self.vision.set('idle')
        self.move = Move.rock
        self.goal.set('waiting')

    def play(self, move):
        if move in Move.__members__:
            outcome = self.move.outcome(Move[move])
            self.dm.add(f'move:{self.move.name} outcome:{outcome}')
        self.vision.set(move)


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
    return pd.Series([move.value for move in api.history], index=dyad.self_move.index)
