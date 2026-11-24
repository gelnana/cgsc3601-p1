from rps import Move
from actr import Agent


class Baseline(Agent):
    def rock(vision='rock'):
        pass

    def paper(vision='paper'):
        pass

    def scissors(vision='scissors'):
        pass

    def no_response(vision='none'):
        pass

class WSLC(Agent):
    def perceive(vision='?x', goal='waiting'):
        self.vision.clear()
        self.dm.request('move:? outcome:?')
        self.goal.set('retrieving')

    def stay(goal='retrieving', retrieval='outcome:win'):
        self.goal.set('waiting')
        self.opponent.play(self.move)

    def shift(goal='retrieving', retrieval='outcome:!win'):
        self.move = self.random.choice([m for m in type(self.move) if m != self.move])
        self.goal.set('waiting')
        self.opponent.play(self.move)

    def forget(goal='retrieving', retrieval=None):
        self.move = self.random.choice(list(type(self.move)))
        self.goal.set('waiting')
        self.opponent.play(self.move)

class WSLC_I(Agent):
    def rock(vision='rock'):
        pass

    def paper(vision='paper'):
        pass

    def scissors(vision='scissors'):
        pass

    def no_response(vision='none'):
        pass


class WSLC_C(Agent):
    def rock(vision='rock'):
        pass

    def paper(vision='paper'):
        pass

    def scissors(vision='scissors'):
        pass

    def no_response(vision='none'):
        pass

class WCLS(Agent):
    def rock(vision='rock'):
        pass

    def paper(vision='paper'):
        pass

    def scissors(vision='scissors'):
        pass

    def no_response(vision='none'):
        pass
