from actr import Agent, DMReward, fresh_dm

class Baseline(Agent):
    def wildcard(goal='retrieving'):
        self.move = self.random.choice([m for m in self.Move])
        self.goal.set('waiting')
        self.opponent.play(self.move)
class WSLC(Agent):
    # biased toward win chunks
    dm = fresh_dm()
    DMReward(dm, win=0.2, lose=-0.2)

    def stay(goal='retrieving', retrieval='outcome:win'):
        self.goal.set('waiting')
        self.opponent.play(self.move)

    def shift(goal='retrieving', retrieval='outcome:!win'):
        self.move = self.random.choice([m for m in self.Move if m != self.move])
        self.goal.set('waiting')
        self.opponent.play(self.move)

class WCLS(Agent):
    # biased towards loss chunks
    dm = fresh_dm(associate_weight=0.0)
    DMReward(dm, win=-0.2, lose=0.2)

    def stay(goal='retrieving', retrieval='outcome:!win'):
        self.goal.set('waiting')
        self.opponent.play(self.move)

    def shift(goal='retrieving', retrieval='outcome:win'):
        self.move = self.random.choice([m for m in self.Move if m != self.move])
        self.goal.set('waiting')
        self.opponent.play(self.move)

class Mimicking(WSLC):
    def transform(self, m):
        return m

    def mimic(goal='retrieving', retrieval='opponent:?opp'):
        self.move = self.transform(self.Move[opp])
        self.goal.set('waiting')
        self.opponent.play(self.move)

class WSLC_I(Mimicking):
    dm = fresh_dm(associate_weight=2.0)
    DMReward(dm, win=-0.1, lose=0.1)

class WSLC_C(Mimicking):
    dm = fresh_dm(associate_weight=0.0)
    DMReward(dm, win=-0.2, lose=0.2)

    def transform(self, m):
        return m.counter
