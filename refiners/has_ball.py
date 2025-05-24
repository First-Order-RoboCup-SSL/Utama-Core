from refiners.base_refiner import BaseRefiner


class HasBallRefiner(BaseRefiner):

    def refine(self, game, data):
        return game
