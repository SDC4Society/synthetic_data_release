""" Parent class for sanitisers """


class Sanitiser(object):

    seed = None

    def set_seed(self, seed: int | None):
        """Set a seed for reproducibility"""
        self.seed = seed

    def sanitise(self, data):
        """ Apply a privacy policy to the data. """
        return NotImplementedError('Method needs to be overwritten by a subclass')
