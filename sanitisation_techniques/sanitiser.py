""" Parent class for sanitisers """


class Sanitiser(object):

    def sanitise(self, data):
        """ Apply a privacy policy to the data. """
        return NotImplementedError('Method needs to be overwritten by a subclass')
