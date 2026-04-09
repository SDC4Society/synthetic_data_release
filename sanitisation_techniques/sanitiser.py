""" Parent class for sanitisers """


class Sanitiser(object):

    def sanitise(self, data):
        """ Apply a privacy policy to the data. """
        return NotImplementedError('Method needs to be overwritten by a subclass')

    def get_output_metadata(self, input_metadata):
        """Return metadata describing the sanitiser's output schema.

        Default: identity (output schema matches input).
        Override in subclasses that transform the schema.
        """
        return input_metadata
