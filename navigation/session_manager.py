"""
Defines context managers for Tensorflow sessions, which can be used by different
algorithms to construct new sessions for a reusable computation graph
"""


class TensorflowManager:
    """
    A context manager for Tensorflow sessions
    """

    def __init__(self, graph, builder):
        """
        Initializes the session manager.

        :param graph: the computation graph with which to associate each session
        :param builder: the builder function to call on the constructed session
        """

        self._graph = graph
        self._builder = builder

    def __enter__(self):

        # Initialize session
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self._session = tf.Session(graph=self._graph, config=config)

        try:
             obj = self._builder(session)
        except Exception as e:
            self._session.close()
            raise e

        return obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()

        return False
