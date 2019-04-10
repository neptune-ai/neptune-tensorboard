neptune-tensorboard: TensorBoard integration with Neptune
===========================================
 
This library integrates `TensorBoard`_ with `Neptune`_ to let you get the best of both worlds.
Enjoy tracking experience of `TensorBoard`_ with organizion and collaboration of `Neptune`_.

With `neptune-mlflow` you can have your `TensorBoard`_ experiment runs hosted in a beatutiful knowledge repo that lets you invite and manage project contributors. 
 
With one simple command:

    neptune tensorboard /path/to/logdir --project USER_NAME/PROJECT_NAME
    
Organize your TensorBoard experiments: 

.. figure:: https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/9412c0124439f34b42737a7f3760849761c42dc4/tensorboard_1.png
   :alt: image
   
   
and compare your TensorBoard runs,

.. figure:: https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/9412c0124439f34b42737a7f3760849761c42dc4/tensorboard_2.png
   :alt: image
   
and share your work with others by sending a `experiment link`_

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   neptune_app
   library
   
   
.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/examples_index


.. toctree::
   :maxdepth: 1
   :caption: User Guide

   sync <user_guide/data_sync>
   Keras integration <user_guide/keras_integration>
   Tensorflow integration <user_guide/tensorflow_integration>

   
Bug Reports and Questions
-----------------------

neptune-mlflow is an Apache Licence 2.0 project and the source code is available on `GitHub`_. If you
find yourself in any trouble drop an isse on `Git Issues`_, fire a feature request on
`Git Feature Request`_ or ask us on the `Neptune community forum`_ or `Neptune Slack`_.

You can join our slack by clicking on the `Neptune Slack Join`_.


Contribute
-----------------------

We keep an updated list of open issues/feature ideas on github project page `Github projects`_.
If you feel like taking a shot at one of those do go for it!
In case of any trouble please talk to us on the `Neptune Slack`_.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`


.. _GitHub: https://github.com/neptune-ml/neptune-tensorboard
.. _Git Issues: https://github.com/neptune-ml/neptune-tensorboard/issues
.. _Git Feature Request: https://github.com/neptune-ml/neptune-tensorboard/issues
.. _TensorBoard: https://www.tensorflow.org/guide/summaries_and_tensorboard
.. _Neptune: http://bit.ly/2uUd9AB
.. _Neptune community forum: https://community.neptune.ml/
.. _experiment link: https://ui.neptune.ml/jakub-czakon/tensorboard-integration/compare?shortId=%5B%22TEN-41%22%2C%22TEN-40%22%2C%22TEN-39%22%2C%22TEN-38%22%2C%22TEN-37%22%2C%22TEN-36%22%2C%22TEN-35%22%2C%22TEN-34%22%2C%22TEN-33%22%2C%22TEN-32%22%5D
.. _Github projects: https://github.com/neptune-ml/neptune-tensorboard/projects
.. _Neptune Slack: https://neptune-community.slack.com
.. _Neptune Slack Join: https://join.slack.com/t/neptune-community/shared_invite/enQtNTI4Mjg3ODk2MjQwLWE5YjI0YThiODViNDY4MDBlNmRmZTkwNTE3YzNiMjQ5MGM2ZTFhNzhjN2YzMTIwNDM3NjQyZThmMDk1Y2Q1ZjY
