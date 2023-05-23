# neptune-tensorboard

> **Note**
>
> _This integration is still being updated for the main Neptune client library. It is currently only available for the Neptune legacy API._

---

[![PyPI version](https://badge.fury.io/py/neptune-tensorboard.svg)](https://badge.fury.io/py/neptune-tensorboard)

![TensorBoard neptune.ai integration](docs/_static/tensorboard_neptuneml.png)

## Overview
`neptune-tensorboard` integrates TensorBoard with Neptune to give you the best of both worlds.
Enjoy tracking from TensorBoard with the organization and collaboration of Neptune.

With `neptune-tensorboard` you can have your TensorBoard experiment runs hosted in a beautiful knowledge repo that lets you invite and manage project contributors.

All you need to do to convert your past runs from TensorBoard logdir is run:

```bash
neptune tensorboard /path/to/logdir --project USER_NAME/PROJECT_NAME
```

You can connect Neptune to your TensorBoard and log all future experiments by adding the following to your scripts:

```python
import neptune
import neptune_tensorboard as neptune_tb

neptune.init(api_token='YOUR_TOKEN', project_qualified_name='USER_NAME/PROJECT_NAME') # credentials
neptune_tb.integrate_with_tensorflow()

neptune.create_experiment()
```

You will have your experiments hosted on Neptune and easily shareable with the world.

## Documentation

See [neptune-tensorboard docs](https://docs-legacy.neptune.ai/integrations/tensorboard.html) for more info.

## Get started

### Register
Go to [neptune.ai](http://bit.ly/2uUd9AB) and sign up.

It is completely free for individuals and non-organizations, and you can invite others to join your team!

### Get your API token

 In the bottom-left corner, click your user menu and select **Get your API token**.

### Set NEPTUNE_API_TOKEN environment variable

Go to your console and run:

```
export NEPTUNE_API_TOKEN='your_long_api_token'
```

### Create your first project

Click **All projects** &rarr; **New project**. Choose a name for it and whether you want it public or private.

### Install lib

```bash
pip install neptune-tensorboard
```

### Sync your TensorBoard logdir with Neptune

```bash
neptune tensorboard /path/to/logdir --project USER_NAME/PROJECT_NAME
```

### Connect Neptune to TensorBoard to log future runs

You can connect Neptune to your TensorBoard and log all future experiments by adding the following to your scripts:

```python
import neptune
import neptune_tensorboard as neptune_tb

neptune.init(api_token='YOUR_TOKEN', project_qualified_name='USER_NAME/PROJECT_NAME')  # credentials
neptune_tb.integrate_with_tensorflow()

neptune.create_experiment()
```

### Explore and Share

You can now explore and organize your experiments in Neptune, and share it with anyone:

* by sending a link to your project, experiment or chart if it is public
* or invite people to your project if you want to keep it private!

## Getting help

If you get stuck, don't worry. We are here to help.

The best order of communication is:

 * [GitHub issues](https://github.com/neptune-ai/neptune-tensorboard/issues)
 * [Email](mailto:support@neptune.ai)

## Contributing

If you see something that you don't like, you are more than welcome to contribute!

There are many options:

* Submit a feature request or a bug here, on Github
* Submit a pull request that deals with an open feature request or bug
* Spread the word about Neptune in your community
