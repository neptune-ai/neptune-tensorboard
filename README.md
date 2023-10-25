![Neptune and TensorFlow logos](https://neptune.ai/wp-content/uploads/2023/09/tensorboard_tensorflow.svg)

# Neptune-TensorBoard integration

Log TensorBoard-tracked metadata to neptune.ai.

## What will you get with this integration?

* Log, organize, visualize, and compare ML experiments in a single place
* Monitor model training live
* Version and query production-ready models and associated metadata (e.g. datasets)
* Collaborate with the team and across the organization

## What will be logged to Neptune?

* Model summary and predictions
* Training code and Git information
* System metrics and hardware consumption

You can also log:

* Existing TensorBoard logs
* [Other metadata](https://docs.neptune.ai/logging/what_you_can_log)

![Dashboard with TensorBoard metadata](https://docs.neptune.ai/img/app/integrations/tensorboard.png)

## Resources

* [Documentation](https://docs.neptune.ai/integrations/tensorboard/)
* [Code example on GitHub](https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/tensorboard)
* [Example project in the Neptune app](https://app.neptune.ai/o/common/org/tensorboard-integration/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=99830fca-15f8-4431-baea-808ae13c0120&shortId=TBOARD-880&type=run)

## Example

Install Neptune and the integration:

```sh
pip install -U "neptune[tensorboard]"
```

Enable Neptune logging:

```python
import neptune
from neptune_tensorboard import enable_tensorboard_logging

neptune_run = neptune.init_run(
    project="workspace-name/project-name",  # replace with your own
    tags = ["tensorboard", "test"],  # optional
    dependencies="infer",  # optional
)

enable_tensorboard_logging(neptune_run)
```

Export existing TensorBoard logs:

```sh
neptune tensorboard --api_token YourNeptuneApiToken --project YourNeptuneProjectName logs
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting_help).
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! In the Neptune app, click the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP).
* You can just shoot us an email at [support@neptune.ai](mailto:support@neptune.ai).
