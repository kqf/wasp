import click


@click.group()
def wasp():
    """Things related to wasp"""
    pass


@wasp.command()
def track():
    """Start the experiments with tracker"""
    from wasp.tracker.main import main  # noqa

    main()


@wasp.command()
def train():
    """Train the detector"""
    from wasp.retinaface.train import main  # noqa

    main()
