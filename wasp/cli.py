import click


@click.group()
def wasp():
    """Things related to wasp"""
    pass


@wasp.command()
def track():
    from wasp.tracker.main import main  # noqa

    main()
