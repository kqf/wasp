import click


@click.group()
def wasp():
    """Things related to wasp"""
    pass


@wasp.command()
def tracker():
    from wasp.tracker.main import main  # noqa

    main()
