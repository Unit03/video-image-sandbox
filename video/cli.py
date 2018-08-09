import click

from ._version import version


@click.group()
@click.pass_context
@click.option("--debug", is_flag=True)
@click.version_option(version)
def video(ctx, debug):
    ctx.obj = dict()
    ctx.obj["debug_flag"] = debug
