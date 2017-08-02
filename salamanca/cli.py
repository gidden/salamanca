import argparse
import logging

from salamanca import data
from salamanca import currency

COMMANDS = {}

#
# Download wb data
#


def download_wb_cli(parser):
    log = 'Print log output during download.'
    parser.add_argument('--log', help=log, action="store_true")
    overwrite = 'Overwrite local files if they exist.'
    parser.add_argument('--overwrite', help=overwrite, action="store_true")


def download_wb(log=False, overwrite=False, **kwargs):
    if log:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    wb = data.WorldBank()
    wb.iso_metadata(overwrite=overwrite)
    for ind in data.INDICATORS_WB:
        wb.query(ind, overwrite=overwrite)


COMMANDS['download_wb'] = (
    """Download national World Bank data to your machine""",
    download_wb_cli,
    download_wb,
)


#
# Currency Exchange
#

# implement this

def main():
    descr = """
    Main CLI for salamanca.
    """
    parser = argparse.ArgumentParser(
        description=descr,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    for cmd in COMMANDS:
        cli_help = COMMANDS[cmd][0]
        cli_func = COMMANDS[cmd][1]
        subparser = subparsers.add_parser(cmd, help=cli_help)
        cli_func(subparser)

    args = parser.parse_args()
    cmd = args.command
    cmd_func = COMMANDS[cmd][2]
    cmd_func(**vars(args))

if __name__ == '__main__':
    main()
