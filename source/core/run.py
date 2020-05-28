import argparse as ap

from source.utils import file_manager, learning_manager
from source.graphing import grapher


def main(args):
    assert isinstance(args, ap.Namespace)

    b = True if args.force_rebuild else False
    for s in args.split:
        file_manager.set_split(str(s))
        file_manager.read(args.input)

        if args.all:
            learning_manager.build_all(b=b, percent=s, noise=args.noise)

        if args.method:
            learning_manager.build_algorithms(args.method, b=b, percent=s, noise=args.noise)

        if args.graph and args.all:
            grapher.plot_all()

        if args.graph and args.method:
            grapher.plot(args.method)

        if args.create_table and args.all:
            grapher.table_all()

        if args.create_table and args.method:
            grapher.table(args.method)


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument('-i', '--input', nargs="+", required=True)
    parser.add_argument('-g', '--graph', action="store_true")
    parser.add_argument('-f', '--force-rebuild', action="store_true")
    parser.add_argument('-t', '--create-table', action="store_true")
    parser.add_argument('-s', '--split', nargs="+", type=float, required=True)
    parser.add_argument('-n', '--noise', type=float, default=0)

    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument('-m', '--method', nargs="+")
    command_group.add_argument('-a', '--all', action="store_true")

    a = parser.parse_args()
    main(a)
