import argparse as ap

from source.utils import file_manager, learning_manager


def main(args):
    assert isinstance(args, ap.Namespace)

    b = True if args.force_rebuild else False

    if args.input:
        file_manager.read(args.input)

    if args.all:
        learning_manager.build_all(b=b)

    if args.method:
        learning_manager.build_algorithms(args.method, b=b)


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument('-i', '--input', nargs="+", required=True)
    parser.add_argument('-g', '--graph', action="store_true")
    parser.add_argument('-f', '--force-rebuild', action="store_true")

    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument('-m', '--method', nargs="+")
    command_group.add_argument('-a', '--all', action="store_true")

    a = parser.parse_args()
    main(a)
