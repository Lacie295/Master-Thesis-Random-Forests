import argparse as ap

from source.utils import file_manager


def main(args):
    assert isinstance(args, ap.Namespace)

    if args.input:
        file_manager.read(args.input)


if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument('-m', '--method', nargs="+")
    parser.add_argument('-i', '--input', nargs="+")
    parser.add_argument('-g', '--graph', action="store_true")
    parser.add_argument('-a', '--graph', action="store_true")

    a = parser.parse_args()
    main(a)
