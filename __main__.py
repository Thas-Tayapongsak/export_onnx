from exporters import export_onnx, parse_arguments

def main(args):
    export_onnx(args.repo_id, args.task, args.output)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)