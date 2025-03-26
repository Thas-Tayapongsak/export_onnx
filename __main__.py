from exporters import export_onnx, parse_arguments


def main(args):
    onnx_inputs, onnx_outputs = export_onnx(args.repo_id, args.task, args.output_path, args.abs_path, args.do_validation)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)