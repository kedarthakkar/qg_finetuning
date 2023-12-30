from model.qg_fine_tune import QGFineTune
import argparse

def main(model_filepath, example):
    qg_fine_tune = QGFineTune(
        model_filepath=model_filepath,
    )
    qg_fine_tune.infer(example)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with QG model.')
    parser.add_argument('model_filepath', type='str', help='Filepath of the QG model.')
    parser.add_argument('example', type='str', help='Example to run QG inference on.')
    args = parser.parse_args()
    main(args.model_filepath, args.example)