from src.config import config
from src.training import run_training_pipeline


def main():
    run_training_pipeline(config)


if __name__ == '__main__':
    main()