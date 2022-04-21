import toml
import utility_functions.classification_models as model_zoo


def load_model(cfg):
    if "model_name" in cfg.keys():
        method_to_call = getattr(model_zoo, cfg["model_name"])
        model = method_to_call()
        # TODO how to handle exceptions of getattr?
        return model
    else:
        print(
            "ERROR: No model specified, you have to specify model_name in the config")
        exit(1)


def main(config_name):
    with open('classification.toml', 'r') as f:
        config_dict = toml.load(f)

    cfg = config_dict[config_name]
    model = load_model(cfg)
    model.summary()


if __name__ == '__main__':
    main("classification_16_color_combined")
