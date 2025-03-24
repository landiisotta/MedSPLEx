import yaml
import pandas as pd

def update_config_from_annotations(ann_path, config_path,
                                   keyword_column = 'keyword', label_column='label',
                                   new_config_path = None,
                                   label_mapping={0:'neutral', 1:'stigmatizing', 2:'privileging'},
                                   save=False):
    """
    Update config valence for each keyword based on the annotated dataset
    :param ann_path: path to annotated file (csv)
    :param config_path: path to keywrod config
    :param keyword_column: column name for keyword
    :param label_column: column name for label
    :param new_config_path: new config file name. Defaults to <config_path>_updated.yml
    :param label_mapping: mapping from label column values to neutral, stigmatizing and privileging
    :param save: whether to save the new config
    :return:
    """
    annotations = pd.read_csv(ann_path)
    annotations[label_column] = annotations[label_column].replace(label_mapping)
    defined_keyword = annotations[keyword_column].unique().tolist()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for keyword in config.keys():
        if keyword in defined_keyword:
            labs = annotations.loc[annotations[keyword_column]==keyword,label_column].unique().tolist()
            if 'privileging' in labs and 'stigmatizing' in labs and len(labs)==3:
                config[keyword]['valence'] = 'ambiguous_bivalent'
            elif 'stigmatizing' in labs and 'neutral' in labs:
                config[keyword]['valence'] = 'ambiguous_stigmatizing'
            elif 'privileging' in labs and 'neutral' in labs:
                config[keyword]['valence'] = 'ambiguous_privileging'
            elif len(labs)==1:
                config[keyword]['valence'] = labs[0]
        else:
            if not config[keyword]['valence'] == 'undefined':
                config[keyword]['valence'] = 'undefined'

    if save:
        if new_config_path is None:
            new_config_path = f'{config_path.rpartition(".")[0]}_updated.yaml'
        print(f'saving new config to {new_config_path}')
        with open(new_config_path, 'w') as f:
            yaml.dump(config, f)

    return config
