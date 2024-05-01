import yaml
# from easydict import EasyDict as edict
def Config(args):

    with open(args.config) as f:  # args.config 就是指 config.yaml文件在python中的avatar对象
        # config = yaml.load(f)
        config = yaml.safe_load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('{0}: {1}'.format(k, v))
    print('='*20)
    return args