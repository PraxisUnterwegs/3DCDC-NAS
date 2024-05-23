import yaml
import subprocess
import os
# from easydict import EasyDict as edict

def get_git_root():
    try:
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            stderr=subprocess.STDOUT
        ).strip().decode('utf-8')
        return git_root
    except subprocess.CalledProcessError:
        return None

def resolveEnv(v):
    if isinstance(v, str):
        if "$HOME" in v:
            v = v.replace("$HOME", os.environ.get('HOME'))
        if "~" in v:
            v = v.replace("~", os.environ.get('HOME'))
        if "$GITDIR" in v:
            v = v.replace("$GITDIR", get_git_root())
    return v

def Config(args):
    with open(args.config) as f:
        # config = yaml.load(f)
        config = yaml.safe_load(f)
    for k, v in config['common'].items():
        ### env var replacement
        v = resolveEnv(v)
        ### set attribute
        setattr(args, k, v)
        print('{0}: {1}'.format(k, v))
    print('='*20)
    return args

if __name__ == "__main__":
    print(get_git_root())
