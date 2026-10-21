import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco14', help='', choices=['coco14', 'voc'])
    parser.add_argument('--config_root', type=str, default='', help='config directory for default templates')
    parser.add_argument('--template', type=str, default='', help='explicit template path (overrides --config_root)')
    parser.add_argument('--output', type=str, default='', help='explicit output path')
    parser.add_argument('--shot', type=int, default=1, help='shot to run experiments over')
    parser.add_argument('--seed', type=int, default=0, help='seed to run experiments over')
    parser.add_argument('--setting', type=str, default='fsod', choices=['fsod'])
    parser.add_argument('--split', type=int, default=1, help='only for voc')
    args = parser.parse_args()
    return args


def load_config_file(yaml_path):
    fpath = os.path.join(yaml_path)
    yaml_info = open(fpath).readlines()
    return yaml_info


def save_config_file(yaml_info, yaml_path):
    wf = open(yaml_path, 'w')
    for line in yaml_info:
        wf.write('{}'.format(line))
    wf.close()


def build_default_template_path(args):
    if args.dataset in ['voc']:
        name_template = 'defrcn_{}_r101_novelx_{}shot_seedx.yaml'
        return os.path.join(args.config_root, name_template.format(args.setting, args.shot))
    if args.dataset in ['coco14']:
        name_template = 'defrcn_{}_r101_novel_{}shot_seedx.yaml'
        return os.path.join(args.config_root, name_template.format(args.setting, args.shot))
    raise NotImplementedError


def infer_output_path(template_path, args):
    if args.output:
        return args.output

    output_path = template_path.replace('seedx', 'seed{}'.format(args.seed))
    if args.dataset in ['voc']:
        output_path = output_path.replace('novelx', 'novel{}'.format(args.split))
    return output_path


def main():
    args = parse_args()
    suffix = 'novel' if args.setting == 'fsod' else 'all'
    template_path = args.template or build_default_template_path(args)

    yaml_info = load_config_file(template_path)
    output_path = infer_output_path(template_path, args)

    if args.dataset in ['voc']:
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                _str_ = '  TRAIN: ("voc_2007_trainval_{}{}_{}shot_seed{}", )\n'
                yaml_info[i] = _str_.format(suffix, args.split, args.shot, args.seed)
            if '  TEST: ' in lineinfo:
                _str_ = '  TEST: ("voc_2007_test_{}{}",)\n'
                yaml_info[i] = _str_.format(suffix, args.split)
    elif args.dataset in ['coco14']:
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                _str_ = '  TRAIN: ("coco14_trainval_{}_{}shot_seed{}", )\n'
                yaml_info[i] = _str_.format(suffix, args.shot, args.seed)
    else:
        raise NotImplementedError

    save_config_file(yaml_info, output_path)


if __name__ == '__main__':
    main()
